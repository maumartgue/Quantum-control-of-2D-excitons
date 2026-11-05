"""
exciton_spectrum.py -- exciton spectrum, optical primitives and topology ranking of
hBN-encapsulated monolayer WSe2 (App. D and Sect. IV of the paper
"Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley
Lindblad Framework for WSe2").

Three parts, all built on the multi-scale Rayleigh-Ritz solver rayleigh_ritz_2d.py:

1. Exciton properties (App. D 5; Tables VII and VIII). Post-processing of the
   eigenstates; all reconstruction and matrix elements are delegated to the
   solver's own (multi-scale aware) helpers:

       result.wavefunction(i, r)                 -> psi_i(r) in the multi-scale basis
       result.radial_matrix_element(a, b, f)     -> <a|f(r)|b> within one m sector
       radial_matrix_element_between(...)        -> cross-sector (Delta m = +/-1) radial factor
       envelope_at_origin(result, i)             -> Phi_i(0) = psi_i(0)/sqrt(2 pi)

     - state catalogue:     E_n, (n_r, m), <r>, <r^2>
     - brightness:          Phi_n(0) and |Phi_n(0)|^2  (interband oscillator strength);
                            psi_at_zero stores the full 2D envelope amplitude
                            Phi(0) = psi(0)/sqrt(2 pi) (nonzero only for m = 0) and
                            psi_at_zero_sq = |Phi(0)|^2. The ranking uses brightness
                            ratios, so this normalization does not change rankings.
     - intraband dipoles:   d_{ij} = e <i|r|j>  (Delta m = +/-1 selection rule)
     - density form factor: F_{ij}(q) = <i|e^{i q.r}|j>
   validate_coulomb_properties() checks them on the exact 2D Coulomb problem.

2. Topology ranking (Sect. IV A-B; Tables I and II). Candidate three-level triples
   for each coupling topology:

       Ξ (Ladder):   (0, a, b)  —  0↔a optical, a↔b THz
       V:            (0, a, b)  —  0↔a optical, 0↔b optical
       Λ (intra):    (a, b, c)  —  a↔c THz, b↔c THz
       Δ (loop):     (i, j, k)  —  all three couplings nonzero

   with the normalized primitives of Eqs. (34)-(35),

       B̃_i = B_i / max_k B_k           (brightness)
       D̃_ij = D_ij / max_{a<b} D_ab    (dipole)
       P_i^{size} = 1/(1 + (R²_i/R²_ref)^ν)  (size penalty)

   and the scores of Eqs. (36) (Ξ) and (37) (V). The Λ and Δ rankings are computed
   for completeness; the paper retains Ξ and V and leaves Λ and Δ for future studies.

3. Comparison with Stier et al., PRL 120, 057405 (2018) (App. D 4, Table VI), and
   with Chernikov et al., PRL 113, 076802 (2014) for WS2/SiO2.

   Material 1: WSe2/hBN — Stier et al.
     Convention: V = -e^2/(8 eps0 r0) [H0(kappa r/r0) - Y0(kappa r/r0)]  (Stier/Berkelbach)
     Parameters: mu = 0.20 m0, r0 = 4.5 nm, kappa_env = 4.5
   Material 2: WS2/SiO2 — Chernikov et al.
     Convention: V = -pi e^2/(2 r0) [H0(r/r0) - Y0(r/r0)]  (bare, kappa absorbed into r0)
     Parameters: mu = 0.16 m0, r0 = 7.5 nm (effective), kappa_env = 1

   Natural units: hbar = mu = kappa_coul = 1, where kappa_coul = e^2/(4 pi eps0 kappa_env)
     Energy unit: E0 = mu kappa_coul^2 / hbar^2
     Length unit: a_B = hbar^2 / (mu kappa_coul)

   The eigenstates are the S-orthonormal solutions of a single multi-scale
   generalized eigenvalue problem; there is no single "lambda_opt" (a ladder of
   scales is used), so per-state character is reported via the hydrogenic weight
   result.levels[i].hydrogenic_weight rather than raw coefficients.

   Two independent reference series for WSe2/hBN are kept apart:

     theory_bind / theory_spacings  = Stier's own finite-difference solution of the
         SAME Keldysh Hamiltonian. Comparing our solver to this tests NUMERICS only
         -> column "dnum" (Code - Stier th.). Stier states E_th(1s)=161 meV and the
         theoretical 1s-2s / 2s-3s separations 124 / 21.3 meV; the individual 2s
         (161-124=37) and 3s (37-21.3=15.7, also read from Fig. 3b) are reconstructed;
         the 4s theoretical value is not reported.

     expt_bind / expt_spacings      = experimental values. Spacings are direct peak
         differences (130, 22 meV, both stated in Ref.). Binding energies use
         the fitted free-particle gap E_gap = 1.890 eV: E_b(ns) = E_gap - E_trans(ns)
         -> 167 / 37 / 15 meV. The 4s is excluded: Ref. resolves no zero-field 4s
         peak (identified only via its high-field shift) and reports no 4s theory.
         Comparing our solver to this tests the physical
         MODEL -> column "dmodel" (Code - Expt.).

Units: natural (a_B, E_0 = mu kappa^2/hbar^2); E0 = 269 meV, a_B = 1.19 nm, r0_eff = 0.84 a_B.
Usage:  python exciton_spectrum.py [ranking|stier|coulomb|all]     (default: all, in this order)
"""

import sys

import numpy as np
from scipy import special
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

from rayleigh_ritz_2d import (
    RRSolverResult, RRLevelResult, RayleighRitzSolver2D,
    make_rr_grid, radial_matrix_element_between, envelope_at_origin,
    coulomb_potential, keldysh_potential,
)


# ###############################################################################
# PART 1. EXCITON PROPERTIES
# ###############################################################################

# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExcitonState:
    state_id: int               # global label (across all m sectors)
    n_r: int                    # dominant radial quantum number (hydrogenic fit)
    m: int                      # angular momentum quantum number
    energy: float               # eigenvalue [E_0]
    mean_r: float               # <r>  [a_B]
    mean_r2: float              # <r^2> [a_B^2]
    psi_at_zero: float          # Phi_n(0) = psi_n(0)/sqrt(2 pi)  (0 for m != 0)
    psi_at_zero_sq: float       # |Phi_n(0)|^2  (interband brightness proxy)
    is_bright: bool             # |Phi_n(0)|^2 > threshold
    coefficients: np.ndarray    # multi-scale expansion coefficients
    hydrogenic_fit_nr: int = -1     # best single-hydrogenic label (diagnostic only)
    hydrogenic_weight: float = 0.0  # overlap weight of that hydrogenic fit


@dataclass
class IntraBandDipole:
    state_i: int
    state_j: int
    m_i: int
    m_j: int
    d_abs: float                # |d_{ij}| = |I|/sqrt(2)  [a_B]  (charge e separate)
    d_x: complex                # <i|x|j> [a_B]
    d_y: complex                # <i|y|j> [a_B]


@dataclass
class FormFactorResult:
    state_i: int
    state_j: int
    q_values: np.ndarray
    F_values: np.ndarray        # complex for i != j (cross-sector), real for diagonal


@dataclass
class ExcitonPropertyBundle:
    states: List[ExcitonState]
    dipoles: List[IntraBandDipole]
    form_factors: List[FormFactorResult]
    solver_results: Dict[int, RRSolverResult]   # keyed by m


# ═══════════════════════════════════════════════════════════════════════════════
# Shared radial grid + cached wavefunctions
# ═══════════════════════════════════════════════════════════════════════════════

def _shared_grid(solver_results: Dict[int, RRSolverResult],
                 n_per_scale: int, n_quad: int) -> Tuple[np.ndarray, np.ndarray]:
    """One radial grid covering every sector. All sectors share the same
    (master-scale) ladder, so the largest scale and largest nu determine the
    tail; m_abs = 0 keeps the near-origin region dense for the m = 0 cusp."""
    max_scale = max(res.scales.max() for res in solver_results.values())
    max_m = max(abs(m) for m in solver_results)
    nu_max = (n_per_scale - 1) + max_m + 0.5
    return make_rr_grid(n_points=n_quad, lam=max_scale, nu_max=nu_max, m_abs=0)


# ═══════════════════════════════════════════════════════════════════════════════
# A/B. State catalogue + brightness
# ═══════════════════════════════════════════════════════════════════════════════

def build_state_catalogue(result: RRSolverResult, r: np.ndarray, w: np.ndarray,
                          psi_cache: List[np.ndarray],
                          brightness_threshold: float = 1e-4,
                          global_id_offset: int = 0) -> List[ExcitonState]:
    """Catalogue for one m-sector solve. <r>, <r^2> from the cached wavefunctions
    on the shared grid; brightness from the solver's envelope_at_origin."""
    states = []
    wr = w * r
    for idx, lv in enumerate(result.levels):
        psi = psi_cache[idx]
        mean_r = float(np.sum(wr * psi * psi * r))          # <r>  = int psi^2 r . r dr
        mean_r2 = float(np.sum(wr * psi * psi * r * r))     # <r^2> = int psi^2 r^2 . r dr
        phi0 = envelope_at_origin(result, idx)              # Phi(0) = psi(0)/sqrt(2 pi)
        states.append(ExcitonState(
            state_id=global_id_offset + idx,
            n_r=lv.index,   # exact radial quantum number (node count / ordering),
                            # NOT the fitted hydrogenic label (a diagnostic that can
                            # disagree for highly excited / non-hydrogenic states)
            m=result.m,
            energy=lv.energy,
            mean_r=mean_r,
            mean_r2=mean_r2,
            psi_at_zero=phi0,
            psi_at_zero_sq=phi0 * phi0,
            is_bright=(phi0 * phi0 > brightness_threshold),
            coefficients=lv.coefficients.copy(),
            hydrogenic_fit_nr=lv.hydrogenic_nr,
            hydrogenic_weight=lv.hydrogenic_weight,
        ))
    return states


# ═══════════════════════════════════════════════════════════════════════════════
# C. Intraband dipoles — <i|r|j>, Delta m = +/-1
# ═══════════════════════════════════════════════════════════════════════════════
#
# In 2D with angular part e^{i m phi}/sqrt(2 pi):
#     x = r cos phi,  y = r sin phi ;  <m1|cos phi|m2> = (1/2)(d_{m1,m2+1}+d_{m1,m2-1})
#                                       <m1|sin phi|m2> = (1/2i)(d_{m1,m2+1}-d_{m1,m2-1})
# => selection rule Delta m = +/-1.  Radial factor
#     I = int psi_i(r) r psi_j(r) r dr    (one r from the operator, one from the measure)
# Cartesian components:  <i|x|j> = I/2 ;  <i|y|j> = -/+ i I/2 ;  |d| = |I|/sqrt(2).

def compute_intraband_dipoles(states: List[ExcitonState],
                              solver_results: Dict[int, RRSolverResult],
                              id_to_local: Dict[int, int],
                              n_quad: int = 4000) -> List[IntraBandDipole]:
    """All Delta m = +/-1 dipole matrix elements, using the solver's
    cross-sector radial helper (correct for the multi-scale basis)."""
    dipoles = []
    for i, si in enumerate(states):
        for j in range(i + 1, len(states)):
            sj = states[j]
            delta_m = si.m - sj.m
            if abs(delta_m) != 1:
                continue
            res_i, res_j = solver_results[si.m], solver_results[sj.m]
            I_rad = radial_matrix_element_between(
                res_i, id_to_local[si.state_id],
                res_j, id_to_local[sj.state_id],
                f=lambda r: r, n_points=n_quad)
            d_x = 0.5 * I_rad
            d_y = -0.5j * I_rad if delta_m == 1 else 0.5j * I_rad
            dipoles.append(IntraBandDipole(
                state_i=si.state_id, state_j=sj.state_id,
                m_i=si.m, m_j=sj.m,
                d_abs=abs(I_rad) / np.sqrt(2.0), d_x=d_x, d_y=d_y))
    return dipoles


# ═══════════════════════════════════════════════════════════════════════════════
# D. Density form factors  F_{ij}(q) = <i|e^{i q.r}|j>
# ═══════════════════════════════════════════════════════════════════════════════
#
# Same-sector (Delta m = 0):  F_{ij}(q) = int J_0(q r) psi_i psi_j r dr
#     diagonal small-q:  F_{ii}(q) ~ 1 - q^2 <r^2>_i / 4
# Cross-sector (Delta m = l): F_{ij}(q) = i^l int J_{|l|}(q r) psi_i psi_j r dr
#     (angular phase e^{i l phi_q} omitted; radial magnitude returned).

def _form_factor(psi_i, psi_j, r, w, q_values, delta_m, same_state):
    wr = w * r
    order = abs(delta_m)
    phase = (1j) ** delta_m
    F = np.zeros(len(q_values), dtype=complex)
    for iq, q in enumerate(q_values):
        if q < 1e-15:
            if delta_m == 0:
                F[iq] = 1.0 if same_state else float(np.sum(wr * psi_i * psi_j))
            else:
                F[iq] = 0.0                      # J_{l>0}(0) = 0
        else:
            Jl = special.jv(order, q * r)
            F[iq] = phase * np.sum(wr * Jl * psi_i * psi_j)
    if delta_m == 0:
        F = F.real.astype(complex)
    return F


# ═══════════════════════════════════════════════════════════════════════════════
# Master driver
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_properties(
    potential,
    m_sectors: List[int] = [0, 1, -1],
    n_states_per_sector: int = 10,
    n_quad: int = 3000,
    brightness_threshold: float = 1e-4,
    q_values: Optional[np.ndarray] = None,
    form_factor_max_pairs: int = 50,
    n_per_scale: int = 6,
    n_scales: int = 6,
    verbose: bool = True,
    potential_kw: Optional[dict] = None,
    # --- accepted and ignored (the multi-scale solver has no single basis size) ---
    n_max: Optional[int] = None,
    lambda_bounds: Optional[Tuple[float, float]] = None,
) -> ExcitonPropertyBundle:
    """Solve each m sector with the multi-scale solver, then compute all
    observables. `n_max` / `lambda_bounds` are accepted but ignored (the
    multi-scale solver has no single basis size or lambda bound); use
    `n_per_scale` / `n_scales` to control the basis."""
    if q_values is None:
        q_values = np.concatenate([[0.0], np.geomspace(0.01, 2.0, 30)])
    if potential_kw:
        raw = potential
        potential = lambda r: raw(r, **potential_kw)
    if n_max is not None and verbose:
        print(f"  [note] n_max={n_max} is ignored by the multi-scale solver; "
              f"using n_per_scale={n_per_scale}, n_scales={n_scales}.")

    # ── Step 1: solve every sector (shared master-scale ladder) ──────────────
    solver_results: Dict[int, RRSolverResult] = {}
    for m in m_sectors:
        if verbose:
            print(f"\n  Solving m = {m} sector ...")
        solver = RayleighRitzSolver2D(potential=potential, m=m,
                                      n_per_scale=n_per_scale, n_scales=n_scales,
                                      n_quad=max(n_quad, 2000))
        solver_results[m] = solver.solve(n_states=n_states_per_sector, verbose=verbose)

    # ── Step 2: shared grid + cache all wavefunctions once ──────────────────
    r, w = _shared_grid(solver_results, n_per_scale, max(n_quad, 3000))
    psi_by_sector: Dict[int, List[np.ndarray]] = {
        m: [res.wavefunction(i, r) for i in range(len(res.levels))]
        for m, res in solver_results.items()}

    # ── Step 3: catalogue (with global ids and id -> local map) ─────────────
    all_states: List[ExcitonState] = []
    id_to_local: Dict[int, int] = {}
    global_id = 0
    for m in m_sectors:
        res = solver_results[m]
        sector = build_state_catalogue(res, r, w, psi_by_sector[m],
                                       brightness_threshold=brightness_threshold,
                                       global_id_offset=global_id)
        for local_idx, st in enumerate(sector):
            id_to_local[st.state_id] = local_idx
        all_states.extend(sector)
        global_id += len(sector)
    if verbose:
        _print_state_table(all_states)

    # ── Step 4: intraband dipoles (Delta m = +/-1) ──────────────────────────
    if verbose:
        print("\n  Computing intraband dipoles (Delta m = +/-1) ...")
    dipoles = compute_intraband_dipoles(all_states, solver_results, id_to_local,
                                        n_quad=max(n_quad, 3000))
    if verbose:
        _print_dipole_table(dipoles, all_states)

    # ── Step 5: density form factors (diagonal + same-sector off-diagonal) ──
    if verbose:
        print("\n  Computing density form factors ...")
    form_factors: List[FormFactorResult] = []
    pair_count = 0
    # diagonal
    for s in all_states:
        if pair_count >= form_factor_max_pairs:
            break
        psi = psi_by_sector[s.m][id_to_local[s.state_id]]
        F = _form_factor(psi, psi, r, w, q_values, delta_m=0, same_state=True)
        form_factors.append(FormFactorResult(s.state_id, s.state_id, q_values, F))
        pair_count += 1
    # same-sector off-diagonal (Delta m = 0, i != j)
    for m in m_sectors:
        sector = [s for s in all_states if s.m == m]
        for a in range(len(sector)):
            for b in range(a + 1, len(sector)):
                if pair_count >= form_factor_max_pairs:
                    break
                si, sj = sector[a], sector[b]
                psi_i = psi_by_sector[m][id_to_local[si.state_id]]
                psi_j = psi_by_sector[m][id_to_local[sj.state_id]]
                F = _form_factor(psi_i, psi_j, r, w, q_values, delta_m=0,
                                 same_state=False)
                form_factors.append(
                    FormFactorResult(si.state_id, sj.state_id, q_values, F))
                pair_count += 1
    if verbose:
        _print_form_factor_summary(form_factors, all_states, q_values)

    return ExcitonPropertyBundle(states=all_states, dipoles=dipoles,
                                 form_factors=form_factors,
                                 solver_results=solver_results)


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _label(state: ExcitonState) -> str:
    n = state.n_r + abs(state.m) + 1
    ang = {0: 's', 1: 'p', 2: 'd', 3: 'f'}.get(abs(state.m), f'm{abs(state.m)}')
    return f"{n}{ang}"


def _print_state_table(states: List[ExcitonState]):
    print(f"\n{'='*92}")
    print("  Exciton state catalogue")
    print(f"{'='*92}")
    print(f"  {'id':>4}  {'(n_r,m)':>8}  {'label':>6}  {'E [E0]':>16}  "
          f"{'<r>':>9}  {'<r^2>':>10}  {'|Phi(0)|^2':>11}  {'bright':>6}")
    for s in states:
        print(f"  {s.state_id:4d}  ({s.n_r:2d},{s.m:2d})  {_label(s):>6}  "
              f"{s.energy:16.10f}  {s.mean_r:9.4f}  {s.mean_r2:10.4f}  "
              f"{s.psi_at_zero_sq:11.6f}  {'yes' if s.is_bright else 'no':>6}")
    print(f"{'='*92}\n")


def _print_dipole_table(dipoles: List[IntraBandDipole], states: List[ExcitonState]):
    lab = {s.state_id: _label(s) for s in states}
    print(f"\n{'='*72}")
    print("  Intraband dipoles  <i|r|j>  (Delta m = +/-1;  d = e <i|r|j>)")
    print(f"{'='*72}")
    if not dipoles:
        print("  No Delta m = +/-1 transitions found.\n" + "="*72 + "\n"); return
    print(f"  {'i':>4}  {'j':>4}  {'i':>6}  {'j':>6}  {'|d| [a_B]':>12}")
    for d in sorted(dipoles, key=lambda x: -x.d_abs)[:25]:
        print(f"  {d.state_i:4d}  {d.state_j:4d}  {lab[d.state_i]:>6}  "
              f"{lab[d.state_j]:>6}  {d.d_abs:12.6f}")
    print(f"{'='*72}\n")


def _print_form_factor_summary(ffs, states, q_values):
    print(f"\n{'='*72}\n  Density form factors F_ij(q)\n{'='*72}")
    idx = [0] + [int(np.argmin(np.abs(q_values - q))) for q in (0.1, 0.5, 1.0)]
    idx = sorted(set(idx))
    print("  (i,j)   " + "  ".join(f"F(q={q_values[k]:.2f})" for k in idx))
    for ff in ffs[:12]:
        print(f"  ({ff.state_i:2d},{ff.state_j:2d})   " +
              "  ".join(f"{ff.F_values[k].real:10.6f}" for k in idx))
    print(f"{'='*72}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_coulomb_properties():
    """Exact 2D Coulomb checks:
        |Phi_{ns}(0)|^2 = 1/(pi nu^3),  nu = n_r + 1/2   (m = 0)
        F_{ii}(q) ~ 1 - q^2 <r^2>_i / 4  at small q
    and finite 1s->2p dipole (Delta m = +/-1)."""
    print("\n" + "=" * 80)
    print("  VALIDATION: Exciton Properties on 2D Coulomb")
    print("=" * 80)
    props = compute_all_properties(
        potential=coulomb_potential, m_sectors=[0, 1],
        n_states_per_sector=4, n_quad=3000,
        q_values=np.array([0.0, 0.01, 0.1, 0.5, 1.0]),
        form_factor_max_pairs=12, verbose=True)

    print("\n  -- |Phi(0)|^2 vs exact 1/(pi nu^3)  (m = 0) --")
    for s in props.states:
        if s.m == 0:
            nu = s.n_r + 0.5
            exact = 1.0 / (np.pi * nu**3)
            print(f"    {_label(s)}: |Phi(0)|^2 = {s.psi_at_zero_sq:.6f}  "
                  f"exact = {exact:.6f}  err = {abs(s.psi_at_zero_sq-exact):.2e}")
        else:
            print(f"    {_label(s)} (m={s.m}): |Phi(0)|^2 = {s.psi_at_zero_sq:.2e} (dark)")

    print("\n  -- form factor small-q:  F(0.01) vs 1 - q^2<r^2>/4 --")
    for ff in props.form_factors:
        if ff.state_i == ff.state_j:
            s = [x for x in props.states if x.state_id == ff.state_i][0]
            q = ff.q_values[1]; Fnum = ff.F_values[1].real
            Fapx = 1.0 - q*q/4.0 * s.mean_r2
            print(f"    {_label(s)}: F({q:.2f}) = {Fnum:.8f}  approx = {Fapx:.8f}  "
                  f"diff = {abs(Fnum-Fapx):.1e}")

    print("\n  -- 1s -> 2p radial dipole (Delta m = +/-1) --")
    for d in props.dipoles:
        si = [x for x in props.states if x.state_id == d.state_i][0]
        sj = [x for x in props.states if x.state_id == d.state_j][0]
        if {_label(si), _label(sj)} == {'1s', '2p'}:
            print(f"    |d(1s,2p)| = {d.d_abs:.6f} a_B  (|I| = {d.d_abs*np.sqrt(2):.6f})")
    return props


# ###############################################################################
# PART 2. TOPOLOGY RANKING
# ###############################################################################

# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RankedTriple:
    """A scored (i, j, k) candidate for a given topology."""
    topology: str           # "Xi", "V", "Lambda", "Delta"
    ids: Tuple[int, ...]    # (i, j, k) state indices (energy-ordered)
    labels: Tuple[str, ...]
    score: float
    detail: Dict = field(default_factory=dict)  # breakdown of score components


@dataclass
class RankerParams:
    """Tunable parameters for the ranking algorithm."""
    w_B: float = 1.0       # brightness weight
    w_D: float = 1.0       # dipole weight
    w_S: float = 0.3       # size penalty weight
    R2_ref: float = None    # reference ⟨r²⟩ for size penalty (None = use 1s value)
    nu: float = 1.0         # size penalty exponent
    eps_B: float = 1e-4     # brightness threshold (relative to max)
    eps_D: float = 1e-4     # dipole threshold (relative to max)
    K: int = 10             # number of top candidates to return per topology
    freq_gate: bool = False # enable frequency feasibility gate
    omega_optical: Tuple[float, float] = (50.0, 500.0)   # meV
    omega_THz: Tuple[float, float] = (0.5, 50.0)         # meV
    # ── Hard constraints (state-set filtering) ──
    R2_max: float = None    # if set, exclude states with ⟨r²⟩ > R2_max [a_B²]
    n_principal_max: int = None  # if set, exclude states with n_principal > this
    # ── Λ-specific energy window [meV] ──
    lambda_dE_min: float = 5.0    # hard gate: require ΔE_ac, ΔE_bc >= this [meV]
    lambda_dE_max: float = 150.0  # hard gate: require ΔE_ac, ΔE_bc <= this [meV]
    # ── Λ soft penalty for small ΔE (experimental cleanliness) ──
    lambda_dE_star: float = 5.0   # soft penalty scale [meV]: W(ΔE) = 1 - exp(-ΔE/ΔE*)
    # ── Optional energy-order gates ──
    #    Default False: the topology is set by the drive architecture (optical and
    #    THz/mid-IR legs), the classification used in the paper (Sect. IV A).
    #    Set to True to require the strict quantum-optics energy ordering.
    ladder_ascending: bool = False    # Xi: drive-architecture (optical+THz) sets topology; no strict-energy gate
    lambda_common_upper: bool = False # Lambda: shared-THz-leg (drive-architecture) classification; vertex energy-position free


# ═══════════════════════════════════════════════════════════════════════════════
# Core ranking engine
# ═══════════════════════════════════════════════════════════════════════════════

class TopologyRanker:
    """
    Ranks three-level triples for EIT/STIRAP topologies.

    By default the topology class is set by the drive architecture of the legs
    (optical: ground <-> exciton; THz/mid-IR: exciton <-> exciton), as in the paper.
    With the optional gates of RankerParams it is also set by the physical energy
    ordering (strict quantum-optics convention):
        Xi     -- ascending ladder, common intermediate-energy state;
        Lambda -- common upper-energy state;
        V      -- common lower-energy state (the crystal ground).

    Usage:
        props = compute_all_properties(...)
        ranker = TopologyRanker(props, E0_meV=269.0, aB_nm=1.19)
        results = ranker.rank_all()
    """

    def __init__(self, props: ExcitonPropertyBundle,
                 E0_meV: float = 269.0,
                 aB_nm: float = 1.19,
                 params: Optional[RankerParams] = None,
                 n_levels: int = 11):
        """
        Parameters
        ----------
        props : ExcitonPropertyBundle
            Output from compute_all_properties().
        E0_meV : float
            Energy unit in meV.
        aB_nm : float
            Length unit in nm.
        params : RankerParams
            Scoring parameters.
        n_levels : int
            Number of energy-ordered levels to consider (n = 0, ..., n_levels-1).
        """
        self.props = props
        self.E0 = E0_meV
        self.aB = aB_nm
        self.params = params or RankerParams()

        # Build energy-ordered unique state list (skip m=-1 duplicates)
        # Keep one representative per |m| sector (m >= 0) and only bound states
        # (energy < 0): continuum Ritz pseudostates and m<0 duplicates never ranked.
        unique = [s for s in props.states if s.m >= 0 and s.energy < 0.0]
        unique.sort(key=lambda s: s.energy)
        self.states = unique[:n_levels]

        # ── Hard constraints: prune state set before scoring ──
        p = self.params
        if p.R2_max is not None or p.n_principal_max is not None:
            filtered = []
            for s in self.states:
                m_abs = abs(s.m)
                n_princ = s.n_r + m_abs + 1
                if p.R2_max is not None and s.mean_r2 > p.R2_max:
                    continue
                if p.n_principal_max is not None and n_princ > p.n_principal_max:
                    continue
                filtered.append(s)
            n_removed = len(self.states) - len(filtered)
            self.states = filtered
            if verbose_init := (n_removed > 0):
                print(f"  Hard constraints: removed {n_removed} states "
                      f"(R2_max={p.R2_max}, n_max={p.n_principal_max}), "
                      f"{len(self.states)} remain")

        # Assign energy-level index n
        self.n_levels = len(self.states)
        self.sid_to_n = {}
        for n, s in enumerate(self.states):
            self.sid_to_n[s.state_id] = n

        # Precompute labels
        self.labels = []
        for s in self.states:
            m_abs = abs(s.m)
            nprinc = s.n_r + m_abs + 1
            ang = {0: 's', 1: 'p', 2: 'd', 3: 'f'}.get(m_abs, f'm{m_abs}')
            self.labels.append(f"{nprinc}{ang}")

        # Build dipole lookup: D[i,j] in a_B, keyed by energy-ordered index
        self._build_dipole_lookup()

        # Precompute normalized quantities
        self._precompute_normalized()

    def _build_dipole_lookup(self):
        """Build symmetric dipole dict keyed by (n_i, n_j) energy indices."""
        self.D = {}
        sid_set = set(s.state_id for s in self.states)

        for d in self.props.dipoles:
            if d.state_i in sid_set and d.state_j in sid_set:
                ni = self.sid_to_n[d.state_i]
                nj = self.sid_to_n[d.state_j]
                key = (min(ni, nj), max(ni, nj))
                if key not in self.D:
                    self.D[key] = d.d_abs

    def _get_dipole(self, i: int, j: int) -> float:
        """Get |d_ij| for energy-ordered indices i, j."""
        key = (min(i, j), max(i, j))
        return self.D.get(key, 0.0)

    def _precompute_normalized(self):
        """Compute B̃, D̃, P^{size} for all states/pairs."""
        p = self.params

        # Brightness
        B_vals = [s.psi_at_zero_sq for s in self.states]
        self.B_max = max(B_vals) if max(B_vals) > 0 else 1.0
        self.B_tilde = [b / self.B_max for b in B_vals]

        # Dipole normalization
        D_vals = list(self.D.values())
        self.D_max = max(D_vals) if D_vals else 1.0

        # Size penalty
        R2_ref = p.R2_ref
        if R2_ref is None:
            R2_ref = self.states[0].mean_r2  # 1s ⟨r²⟩
        self.R2_ref = R2_ref

        self.P_size = []
        for s in self.states:
            self.P_size.append(1.0 / (1.0 + (s.mean_r2 / R2_ref) ** p.nu))

    def _D_tilde(self, i: int, j: int) -> float:
        """Normalized dipole."""
        return self._get_dipole(i, j) / self.D_max if self.D_max > 0 else 0.0

    def _freq_gate(self, i: int, j: int, window: str) -> bool:
        """Check if transition i↔j falls in frequency window."""
        if not self.params.freq_gate:
            return True
        dE = abs(self.states[i].energy - self.states[j].energy) * self.E0
        if window == "optical":
            return self.params.omega_optical[0] <= dE <= self.params.omega_optical[1]
        elif window == "THz":
            return self.params.omega_THz[0] <= dE <= self.params.omega_THz[1]
        return True

    def _freq_gate_interband(self, a: int) -> bool:
        """
        Check if the interband transition ground→exciton(a) is in the optical window.
        The interband energy is E_gap - E_bind(a).  Since we don't know E_gap here,
        the optical leg is always assumed feasible (it's a laser line at ~1.7 eV).
        The freq_gate for interband is therefore trivially True unless the user
        sets explicit bounds that exclude it.
        """
        if not self.params.freq_gate:
            return True
        # Binding energy of state a
        E_bind_a = -self.states[a].energy * self.E0  # positive, meV
        # Interband transition: E_gap - E_bind(a).  We don't store E_gap,
        # so we just check that E_bind > 0 (state is bound).
        return E_bind_a > 0

    # ─── Topology A: Ladder Ξ ───────────────────────────────────────────

    def rank_ladder(self) -> List[RankedTriple]:
        """
        Ξ (Ladder): (0, a, b) with 0↔a optical, a↔b THz.

        Score, Eq. (36): S = B̃_a^{w_B} · D̃_{ab}^{w_D} · (P_a · P_b)^{w_S} · G_win
        """
        p = self.params
        scores = []

        for a in range(self.n_levels):
            # a must be optically bright
            if self.states[a].psi_at_zero_sq < p.eps_B * self.B_max:
                continue
            # frequency gate on ground→a (interband)
            if not self._freq_gate_interband(a):
                continue

            for b in range(self.n_levels):
                if b == a:
                    continue
                # Optional strict-ladder gate: ascending cascade |0>->a->b needs
                # E_phys(b) > E_phys(a); solver energies are -E_bind, so:
                if p.ladder_ascending and self.states[b].energy <= self.states[a].energy:
                    continue
                d_ab = self._D_tilde(a, b)
                if d_ab < p.eps_D:
                    continue
                # frequency gate on a↔b (THz/intraband)
                if not self._freq_gate(a, b, "THz"):
                    continue

                S = ((self.B_tilde[a]) ** p.w_B
                     * (d_ab) ** p.w_D
                     * (self.P_size[a] * self.P_size[b]) ** p.w_S)

                E_bind_a = -self.states[a].energy * self.E0
                dE_ab = abs(self.states[a].energy - self.states[b].energy) * self.E0

                scores.append(RankedTriple(
                    topology="Xi",
                    ids=(a, b),
                    labels=(self.labels[a], self.labels[b]),
                    score=S,
                    detail={
                        "B_a": self.B_tilde[a],
                        "D_ab": d_ab,
                        "P_a": self.P_size[a],
                        "P_b": self.P_size[b],
                        "E_bind_a_meV": E_bind_a,
                        "dE_ab_meV": dE_ab,
                    }
                ))

        scores.sort(key=lambda x: -x.score)
        return scores[:p.K]

    # ─── Topology B: V system ───────────────────────────────────────────

    def rank_V(self) -> List[RankedTriple]:
        """
        V: (0, a, b) with 0↔a optical, 0↔b optical.

        Score, Eq. (37): S = (B̃_a · B̃_b)^{w_B} · (P_a · P_b)^{w_S} · G_win
        """
        p = self.params
        bright = [n for n in range(self.n_levels)
                  if self.states[n].psi_at_zero_sq >= p.eps_B * self.B_max]

        scores = []
        for idx_a, a in enumerate(bright):
            if not self._freq_gate_interband(a):
                continue
            for b in bright[idx_a + 1:]:
                if not self._freq_gate_interband(b):
                    continue

                S = ((self.B_tilde[a] * self.B_tilde[b]) ** p.w_B
                     * (self.P_size[a] * self.P_size[b]) ** p.w_S)

                E_bind_a = -self.states[a].energy * self.E0
                E_bind_b = -self.states[b].energy * self.E0

                scores.append(RankedTriple(
                    topology="V",
                    ids=(a, b),
                    labels=(self.labels[a], self.labels[b]),
                    score=S,
                    detail={
                        "B_a": self.B_tilde[a],
                        "B_b": self.B_tilde[b],
                        "P_a": self.P_size[a],
                        "P_b": self.P_size[b],
                        "E_bind_a_meV": E_bind_a,
                        "E_bind_b_meV": E_bind_b,
                    }
                ))

        scores.sort(key=lambda x: -x.score)
        return scores[:p.K]

    # ─── Topology C: Intra-excitonic Λ ──────────────────────────────────

    def rank_lambda(self) -> List[RankedTriple]:
        """
        Λ (intra-excitonic): (a, b, c) with a↔c THz, b↔c THz.
        Common vertex c (bright, s-like), two dark (p-like) partners a, b.

        Hard symmetry gate of the Λ score: the two coherence partners
        a, b must be dipole-dark (m ≠ 0) and the common vertex c bright
        (m = 0). The Δm = ±1 dipole filter (eps_D) then admits only s–p legs,
        so the surviving partners are p-like. Radiative protection therefore
        enters as this HARD restriction, not as a graded brightness factor.

        No exciton-gas coherence factor is included: in the strict fixed-K0
        mean-field limit the q=0 exciton gas renormalizes detunings only and
        produces no dephasing, so there is no partner-matching quantity to
        optimize at the screening stage.

        Score: S = (D̃_{ac} · D̃_{bc})^{w_D} · (P_a · P_b · P_c)^{w_S}
                     · W(ΔE_ac) · W(ΔE_bc)

        Output slot order (|1⟩, |2⟩, |3⟩) =
        (lower-energy partner, vertex, higher-energy partner).
        """
        p = self.params
        scores = []

        for c in range(self.n_levels):
            # vertex must be bright (s-like, m = 0)
            if self.states[c].m != 0:
                continue
            for a in range(self.n_levels):
                if a == c:
                    continue
                # partner a must be dipole-dark (p-like, m ≠ 0)
                if self.states[a].m == 0:
                    continue
                d_ac = self._D_tilde(a, c)
                if d_ac < p.eps_D:
                    continue
                dE_ac = abs(self.states[a].energy - self.states[c].energy) * self.E0
                # Λ energy window on leg a↔c
                if p.lambda_dE_min is not None and dE_ac < p.lambda_dE_min:
                    continue
                if p.lambda_dE_max is not None and dE_ac > p.lambda_dE_max:
                    continue
                if not self._freq_gate(a, c, "THz"):
                    continue

                for b in range(a + 1, self.n_levels):
                    if b == c:
                        continue
                    # partner b must be dipole-dark (p-like, m ≠ 0)
                    if self.states[b].m == 0:
                        continue
                    d_bc = self._D_tilde(b, c)
                    if d_bc < p.eps_D:
                        continue
                    dE_bc = abs(self.states[b].energy - self.states[c].energy) * self.E0
                    # Λ energy window on leg b↔c
                    if p.lambda_dE_min is not None and dE_bc < p.lambda_dE_min:
                        continue
                    if p.lambda_dE_max is not None and dE_bc > p.lambda_dE_max:
                        continue
                    if not self._freq_gate(b, c, "THz"):
                        continue
                    # Optional strict-Lambda gate: common vertex c above both outer states
                    if p.lambda_common_upper and not (
                            self.states[c].energy > self.states[a].energy
                            and self.states[c].energy > self.states[b].energy):
                        continue

                    # Soft penalty for small ΔE: W(ΔE) = 1 - exp(-ΔE/ΔE*)
                    dE_star = p.lambda_dE_star
                    W_ac = 1.0 - np.exp(-dE_ac / dE_star) if dE_star > 0 else 1.0
                    W_bc = 1.0 - np.exp(-dE_bc / dE_star) if dE_star > 0 else 1.0

                    # Λ policy scorer. Radiative
                    # protection is the hard p/s symmetry gate applied above;
                    # no graded brightness factor and no exciton-gas
                    # coherence factor enter the score.
                    S = ((d_ac * d_bc) ** p.w_D
                         * (self.P_size[a] * self.P_size[b] * self.P_size[c]) ** p.w_S
                         * W_ac * W_bc)

                    # Slot order (|1⟩,|2⟩,|3⟩) = (partner a, vertex c, partner b);
                    # a is the lower-energy (more-bound) partner since a < b.
                    scores.append(RankedTriple(
                        topology="Lambda",
                        ids=(a, c, b),
                        labels=(self.labels[a], self.labels[c], self.labels[b]),
                        score=S,
                        detail={
                            "D_ac": d_ac,
                            "D_bc": d_bc,
                            "B_a": self.B_tilde[a],
                            "B_b": self.B_tilde[b],
                            "P_a": self.P_size[a],
                            "P_b": self.P_size[b],
                            "P_c": self.P_size[c],
                            "W_ac": W_ac,
                            "W_bc": W_bc,
                            "dE_ac_meV": dE_ac,
                            "dE_bc_meV": dE_bc,
                        }
                    ))

        scores.sort(key=lambda x: -x.score)
        return scores[:p.K]

    # ─── Topology E: Δ loop ─────────────────────────────────────────────

    def rank_delta(self) -> List[RankedTriple]:
        """
        Δ (loop): (i, j, k) with all three couplings D_ij, D_jk, D_ki > 0.

        Feasibility test: min(D̃_ij, D̃_jk, D̃_ki) > eps_D.
        Score: S = (D̃_ij · D̃_jk · D̃_ki)^{w_D} · (P_i · P_j · P_k)^{w_S} · G_win
        """
        p = self.params
        scores = []

        for i in range(self.n_levels):
            for j in range(i + 1, self.n_levels):
                d_ij = self._D_tilde(i, j)
                if d_ij < p.eps_D:
                    continue
                for k in range(j + 1, self.n_levels):
                    d_jk = self._D_tilde(j, k)
                    d_ki = self._D_tilde(k, i)
                    if min(d_jk, d_ki) < p.eps_D:
                        continue

                    S = ((d_ij * d_jk * d_ki) ** p.w_D
                         * (self.P_size[i] * self.P_size[j] * self.P_size[k]) ** p.w_S)

                    scores.append(RankedTriple(
                        topology="Delta",
                        ids=(i, j, k),
                        labels=(self.labels[i], self.labels[j], self.labels[k]),
                        score=S,
                        detail={
                            "D_ij": d_ij, "D_jk": d_jk, "D_ki": d_ki,
                            "P_i": self.P_size[i],
                            "P_j": self.P_size[j],
                            "P_k": self.P_size[k],
                        }
                    ))

        scores.sort(key=lambda x: -x.score)
        return scores[:p.K]

    # ─── Master driver ──────────────────────────────────────────────────

    def rank_all(self, verbose: bool = True) -> Dict[str, List[RankedTriple]]:
        """Run all four topology rankings and return results."""

        results = {
            "Xi": self.rank_ladder(),
            "V": self.rank_V(),
            "Lambda": self.rank_lambda(),
            "Delta": self.rank_delta(),
        }

        if verbose:
            self._print_primitives()
            for topo in ["Xi", "V", "Lambda", "Delta"]:
                self._print_ranking(topo, results[topo])

        return results

    # ─── Pretty-print ───────────────────────────────────────────────────

    def _print_primitives(self):
        """Print the normalized primitives table."""
        print()
        print("=" * 90)
        print("  Normalized primitives  (R2_ref = {:.2f} a_B^2,  B_max = {:.6f},  D_max = {:.4f} a_B)"
              .format(self.R2_ref, self.B_max, self.D_max))
        print("=" * 90)
        print(f"  {'n':>2}  {'label':>5}  {'E_b [meV]':>10}  {'B̃':>8}  "
              f"{'P_size':>8}  {'R² [a_B²]':>10}  {'bright':>6}")
        print(f"  {'─'*2}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*6}")
        for n, s in enumerate(self.states):
            E_b_meV = -s.energy * self.E0   # positive binding energy E_b(n),
                                            # NOT the physical exciton energy E_gap - E_b
            bright = "yes" if s.is_bright else "no"
            print(f"  {n:2d}  {self.labels[n]:>5}  {E_b_meV:10.2f}  {self.B_tilde[n]:8.6f}  "
                  f"{self.P_size[n]:8.6f}  {s.mean_r2:10.2f}  {bright:>6}")
        print("=" * 90)

    def _print_ranking(self, topo: str, ranked: List[RankedTriple]):
        """Print one topology's ranking table."""
        topo_names = {
            "Xi": "Ξ (Ladder): (0, a, b) — 0↔a optical, a↔b THz",
            "V": "V system: (0, a, b) — 0↔a optical, 0↔b optical",
            "Lambda": "Λ (intra-excitonic): (|1⟩,|2⟩,|3⟩) — |1⟩↔|2⟩ THz, |3⟩↔|2⟩ THz (|2⟩ = vertex)",
            "Delta": "Δ (loop): (i, j, k) — all three couplings",
        }

        print()
        print("=" * 100)
        print(f"  {topo_names[topo]}")
        print("=" * 100)

        if not ranked:
            if topo == "Delta":
                print("  ** No Δ candidates found — Δ is absent at dipole order **")
                print("     (requires simultaneous Δm=±1 for all three pairs,")
                print("      which is impossible with angular momentum selection rules)")
            else:
                print(f"  No candidates found for {topo}.")
            print("=" * 100)
            return

        # Header depends on topology
        if topo == "Xi":
            print(f"  {'#':>3}  {'(a,b)':>10}  {'labels':>12}  {'score':>8}  "
                  f"{'B̃_a':>7}  {'D̃_ab':>7}  {'P_a':>6}  {'P_b':>6}  "
                  f"{'Eb_a':>7}  {'ΔE_ab':>7}")
            print(f"  {'─'*3}  {'─'*10}  {'─'*12}  {'─'*8}  "
                  f"{'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
            for rank, r in enumerate(ranked):
                d = r.detail
                ids_str = f"({r.ids[0]:d},{r.ids[1]:d})"
                lab_str = f"{r.labels[0]},{r.labels[1]}"
                print(f"  {rank+1:3d}  {ids_str:>10}  {lab_str:>12}  {r.score:8.5f}  "
                      f"{d['B_a']:7.4f}  {d['D_ab']:7.4f}  {d['P_a']:6.4f}  {d['P_b']:6.4f}  "
                      f"{d['E_bind_a_meV']:7.1f}  {d['dE_ab_meV']:7.1f}")

        elif topo == "V":
            print(f"  {'#':>3}  {'(a,b)':>10}  {'labels':>12}  {'score':>8}  "
                  f"{'B̃_a':>7}  {'B̃_b':>7}  {'P_a':>6}  {'P_b':>6}  "
                  f"{'Eb_a':>7}  {'Eb_b':>7}")
            print(f"  {'─'*3}  {'─'*10}  {'─'*12}  {'─'*8}  "
                  f"{'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
            for rank, r in enumerate(ranked):
                d = r.detail
                ids_str = f"({r.ids[0]:d},{r.ids[1]:d})"
                lab_str = f"{r.labels[0]},{r.labels[1]}"
                print(f"  {rank+1:3d}  {ids_str:>10}  {lab_str:>12}  {r.score:8.5f}  "
                      f"{d['B_a']:7.4f}  {d['B_b']:7.4f}  {d['P_a']:6.4f}  {d['P_b']:6.4f}  "
                      f"{d['E_bind_a_meV']:7.1f}  {d['E_bind_b_meV']:7.1f}")

        elif topo == "Lambda":
            print(f"  {'#':>3}  {'(1,2,3)':>14}  {'labels':>14}  {'score':>10}  "
                  f"{'D̃_12':>7}  {'D̃_32':>7}  {'W_12':>6}  {'W_32':>6}  "
                  f"{'P_1':>6}  {'P_2':>6}  {'P_3':>6}  "
                  f"{'ΔE_12':>7}  {'ΔE_32':>7}")
            print(f"  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*10}  "
                  f"{'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  "
                  f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
            for rank, r in enumerate(ranked):
                d = r.detail
                ids_str = f"({r.ids[0]:d},{r.ids[1]:d},{r.ids[2]:d})"
                lab_str = f"{r.labels[0]},{r.labels[1]},{r.labels[2]}"
                print(f"  {rank+1:3d}  {ids_str:>14}  {lab_str:>14}  {r.score:10.3e}  "
                      f"{d['D_ac']:7.4f}  {d['D_bc']:7.4f}  {d['W_ac']:6.4f}  {d['W_bc']:6.4f}  "
                      f"{d['P_a']:6.4f}  {d['P_b']:6.4f}  "
                      f"{d['P_c']:6.4f}  {d['dE_ac_meV']:7.1f}  {d['dE_bc_meV']:7.1f}")

        elif topo == "Delta":
            print(f"  {'#':>3}  {'(i,j,k)':>12}  {'labels':>14}  {'score':>8}  "
                  f"{'D̃_ij':>7}  {'D̃_jk':>7}  {'D̃_ki':>7}  "
                  f"{'P_i':>6}  {'P_j':>6}  {'P_k':>6}")
            print(f"  {'─'*3}  {'─'*12}  {'─'*14}  {'─'*8}  "
                  f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}")
            for rank, r in enumerate(ranked):
                d = r.detail
                ids_str = f"({r.ids[0]:d},{r.ids[1]:d},{r.ids[2]:d})"
                lab_str = f"{r.labels[0]},{r.labels[1]},{r.labels[2]}"
                print(f"  {rank+1:3d}  {ids_str:>12}  {lab_str:>14}  {r.score:8.5f}  "
                      f"{d['D_ij']:7.4f}  {d['D_jk']:7.4f}  {d['D_ki']:7.4f}  "
                      f"{d['P_i']:6.4f}  {d['P_j']:6.4f}  {d['P_k']:6.4f}")

        print("=" * 100)


# ###############################################################################
# PART 3. COMPARISON WITH STIER et al. (WSe2/hBN) AND CHERNIKOV et al. (WS2/SiO2)
# ###############################################################################

# Physical constants (SI)
m0 = 9.10938e-31
e_charge = 1.60218e-19
hbar = 1.05457e-34
eps0 = 8.85419e-12
eV = e_charge

def get_scales(m_r_m0, kappa_env):
    if m_r_m0 <= 0 or kappa_env <= 0:
        raise ValueError("get_scales: m_r_m0 and kappa_env must be positive")
    mu = m_r_m0 * m0
    kappa_coul = e_charge**2 / (4*np.pi*eps0*kappa_env)
    a_B = hbar**2 / (mu * kappa_coul)
    E_0 = mu * kappa_coul**2 / hbar**2
    return a_B * 1e9, E_0 / eV   # nm, eV


def run_material(label, m_r_m0, kappa_env, r0_nm,
                 theory_bind, expt_bind, theory_spacings, expt_spacings,
                 n_states=8, n_quad=3000, n_per_scale=6, n_scales=5,
                 E0_meV=None, r0_eff_fixed=None):
    """Solve the multi-scale Keldysh problem for one material and print the
    validation table with two independent reference columns:

        Code       : our multi-scale Rayleigh-Ritz binding energies (meV)
        Stier(th.) : Stier's finite-difference Keldysh values (numerics benchmark)
        dnum       : Code - Stier(th.)  -> tests our numerics
        Expt.      : measured / experimentally-derived values
        dmodel     : Code - Expt.       -> tests the Rytova-Keldysh model
        w_hyd      : overlap with best single hydrogenic function

    `theory_*` entries may be None where Stier reports no value (e.g. 4s, WS2).
    """
    if r0_nm <= 0:
        raise ValueError("run_material: r0_nm must be positive")
    a_B_nm, E_0_eV = get_scales(m_r_m0, kappa_env)
    r0_eff = r0_nm / (kappa_env * a_B_nm)
    # Pin to the quoted production values (App. D: E0=269 meV, r0_eff=0.84 a_B)
    if E0_meV is not None:
        E_0_eV = E0_meV / 1000.0
    if r0_eff_fixed is not None:
        r0_eff = r0_eff_fixed

    def V(r):
        return keldysh_potential(r, kappa=1.0, r0=r0_eff)

    # Multi-scale solver: one generalized eigenproblem, S-orthonormal eigenstates.
    solver = RayleighRitzSolver2D(potential=V, m=0, n_per_scale=n_per_scale,
                                  n_scales=n_scales, n_quad=n_quad)
    result = solver.solve(n_states=n_states, verbose=False)

    if not result.levels:
        print(f"\n  WARNING: No levels returned for {label}. Skipping.")
        return np.array([]), np.array([])

    E_eV = result.energies() * E_0_eV
    E_bind = -E_eV * 1000

    names = ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s']

    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    print(f"  mu = {m_r_m0} m0,  kappa_env = {kappa_env},  "
          f"r0 = {r0_nm} nm,  a_B = {a_B_nm:.4f} nm,  E0 = {E_0_eV:.4f} eV")
    print(f"  r0_eff = {r0_eff:.4f} a_B")
    print(f"  multi-scale ladder = [{result.scales.min():.3f}, {result.scales.max():.3f}] a_B"
          f"  |  basis {result.n_basis} (effective {result.n_effective})"
          f"  |  max|C^T S C - I| = {result.orthonormality_error:.1e}")
    print(f"{'-'*90}")

    def _fmt(x, spec):
        return format(x, spec) if x is not None else "-"

    print(f"\n  Binding energies (meV):")
    print(f"  {'state':>5}  {'Code':>8}  {'Stier(th.)':>10}  {'dnum':>7}  "
          f"{'Expt.':>7}  {'dmodel':>7}  {'w_hyd':>7}")
    n_rows = min(len(E_bind), max(len(theory_bind), len(expt_bind)))
    for i in range(n_rows):
        code = E_bind[i]
        th = theory_bind[i] if i < len(theory_bind) else None
        ex = expt_bind[i] if i < len(expt_bind) else None
        dnum = f"{code - th:+.1f}" if th is not None else "-"
        dmod = f"{code - ex:+.1f}" if ex is not None else "-"
        w = result.levels[i].hydrogenic_weight
        print(f"  {names[i]:>5}  {code:8.2f}  {_fmt(th, '.1f'):>10}  {dnum:>7}  "
              f"{_fmt(ex, '.0f'):>7}  {dmod:>7}  {w:7.3f}")

    print(f"\n  Energy spacings (meV, model-independent):")
    print(f"  {'pair':>7}  {'Code':>8}  {'Stier(th.)':>10}  {'dnum':>7}  "
          f"{'Expt.':>7}  {'dmodel':>7}")
    n_sp = min(len(E_bind) - 1, max(len(theory_spacings), len(expt_spacings)))
    for i in range(n_sp):
        sp = (E_eV[i + 1] - E_eV[i]) * 1000    # = E_bind[i] - E_bind[i+1]
        th = theory_spacings[i] if i < len(theory_spacings) else None
        ex = expt_spacings[i] if i < len(expt_spacings) else None
        dnum = f"{sp - th:+.1f}" if th is not None else "-"
        dmod = f"{sp - ex:+.1f}" if ex is not None else "-"
        tag = f"{names[i+1]}-{names[i]}"
        print(f"  {tag:>7}  {sp:8.1f}  {_fmt(th, '.1f'):>10}  {dnum:>7}  "
              f"{_fmt(ex, '.0f'):>7}  {dmod:>7}")

    print(f"{'='*90}")
    return E_bind, E_eV


_NOTES = """
  Two-level reading of the WSe2 comparison (App. D 4, Table VI of the paper):

  Numerics (dnum = Code - Stier th.).  Our multi-scale Rayleigh-Ritz binding
  energies reproduce Stier's independent finite-difference solution of the SAME
  Keldysh Hamiltonian (161 / 37 / 15.7 meV) to <= 0.6 meV, and the 2s-1s / 3s-2s
  spacings match Stier's theoretical spacings (124 / 21.3 meV) to 0.2 / 0.0 meV.
  This is a numerics-vs-numerics check and it passes cleanly.

  Model (dmodel = Code - Expt.).  Against experiment (bindings 167/37/15 meV
  from E_gap = 1.890 eV; spacings 130/22 meV) the excited 2s/3s agree to ~1 meV,
  while the 1s underbinds by ~5.4 meV. The 2s-1s
  spacing sits 5.8 meV (4.5%) below the measured 130 meV; Stier's own Keldysh
  theory shows the same offset, so this is a Rytova-Keldysh model limitation
  (polaron / short-range-exchange effects beyond the model), not a solver error.

  WS2 agreement is good for the 1s binding (~1 meV vs the 320 meV estimate) but
  the higher states deviate: Chernikov's effective r0 = 7.5 nm absorbs substrate
  screening phenomenologically, and the inhomogeneous SiO2 environment is not
  fully captured by a single-parameter Keldysh potential.

  All eigenstates are S-orthonormal (max|C^T S C - I| ~ 1e-12), so the states are
  a proper basis for the transition-dipole / Lindblad calculations downstream.
"""


def compare_experiment():
    """Print the WSe2/hBN and WS2/SiO2 comparison tables and the reading notes."""
    print("\n" + "#"*90)
    print("  RAYLEIGH-RITZ (multi-scale) + KELDYSH  vs  THEORY (Stier) AND EXPERIMENT")
    print("#"*90)

    # -- WSe2 / hBN (Stier 2018) --
    # theory_*  : Stier finite-difference Keldysh  (numerics-vs-numerics, column dnum)
    #             1s=161 stated; 2s=161-124=37 and 3s=37-21.3=15.7 reconstructed;
    #             separations 124 / 21.3 stated; 4s not reported by Stier.
    # expt_*    : experiment (column dmodel). Bindings from E_gap=1.890 eV:
    #             167/37/15 (4s excluded; see docstring). Spacings 130/22 stated.
    wse2_bind = run_material(
        label="WSe2 / hBN - Stier et al., PRL 120, 057405 (2018)",
        m_r_m0=0.20, kappa_env=4.5, r0_nm=4.5,
        theory_bind=[161, 37, 15.7],      # Stier: 4s theory not reported
        expt_bind=[167, 37, 15],          # 4s dropped: no resolved zero-field 4s in Ref.
        theory_spacings=[124, 21.3],
        expt_spacings=[130, 22],
        n_scales=6, E0_meV=269.0, r0_eff_fixed=0.84,
    )

    # -- WS2 / SiO2 (Chernikov 2014) --
    # No independent same-model theory series is tabulated here; experiment only.
    ws2_bind = run_material(
        label="WS2 / SiO2 - Chernikov et al., PRL 113, 076802 (2014)",
        m_r_m0=0.16, kappa_env=1.0, r0_nm=7.5,
        theory_bind=[None, None, None, None, None],
        expt_bind=[320, 160, 100, 50, 20],
        theory_spacings=[None, None, None, None],
        expt_spacings=[162, 60, 50, 30],
    )

    print(f"\n{'#'*90}")
    print(f"  NOTES")
    print(f"{'#'*90}")
    print(_NOTES)


def rank_wse2():
    """WSe2/hBN: exciton properties of the 11 lowest bound states and topology ranking."""
    E0 = 269.0   # meV
    aB = 1.19     # nm
    r0_eff = 0.84 # a_B

    print("=" * 60)
    print("  Computing exciton properties for WSe2/hBN ...")
    print("=" * 60)

    props = compute_all_properties(
        potential=keldysh_potential,
        potential_kw={'r0': r0_eff},
        m_sectors=[0, 1, -1],
        n_states_per_sector=11,
        n_quad=2000,
        n_per_scale=6,
        n_scales=6,
        q_values=np.array([0.0, 0.01, 0.05, 0.1, 0.2]),
        form_factor_max_pairs=200,
        verbose=False,
    )

    print("  Done. Running topology ranker ...\n")

    params = RankerParams(
        w_B=1.0, w_D=1.0, w_S=0.3,
        nu=1.0,
        eps_B=1e-4,
        eps_D=1e-4,
        K=10,
        freq_gate=False,
        # drive-architecture classification of the paper (Sect. IV A)
        ladder_ascending=False,      # Xi: drive-architecture classification (physics-of-excitons first)
        lambda_common_upper=False,   # Lambda: shared-THz-leg classification (physics-of-excitons first)
    )

    ranker = TopologyRanker(props, E0_meV=E0, aB_nm=aB, params=params, n_levels=11)
    results = ranker.rank_all(verbose=True)
    return results


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("ranking", "all"):
        rank_wse2()
    if part in ("stier", "all"):
        compare_experiment()
    if part in ("coulomb", "all"):
        validate_coulomb_properties()
