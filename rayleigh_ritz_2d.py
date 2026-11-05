"""
2D Radial Schrödinger Equation — Rayleigh–Ritz Variational Solver
  (multi-scale / even-tempered hydrogenic basis)
================================================================

Solves the 2D radial Schrödinger equation for a radial potential V(r) at fixed
angular momentum m, via a Rayleigh–Ritz expansion in 2D hydrogenic radial
functions and a SINGLE generalized eigenvalue problem  H c = E S c.

The basis spans SEVERAL length scales at once (an even-tempered ladder λ_k), so
each state reaches its natural extent while all states remain eigenvectors of one
Hermitian pencil (H, S) and are therefore S-orthonormal: c_a^T S c_b = δ_ab.
This is simultaneously CONVERGED and ORTHONORMAL — see validate_coulomb_rr()
below and App. D of the paper.

Validation: reproduces the exact 2D Coulomb spectrum with relative errors of
order 1e-10 (no variational collapse; App. D 3, Table V of the paper), and the
WSe2/hBN Keldysh Rydberg series in agreement with the finite-difference values of
Stier et al. (arXiv:1709.00123) to ~0.5 meV (App. D 4, Table VI).

TRANSITION MATRIX ELEMENTS (Lindblad use)
-----------------------------------------
In-plane electric-dipole transitions obey the selection rule Δm = ±1 (from
r_± = r e^{±iφ}); they connect states in DIFFERENT m sectors, i.e. different
solver runs. Use the cross-sector helper:

    radial_matrix_element_between(result_s, a, result_p, b, f=lambda r: r)
        -> ∫ ψ_a(r) ψ_b(r) r² dr, the radial factor of <n_a m_a| r_± |n_b m_b>
           (nonzero only when m_a = m_b ± 1; Cartesian x needs an extra 1/2).

Interband (crystal-ground-state -> exciton) brightness is set by the envelope at
zero electron–hole separation, and only m = 0 states are bright:

    envelope_at_origin(result, i)  ->  ψ_{i,0}(0)/√(2π)   (0 for |m| > 0).

The SAME-sector method result.radial_matrix_element(a, b) integrates two states
of ONE m sector (Δm = 0); it is a valid radial integral but NOT a physical
in-plane dipole. result.overlap(a, b) ≈ δ_ab is an orthonormality check.

Units (natural):  μ = κ = ℏ = 1,  a_B = ℏ²/(μκ),  E_0 = μκ²/ℏ².
Basis:  R_{n_r}^{|m|}(r; λ) = N(λ) ξ^|m| e^{-ξ/2} L_{n_r}^{2|m|}(ξ),  ξ = 2r/(ν λ),
        ν = n_r + |m| + 1/2.  Exact 2D Coulomb eigenstates at λ = 1.
"""

import numpy as np
from scipy import special, optimize, linalg
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass, field
from functools import lru_cache
from numpy.polynomial.legendre import leggauss as _leggauss_raw


@lru_cache(maxsize=None)
def _cached_leggauss(n: int):
    """Gauss–Legendre nodes/weights, cached (depend only on n)."""
    return _leggauss_raw(n)


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RRLevelResult:
    """One eigenstate of the multi-scale generalized eigenvalue problem."""
    index: int
    m: int
    energy: float
    coefficients: np.ndarray        # weights in the multi-scale basis (S-orthonormal)
    hydrogenic_nr: int              # n_r of the best single-hydrogenic fit
    hydrogenic_weight: float        # max overlap over the discrete (n_r, λ) search grid


@dataclass
class RRSolverResult:
    """Full multi-scale Rayleigh–Ritz output. Eigenstates are S-orthonormal and
    expressed in the shared basis `basis` = list of (n_r, λ) pairs. Use the
    helper methods for wavefunctions / matrix elements; do not index
    `coefficients` by n_r (the basis mixes several scales)."""
    levels: List[RRLevelResult]
    m: int
    n_basis: int
    scales: np.ndarray
    basis: List[Tuple[int, float]]
    H_matrix: np.ndarray
    S_matrix: np.ndarray
    orthonormality_error: float
    n_effective: int = 0
    potential_name: str = ""

    def energies(self) -> np.ndarray:
        return np.array([lv.energy for lv in self.levels])

    def _basis_on_grid(self, r: np.ndarray) -> np.ndarray:
        return np.array([BasisFunction2D(nr, self.m, lam).eval_R(r)
                         for (nr, lam) in self.basis])

    def wavefunction(self, i: int, r: np.ndarray) -> np.ndarray:
        """Radial wavefunction ψ_i(r) = Σ_k c_{i,k} R_k(r), ∫|ψ_i|² r dr = 1."""
        R = self._basis_on_grid(np.asarray(r, dtype=float))
        return self.levels[i].coefficients @ R

    def _quad_grid(self, n_points: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
        numax = max(nr + abs(self.m) + 0.5 for nr, _ in self.basis)
        return make_rr_grid(n_points=n_points, lam=max(self.scales),
                            nu_max=numax, m_abs=abs(self.m))

    def overlap(self, a: int, b: int, n_points: int = 4000) -> float:
        """<ψ_a|ψ_b> = ∫ ψ_a ψ_b r dr  (≈ δ_ab; orthonormality check)."""
        r, w = self._quad_grid(n_points)
        return float(np.sum(w * r * self.wavefunction(a, r) * self.wavefunction(b, r)))

    def radial_matrix_element(self, a: int, b: int,
                              f: Callable[[np.ndarray], np.ndarray] = lambda r: r,
                              n_points: int = 4000) -> float:
        """<ψ_a| f(r) |ψ_b> = ∫ ψ_a f(r) ψ_b r dr WITHIN this (single-m) sector.

        NOTE: both states share this result's m, so Δm = 0 here. This is a
        radial integral, NOT a physical in-plane electric dipole (which requires
        Δm = ±1 — use radial_matrix_element_between across sectors). Useful for
        intra-sector operators (e.g. <r>, <r²>, form factors).
        """
        r, w = self._quad_grid(n_points)
        pa, pb = self.wavefunction(a, r), self.wavefunction(b, r)
        return float(np.sum(w * r * pa * f(r) * pb))


def make_rr_grid(n_points: int = 1000, lam: float = 1.0,
                 nu_max: float = 20.0, m_abs: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radial quadrature grid for Rayleigh-Ritz.

    Must cover all basis functions simultaneously, from the compact ground
    state (ν_min = |m| + 0.5) to the most extended (ν_max).
    Uses three Gauss-Legendre segments for good coverage:
      [0, r1]        — dense near origin (ground state, cusp for m=0)
      [r1, r2]       — intermediate region
      [r2, r_max]    — far tail for excited states
    """
    nu_min = m_abs + 0.5
    r_char_min = nu_min * lam / 2.0     # smallest decay length
    r_char_max = nu_max * lam / 2.0     # largest decay length

    r1 = max(6.0 * r_char_min, 2.0 * lam)
    r2 = max(8.0 * r_char_max, 20.0 * lam)
    r_max = max(200.0 * r_char_max, 400.0 * lam)

    # Point distribution: heavy near origin for m=0
    if m_abs == 0:
        f1, f2, f3 = 0.45, 0.35, 0.20
    else:
        f1, f2, f3 = 0.25, 0.45, 0.30

    n1 = max(int(f1 * n_points), 80)
    n2 = max(int(f2 * n_points), 80)
    n3 = max(n_points - n1 - n2, 60)

    # Segment 1: [0, r1]
    t1, w1 = _cached_leggauss(n1)
    seg1_r = r1 * (1.0 + t1) / 2.0
    seg1_w = w1 * r1 / 2.0

    # Segment 2: [r1, r2]
    t2, w2 = _cached_leggauss(n2)
    seg2_r = r1 + (r2 - r1) * (1.0 + t2) / 2.0
    seg2_w = w2 * (r2 - r1) / 2.0

    # Segment 3: [r2, r_max]
    t3, w3 = _cached_leggauss(n3)
    seg3_r = r2 + (r_max - r2) * (1.0 + t3) / 2.0
    seg3_w = w3 * (r_max - r2) / 2.0

    r = np.concatenate([seg1_r, seg2_r, seg3_r])
    weights = np.concatenate([seg1_w, seg2_w, seg3_w])
    return r, weights


class BasisFunction2D:
    """
    2D hydrogenic radial basis function R_{n_r}^{|m|}(r; λ).

        R(r; λ) = N(λ) · ξ^|m| · exp(-ξ/2) · L_{n_r}^{2|m|}(ξ)
        ξ = 2r / (ν λ),    ν = n_r + |m| + 1/2

    Normalized: ∫₀^∞ |R|² r dr = 1.
    For Coulomb, exact eigenbasis at λ = a_B = 1.
    """

    def __init__(self, n_r: int, m: int, lam: float):
        self.n_r = n_r
        self.m_abs = abs(m)
        self.lam = lam
        self.nu = n_r + self.m_abs + 0.5
        self.scale = self.nu * lam / 2.0    # ξ = r / scale
        self.alpha = 2 * self.m_abs

    def _log_norm(self) -> float:
        """log(N), with N = (√2/λ) √[n_r! / (ν³ (n_r+2|m|)!)]"""
        n_r, m, nu = self.n_r, self.m_abs, self.nu
        return (
            0.5 * np.log(2.0)
            - np.log(self.lam)
            - 1.5 * np.log(nu)
            + 0.5 * special.gammaln(n_r + 1)
            - 0.5 * special.gammaln(n_r + 2 * m + 1)
        )

    def eval_R(self, r: np.ndarray) -> np.ndarray:
        m = self.m_abs
        xi = r / self.scale
        xi_safe = np.maximum(xi, 1e-300)
        log_N = self._log_norm()
        log_pre = m * np.log(xi_safe) - xi / 2.0 + log_N
        L = special.eval_genlaguerre(self.n_r, self.alpha, xi)
        return np.exp(log_pre) * L

    def eval_R_and_dR(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = self.m_abs
        xi = r / self.scale
        xi_safe = np.maximum(xi, 1e-300)
        log_N = self._log_norm()
        P = np.exp(m * np.log(xi_safe) - xi / 2.0 + log_N)

        L = special.eval_genlaguerre(self.n_r, self.alpha, xi)
        dL = (-special.eval_genlaguerre(self.n_r - 1, self.alpha + 1, xi)
              if self.n_r >= 1 else np.zeros_like(xi))

        R = P * L
        dR = P * ((m / xi_safe - 0.5) * L + dL) / self.scale
        return R, dR


def coulomb_potential(r: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """V(r) = -κ/r. Uses r_safe to avoid division by zero at r=0."""
    r_safe = np.maximum(np.asarray(r, dtype=float), 1e-300)
    return -kappa / r_safe


def soft_coulomb_potential(r: np.ndarray, kappa: float = 1.0,
                           r0: float = 0.1) -> np.ndarray:
    """V(r) = -κ/√(r² + r0²)"""
    return -kappa / np.sqrt(r**2 + r0**2)


def keldysh_potential(r: np.ndarray, kappa: float = 1.0,
                      r0: float = 1.0) -> np.ndarray:
    """
    Rytova-Keldysh potential for 2D semiconductors:
        V(r) = -(κ π)/(2 r0) [H0(r/r0) - Y0(r/r0)]

    Natural-unit usage (this solver): pass kappa=1.0 and r0 = r0_eff, the
    DIMENSIONLESS effective screening length r0_eff = r0_phys / (kappa_env * a_B).
    The physical Coulomb coupling is already absorbed into a_B and E_0, so
    supplying a physical kappa here would double-count it. (The physical
    screening length r0_phys = 2π χ_{2D} must not be passed directly.)
    Uses r_safe to avoid the Y0 singularity at r=0.
    """
    r_safe = np.maximum(np.asarray(r, dtype=float), 1e-300)
    x = r_safe / r0
    H0 = special.struve(0, x)
    Y0 = special.y0(x)
    return -kappa * np.pi / (2.0 * r0) * (H0 - Y0)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-scale Rayleigh–Ritz solver
# ═══════════════════════════════════════════════════════════════════════════════

class RayleighRitzSolver2D:
    """Multi-scale (even-tempered) Rayleigh–Ritz solver for one m sector.

    Parameters
    ----------
    potential : callable         V(r), vectorized, natural units.
    m : int                      angular momentum quantum number.
    n_per_scale : int            radial functions n_r = 0..n_per_scale-1 per scale.
    n_scales : int               number of scales in the ladder.
    scale_span : (float, float)  ladder = geomspace(span[0]·λ_g, span[1]·λ_g,
                                 n_scales), λ_g = MASTER scale (see below). The
                                 default (0.10, 1.40) is deliberately wide so the
                                 diffuse ground state AND the more compact excited
                                 states of every |m| sector are covered.
    n_quad : int                 quadrature points on the shared radial grid.
    svd_threshold : float        RELATIVE cutoff for trimming the near-null space
                                 of the (near-singular) overlap S. NOTE: an
                                 ABSOLUTE floor of 1e-9 is also applied, so
                                 s_cut = max(1e-9, s_max·svd_threshold); values of
                                 svd_threshold below ~1e-9/s_max have no effect.
                                 The floor blocks unstable near-null directions
                                 and should be kept unless a cutoff study warrants
                                 lowering it.

    MASTER SCALE (auto-calibration)
    -------------------------------
    When `scales` is not given, the ladder is centered on λ_g obtained by
    minimizing the m = 0 GROUND-state energy over `ground_bounds`. The m = 0
    scale is a robust material length scale; calibrating in-sector for |m| ≥ 2
    is NOT reliable (the ground energy is flat in λ once many functions are
    included and the minimizer rails to the search boundary, yielding a too-
    compact ladder and spurious unbound states). Centering every sector on the
    m = 0 master scale, with the wide `scale_span`, avoids this. A boundary hit
    during calibration raises RuntimeError so the failure is never silent.
    """

    def __init__(self, potential: Callable, m: int = 0,
                 n_per_scale: int = 6, n_scales: int = 6,
                 scale_span: Tuple[float, float] = (0.10, 1.40),
                 n_quad: int = 3000, svd_threshold: float = 1e-10):
        self.potential = potential
        self.m = m
        self.n_per_scale = n_per_scale
        self.n_scales = n_scales
        self.scale_span = scale_span
        self.n_quad = n_quad
        self.svd_threshold = svd_threshold

    # -- master scale from the m = 0 ground state (robust for every target m) ---
    def _master_scale(self, bounds: Tuple[float, float],
                      n_probe: int = 40, n_basis: int = 20,
                      n_quad: int = 1200) -> float:
        scan = np.geomspace(bounds[0], bounds[1], n_probe)
        costs = [self._single_scale_ground(lam, n_basis, n_quad, m_abs=0)
                 for lam in scan]
        lam_g = float(scan[int(np.argmin(costs))])
        if (np.isclose(lam_g, bounds[0], rtol=0.05) or
                np.isclose(lam_g, bounds[1], rtol=0.05)):
            raise RuntimeError(
                f"master-scale calibration reached a search boundary "
                f"(λ_g={lam_g:.3f} in {bounds}). Widen ground_bounds or pass an "
                f"explicit `scales=` ladder to solve().")
        return lam_g

    def _single_scale_ground(self, lam, n_basis, n_quad, m_abs):
        numax = (n_basis - 1) + m_abs + 0.5
        r, w = make_rr_grid(n_points=n_quad, lam=lam, nu_max=numax, m_abs=m_abs)
        R = np.empty((n_basis, len(r))); dR = np.empty((n_basis, len(r)))
        for i in range(n_basis):
            R[i], dR[i] = BasisFunction2D(i, 0 if m_abs == 0 else m_abs, lam).eval_R_and_dR(r)
        H, S = self._matrices(R, dR, r, w, m_abs=m_abs)
        E = self._geneig(H, S)[0]
        return E[0] if len(E) else np.inf

    def _matrices(self, R, dR, r, w, m_abs=None):
        m = abs(self.m) if m_abs is None else m_abs
        W = w * r
        S = (R * W) @ R.T
        V = (R * (W * self.potential(r))) @ R.T
        T = 0.5 * ((dR * W) @ dR.T)
        if m != 0:
            r_safe = np.maximum(r, 1e-300)
            T = T + (R * (0.5 * m * m * w / r_safe)) @ R.T
        H = 0.5 * ((T + V) + (T + V).T)
        S = 0.5 * (S + S.T)
        return H, S

    def _geneig(self, H, S):
        """H c = E S c via symmetric orthogonalization with SVD trimming.
        Returns (energies, coeffs, n_kept) with coeffs^T S coeffs = I."""
        es, Vs = linalg.eigh(S)
        thr = max(1e-9, es.max() * self.svd_threshold)   # absolute floor 1e-9
        mask = es > thr
        U = Vs[:, mask] / np.sqrt(es[mask])
        Ho = U.T @ H @ U
        Ho = 0.5 * (Ho + Ho.T)
        E, co = linalg.eigh(Ho)
        return E, U @ co, int(np.sum(mask))

    def solve(self, n_states: int = 5, scales: Optional[np.ndarray] = None,
              ground_bounds: Tuple[float, float] = (0.15, 25.0),
              verbose: bool = True) -> RRSolverResult:
        """Compute the lowest `n_states` eigenstates in the multi-scale basis.

        scales : explicit λ ladder; if None, an even-tempered ladder is built
                 around the m = 0 master scale (see class docstring). Passing an
                 explicit ladder is recommended for unusual potentials.
        """
        if n_states < 1:
            raise ValueError("n_states must be >= 1")

        if scales is None:
            lam_g = self._master_scale(ground_bounds)
            scales = np.geomspace(self.scale_span[0] * lam_g,
                                  self.scale_span[1] * lam_g, self.n_scales)
        scales = np.asarray(scales, dtype=float)

        basis = [(nr, lam) for lam in scales for nr in range(self.n_per_scale)]
        m_abs = abs(self.m)
        numax = max(nr + m_abs + 0.5 for nr, _ in basis)
        r, w = make_rr_grid(n_points=self.n_quad, lam=float(scales.max()),
                            nu_max=numax, m_abs=m_abs)
        Nb = len(basis)
        R = np.empty((Nb, len(r))); dR = np.empty((Nb, len(r)))
        for k, (nr, lam) in enumerate(basis):
            R[k], dR[k] = BasisFunction2D(nr, self.m, lam).eval_R_and_dR(r)

        H, S = self._matrices(R, dR, r, w)
        E, C, n_eff = self._geneig(H, S)

        n_ret = min(n_states, C.shape[1])
        Csel = C[:, :n_ret]
        orth = float(np.max(np.abs(Csel.T @ S @ Csel - np.eye(n_ret))))

        levels = []
        for i in range(n_ret):
            hnr, hw = self._hydrogenic_fit(Csel[:, i], basis, i, r, w)
            levels.append(RRLevelResult(
                index=i, m=self.m, energy=float(E[i]),
                coefficients=Csel[:, i], hydrogenic_nr=hnr, hydrogenic_weight=hw))

        result = RRSolverResult(
            levels=levels, m=self.m, n_basis=Nb, scales=scales, basis=basis,
            H_matrix=H, S_matrix=S, orthonormality_error=orth, n_effective=n_eff)
        if verbose:
            self._print_results(result)
        return result

    def _hydrogenic_fit(self, coeffs, basis, index, r, w):
        """(n_r*, weight): best single-hydrogenic fit. `weight` = max overlap over
        the DISCRETE search grid n_r ∈ {i-1,i,i+1}, 30 log-spaced λ — a diagnostic
        of hydrogenic character, not an exact basis-independent observable."""
        psi = np.zeros_like(r)
        for ck, (nr, lam) in zip(coeffs, basis):
            psi += ck * BasisFunction2D(nr, self.m, lam).eval_R(r)
        n = np.sqrt(np.sum(w * r * psi * psi)); psi = psi / n if n > 0 else psi
        best_w, best_nr = 0.0, index
        for nr in range(max(0, index - 1), index + 2):
            for lam in np.geomspace(0.5 * max(basis[0][1], 1e-3), 2.0 * basis[-1][1], 30):
                phi = BasisFunction2D(nr, self.m, lam).eval_R(r)
                phi = phi / np.sqrt(np.sum(w * r * phi * phi))
                ov = np.sum(w * r * psi * phi) ** 2
                if ov > best_w:
                    best_w, best_nr = float(ov), nr
        return best_nr, best_w

    def _print_results(self, result: RRSolverResult):
        print(f"\n{'═'*78}")
        print(f"  Rayleigh–Ritz 2D (multi-scale)  │  m = {result.m}  │  "
              f"basis {result.n_basis} (eff {result.n_effective})  │  "
              f"scales [{result.scales.min():.2f}, {result.scales.max():.2f}]")
        print(f"  max|CᵀSC − I| = {result.orthonormality_error:.1e}")
        print(f"{'═'*78}")
        print(f"  {'idx':>4}  {'E [μκ²/ℏ²]':>18}  {'best n_r':>8}  {'hydrogenic wt':>13}")
        print(f"  {'─'*4}  {'─'*18}  {'─'*8}  {'─'*13}")
        for lv in result.levels:
            print(f"  {lv.index:4d}  {lv.energy:18.12f}  {lv.hydrogenic_nr:8d}  "
                  f"{lv.hydrogenic_weight:13.4f}")
        print(f"{'═'*78}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Transition matrix elements (cross-sector) and interband brightness
# ═══════════════════════════════════════════════════════════════════════════════

def radial_matrix_element_between(result_a, state_a, result_b, state_b,
                                  f: Callable = lambda r: r,
                                  n_points: int = 5000) -> float:
    """Radial factor of a cross-sector matrix element:
        ∫ ψ_{state_a}^{(m_a)}(r) f(r) ψ_{state_b}^{(m_b)}(r) r dr.

    With f(r) = r this is the radial factor of the in-plane electric-dipole
    element <n_a m_a| r_± |n_b m_b>, which is nonzero only for m_a = m_b ± 1
    (enforce the Δm = ±1 selection rule at the call site). For Cartesian x,
    x = (r_+ + r_-)/2, include an extra factor 1/2.

    result_a and result_b are separate solver runs (generally different m).
    """
    m_grid = max(abs(result_a.m), abs(result_b.m))
    numax_a = max(nr + abs(result_a.m) + 0.5 for nr, _ in result_a.basis)
    numax_b = max(nr + abs(result_b.m) + 0.5 for nr, _ in result_b.basis)
    r, w = make_rr_grid(n_points=n_points,
                        lam=max(result_a.scales.max(), result_b.scales.max()),
                        nu_max=max(numax_a, numax_b), m_abs=m_grid)
    psi_a = result_a.wavefunction(state_a, r)
    psi_b = result_b.wavefunction(state_b, r)
    return float(np.sum(w * r * psi_a * f(r) * psi_b))


def envelope_at_origin(result, state: int) -> float:
    """Interband brightness factor Φ_{n0}(0) = ψ_{n0}(0)/√(2π).

    At dipole order the oscillator strength for creating an exciton from the
    crystal ground state is set by the envelope at zero electron–hole
    separation. Only m = 0 states are bright: ψ_{nm}(0) = 0 for |m| > 0, so this
    returns 0.0 for any |m| > 0.
    """
    if result.m != 0:
        return 0.0
    return float(result.wavefunction(state, np.array([0.0]))[0] / np.sqrt(2.0 * np.pi))


def radial_wavefunction(r: np.ndarray, result: RRSolverResult, i: int) -> np.ndarray:
    """Convenience wrapper: ψ_i(r) for eigenstate i of a multi-scale result."""
    return result.wavefunction(i, r)


# ═══════════════════════════════════════════════════════════════════════════════
# Validation and units
# ═══════════════════════════════════════════════════════════════════════════════

def validate_coulomb_rr(m_values: Optional[List[int]] = None, n_states: int = 6):
    """Validate against the exact 2D Coulomb spectrum E_n = -1/[2(n_r+|m|+1/2)²].

    A multi-scale ladder that includes λ = 1 contains the exact eigenbasis, so
    energies are reproduced with relative errors of order 1e-10 (limited by
    quadrature and the overlap cutoff — NOT machine precision) and never fall
    below the exact values (no variational collapse).
    """
    if m_values is None:
        m_values = [0, 1, 2, 3]
    print("\n" + "═" * 80)
    print("  VALIDATION: multi-scale Rayleigh–Ritz  │  2D Coulomb  V(r) = -1/r")
    print("═" * 80)
    all_errors = []
    for m in m_values:
        print(f"\n  ─── m = {m} ───")
        # explicit ladder containing λ = 1 so the exact eigenbasis is spanned
        scales = np.unique(np.concatenate([np.geomspace(0.3, 3.0, 6), [1.0]]))
        result = RayleighRitzSolver2D(potential=coulomb_potential, m=m).solve(
            n_states=n_states, scales=scales, verbose=False)
        worst = 0.0; collapse = False
        for lv in result.levels:
            nu = lv.index + abs(m) + 0.5
            E_exact = -1.0 / (2.0 * nu ** 2)
            dE = abs(lv.energy - E_exact) / abs(E_exact); worst = max(worst, dE)
            if lv.energy < E_exact - 1e-7:
                collapse = True
            all_errors.append(dE)
        print(f"  worst rel. error = {worst:.1e}   collapse = {collapse}   "
              f"max|CᵀSC−I| = {result.orthonormality_error:.1e}")
    w = max(all_errors) if all_errors else 0.0
    print(f"\n  Overall worst relative error: {w:.2e}  (order 1e-10 expected)")
    print(f"  Status: {'PASS ✓' if w < 1e-6 else 'CHECK'}")
    print("═" * 80 + "\n")
    return all_errors


def to_physical_energy(E_natural, mu, kappa, hbar=1.0):
    """E_phys = E_natural × μ κ² / ℏ²."""
    return E_natural * mu * kappa ** 2 / hbar ** 2


def to_physical_length(r_natural, mu, kappa, hbar=1.0):
    """r_phys = r_natural × ℏ² / (μ κ)."""
    return r_natural * hbar ** 2 / (mu * kappa)


if __name__ == "__main__":
    validate_coulomb_rr(m_values=[0, 1, 2, 3], n_states=6)

    print("Example: WSe2/hBN Keldysh (r0_eff = 0.84)")
    Vk = lambda r: keldysh_potential(r, kappa=1.0, r0=0.84)
    E0 = 269.0
    res_s = RayleighRitzSolver2D(potential=Vk, m=0).solve(n_states=4, verbose=False)
    res_p = RayleighRitzSolver2D(potential=Vk, m=1).solve(n_states=4, verbose=False)
    for lv in res_s.levels:
        print(f"  {lv.index+1}s: E_b = {-lv.energy*E0:8.3f} meV")
    for lv in res_p.levels:
        print(f"  {lv.index+2}p: E_b = {-lv.energy*E0:8.3f} meV")

    # Physical in-plane dipole (Δm = ±1): 1s (m=0) -> 2p (m=1)
    I_1s2p = radial_matrix_element_between(res_s, 0, res_p, 0, f=lambda r: r)
    print(f"  radial dipole factor |I(1s,2p)| = {abs(I_1s2p):.4f} a_B  (Δm=±1, allowed)")
    # Same-sector integral is NOT a dipole (Δm = 0):
    print(f"  same-sector radial integral <1s|r|2s> = "
          f"{res_s.radial_matrix_element(0, 1):.4f}  (Δm=0 -> NOT an optical dipole)")
    # Interband brightness (only m = 0 bright):
    print(f"  envelope Φ_1s(0) = {envelope_at_origin(res_s, 0):.4f}  |  "
          f"Φ_2p(0) = {envelope_at_origin(res_p, 0):.4f} (m≠0 dark)")
