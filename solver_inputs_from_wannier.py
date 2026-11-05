"""
===============================================================================
 solver_inputs_from_wannier.py
 Wannier-solver-derived inputs for the Xi and V Lindblad solvers
 (paper "Dark-State Coherent Control of Excitons in TMD Monolayers:
  A Reduced Intravalley Lindblad Framework for WSe2")
===============================================================================

PURPOSE
-------
Every number the Lindblad solver needs that follows from the exciton model is
computed here from the Rayleigh-Ritz Wannier solver and written to a single
parameter file (params.yaml). This script does not run the Lindblad solver
and does not integrate any master equation.

It computes:
  1. Wannier spectrum and matrix elements (App. D; Tables VII-VIII):
     binding energies, |phi_n(0)|^2, intraband dipoles |<n|x|m>|.
  2. Bohr frequencies of the retained legs (Sect. IV C 3).
  3. Photon-reservoir rates (App. C 2, Eqs. C22, C24-C26) at T_gamma = 4, 77,
     300 K. Downward and upward, interband and intra-excitonic; the
     intra-excitonic rate uses the baseline normalization area A.
  4. Exciton-gas exchange vertices zeta_ab (App. D 6, Eqs. D8-D10; Table IX),
     read from the vertex file written by exchange_mean_field.py
     (like-carrier part, electron-hole Coulomb term of the carrier exchange
     with the Hermitian prescription, and interband exchange), and the diagonal
     self-energies Sigma_a(n_X) = n_X zeta_aa (Eq. 62).
  5. Laser calibration: carrier photon energies that put every renormalized
     detuning at zero (Delta~ = 0) at a calibration density, and the
     renormalized detunings produced when the actual density differs
     (n_X calibration-mismatch scan).
     Detunings are written in the SIGN CONVENTION OF THE PAPER, Eqs. (65),
     (72): Delta~_ab = Delta_ab + [Sigma(...)]/hbar, so that lasers calibrated
     at n_cal and actual density n_X give
         hbar*Delta~ = Sigma(n_X) - Sigma(n_cal)   (per leg / two-photon).
  6. Rabi-frequency -> field conversion (Eqs. 44-47; A6) at the baseline
     area A, for reporting fields and intensities. The solver input is hbar*Omega.
  7. Spectral-selectivity bounds (Sect. IV C 1).
  8. Dimensionless (natural-unit) versions of all solver inputs: energies in
     E0, times in tau0 = hbar/E0, rates as gamma*tau0.

The fixed protocol choices (envelopes, scans, temperatures, numerical
controls) are written into the "protocol" block of params.yaml, so that
params.yaml is the single source of truth for the solver. Every scan grid is
written out as an explicit list, so the Lindblad solver reads grids and does
not rebuild them.

PROTOCOL (Sect. VI, Table III)
--------
  * Xi conversion with a WEAK optical leg,
    hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2) (the weak-probe
    rule), and the THz leg hbar*Omega23 scanned over 1-5 meV (baseline 5 meV).
    The CPT plateau uses the same weak optical leg with hbar*Omega23 = 1 meV.
  * The one-exciton Fock space of Sect. III is kept; validity is checked by
    max_t(1 - rho11) << 1.
  * Case 3 (exchange mean field with bare-tuned lasers) at n_X = 1e11 cm^-2.

DEPENDENCIES
------------
  rayleigh_ritz_2d.py   -- the Rayleigh-Ritz Wannier solver
  vertices_keldysh_200_8000_250_100.json -- exchange vertices, written by
                           exchange_mean_field.py
  numpy, scipy, pyyaml

  2p-ionization coefficient: computed separately by the ionization estimate
  of analytic_estimates.py (it needs continuum states, which are outside the
  bound-state RR basis). Its result is entered here as a constant.

CONVENTIONS
-----------
  Natural units of App. D: E0 = 269 meV, a_B = 1.19 nm.
  Intraband dipole: d_nm = |<n|x|m_x>| = |I_nm|/sqrt(2) (Table VIII convention).
  Interband dipole: |d_cv|^2 = e^2 hbar^2 / (2 mu Eg) (Eq. A6, band edge),
  with co-circular polarization so that |e . d_cv| = |d_cv|.

USAGE
-----
  python solver_inputs_from_wannier.py            # writes params.yaml + report
  python solver_inputs_from_wannier.py --out my_params.yaml
===============================================================================
"""
import argparse
import datetime
import json
import sys

import numpy as np
import yaml
from scipy import constants as C

from rayleigh_ritz_2d import (RayleighRitzSolver2D, keldysh_potential,
                              radial_matrix_element_between, envelope_at_origin)

VERTEX_FILE = "vertices_keldysh_200_8000_250_100.json"

# =============================================================================
# 0. MATERIAL AND MODEL PARAMETERS  (App. D 4)
# =============================================================================
MAT = dict(
    material="hBN-encapsulated monolayer WSe2",
    mu_m0=0.20,          # reduced mass (Stier et al. 2018), used by the Wannier problem
    kappa_env=4.5,       # environmental dielectric constant
    r0_nm=4.5,           # Rytova-Keldysh screening length
    Eg_meV=1890.0,       # quasiparticle gap (Stier et al. 2018)
    me_m0=0.39,          # Jin et al. PRB 90, 045422 (2014), Table I
    mh_m0=0.51,
    E0_meV=269.0,        # natural energy unit (value of the paper)
    aB_nm=1.19,          # natural length unit (value of the paper)
    r0_eff=0.84,         # r0 / (kappa_env a_B), natural units
    A_um2=np.pi * 1.75**2,   # baseline area: 3.5 um fibre core of Stier et al., PRL 120, 057405 (2018)
)
T_LIST = [4.0, 77.0, 300.0]                    # K; 4 K baseline
NX_CAL = [2.5e10, 5e10, 1e11]                  # nominal densities, cm^-2
NX_DEV = [-0.10, -0.05, 0.0, +0.05, +0.10]     # actual/nominal - 1
OMEGA_REPORT_meV = [0.05, 0.08, 0.1, 0.5, 1.0, 1.5, 3.0, 4.0, 5.0]   # for field/intensity tables
KAPPA_ION = 0.304    # hbar*Gamma_ion = KAPPA_ION (hbar*Omega23)^2 / E0; Eq. (57),
                     # from the ionization estimate of analytic_estimates.py (S_d(E_f)=0.141 a_B^2/E0)

# Protocol choices of the paper (Sect. VI, Table III). Fixed, not computed.
PROTO = dict(
    # Xi conversion grids (weak optical leg; THz leg scanned)
    xi_delta_meV=dict(start=-1.0, stop=1.0, step=0.2),         # 11 points incl. 0
    xi_Omega23_meV=dict(start=1.0, stop=5.0, step=0.5),        # {1.0, 1.5, ..., 5.0}
    xi_delay_over_sigma=dict(start=1.0, stop=1.5, step=0.25),  # {1.0, 1.25, 1.5}
    xi_sigma_ps=[1.1, 1.5, 1.9],
    xi_sigma_base_ps=1.5, xi_delay_base=1.25, xi_Omega23_base_meV=5.0,
    # Xi response (EIT)
    eit_Omega23_meV=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.6, 2.0, 3.0, 4.0, 5.0],
    eit_claim_max_meV=1.5, eit_crossover_meV=1.6,
    eit_probe_factor=0.05,                                     # weak-optical-leg rule
    probe_Delta12=dict(span=15.0, inner=1.0, d_inner=0.01, d_outer=0.1),
    # V response
    v_Omega12_meV=0.1, v_Omega13_meV=0.01,
    v_delta=dict(span=5.0, inner=1.0, d_inner=0.01, d_outer=0.1),
    # V beat
    vb_sigma_ps=[0.093, 0.12, 0.15, 0.18], vb_sigma_base_ps=0.12, vb_Omega_meV=1.0,
    # cases 1-3 at the baseline (Figs. 3-4); case-3 density; CPT plateau
    pa_nX_case3_cm2=1e11, pa_A4_Omega23_meV=1.0,
    # elevated-temperature runs
    eit_Omega23_elevated_T_meV=1.0,
)

# States retained or used for diagnostics: label -> (m-sector, index in sector)
STATE = {'1s': (0, 0), '2s': (0, 1), '3s': (0, 2), '4s': (0, 3),
         '2p': (1, 0), '3p': (1, 1), '4p': (1, 2)}

HB_meVps = C.hbar / C.e * 1e3 * 1e12           # hbar in meV ps (0.6582...)
KB_meV_K = C.k / C.e * 1e3                     # k_B in meV/K


# =============================================================================
# helpers
# =============================================================================
def n_bose(hw_meV, T):
    """Bose-Einstein photon occupation; exactly 0 below double precision."""
    x = hw_meV / (KB_meV_K * T)
    return 0.0 if x > 700.0 else float(1.0 / np.expm1(x))


def f(x, nd=6):
    """float with fixed significant digits, for YAML readability."""
    return float(f"{x:.{nd}g}")


# =============================================================================
# 1. WANNIER SPECTRUM AND MATRIX ELEMENTS
# =============================================================================
def solve_wannier():
    Vk = lambda r: keldysh_potential(r, kappa=1.0, r0=MAT["r0_eff"])
    res = {0: RayleighRitzSolver2D(Vk, m=0).solve(n_states=6, verbose=False),
           1: RayleighRitzSolver2D(Vk, m=1).solve(n_states=5, verbose=False)}
    E0 = MAT["E0_meV"]
    Eb = {l: -res[m].levels[i].energy * E0 for l, (m, i) in STATE.items()}
    phi0sq = {l: envelope_at_origin(res[0], i) ** 2          # a_B^-2
              for l, (m, i) in STATE.items() if m == 0}

    def d(a, b):   # |<a|x|b_x>| = |I_ab| / sqrt(2), a in m=0, b in m=1
        return abs(radial_matrix_element_between(res[0], STATE[a][1],
                                                 res[1], STATE[b][1])) / np.sqrt(2)
    dip = {"1s-2p": d('1s', '2p'), "1s-3p": d('1s', '3p'),
           "2s-2p": d('2s', '2p'), "2s-3p": d('2s', '3p'), "3s-2p": d('3s', '2p')}
    orth = {m: res[m].orthonormality_error for m in res}
    return Eb, phi0sq, dip, orth


# =============================================================================
# 2-3. BOHR FREQUENCIES AND PHOTON-RESERVOIR RATES
# =============================================================================
def radiative_rates(Eb, phi0sq, dip):
    E0, aB = MAT["E0_meV"], MAT["aB_nm"] * 1e-9
    Eg = MAT["Eg_meV"]
    # Bohr energies (bare Wannier; Sect. IV C 3). Rates are sampled at the
    # physical Bohr frequencies of H0 (App. B); the mean-field shift of the
    # interband line (a few meV of 1728 meV) changes them by < 0.2 %.
    hw = {"0-1s": Eg - Eb['1s'], "0-2s": Eg - Eb['2s'], "0-3s": Eg - Eb['3s'],
          "1s-2p": Eb['1s'] - Eb['2p'], "2p-2s": Eb['2p'] - Eb['2s'],
          "1s-2s": Eb['1s'] - Eb['2s'], "1s-3p": Eb['1s'] - Eb['3p']}

    dcv2 = C.e**2 * C.hbar**2 / (2 * MAT["mu_m0"] * C.m_e * Eg * 1e-3 * C.e)   # C^2 m^2
    pref = 1.0 / (C.hbar * C.epsilon_0 * C.c)

    def w(hw_meV):
        return hw_meV * 1e-3 * C.e / C.hbar          # rad/s

    # interband, K0 = 0: gamma = g_n omega (n+1), g_n = |phi_n(0)|^2 |d_cv|^2/(hbar eps0 c)
    g = {l: phi0sq[l] / aB**2 * dcv2 * pref for l in ('1s', '2s', '3s')}
    A = MAT["A_um2"] * 1e-12
    # intra-excitonic, polarization-aligned: gamma = |D|^2 omega /(hbar eps0 c A) (n+1)
    Dintra2 = (C.e * dip["1s-2p"] * aB) ** 2
    Domit2 = (C.e * dip["2s-2p"] * aB) ** 2

    rates = {}
    for T in T_LIST:
        n1 = n_bose(hw["0-1s"], T); n2 = n_bose(hw["0-2s"], T)
        ni = n_bose(hw["1s-2p"], T); no = n_bose(hw["2p-2s"], T)
        g1s = g['1s'] * w(hw["0-1s"]); g2s = g['2s'] * w(hw["0-2s"])
        gi = Dintra2 * w(hw["1s-2p"]) * pref / A
        rates[T] = {
            # Xi: |1>=0, |2>=1s, |3>=2p ;  V: |1>=0, |2>=1s, |3>=2s   (s^-1)
            "gamma_1s_to_0": g1s * (n1 + 1), "gamma_0_to_1s": g1s * n1,
            "gamma_2s_to_0": g2s * (n2 + 1), "gamma_0_to_2s": g2s * n2,
            "gamma_2p_to_1s": gi * (ni + 1), "gamma_1s_to_2p": gi * ni,
            "n_gamma": {"0-1s": n1, "0-2s": n2, "1s-2p": ni, "2p-2s": no},
            # Sect. V item 6: omitted 2p->2s upward channel / retained 2p->1s downward
            "omitted_2p_2s_ratio": (Domit2 * hw["2p-2s"]) / (Dintra2 * hw["1s-2p"])
                                   * no / (1 + ni),
        }
    extras = {"dcv2_C2m2": dcv2, "g_1s": g['1s'], "g_2s": g['2s'],
              "gammaA_2p1s_m2_per_s": Dintra2 * w(hw["1s-2p"]) * pref}
    return hw, rates, extras


# =============================================================================
# 4. EXCHANGE VERTICES AND SELF-ENERGIES
# =============================================================================
def exchange_vertices():
    """Read the exchange vertices (E0 aB^2): like-carrier (cc_r), electron-hole
    Coulomb term of the carrier exchange (cc_a, Hermitian prescription), interband
    exchange (eh), total. The in/out (prior/post) totals are kept for the
    off-diagonal accuracy statement."""
    V = json.load(open(VERTEX_FILE))["vertices"]
    pairs = [('1s', '1s'), ('2s', '2s'), ('2p', '2p'), ('3p', '3p'),
             ('1s', '2s'), ('2p', '3p'), ('1s', '2p')]
    z = {}
    for a, b in pairs:
        v = V[f"{a},{b}"]
        tot = v["ccr"] + v["cca_herm"] + v["eh"]
        z[f"{a},{b}"] = {"cc_r": v["ccr"], "cc_a": v["cca_herm"], "eh": v["eh"], "tot": tot,
                         "tot_prior": v["ccr"] + v["cca_prior"] + v["eh"],
                         "tot_post": v["ccr"] + v["cca_post"] + v["eh"]}
    return z


def sigma_meV(zeta_tot, n_cm2):
    aB_cm = MAT["aB_nm"] * 1e-7
    return MAT["E0_meV"] * n_cm2 * aB_cm**2 * zeta_tot


# =============================================================================
# 5. LASER CALIBRATION AND MISMATCH DETUNINGS  (sign convention of the paper)
# =============================================================================
def calibration(hw, z):
    S = lambda st, n: sigma_meV(z[f"{st},{st}"]["tot"], n)

    def laser(n_cal):     # carrier photon energies giving Delta~ = 0 at n_cal (meV)
        return {"Xi_opt_0-1s": hw["0-1s"] + S('1s', n_cal),
                "Xi_THz_1s-2p": hw["1s-2p"] + S('2p', n_cal) - S('1s', n_cal),
                "V_opt_0-1s": hw["0-1s"] + S('1s', n_cal),
                "V_opt_0-2s": hw["0-2s"] + S('2s', n_cal)}

    def detunings(n_act, n_cal):   # hbar*Delta~ (meV) in the convention of Eqs. (65),(72)
        d = lambda st: S(st, n_act) - S(st, n_cal)
        return {"Xi_Delta12": d('1s'), "Xi_Delta23": d('2p') - d('1s'),
                "Xi_delta": d('2p'),
                "V_Delta12": d('1s'), "V_Delta13": d('2s'), "V_delta": d('2s') - d('1s')}

    table = []
    for n0 in NX_CAL:
        rows = [{"n_cal": 0.0, "n_act": n0, "label": "bare-tuned",
                 **{k: f(v) for k, v in detunings(n0, 0.0).items()}}]
        for dev in NX_DEV:
            na = n0 * (1 + dev)
            rows.append({"n_cal": n0, "n_act": na, "label": f"{dev:+.0%}",
                         **{k: f(v) for k, v in detunings(na, n0).items()}})
        table.append({"n_X_nominal_cm2": n0,
                      "laser_photon_energy_meV": {k: f(v, 7) for k, v in laser(n0).items()},
                      "rows": rows})
    bare = {k: f(v, 7) for k, v in laser(0.0).items()}
    slopes = {st: sigma_meV(z[f"{st},{st}"]["tot"], 1e11) for st in ('1s', '2s', '2p')}
    return bare, table, slopes


# =============================================================================
# 6. RABI FREQUENCY -> FIELD CONVERSION (reporting only)
# =============================================================================
def field_tables(phi0sq, dip, extras):
    aB = MAT["aB_nm"] * 1e-9
    A = MAT["A_um2"] * 1e-12
    dcv = np.sqrt(extras["dcv2_C2m2"])
    per_meV = 1e-3 * C.e
    E_thz = per_meV / (C.e * dip["1s-2p"] * aB)                    # V/m per meV, Eqs. (46)-(47)
    E_opt = {l: per_meV / (np.sqrt(A) * np.sqrt(phi0sq[l]) / aB * dcv)   # Eqs. (44)-(45)
             for l in ('1s', '2s')}
    I = lambda E: 0.5 * C.c * C.epsilon_0 * E**2                   # W/m^2, vacuum-equivalent
    tab = []
    for W in OMEGA_REPORT_meV:
        tab.append({"hbar_Omega_meV": W,
                    "E_THz_kV_per_cm": f(W * E_thz / 1e5),
                    "I_THz_MW_per_cm2": f(I(W * E_thz) / 1e10),
                    "E_opt_1s_V_per_cm": f(W * E_opt['1s'] / 1e2),
                    "I_opt_1s_W_per_cm2": f(I(W * E_opt['1s']) / 1e4),
                    "E_opt_2s_V_per_cm": f(W * E_opt['2s'] / 1e2),
                    "I_opt_2s_W_per_cm2": f(I(W * E_opt['2s']) / 1e4)})
    return {"E_THz_kV_per_cm_per_meV": f(E_thz / 1e5),
            "E_opt_1s_V_per_cm_per_meV": f(E_opt['1s'] / 1e2),
            "E_opt_2s_V_per_cm_per_meV": f(E_opt['2s'] / 1e2),
            "note": "Intensities are vacuum-equivalent (I = c eps0 E^2 / 2); "
                    "E_opt scales as A^-1/2, E_THz is independent of A.",
            "table": tab}


# =============================================================================
# 7. SPECTRAL-SELECTIVITY BOUNDS (Sect. IV C 1), margin n = 3
# =============================================================================
def selectivity(Eb, phi0sq, dip):
    n = 3.0
    comp = {"THz 1s->2p vs 1s->3p": (Eb['2p'] - Eb['3p'], dip["1s-3p"] / dip["1s-2p"]),
            "opt 0->1s vs 0->2s": (Eb['1s'] - Eb['2s'], np.sqrt(phi0sq['2s'] / phi0sq['1s'])),
            "opt 0->2s vs 0->3s": (Eb['2s'] - Eb['3s'], np.sqrt(phi0sq['3s'] / phi0sq['2s']))}
    out = {}
    for k, (D, ratio) in comp.items():
        out[k] = {"detuning_meV": f(D), "amplitude_ratio": f(ratio, 3),
                  "sigma_min_ps": f(n * HB_meVps / D, 3),
                  "tanh_ramp_min_ps": f(4.5594 * HB_meVps / D, 3)}
    return out


# =============================================================================
# 7b. PROTOCOL BLOCK  (fixed choices, grids made explicit)
# =============================================================================
def uniform_grid(start, stop, step):
    """Inclusive uniform grid, rounded to kill float drift (e.g. -0.2+0.2 -> 0.0)."""
    n = int(round((stop - start) / step)) + 1
    return [round(start + k * step, 6) + 0.0 for k in range(n)]


def two_step_grid(span, inner, d_inner, d_outer):
    """Symmetric grid on [-span, span]: step d_inner on [-inner, inner], d_outer outside."""
    mid = uniform_grid(-inner, inner, d_inner)
    n_out = int(round((span - inner) / d_outer))
    right = [round(inner + k * d_outer, 6) for k in range(1, n_out + 1)]
    left = [-x for x in reversed(right)]
    return left + mid + right


def protocol_block(rates, cal):
    P = PROTO
    hbar_g1s = rates[4.0]["gamma_1s_to_0"] * C.hbar / C.e * 1e3      # meV
    hbar_g2s = rates[4.0]["gamma_2s_to_0"] * C.hbar / C.e * 1e3
    G12, G13 = hbar_g1s / 2, hbar_g2s / 2                              # coherence half-widths
    sat = lambda W, G: W**2 / (2 * G**2)                               # s = Omega^2/(2 Gamma^2)
    area = lambda W, s: W * s * np.sqrt(2 * np.pi) / HB_meVps          # Gaussian pulse area, rad

    xi_delta = uniform_grid(**P["xi_delta_meV"])
    xi_W23 = uniform_grid(**P["xi_Omega23_meV"])
    xi_x = uniform_grid(**P["xi_delay_over_sigma"])
    probe = two_step_grid(**P["probe_Delta12"])
    vdel = two_step_grid(**P["v_delta"])

    # case 3: bare-tuned row of the mismatch table at the chosen density
    c3 = next(c for c in cal if c["n_X_nominal_cm2"] == P["pa_nX_case3_cm2"])
    bt = next(r for r in c3["rows"] if r["label"] == "bare-tuned")

    weak = lambda W23: f(P["eit_probe_factor"] * min(W23, hbar_g1s / 2), 6)
    G23 = G12 + G13
    W12_base = weak(P["xi_Omega23_base_meV"])

    return {
        "source": "protocol choices of the paper (Sect. VI, Table III); "
                  "every grid written out explicitly",
        "initial_state": "rho(0) = |0><0|, t0 = 0, phases phi_ij = 0",
        "inputs": "renormalized detunings hbar*Delta~ and Rabi energies hbar*Omega (meV); "
                  "case 3 (photon dissipator + exchange mean field) uses bare laser detunings Delta = 0 "
                  "plus diagonal Sigma_a",
        "weak_optical_leg_rule": "hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2) for every "
                                 "Xi run (conversion, CPT plateau, spectra)",
        "validity_condition": "one-exciton Fock space of Sect. III kept; every run reports "
                              "max_t (1 - rho11), required << 1",
        "temperatures_K": {"baseline": 4, "comparison": 77, "single_run": 300},

        "Part_A": {
            "temperature_K": 4,
            "cases": {"1": "coherent only: no dissipator, Sigma = 0",
                      "2": "photon dissipator, Sigma = 0",
                      "3": "photon dissipator + diagonal exchange mean field (bare-tuned lasers)"},
            "Xi_conversion_A1_A3": {"envelope": "Gaussian", "sigma_ps": P["xi_sigma_base_ps"],
                                    "delay_over_sigma": P["xi_delay_base"],
                                    "hbar_Omega23_meV": P["xi_Omega23_base_meV"],
                                    "hbar_Omega12_meV": W12_base,
                                    "orderings": ["counterintuitive (t23 < t12)",
                                                  "sequential (t12 < t23), control"]},
            "case3": {"n_X_cm2": P["pa_nX_case3_cm2"],
                      "source": "calibration.mismatch_scan, bare-tuned row (derived)",
                      "Xi": {"Delta12_meV": bt["Xi_Delta12"], "Delta23_meV": bt["Xi_Delta23"],
                             "delta_meV": bt["Xi_delta"]},
                      "V": {"Delta12_meV": bt["V_Delta12"], "Delta13_meV": bt["V_Delta13"],
                            "delta_meV": bt["V_delta"]}},
            "A4_Xi_plateau_CPT": {"envelope": "plateau",
                                  "hbar_Omega12_meV": weak(P["pa_A4_Omega23_meV"]),
                                  "hbar_Omega23_meV": P["pa_A4_Omega23_meV"],
                                  "tau_on_off_ps": 0.5, "T_pulse_ps": 50.0},
            "A5_V": {"envelope": "synchronous Gaussian pair, t12 = t13",
                     "sigma_ps": P["vb_sigma_base_ps"],
                     "hbar_Omega12_meV": P["vb_Omega_meV"], "hbar_Omega13_meV": P["vb_Omega_meV"]},
        },

        "Xi_conversion": {
            "envelope": "Gaussian", "ordering": "t23 < t12 (counterintuitive)",
            "figure_of_merit": "eta = rho33(t_f)/(rho33(t_f) + N_emit); vertex suppression "
                               "max rho22 / rho33(t_f); P_ion / rho33(t_f)",
            "baseline": {"sigma_ps": P["xi_sigma_base_ps"], "delay_over_sigma": P["xi_delay_base"],
                         "hbar_Omega23_meV": P["xi_Omega23_base_meV"],
                         "hbar_Omega12_meV": W12_base, "delta_Xi_meV": 0.0,
                         "optical_pulse_area_rad": f(area(W12_base, P["xi_sigma_base_ps"]), 3)},
            "grids": {"delta_Xi_meV": xi_delta, "hbar_Omega23_meV": xi_W23,
                      "hbar_Omega12_meV_derived": [weak(W) for W in xi_W23],
                      "delay_over_sigma": xi_x, "sigma_ps": P["xi_sigma_ps"]},
            "scans": {
                "robustness_map": {"axes": ["delta_Xi_meV", "hbar_Omega23_meV"],
                                   "fixed": {"sigma_ps": P["xi_sigma_base_ps"],
                                             "delay_over_sigma": P["xi_delay_base"]},
                                   "n_runs": len(xi_delta) * len(xi_W23)},
                "adiabaticity_grid": {"axes": ["hbar_Omega23_meV", "sigma_ps", "delay_over_sigma"],
                                      "fixed": {"delta_Xi_meV": 0.0},
                                      "n_runs": len(xi_W23) * len(P["xi_sigma_ps"]) * len(xi_x)},
                "ordering_control": "sequential order run at every grid point"},
        },

        "Xi_response": {
            "envelope": "plateau", "tau_on_off_ps": 0.5, "T_pulse_ps": 50.0, "Delta23": 0.0,
            "hbar_Omega23_meV": P["eit_Omega23_meV"],
            "eit_claims_only_for_hbar_Omega23_le_meV": P["eit_claim_max_meV"],
            "eit_at_crossover_meV": P["eit_crossover_meV"],
            "weak_probe_rule": "hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2) "
                               "(probe nonlinearity <= 0.32 %)",
            "hbar_Omega12_meV_derived": [weak(W) for W in P["eit_Omega23_meV"]],
            "linearity_check_factor": 0.5,
            "probe_Delta12_grid": {"spec": "0.01 meV within +-1 meV, 0.1 meV outside, +-15 meV",
                                   "n_points": len(probe), "values_meV": probe},
            "chi_source": "Liouvillian null-space steady state for all production spectra",
        },

        "V_response": {
            "envelope": "plateau", "tau_on_off_ps": 0.2, "T_pulse_ps_min": 20.0,
            "Delta12": 0.0, "scan": "delta_V via the 2s tone",
            "hbar_Omega12_meV": P["v_Omega12_meV"], "hbar_Omega13_meV": P["v_Omega13_meV"],
            "saturation_derived": {"Gamma12_meV": f(G12), "Gamma13_meV": f(G13),
                                   "s12": f(sat(P["v_Omega12_meV"], G12), 3),
                                   "s13": f(sat(P["v_Omega13_meV"], G13), 3),
                                   "cross_Omega12sq_over_4G23G13": f(P["v_Omega12_meV"]**2 / (4 * G23 * G13), 3)},
            "delta_V_grid": {"spec": "0.01 meV within +-1 meV, 0.1 meV outside, +-5 meV",
                             "n_points": len(vdel), "values_meV": vdel},
            "chi_source": "Liouvillian null-space steady state",
        },

        "V_beat": {
            "envelope": "synchronous Gaussian pair, t12 = t13",
            "sigma_ps_baseline": P["vb_sigma_base_ps"], "sigma_ps_grid": P["vb_sigma_ps"],
            "hbar_Omega12_meV": P["vb_Omega_meV"], "hbar_Omega13_meV": P["vb_Omega_meV"],
            "pulse_area_rad_derived": [f(area(P["vb_Omega_meV"], s), 3) for s in P["vb_sigma_ps"]],
            "amplitude_check_hbar_Omega_meV": 0.5,
        },

        "nX_mismatch": {"n_cal_cm2": NX_CAL, "deviations": NX_DEV,
                        "bare_tuned_reference": True, "applied_to": "Xi conversion",
                        "operating_point": {"sigma_ps": P["xi_sigma_base_ps"],
                                            "delay_over_sigma": P["xi_delay_base"],
                                            "hbar_Omega23_meV": P["xi_Omega23_base_meV"],
                                            "hbar_Omega12_meV": W12_base}},

        "numerics": {
            "gaussian_centres": ">= 6 sigma after t0", "plateau_t_on": ">= 6 tau_on",
            "t_end": "6 sigma after last pulse / plateau end",
            "integrator": "DOP853 (scipy solve_ivp), rtol 1e-8, atol 1e-10",
            "rtol": 1e-8, "atol": 1e-10,
            "coherent_only_rtol": 1e-10, "coherent_only_atol": 1e-12,
            "weak_drive_note": "with the weak optical leg the excited populations are ~1e-3; "
                               "atol is lowered to 1e-14 for those runs",
            "max_step": "min(sigma, tau_on) / 20",
            "breakpoints": "pulse centres +- 6 sigma, t_on, t_off (piecewise integration)",
            "accumulators": ["N_emit = int gamma_2->1 rho22 dt", "int rho22 dt",
                             "P_ion = int Gamma_ion(t) rho33 dt"],
            "validation_thresholds": {"trace": 1e-8, "hermiticity": 1e-8,
                                      "min_eigenvalue": -1e-9}},
    }


# =============================================================================
# 8. ASSEMBLE, VALIDATE, WRITE
# =============================================================================
DRAFT = {  # values quoted in the paper (Tables VII-IX, App. C 2, Sect. V item 6, Eq. 62)
    "Eb_1s": 161.59, "Eb_2p": 48.39, "Eb_2s": 37.43, "Eb_3p": 19.09, "Eb_3s": 16.13,
    "phi0sq_1s": 0.2842, "phi0sq_2s": 0.02985, "phi0sq_3s": 8.45e-3,
    "d_1s-2p": 0.854, "d_1s-3p": 0.266, "d_2s-2p": 2.740,
    "gamma_1s": 4.87e12, "gamma_2s": 5.48e11,
    "zeta_1s,1s": 5.57, "zeta_2s,2s": 6.68, "zeta_2p,2p": 4.83, "zeta_1s,2s": 3.76,
    "zeta_cc_r_1s,1s": -32.24, "zeta_cc_a_1s,1s": 37.60,
    "Sigma1s_per_nX_ueV_um2": 2.12, "ngamma_intra_300K": 1.27e-2, "omit_ratio_300K": 1.86,
}


def validate(Eb, phi0sq, dip, rates, z):
    checks = [
        ("Eb(1s) [meV]", Eb['1s'], DRAFT["Eb_1s"], 0.01),
        ("Eb(2p) [meV]", Eb['2p'], DRAFT["Eb_2p"], 0.01),
        ("Eb(2s) [meV]", Eb['2s'], DRAFT["Eb_2s"], 0.01),
        ("Eb(3p) [meV]", Eb['3p'], DRAFT["Eb_3p"], 0.01),
        ("Eb(3s) [meV]", Eb['3s'], DRAFT["Eb_3s"], 0.01),
        ("|phi_1s(0)|^2", phi0sq['1s'], DRAFT["phi0sq_1s"], 1e-3),
        ("|phi_2s(0)|^2", phi0sq['2s'], DRAFT["phi0sq_2s"], 1e-3),
        ("|phi_3s(0)|^2", phi0sq['3s'], DRAFT["phi0sq_3s"], 5e-3),
        ("|d_1s,2p| [a_B]", dip["1s-2p"], DRAFT["d_1s-2p"], 1e-3),
        ("|d_1s,3p| [a_B]", dip["1s-3p"], DRAFT["d_1s-3p"], 2e-3),
        ("|d_2s,2p| [a_B]", dip["2s-2p"], DRAFT["d_2s-2p"], 1e-3),
        ("gamma_1s->0 [s^-1]", rates[4.0]["gamma_1s_to_0"], DRAFT["gamma_1s"], 5e-3),
        ("gamma_2s->0 [s^-1]", rates[4.0]["gamma_2s_to_0"], DRAFT["gamma_2s"], 5e-3),
        ("zeta(1s,1s)", z["1s,1s"]["tot"], DRAFT["zeta_1s,1s"], 1e-3),
        ("zeta(2s,2s)", z["2s,2s"]["tot"], DRAFT["zeta_2s,2s"], 1e-3),
        ("zeta(2p,2p)", z["2p,2p"]["tot"], DRAFT["zeta_2p,2p"], 1e-3),
        ("zeta(1s,2s)", z["1s,2s"]["tot"], DRAFT["zeta_1s,2s"], 2e-3),
        ("zeta^{cc,r}(1s,1s)", z["1s,1s"]["cc_r"], DRAFT["zeta_cc_r_1s,1s"], 1e-3),
        ("zeta^{cc,a}(1s,1s)", z["1s,1s"]["cc_a"], DRAFT["zeta_cc_a_1s,1s"], 1e-3),
        ("Sigma_1s/n_X [ueV um^2]",
         MAT["E0_meV"] * (MAT["aB_nm"] * 1e-3) ** 2 * z["1s,1s"]["tot"] * 1e3,
         DRAFT["Sigma1s_per_nX_ueV_um2"], 5e-3),
        ("n_gamma(113.19 meV, 300 K)", rates[300.0]["n_gamma"]["1s-2p"],
         DRAFT["ngamma_intra_300K"], 1e-2),
        ("omitted/retained ratio 300 K", rates[300.0]["omitted_2p_2s_ratio"],
         DRAFT["omit_ratio_300K"], 1e-2),
    ]
    out, ok = [], True
    for name, val, ref, rtol in checks:
        rel = abs(val - ref) / abs(ref)
        passed = rel <= rtol
        ok &= passed
        out.append({"quantity": name, "computed": f(val), "draft": ref,
                    "rel_dev": f(rel, 2), "tolerance": rtol, "pass": bool(passed)})
    # structural checks
    sel = abs(z["1s,2p"]["tot"])
    out.append({"quantity": "zeta(1s,2p) selection rule (must vanish)", "computed": f(sel, 2),
                "draft": 0.0, "rel_dev": None, "tolerance": 1e-8, "pass": bool(sel < 1e-8)})
    ok &= sel < 1e-8
    return ok, out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wannier-solver-derived inputs for the Lindblad solvers")
    ap.add_argument("--out", default="params.yaml")
    ap.add_argument("--report", default="solver_inputs_report.md")
    args = ap.parse_args(argv)

    E0 = MAT["E0_meV"]
    tau0_ps = HB_meVps / E0

    print("[1] Wannier spectrum ...")
    Eb, phi0sq, dip, orth = solve_wannier()
    print("[2-3] Bohr frequencies and radiative rates ...")
    hw, rates, extras = radiative_rates(Eb, phi0sq, dip)
    print("[4] exchange vertices (vertex file) ...")
    z = exchange_vertices()
    print("[5] calibration and mismatch detunings ...")
    bare, cal, slopes = calibration(hw, z)
    print("[6-7] fields and selectivity ...")
    fields = field_tables(phi0sq, dip, extras)
    selec = selectivity(Eb, phi0sq, dip)
    print("[8] validation against the values of the paper ...")
    ok, val = validate(Eb, phi0sq, dip, rates, z)

    def rate_block(T):
        r = rates[T]
        keys = ["gamma_1s_to_0", "gamma_0_to_1s", "gamma_2s_to_0", "gamma_0_to_2s",
                "gamma_2p_to_1s", "gamma_1s_to_2p"]
        return {"SI_s^-1": {k: f(r[k]) for k in keys},
                "hbar_gamma_meV": {k: f(r[k] * C.hbar / C.e * 1e3) for k in keys},
                "dimensionless_gamma_tau0": {k: f(r[k] * tau0_ps * 1e-12) for k in keys},
                "n_gamma": {k: f(v) for k, v in r["n_gamma"].items()},
                "omitted_2p_2s_over_retained": f(r["omitted_2p_2s_ratio"], 3)}

    params = {
        "meta": {
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "generator": "solver_inputs_from_wannier.py",
            "draft": "validated against the values quoted in the paper (block validation)",
            "decisions": "protocol choices of the paper (Sect. VI, Table III)",
            "validation_passed": bool(ok),
        },
        "units": {"E0_meV": E0, "aB_nm": MAT["aB_nm"], "tau0_ps": f(tau0_ps, 8),
                  "tau0_fs": f(tau0_ps * 1e3, 6), "hbar_meV_ps": f(HB_meVps, 10),
                  "note": "E0 and tau0 = hbar/E0 are the natural units of App. D; the Lindblad solver works in meV and ps."},
        "material": {**{k: (f(v) if isinstance(v, float) else v) for k, v in MAT.items()},
                     "M_m0": MAT["me_m0"] + MAT["mh_m0"],
                     "alpha_e": f(MAT["me_m0"] / (MAT["me_m0"] + MAT["mh_m0"]), 4),
                     "alpha_h": f(MAT["mh_m0"] / (MAT["me_m0"] + MAT["mh_m0"]), 4),
                     "A_note": "baseline normalization area = 3.5 um fibre core of Stier et al. "
                               "(PRL 120, 057405 (2018)); enters only field reporting, the n_X "
                               "check (iii) and gamma_2p->1s ~ A^-1."},
        "wannier": {"Eb_meV": {k: f(v, 7) for k, v in Eb.items()},
                    "phi0sq_aB-2": {k: f(v) for k, v in phi0sq.items()},
                    "dipole_x_aB": {k: f(v) for k, v in dip.items()},
                    "orthonormality_error": {int(k): f(v, 2) for k, v in orth.items()}},
        "bohr_energies_meV": {k: f(v, 7) for k, v in hw.items()},
        "manifolds": {
            "Xi": {"states": ["0", "1s,K0", "2p_d,K0"], "legs": {"(1,2)": "interband optical",
                                                                "(2,3)": "intra-excitonic THz, e_d"},
                   "dissipator_legs": ["2->1 (1s->0)", "3->2 (2p->1s)"]},
            "V": {"states": ["0", "1s,K0", "2s,K0"], "legs": {"(1,2)": "interband optical",
                                                             "(1,3)": "interband optical"},
                  "dissipator_legs": ["2->1 (1s->0)", "3->1 (2s->0)"]}},
        "rates": {f"T_{int(T)}K": rate_block(T) for T in T_LIST},
        "rates_note": "Interband rates at K0 = 0 (Eqs. C24-C26); intra-excitonic rate at the "
                      "baseline A (Eq. C22). Offdiagonal Sigma_1s,2s neglected (Sect. V A), "
                      "so no density rescaling of the rates.",
        "exchange": {
            "zeta_E0aB2": {k: {kk: f(vv) for kk, vv in v.items()} for k, v in z.items()},
            "Sigma_at_1e11_meV": {k: f(v) for k, v in slopes.items()},
            "source": "exchange_mean_field.py -> " + VERTEX_FILE,
            "note": "Sigma_a = n_X zeta_aa E0 aB^2 (Eq. 62); zeta = zeta^{cc,r} + zeta^{cc,a} + "
                    "zeta^{eh} (Table IX, Hermitian prescription). Only diagonal elements enter "
                    "the solver; zeta(1s,2s) is recorded for the accuracy statement of Sect. V A."},
        "calibration": {
            "convention": "hbar*Delta~ = Sigma(n_act) - Sigma(n_cal), Eqs. (65),(72). "
                          "Lasers at the tabulated photon energies give Delta~ = 0 at n_cal.",
            "laser_photon_energy_bare_meV": bare,
            "mismatch_scan": cal},
        "fields": fields,
        "ionization": {"kappa_ion": KAPPA_ION,
                       "hbar_Gamma_ion_meV_per_meV2": f(KAPPA_ION / E0),
                       "note": "hbar*Gamma_ion = kappa_ion (hbar*Omega23)^2/E0, Eq. (57); "
                               "P_ion = int Gamma_ion(t) rho33(t) dt evaluated a posteriori."},
        "selectivity_n3": selec,
        "protocol": protocol_block(rates, cal),
        "validation": val,
    }

    with open(args.out, "w") as fh:
        yaml.safe_dump(params, fh, sort_keys=False, allow_unicode=True, width=100)

    # ---------------------------------------------------------------- report
    L = [f"# Solver inputs from the Wannier solver\n",
         f"Generated {params['meta']['generated']} by `solver_inputs_from_wannier.py`. "
         f"Validation against the values of the paper: **{'PASS' if ok else 'FAIL'}**.\n",
         "## Validation\n", "| quantity | computed | paper | rel. dev. | pass |", "|---|---|---|---|---|"]
    for v in val:
        L.append(f"| {v['quantity']} | {v['computed']} | {v['draft']} | {v['rel_dev']} | "
                 f"{'yes' if v['pass'] else 'NO'} |")
    L += ["\n## Exchange (E0 aB^2) and Sigma at 1e11 cm^-2\n",
          "| pair | zeta^{cc,r} | zeta^{cc,a} | zeta^{eh} | zeta |", "|---|---|---|---|---|"]
    for k, v in z.items():
        L.append(f"| {k} | {v['cc_r']:.3f} | {v['cc_a']:.3f} | {v['eh']:.3f} | {v['tot']:.3f} |")
    L.append("\nSigma at 1e11 cm^-2 (meV): " + ", ".join(f"{k} {v:+.4f}" for k, v in slopes.items()))
    L += ["\n## Mismatch detunings ħΔ̃ (meV), convention of the paper\n",
          "| n_cal | n_act | case | Ξ Δ̃12 | Ξ Δ̃23 | Ξ δ̃ | V δ̃ |", "|---|---|---|---|---|---|---|"]
    for c in cal:
        for r in c["rows"]:
            L.append(f"| {r['n_cal']:.2e} | {r['n_act']:.3e} | {r['label']} | {r['Xi_Delta12']} | "
                     f"{r['Xi_Delta23']} | {r['Xi_delta']} | {r['V_delta']} |")
    pr = params["protocol"]
    xc, pa = pr["Xi_conversion"], pr["Part_A"]
    L += ["\n## Protocol\n", "| item | value |", "|---|---|",
          f"| weak optical leg | {pr['weak_optical_leg_rule']} |",
          f"| validity | {pr['validity_condition']} |",
          f"| cases 1-3 | σ = {pa['Xi_conversion_A1_A3']['sigma_ps']} ps, x = "
          f"{pa['Xi_conversion_A1_A3']['delay_over_sigma']}, ħΩ23 = "
          f"{pa['Xi_conversion_A1_A3']['hbar_Omega23_meV']} meV, ħΩ12 = "
          f"{pa['Xi_conversion_A1_A3']['hbar_Omega12_meV']} meV |",
          f"| case 3 (n_X = {pa['case3']['n_X_cm2']:.1e}) | Ξ Δ̃12, Δ̃23, δ̃ = "
          f"{pa['case3']['Xi']['Delta12_meV']}, {pa['case3']['Xi']['Delta23_meV']}, "
          f"{pa['case3']['Xi']['delta_meV']} meV; V Δ̃12, Δ̃13 = "
          f"{pa['case3']['V']['Delta12_meV']}, {pa['case3']['V']['Delta13_meV']} meV |",
          f"| A4 CPT plateau | ħΩ12 = {pa['A4_Xi_plateau_CPT']['hbar_Omega12_meV']} meV, ħΩ23 = "
          f"{pa['A4_Xi_plateau_CPT']['hbar_Omega23_meV']} meV, 50 ps |",
          f"| Ξ conversion baseline | {xc['baseline']} |",
          f"| Ξ ħΩ23 grid (meV) | {xc['grids']['hbar_Omega23_meV']} |",
          f"| Ξ δ̃ grid (meV) | {xc['grids']['delta_Xi_meV']} |"]
    with open(args.report, "w") as fh:
        fh.write("\n".join(L) + "\n")

    print(f"\nvalidation: {'PASS' if ok else 'FAIL'}   ->  {args.out}, {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
