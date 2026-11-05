"""
===============================================================================
 appE_validation.py
 Validation of the Lindblad solver lindblad_solver.py (App. E of the paper
 "Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced
 Intravalley Lindblad Framework for WSe2")
===============================================================================
Three parts:

lindblad   checks V0-V7 of the solver; writes
               validation_report.md   PASS/FAIL table + details
               validation_results.json
               validation_runs.h5     representative trajectories (HDF5 writer test)
  V0  generic generator == hand-coded component equations, Eqs. (68a-f), (75a-f)
  V1  trace, Hermiticity, positivity over every trajectory of V2-V7
  V2  no-drive limit: population rate equations, coherence damping, mean field =
      pure precession, no population transfer
  V3  Xi weak-probe spectrum vs analytic homogeneous cascade susceptibility
  V4  V beat: free decay at (gamma_1s + gamma_2s)/2, lab-frame period 33.3 fs
  V4b V steady state: rho23 = rho21 rho13, width set by gamma_2s, chi12 flat
  V5  dark-state following, Eq. (69): H|D> = 0, 1 - P_D(t) against first-order
      adiabatic perturbation theory
  V6  long plateau converges to the null-space steady state
  V7  cross-checks: Radau and QuTiP mesolve on the strong-drive test transfer
  Pass criteria are fixed in CRIT below.

reference  the conversion, CPT-plateau and V-beat runs of the solver against reference
           values from an independent implementation of the same master equation in
           natural units (E0, hbar/E0), on a fixed test set: strong drive hbar*Omega0 = 5 meV
           on both legs, sigma = 1.5 ps, tau_d = 1.25 sigma, 4 K; case 3 with the fixed test
           detunings Delta12 = -3.04997, Delta23 = 0.460013 meV (V: Delta12 = -3.04997,
           Delta13 = -2.93092 meV). Exits with an error if a relative deviation exceeds
           1e-4 (absolute 1e-6 for values below 1e-3). Writes reference_check.json.

amplitude  the weak-excitation Xi conversion runs (Sect. VI B-C) against first-order
           amplitude equations. In the weak-excitation regime the ground-state amplitude
           stays c1 = 1 to first order in Omega12, and the excited amplitudes obey a
           two-component Schroedinger equation with a non-Hermitian vertex energy
           -i hbar*gamma_{1s->0}/2 (population that decays returns to |1> only at second
           order in Omega12):
             i hbar dc2/dt = (hbar*D12 - i hbar*g2/2) c2 + (hbar*Omega12/2) + (hbar*Omega23/2) c3
             i hbar dc3/dt = (hbar*(D12+D23) - i hbar*g3/2) c3 + (hbar*Omega23/2) c2
           so rho33(t_f) = |c3|^2, N_emit = g2 * int |c2|^2 dt, eta = rho33/(rho33 + N_emit) to
           leading order. Integrated with a different integrator (RK45, real and imaginary
           parts) and compared with the Lindblad runs of fig3_R1.py, fig4_R1b.py (cases 2-3,
           both orders) and fig5_R3.py (grid), read from fig3_R1.json, fig4_R1b.json and
           fig5_R3.json. Writes xi5_amplitude_check.json.

Strong-drive test transfer (V5, V7, V1 and the reference set): Xi Gaussian pair of equal
widths, hbar*Omega12 = hbar*Omega23 = 5 meV, tau_d = 1.25 sigma, sigma = 1.5 ps
(V5 also at 3 and 6 ps), counterintuitive order. It is a solver test, not a protocol of
the paper, and is fixed below in XI_TRANSFER_TEST.

Usage:  python appE_validation.py [lindblad|reference|amplitude|all]   (default all; reads params.yaml)
===============================================================================
"""
import json
import sys
import time

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import lindblad_solver as ls
from lindblad_solver import HB

XI_TRANSFER_TEST = dict(sigma_ps=1.5, delay_over_sigma=1.25, hbar_Omega0_meV=5.0)

# Checks V0-V7 and V1: generator rates (and the analytic widths of V3, V4b) from the
# 'dimensionless_gamma_tau0' block of params.yaml (x E0); the reference rates of V4 and
# of the V4b drive checks from 'hbar_gamma_meV'. The reference and amplitude parts use
# the conversion runs, with 'hbar_gamma_meV'.
RB = ls.RATES_TAU0

CRIT = {
    "V0_max_abs": 1e-13,
    "V1_trace": 1e-8, "V1_herm": 1e-8, "V1_min_eig": -1e-9,
    "V2_max_abs": 1e-7, "V2_meanfield_pop": 1e-9,
    "V3_rel_dev": 1e-2, "V3_ratio": (3.5, 4.5),
    "V4_rel_gamma": 1e-5, "V4_rel_period": 1e-6, "V4_amp_rel_gamma": 1e-5,
    "V4b_rel_dev": 0.10, "V4b_ratio": (3.0, 5.0),
    "V4b_hwhm_rel_approved": 0.05, "V4b_hwhm_rel_weak": 5e-3, "V4b_chi12_flat": 0.02,
    "V5_HD": 1e-14, "V5_ratio_err_sigma6": 0.05,
    "V6_residual": 1e-6,
    "V7_max_abs": 1e-6,
    "V3_extrap_rel_dev": 1e-3,
    "V4b_weak_rel_dev": 1e-2, "V4b_weak_ratio": (3.5, 4.5),
    "V5_pert_rel_dev": 0.03,
}

# Notes on the criteria (reported verbatim).
NOTES = {
    "V3": "At the weak probe of the paper the deviation from the first-order analytic formula is "
          "about 1 % and scales exactly as Omega12^2 (ratio ~4 on halving): the probe's own "
          "saturation, not a solver error. The solver check compares the Richardson-extrapolated "
          "(Omega12 -> 0) spectrum with the analytic one (<= 1e-3); the 1 % criterion at the "
          "probe of the paper is a PARAMETER check.",
    "V4b": "At larger V drives the 2s line is cross-broadened through the shared ground state, "
           "extra HWHM = Omega12^2/(4 Gamma23) (0.035 meV at 0.5 meV; parameter "
           "Omega12^2/(4 Gamma23 Gamma13) = 0.19). Solver checks therefore run in the linear "
           "regime (x0.5, x0.25 of the V-response drive); the V-response drive and a stronger "
           "candidate are PARAMETER / information checks. |rho23| has HWHM sqrt(3) Gamma13; the "
           "Lorentzian of HWHM Gamma13 is |rho23|^2 (and Re rho23).",
    "V5": "For equal-width Gaussians the non-adiabatic loss is created in the pulse wings "
          "(theta_dot/Omega0 grows there), not at tbar, and interferes (Stueckelberg oscillation), "
          "so 1 - P_D is compared with first-order adiabatic perturbation theory, "
          "1 - P_D(t) = |int theta_dot e^{i Phi} dt|^2, Phi = int Omega0/2 dt, over the whole "
          "pulse (<= 3 %, second-order terms O(eps)).",
    "V6": "A plateau must be long compared with the inverse spectral gap; the V plateau is 60 ps "
          "(gap 0.22 meV).",
    "V1": "Positivity is reported separately for dissipative and coherent-only (no-dissipator) "
          "runs. In coherent-only runs rho stays rank-1 and the negative eigenvalue is "
          "integrator error that scales with rtol (see the information line); these runs use "
          "rtol 1e-10, atol 1e-12.",
    "V7": "scipy's implicit methods need real states; the propagator integrates [Re y, Im y] for them.",
}

RESULTS = {}
TRAJ = []          # (name, traj, gen) for V1 and the HDF5 file


def record(key, passed, kind="solver", **info):
    RESULTS[key] = {"pass": bool(passed), "kind": kind, **info}
    print(f"  [{kind[:5]}] {key:<40s} {'PASS' if passed else 'FAIL'}  "
          + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in info.items() if not isinstance(v, (list, dict))))


def rand_herm(rng, psd=False):
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    if psd:
        R = A @ A.conj().T
        return R / np.trace(R).real
    return 0.5 * (A + A.conj().T)


# =============================================================================
# V0  generator vs printed component equations
# =============================================================================
def eqs68(r, W12, W23, p12, p23, D12, D23, g12, g21, g23, g32):
    """Eqs. (68a)-(68f) of the paper verbatim (gAB = gamma_{A->B}, hbar = 1); returns d rho."""
    d = D12 + D23
    e = np.exp
    dr = np.zeros((3, 3), complex)
    r11, r22, r33 = r[0, 0], r[1, 1], r[2, 2]
    r12, r21, r23, r32, r13 = r[0, 1], r[1, 0], r[1, 2], r[2, 1], r[0, 2]
    dr[0, 0] = 0.5j * (W12 * e(1j * p12) * r12 - W12 * e(-1j * p12) * r21) - g12 * r11 + g21 * r22
    dr[1, 1] = (0.5j * (W12 * e(-1j * p12) * r21 - W12 * e(1j * p12) * r12
                        + W23 * e(1j * p23) * r23 - W23 * e(-1j * p23) * r32)
                + g12 * r11 - (g21 + g23) * r22 + g32 * r33)
    dr[2, 2] = 0.5j * (W23 * e(-1j * p23) * r32 - W23 * e(1j * p23) * r23) + g23 * r22 - g32 * r33
    dr[0, 1] = (1j * D12 * r12 - 0.5j * W12 * e(-1j * p12) * (r22 - r11)
                + 0.5j * W23 * e(1j * p23) * r13 - 0.5 * (g12 + g21 + g23) * r12)
    dr[1, 2] = (1j * D23 * r23 - 0.5j * W12 * e(1j * p12) * r13
                - 0.5j * W23 * e(-1j * p23) * (r33 - r22) - 0.5 * (g21 + g23 + g32) * r23)
    dr[0, 2] = (1j * d * r13 - 0.5j * W12 * e(-1j * p12) * r23
                + 0.5j * W23 * e(-1j * p23) * r12 - 0.5 * (g12 + g32) * r13)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        dr[b, a] = np.conj(dr[a, b])
    return dr


def eqs75(r, W12, W13, p12, p13, D12, D13, g12, g21, g13, g31):
    """Eqs. (75a)-(75f) of the paper verbatim."""
    dV = D13 - D12
    e = np.exp
    dr = np.zeros((3, 3), complex)
    r11, r22, r33 = r[0, 0], r[1, 1], r[2, 2]
    r12, r21, r13, r31, r23, r32 = r[0, 1], r[1, 0], r[0, 2], r[2, 0], r[1, 2], r[2, 1]
    dr[0, 0] = (0.5j * (W12 * e(1j * p12) * r12 - W12 * e(-1j * p12) * r21
                        + W13 * e(1j * p13) * r13 - W13 * e(-1j * p13) * r31)
                - (g12 + g13) * r11 + g21 * r22 + g31 * r33)
    dr[1, 1] = 0.5j * (W12 * e(-1j * p12) * r21 - W12 * e(1j * p12) * r12) + g12 * r11 - g21 * r22
    dr[2, 2] = 0.5j * (W13 * e(-1j * p13) * r31 - W13 * e(1j * p13) * r13) + g13 * r11 - g31 * r33
    dr[0, 1] = (1j * D12 * r12 - 0.5j * W12 * e(-1j * p12) * (r22 - r11)
                - 0.5j * W13 * e(-1j * p13) * r32 - 0.5 * (g12 + g13 + g21) * r12)
    dr[0, 2] = (1j * D13 * r13 - 0.5j * W12 * e(-1j * p12) * r23
                - 0.5j * W13 * e(-1j * p13) * (r33 - r11) - 0.5 * (g12 + g13 + g31) * r13)
    dr[1, 2] = (1j * dV * r23 - 0.5j * W12 * e(1j * p12) * r13
                + 0.5j * W13 * e(-1j * p13) * r21 - 0.5 * (g21 + g31) * r23)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        dr[b, a] = np.conj(dr[a, b])
    return dr


def V0(P, rng):
    print("V0  generator vs Eqs. (68), (75)")
    E0 = P.E0
    worst = {"Xi": 0.0, "V": 0.0}
    for _ in range(200):
        # random drives, phases, detunings, rates in natural units (E0); the solver gets meV
        W = rng.uniform(0.001, 0.05, 2); ph = rng.uniform(-np.pi, np.pi, 2)
        D = rng.uniform(-0.05, 0.05, 2); g = rng.uniform(0.0, 0.02, 4)
        r = rand_herm(rng)
        # Xi: frame (h2, h3) = (D12, D12 + D23)
        rates = {"gamma_1s_to_0": g[0] * E0, "gamma_0_to_1s": g[1] * E0,
                 "gamma_2p_to_1s": g[2] * E0, "gamma_1s_to_2p": g[3] * E0}
        gen = ls.Generator(P, ls.XI, frame_meV=ls.xi_frame(*(D * E0)), phases=ph, rates=rates)
        num = ls.unvec(gen.L(W * E0) @ ls.vec(r)) * HB / E0          # d rho / d(t E0/hbar)
        ref = eqs68(r, W[0], W[1], ph[0], ph[1], D[0], D[1], g[1], g[0], g[3], g[2])
        worst["Xi"] = max(worst["Xi"], np.abs(num - ref).max())
        # V: frame (h2, h3) = (D12, D13)
        rates = {"gamma_1s_to_0": g[0] * E0, "gamma_0_to_1s": g[1] * E0,
                 "gamma_2s_to_0": g[2] * E0, "gamma_0_to_2s": g[3] * E0}
        gen = ls.Generator(P, ls.V, frame_meV=tuple(D * E0), phases=ph, rates=rates)
        num = ls.unvec(gen.L(W * E0) @ ls.vec(r)) * HB / E0
        ref = eqs75(r, W[0], W[1], ph[0], ph[1], D[0], D[1], g[1], g[0], g[3], g[2])
        worst["V"] = max(worst["V"], np.abs(num - ref).max())
    record("V0 Xi vs Eqs.(68a-f)", worst["Xi"] < CRIT["V0_max_abs"], max_abs=worst["Xi"], samples=200)
    record("V0 V vs Eqs.(75a-f)", worst["V"] < CRIT["V0_max_abs"], max_abs=worst["V"], samples=200)


# =============================================================================
# V2  no-drive limit
# =============================================================================
def rate_matrix(gen):
    R = np.zeros((3, 3))
    for src, dst, g, _ in gen.jumps:
        R[dst, src] += g; R[src, src] -= g
    return R


def V2(P):
    print("V2  no-drive limit (rate equations, coherence damping, mean field)")
    rho0 = np.array([[0.2, 0.10 + 0.05j, -0.05 + 0.07j],
                     [0.10 - 0.05j, 0.5, 0.08 - 0.10j],
                     [-0.05 - 0.07j, 0.08 + 0.10j, 0.3]])
    assert np.linalg.eigvalsh(rho0).min() > 0
    # artificial rates gamma*hbar/E0, converted to meV
    art = {"Xi": {"gamma_1s_to_0": 0.0119, "gamma_0_to_1s": 0.004,
                  "gamma_2p_to_1s": 0.008, "gamma_1s_to_2p": 0.003},
           "V": {"gamma_1s_to_0": 0.0119, "gamma_0_to_1s": 0.004,
                 "gamma_2s_to_0": 0.0013, "gamma_0_to_2s": 0.002}}
    for topo in (ls.XI, ls.V):
        for tag, kw in (("4K", {"T": 4}), ("300K", {"T": 300}),
                        ("artificial", {"rates": {k: v * P.E0 for k, v in art[topo.name].items()}}),
                        ("meanfield 2.5e10", {"T": 4, "meanfield": True, "n_X_cm2": 2.5e10})):
            gen = ls.Generator(P, topo, frame_meV=(0.0, 0.0), **kw, rate_block=RB)
            tr = ls.propagate(gen, ls.no_drive_protocol(P, 2.0), rho0=rho0)
            R = rate_matrix(gen)
            p0 = np.real(np.diag(rho0))
            err = 0.0
            Gout = np.array([sum(g for s, _, g, _ in gen.jumps if s == a) for a in range(3)])
            h = np.real(np.diag(gen.H0)) / HB                         # rad/ps
            for k, t in enumerate(tr.t):
                p = expm(R * t) @ p0
                err = max(err, np.abs(np.real(np.diag(tr.rho[k])) - p).max())
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    c = rho0[a, b] * np.exp((-1j * (h[a] - h[b]) - 0.5 * (Gout[a] + Gout[b])) * t)
                    err = max(err, abs(tr.rho[k, a, b] - c))
            TRAJ.append((f"V2_{topo.name}_{tag}", tr, gen))
            record(f"V2 {topo.name} {tag}", err < CRIT["V2_max_abs"], max_abs=err)
        # mean field produces no population transfer
        g0 = ls.Generator(P, topo, T=4, rate_block=RB)
        g1 = ls.Generator(P, topo, T=4, meanfield=True, n_X_cm2=1e11, rate_block=RB)
        a = ls.propagate(g0, ls.no_drive_protocol(P, 2.0), rho0=rho0)
        b = ls.propagate(g1, ls.no_drive_protocol(P, 2.0), rho0=rho0)
        d = np.abs(np.einsum("tii->ti", a.rho) - np.einsum("tii->ti", b.rho)).max()
        record(f"V2 {topo.name} Sigma: no pop. transfer", d < CRIT["V2_meanfield_pop"], max_abs=d)


# =============================================================================
# V3  Xi weak-probe cascade susceptibility
# =============================================================================
def cascade_rho21(W12, W23, D12, D23, g):
    """Homogeneous weak-probe cascade (first order in W12), from Eqs. (68d, f):
    rho12 = (i W12/2) / [G12 - i D12 + (W23^2/4)/(G13 - i delta)]; rho21 = conj (all in meV)."""
    G12 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_1s_to_0"] + g["gamma_1s_to_2p"])
    G13 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_2p_to_1s"])
    delta = D12 + D23
    r12 = (0.5j * W12) / (G12 - 1j * D12 + (W23**2 / 4) / (G13 - 1j * delta))
    return np.conj(r12)


def V3(P):
    print("V3  Xi weak-probe spectrum vs analytic cascade susceptibility")
    pr = P.proto["Xi_response"]
    grid = np.array(pr["probe_Delta12_grid"]["values_meV"])
    g = P.rates(4, RB)
    out = {}
    for W23, W12 in zip(pr["hbar_Omega23_meV"], pr["hbar_Omega12_meV_derived"]):
        if W23 not in (0.5, 1.5, 5.0):
            continue
        num = {}; uniq = True
        ana = np.array([cascade_rho21(W12, W23, D, 0.0, g) for D in grid])
        for fac in (1.0, 0.5):
            v = np.empty(len(grid), complex)
            for k, D12 in enumerate(grid):
                gen = ls.Generator(P, ls.XI, T=4, frame_meV=ls.xi_frame(D12, 0.0), rate_block=RB)
                rho, info = ls.steady_state(gen, [W12 * fac, W23])
                uniq &= info["unique"]
                v[k] = rho[1, 0] / fac          # normalized to the probe of the paper
            num[fac] = v
        dev = {f: np.abs(num[f] - ana).max() / np.abs(ana).max() for f in num}
        ratio = dev[1.0] / dev[0.5]
        extrap = (4 * num[0.5] - num[1.0]) / 3            # Richardson, Omega12 -> 0
        dex = np.abs(extrap - ana).max() / np.abs(ana).max()
        record(f"V3 Omega23={W23}: Omega12->0 vs analytic",
               dex < CRIT["V3_extrap_rel_dev"] and CRIT["V3_ratio"][0] < ratio < CRIT["V3_ratio"][1] and uniq,
               rel_dev_extrap=dex, ratio=ratio, unique=uniq)
        record(f"V3 Omega23={W23}: probe nonlinearity", dev[1.0] < CRIT["V3_rel_dev"], kind="parameter",
               W12_meV=W12, rel_dev=dev[1.0], rel_dev_half_probe=dev[0.5])
        out[W23] = {"grid": grid, "num": num[1.0], "ana": ana}
    return out


# =============================================================================
# V4  V beat
# =============================================================================
def V4(P):
    print("V4  V beat: free decay and lab-frame period")
    vb = P.proto["V_beat"]
    rm = P.rates(4)
    G_ref = 0.5 * (rm["gamma_1s_to_0"] + rm["gamma_2s_to_0"])             # meV
    EL = P.laser_energies_meV(0.0)
    dE = EL["V_opt_0-2s"] - EL["V_opt_0-1s"]
    period_ref_fs = 2 * np.pi * HB / dE * 1e3
    Gfit = {}
    for W in (vb["hbar_Omega12_meV"], vb["amplitude_check_hbar_Omega_meV"]):
        gen = ls.Generator(P, ls.V, T=4, frame_meV=(0.0, 0.0), rate_block=RB)
        pro = ls.v_beat_protocol(P, vb["sigma_ps_baseline"], W, W)
        tr = ls.propagate(gen, pro)
        TRAJ.append((f"V4_beat_W{W}", tr, gen))
        tps = tr.t
        sel = tps > pro.meta["free_decay_from_ps"]
        slope = np.polyfit(tps[sel], np.log(np.abs(tr.rho[sel, 1, 2])), 1)[0]   # ps^-1
        Gfit[W] = -slope * HB
        lab = ls.lab_frame_coherence(tr, EL["V_opt_0-1s"], EL["V_opt_0-2s"])
        ph = np.unwrap(np.angle(lab[sel]))
        wfit = np.polyfit(tps[sel], ph, 1)[0]                                    # rad/ps
        period_fs = 2 * np.pi / abs(wfit) * 1e3
        if W == vb["hbar_Omega12_meV"]:
            rel = abs(Gfit[W] - G_ref) / G_ref
            relp = abs(period_fs - period_ref_fs) / period_ref_fs
            record("V4 decay rate (1 meV)", rel < CRIT["V4_rel_gamma"], Gamma_fit_meV=Gfit[W],
                   Gamma_ref_meV=G_ref, T2_ps=HB / Gfit[W], rel=rel)
            record("V4 lab-frame period", relp < CRIT["V4_rel_period"], period_fs=period_fs,
                   period_ref_fs=period_ref_fs, samples_per_period=period_fs / 3.0)
            beat_example = (tps, tr.rho[:, 1, 2], lab)
    rel2 = abs(Gfit[vb["amplitude_check_hbar_Omega_meV"]] - Gfit[vb["hbar_Omega12_meV"]]) / G_ref
    record("V4 T2 independent of amplitude", rel2 < CRIT["V4_amp_rel_gamma"], rel=rel2)
    return beat_example


# =============================================================================
# V4b  V steady state
# =============================================================================
def hwhm(x, y):
    """Half width at half maximum of a single centred peak (linear interpolation)."""
    k = int(np.argmax(y)); half = y[k] / 2
    def cross(idx):
        for i in idx:
            if y[i] < half:
                j = i - 1 if idx.step > 0 else i + 1
                return x[j] + (half - y[j]) * (x[i] - x[j]) / (y[i] - y[j])
        return np.nan
    right = cross(range(k, len(y), 1)); left = cross(range(k, -1, -1))
    return 0.5 * (right - left)


def v_scan(P, grid, W12, W13):
    g = P.rates(4, RB)
    G12 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_0_to_2s"] + g["gamma_1s_to_0"])
    G13 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_0_to_2s"] + g["gamma_2s_to_0"])
    r23 = np.empty(len(grid), complex); chi12 = np.empty(len(grid), complex)
    for k, dV in enumerate(grid):
        gen = ls.Generator(P, ls.V, T=4, frame_meV=ls.v_frame(0.0, dV), rate_block=RB)
        rho, _ = ls.steady_state(gen, [W12, W13])
        r23[k] = rho[1, 2]; chi12[k] = rho[1, 0] / W12
    ana = W12 * W13 / (4 * G12 * (G13 - 1j * grid))
    return {"r23": r23, "ana": ana, "dev": np.abs(r23 - ana).max() / np.abs(ana).max(),
            "hwhm_abs2": hwhm(grid, np.abs(r23) ** 2), "hwhm_abs": hwhm(grid, np.abs(r23)),
            "hwhm_re": hwhm(grid, r23.real),
            "chi12_var": np.ptp(chi12.imag) / np.abs(chi12.imag).max()}, float(G13)


def V4b(P):
    print("V4b V steady state: rho23 = rho21 rho13, width from gamma_2s, chi12 flat")
    vr = P.proto["V_response"]
    grid = np.array(vr["delta_V_grid"]["values_meV"])
    W12a, W13a = vr["hbar_Omega12_meV"], vr["hbar_Omega13_meV"]
    res = {}
    for fac in (0.5, 0.25):          # linear-regime solver check at 0.05/0.005 and 0.025/0.0025 meV
        res[fac], G13 = v_scan(P, grid, W12a * fac, W13a * fac)
    ratio = res[0.5]["dev"] / res[0.25]["dev"]
    w = res[0.5]
    record("V4b linear regime: rho23 = rho21 rho13",
           w["dev"] < CRIT["V4b_weak_rel_dev"] and CRIT["V4b_weak_ratio"][0] < ratio < CRIT["V4b_weak_ratio"][1],
           W12_meV=W12a * 0.5, W13_meV=W13a * 0.5, rel_dev=w["dev"], ratio=ratio)
    r1 = abs(w["hwhm_abs2"] - G13) / G13
    r2 = abs(w["hwhm_abs"] - np.sqrt(3) * G13) / (np.sqrt(3) * G13)
    record("V4b linear regime: HWHM |rho23|^2 = Gamma13", r1 < CRIT["V4b_hwhm_rel_weak"],
           hwhm_meV=w["hwhm_abs2"], Gamma13_meV=G13, rel=r1)
    record("V4b linear regime: HWHM |rho23| = sqrt3 Gamma13", r2 < CRIT["V4b_hwhm_rel_weak"],
           hwhm_meV=w["hwhm_abs"], sqrt3_Gamma13=np.sqrt(3) * G13, rel=r2)
    # parameter checks: the V-response drive of the paper and a stronger candidate for the 1s tone
    cand = [(W12a, W13a, "V response"), (0.5, 0.05, "stronger")]
    for W12, W13, tag in cand:
        d, _ = v_scan(P, grid, W12, W13)
        res[(W12, W13)] = d
        rel = abs(d["hwhm_abs2"] - G13) / G13
        g = P.rates(4)
        G23 = 0.5 * (g["gamma_1s_to_0"] + g["gamma_2s_to_0"])
        record(f"V4b {tag} drive ({W12}, {W13}) meV: 2s width", rel < CRIT["V4b_hwhm_rel_approved"],
               kind="parameter" if tag == "V response" else "info", hwhm_meV=d["hwhm_abs2"], rel_broadening=rel,
               cross_param=W12 ** 2 / (4 * G23 * G13),
               pred_extra_hwhm_meV=W12 ** 2 / (4 * G23), rel_dev_vs_lowest_order=d["dev"],
               chi12_variation=d["chi12_var"])
    record("V4b Im chi12 flat vs delta_V (V-response drive)", res[(W12a, W13a)]["chi12_var"] < CRIT["V4b_chi12_flat"],
           kind="parameter", rel_variation=res[(W12a, W13a)]["chi12_var"])
    return grid, {k if isinstance(k, float) else f"{k[0]}_{k[1]}": v for k, v in res.items()}


# =============================================================================
# V5  dark-state following
# =============================================================================
def V5(P):
    print("V5  dark-state following (Eq. 69)")
    from scipy.integrate import cumulative_trapezoid as ctz
    xt = XI_TRANSFER_TEST
    x, W0 = xt["delay_over_sigma"], xt["hbar_Omega0_meV"]
    rows = []
    HD = 0.0
    for sigma in (1.5, 3.0, 6.0):
        gen = ls.Generator(P, ls.XI, T=4, dissipator=False, rate_block=RB)
        pro = ls.xi_transfer_protocol(P, sigma, x, W0)
        tr = ls.propagate(gen, pro)
        ob = ls.observables(tr, gen)
        for k in range(0, len(tr.t), 25):                 # H(t)|D(t)> = 0
            W12, W23 = tr.omegas[k]
            if np.hypot(W12, W23) < 1e-6 * W0:
                continue
            D = np.array([W23, 0, -W12]) / np.hypot(W12, W23)
            HD = max(HD, np.abs(gen.H(tr.omegas[k]) @ D).max() / np.hypot(W12, W23))
        # first-order adiabatic perturbation theory (Delta = delta = 0):
        # 1 - P_D(t) = |int_0^t theta_dot e^{i Phi} dt'|^2, Phi = int Omega0/2   (t in ps, Omega in rad/ps)
        s_t = float(sigma); a = x / s_t
        tbar = 0.5 * (pro.meta["t12_ps"] + pro.meta["t23_ps"])
        tf = np.linspace(0.0, pro.t_end, 400001)
        O0 = np.hypot(pro.amps[0] * pro.envs[0](tf), pro.amps[1] * pro.envs[1](tf)) / HB
        thd = 0.5 * a / np.cosh(a * (tf - tbar))
        I = ctz(thd * np.exp(1j * ctz(O0 / 2, tf, initial=0)), tf, initial=0)
        pert = np.interp(tr.t, tf, np.abs(I) ** 2)
        num = 1 - ob["P_D"]; ok = np.isfinite(num)
        dev = np.abs(num[ok] - pert[ok]).max() / num[ok].max()
        k = int(np.argmin(np.abs(tr.t - tbar)))
        eps = 2 * (0.5 * a) / (np.sqrt(2) * (W0 / HB) * np.exp(-x**2 / 8))
        rows.append({"sigma_ps": sigma, "eps": float(eps), "max_1-P_D": float(num[ok].max()),
                     "1-P_D(tbar)": float(num[k]), "1-rho33(tf)": float(1 - ob["rho33"][-1]),
                     "pert_1-rho33(tf)": float(pert[ok][-1]), "rel_dev": float(dev)})
        TRAJ.append((f"V5_case1_sigma{sigma}", tr, gen))
        record(f"V5 sigma={sigma}: 1-P_D(t) vs adiabatic PT", dev < CRIT["V5_pert_rel_dev"],
               rel_dev=dev, max_loss=float(num[ok].max()), final_loss=float(1 - ob["rho33"][-1]),
               final_loss_PT=float(pert[ok][-1]), eps=float(eps))
    record("V5 H|D> = 0 (Eq. 69)", HD < CRIT["V5_HD"], max_rel=HD)
    gen = ls.Generator(P, ls.XI, T=4, rate_block=RB)                    # informative: case 2, sigma = 1.5 ps
    tr = ls.propagate(gen, ls.xi_transfer_protocol(P, xt["sigma_ps"], x, W0))
    TRAJ.append(("V5_case2_sigma1.5", tr, gen))
    sm = ls.trajectory_summary(tr, gen)
    RESULTS["V5 info: case 2 test transfer"] = {"pass": True, "kind": "info",
                                                **{k: sm[k] for k in ("rho33_final", "max_rho22",
                                                   "int_rho22_ps", "N_emit", "P_ion", "P_D_min", "purity_min")}}
    return rows


# =============================================================================
# V6  plateau -> null-space steady state
# =============================================================================
def V6(P):
    print("V6  long plateau converges to the null-space steady state")
    cases = [("Xi EIT (Omega23=1, weak probe)", ls.XI, [0.1, 1.0], 0.5, 50.0),
             ("Xi CPT (Omega12=Omega23=1)", ls.XI, [1.0, 1.0], 0.5, 50.0),
             ("V response (0.1, 0.01)", ls.V, [0.1, 0.01], 0.2, 60.0)]
    out = {}
    for name, topo, W, tau, T in cases:
        gen = ls.Generator(P, topo, T=4, rate_block=RB)
        pro = ls.plateau_protocol(P, W, tau, T)
        tr = ls.propagate(gen, pro)
        TRAJ.append((f"V6_{topo.name}_{name.split()[1]}", tr, gen))
        rss, info = ls.steady_state(gen, W)
        t_check = pro.meta["t_off_ps"] - 10 * tau
        k = int(np.argmin(np.abs(tr.t - t_check)))
        resid_t = np.abs(tr.rho - rss).max(axis=(1, 2))
        res = resid_t[k]
        ok = res < CRIT["V6_residual"] and info["unique"]
        extra = {}
        if topo.name == "Xi":
            extra["P_D_ss"] = float(ls.dressed_populations(rss[None], np.array([W]),
                                                           np.real(np.diag(gen.H0)))["P_D"][0])
            extra["P_D_t"] = float(ls.observables(tr, gen)["P_D"][k])
        record(f"V6 {name}", ok, residual=float(res), t_check_ps=float(tr.t[k]),
               gap_meV=float(info["spectral_gap_per_ps"] * HB), unique=info["unique"], **extra)
        out[name] = (tr.t, resid_t)
    return out


# =============================================================================
# V7  independent integrators
# =============================================================================
def V7(P):
    print("V7  cross-checks: Radau and QuTiP on the strong-drive test transfer (case 2)")
    xt = XI_TRANSFER_TEST
    gen = ls.Generator(P, ls.XI, T=4, rate_block=RB)
    pro = ls.xi_transfer_protocol(P, xt["sigma_ps"], xt["delay_over_sigma"], xt["hbar_Omega0_meV"])
    ref = ls.propagate(gen, pro)
    rad = ls.propagate(gen, pro, method="Radau")
    d = np.abs(ref.rho - rad.rho).max()
    record("V7 Radau vs DOP853", d < CRIT["V7_max_abs"], max_abs=d)
    try:
        import qutip as qt
        H0 = qt.Qobj(gen.H0 / HB); h = [qt.Qobj(x / HB) for x in gen.hleg]       # rad/ps per meV of drive
        e = pro.envs; a = pro.amps
        H = [H0, [h[0], lambda t: a[0] * e[0](t)], [h[1], lambda t: a[1] * e[1](t)]]
        c_ops = [np.sqrt(g) * qt.Qobj(ls._op(dst, src)) for src, dst, g, _ in gen.jumps if g > 0]
        opts = {"atol": 1e-12, "rtol": 1e-10, "max_step": pro.max_step(), "nsteps": 10**7}
        res = qt.mesolve(H, qt.Qobj(ls._op(0, 0)), ref.t, c_ops=c_ops, options=opts)
        rq = np.stack([s.full() for s in res.states])
        dq = np.abs(ref.rho - rq).max()
        record("V7 QuTiP mesolve vs DOP853", dq < CRIT["V7_max_abs"], max_abs=dq,
               qutip=qt.__version__)
    except ImportError:
        RESULTS["V7 QuTiP mesolve vs DOP853"] = {"pass": True, "note": "qutip not installed; skipped"}


# =============================================================================
# V1  over all trajectories
# =============================================================================
def V1(P):
    print("V1  trace / Hermiticity / positivity over all trajectories")
    tr_e = herm = 0.0; lmin = {"dissipative": np.inf, "coherent-only": np.inf}
    for name, tr, gen in TRAJ:
        d = ls.diagnostics(tr.rho)
        tr_e = max(tr_e, d["trace_err"].max()); herm = max(herm, d["herm_err"].max())
        cls = "dissipative" if gen.dissipator else "coherent-only"
        lmin[cls] = min(lmin[cls], d["min_eig"].min())
    record("V1 trace", tr_e <= CRIT["V1_trace"], max_err=float(tr_e), n_traj=len(TRAJ))
    record("V1 Hermiticity", herm <= CRIT["V1_herm"], max_err=float(herm))
    record("V1 positivity (dissipative runs)", lmin["dissipative"] >= CRIT["V1_min_eig"],
           min_eig=float(lmin["dissipative"]))
    rt, at = ls.default_tolerances(ls.Generator(P, ls.XI, dissipator=False, rate_block=RB))
    record(f"V1 positivity (coherent-only runs, rtol {rt:g})", lmin["coherent-only"] >= CRIT["V1_min_eig"],
           min_eig=float(lmin["coherent-only"]))
    # diagnosis: coherent-only runs keep rho rank-1 (two exact zero eigenvalues);
    # the negative eigenvalue is integrator error and must scale with the tolerance
    xt = XI_TRANSFER_TEST
    gen = ls.Generator(P, ls.XI, T=4, dissipator=False, rate_block=RB)
    sc = {}
    for rt, at in ((1e-8, 1e-10), (1e-10, 1e-12), (1e-12, 1e-14)):
        m = min(ls.diagnostics(ls.propagate(gen, ls.xi_transfer_protocol(P, s_, xt["delay_over_sigma"],
                                                                         xt["hbar_Omega0_meV"]),
                                            rtol=rt, atol=at).rho)["min_eig"].min()
                for s_ in (1.5, 3.0, 6.0))
        sc[f"rtol_{rt:g}"] = float(m)
    RESULTS["V1 info: coherent-only min_eig vs rtol"] = {"pass": True, "kind": "info", **sc}
    print("      coherent-only min_eig vs rtol:", sc)


# =============================================================================
def write_report(P, t_run, path="validation_report.md"):
    solver = {k: v for k, v in RESULTS.items() if v.get("kind") == "solver"}
    param = {k: v for k, v in RESULTS.items() if v.get("kind") == "parameter"}
    info = {k: v for k, v in RESULTS.items() if v.get("kind") == "info"}
    ok = all(v["pass"] for v in solver.values())
    def rows(d):
        out = ["| check | result | key numbers |", "|---|---|---|"]
        for k, v in d.items():
            nums = ", ".join(f"{a} = {b:.4g}" if isinstance(b, float) else f"{a} = {b}"
                             for a, b in v.items() if a not in ("pass", "kind") and not isinstance(b, dict))
            out.append(f"| {k} | {'PASS' if v['pass'] else '**FAIL**'} | {nums} |")
        return out
    L = ["# Lindblad solver validation (V0–V7)\n",
         f"`lindblad_solver.py` {ls.__version__}, code SHA-256 `{ls.code_sha256()[:16]}`; "
         f"`params.yaml` SHA-256 `{P.sha256[:16]}`. Run time {t_run:.0f} s.\n",
         f"**Solver validation: {'PASS' if ok else 'FAIL'}** "
         f"({sum(v['pass'] for v in solver.values())}/{len(solver)} solver checks).  "
         f"Parameter checks: {sum(v['pass'] for v in param.values())}/{len(param)} pass "
         "(these test the run parameters of the paper, not the code).\n",
         "## Solver checks\n"] + rows(solver) + ["\n## Parameter checks\n"] + rows(param) + \
        ["\n## Information\n"] + rows(info) + ["\n## Pass criteria\n"] + \
        [f"- `{k}`: {v}" for k, v in CRIT.items()] + ["\n## Notes on the criteria\n"] + \
        [f"- **{k}**: {v}" for k, v in NOTES.items()]
    __import__("pathlib").Path(path).write_text("\n".join(L) + "\n")
    return ok


def run_lindblad(P):
    t0 = time.time()
    rng = np.random.default_rng(20260930)
    V0(P, rng); V2(P); v3 = V3(P); beat = V4(P); v4b = V4b(P); V5(P); v6 = V6(P); V7(P); V1(P)
    t_run = time.time() - t0
    ok = write_report(P, t_run)
    json.dump(RESULTS, open("validation_results.json", "w"), indent=1, default=float)
    keep = {n: (tr, g) for n, tr, g in TRAJ
            if n.startswith(("V4_beat_W1.0", "V5_case1_sigma1.5", "V5_case2", "V6_"))}
    scans = {"V3_Xi_weak_probe": {}, "V4b_V_steady_state": {"delta_V_meV": v4b[0]}}
    for W23, d in v3.items():
        scans["V3_Xi_weak_probe"][f"rho21_num_W23_{W23}"] = d["num"]
        scans["V3_Xi_weak_probe"][f"rho21_ana_W23_{W23}"] = d["ana"]
        scans["V3_Xi_weak_probe"]["Delta12_meV"] = d["grid"]
    for key, d in v4b[1].items():
        scans["V4b_V_steady_state"][f"rho23_num_{key}"] = d["r23"]
        scans["V4b_V_steady_state"][f"rho23_ana_{key}"] = d["ana"]
    ls.write_h5("validation_runs.h5", P, runs=keep, scans=scans,
                meta={"purpose": "validation V0-V7", "results": RESULTS})
    print(f"\nSOLVER VALIDATION: {'PASS' if ok else 'FAIL'}  ({t_run:.0f} s)  -> validation_report.md, "
          f"validation_results.json, validation_runs.h5")
    return ok


# =============================================================================
# reference values of an independent implementation (natural units E0, hbar/E0)
# =============================================================================
REF = {
 "ci1": dict(rho33_f=0.9964730171046922, max_rho22=0.020215502630644526, P_ion=0.02963475780132178, P_D_min=0.9797844940726399),
 "int1": dict(rho33_f=0.38296517475740904, max_rho22=0.9992357105131459),
 "ci2": dict(rho33_f=0.9380176841659907, N_emit=0.36553917993221713, P_ion=0.025660338119340948, P_D_min=0.9381428696965879, purity_min=0.9009603423906585, max_rho22=0.01802160986346212),
 "int2": dict(rho33_f=0.00988020746908565, N_emit=5.28696105799518, max_rho22=0.3750248050951619),
 "ci3": dict(rho33_f=0.2591719035670796, N_emit=3.7684614572449613, max_rho22=0.2374292216272182, P_ion=0.013594929141341984),
 "int3": dict(rho33_f=0.009158439098758152, N_emit=4.643041763177113),
 "plat2": dict(rho33_f=0.49999449990944866, N_emit=1.0000778244450348, max_rho22=0.058025321226870784),
 "plat3": dict(rho33_f=0.0009787107516256812, N_emit=5.109955607405087),
 "V2": dict(N_emit=0.03757918661949, rho23=0.007504933034258138),
 "V3": dict(rho23=0.005569764125559548),
 "V1": dict(rho23=0.05041753664455238),
}
TEST3_XI = (-3.04997, 0.460013)                 # (Delta12, Delta23) meV, case 3 of the test set
TEST3_V = (-3.04997, -2.93092)                  # (Delta12, Delta13) meV


def run_reference(P):
    out, bad = {}, 0

    def cmp(tag, s):
        nonlocal bad
        for k, ref in REF[tag].items():
            v = s[k]; dev = abs(v - ref) / abs(ref) if abs(ref) > 1e-3 else abs(v - ref)
            ok = dev <= (1e-4 if abs(ref) > 1e-3 else 1e-6)
            bad += not ok
            print(f"  {tag:6s} {k:12s} {v:.8g}  reference {ref:.8g}  dev {dev:.1e} {'ok' if ok else 'FAIL'}")
            out.setdefault(tag, {})[k] = (v, ref, dev)

    xt = XI_TRANSFER_TEST
    W0, S0, X0 = xt["hbar_Omega0_meV"], xt["sigma_ps"], xt["delay_over_sigma"]
    for case, diss, (D12, D23), tol in ((1, False, (0, 0), 1e-10), (2, True, (0, 0), 1e-8), (3, True, TEST3_XI, 1e-8)):
        for order, tag in (("counterintuitive", f"ci{case}"), ("intuitive", f"int{case}")):
            r = ls.xi_conversion(P, W0, W0, S0, X0, D12=D12, D23=D23, order=order, dissipator=diss,
                                 rtol=tol, atol=1e-12 if diss else 1e-12)
            if tag in REF: cmp(tag, ls.summary(r))
    for case, (D12, D23) in ((2, (0, 0)), (3, TEST3_XI)):
        r = ls.xi_plateau(P, 1.0, 1.0, D12=D12, D23=D23)
        cmp(f"plat{case}", ls.summary(r))
    for case, diss, (D12, D13) in ((1, False, (0, 0)), (2, True, (0, 0)), (3, True, TEST3_V)):
        r = ls.v_beat(P, 1.0, 1.0, 0.12, D12=D12, D13=D13, dissipator=diss, rtol=1e-10 if not diss else 1e-8)
        s = ls.summary(r)
        k = np.argmin(abs(r["t"] - (r["tc"] + 6 * r["sigma"])))
        s["rho23"] = float(abs(r["rho"][k, 1, 2]))
        cmp(f"V{case}", s)
    json.dump({k: {kk: list(vv) for kk, vv in v.items()} for k, v in out.items()}, open("reference_check.json", "w"), indent=1)
    print("REFERENCE CHECK", "PASS" if bad == 0 else f"FAIL ({bad})")
    return bad == 0


# =============================================================================
# first-order amplitude equations vs the weak-excitation conversion runs
# =============================================================================
def run_amplitude():
    raw = yaml.safe_load(open("params.yaml"))
    R = {k: float(v) for k, v in raw["rates"]["T_4K"]["hbar_gamma_meV"].items()}
    g2, g3 = R["gamma_1s_to_0"] / HB, R["gamma_2p_to_1s"] / HB
    S = {"partA": {}}
    for f in ("fig3_R1.json", "fig4_R1b.json"):
        S["partA"].update(json.load(open(f))["partA"])
    S["grid"] = json.load(open("fig5_R3.json"))["grid"]

    def amp(W12, W23, sigma, x, D12=0.0, D23=0.0, order="counterintuitive"):
        first, second = 6 * sigma, 6 * sigma + x * sigma
        t23, t12 = (first, second) if order == "counterintuitive" else (second, first)
        t_end = second + 6 * sigma
        e = lambda t, tc: np.exp(-0.5 * ((t - tc) / sigma) ** 2)

        def rhs(t, y):
            c2, c3 = y[0] + 1j * y[1], y[2] + 1j * y[3]
            a12, a23 = W12 * e(t, t12) / HB, W23 * e(t, t23) / HB
            d2 = -1j * ((D12 / HB - 0.5j * g2) * c2 + 0.5 * a12 + 0.5 * a23 * c3)
            d3 = -1j * (((D12 + D23) / HB - 0.5j * g3) * c3 + 0.5 * a23 * c2)
            return [d2.real, d2.imag, d3.real, d3.imag, g2 * abs(c2) ** 2]

        s = solve_ivp(rhs, (0, t_end), [0, 0, 0, 0, 0], method="RK45", rtol=1e-10, atol=1e-16,
                      max_step=sigma / 40)
        c3 = s.y[2, -1] + 1j * s.y[3, -1]
        r33, N = abs(c3) ** 2, s.y[4, -1]
        return dict(rho33=r33, N_emit=N, eta=r33 / (r33 + N))

    out = []
    W12b = 0.05 * min(5.0, R["gamma_1s_to_0"] / 2)
    c3 = raw["protocol"]["Part_A"]["case3"]["Xi"]
    for name, kw in (("Xi_ci_case2", {}), ("Xi_seq_case2", dict(order="intuitive")),
                     ("Xi_ci_case3", dict(D12=c3["Delta12_meV"], D23=c3["Delta23_meV"]))):
        a = amp(W12b, 5.0, 1.5, 1.25, **kw); L = S["partA"][name]
        out.append(dict(run=name, amp=a, lindblad=dict(rho33=L["rho33_f"], N_emit=L["N_emit"], eta=L["eta"]),
                        rel_dev_eta=abs(a["eta"] - L["eta"]) / L["eta"]))
    for r in S["grid"]:
        if r["order"] == "counterintuitive" and r["x"] == 1.25 and r["W23"] in (3.0, 5.0):
            W12 = 0.05 * min(r["W23"], R["gamma_1s_to_0"] / 2)
            a = amp(W12, r["W23"], r["sigma"], r["x"])
            out.append(dict(run=f"grid W23={r['W23']} sigma={r['sigma']}", amp=a,
                            lindblad=dict(rho33=r["rho33_f"], N_emit=r["N_emit"], eta=r["eta"]),
                            rel_dev_eta=abs(a["eta"] - r["eta"]) / r["eta"]))
    for o in out:
        print(f"{o['run']:28s} eta amp {o['amp']['eta']:.5f}  Lindblad {o['lindblad']['eta']:.5f}"
              f"  rel dev {o['rel_dev_eta']:.1e}  rho33 {o['amp']['rho33']:.4e} / {o['lindblad']['rho33']:.4e}")
    print("max relative deviation of eta:", max(o["rel_dev_eta"] for o in out))
    json.dump(out, open("xi5_amplitude_check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    P = ls.Params("params.yaml")
    ok = True
    if which in ("lindblad", "all"):
        ok &= run_lindblad(P)
    if which in ("reference", "all"):
        print("\n######## reference check")
        ok &= run_reference(P)
    if which in ("amplitude", "all"):
        print("\n######## amplitude equations")
        run_amplitude()
    sys.exit(0 if ok else 1)
