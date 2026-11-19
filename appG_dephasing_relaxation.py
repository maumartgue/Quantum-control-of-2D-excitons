"""
appG_dephasing_relaxation.py -- App. G and Sect. VI G of the paper "Conditions for Dark-State Coherent
Control of Excitons in TMD Monolayers: A Reduced Intravalley Lindblad Framework for WSe2": sensitivity of the
Xi conversion and of the cascade transparency window to a pure dephasing and to a population decay of the
2p exciton, Fig. S1 (NJP Supplementary Material) / Fig. 9 (arXiv), Table SVII / XI.

These terms are NOT produced by the fixed-K0 model; they are added to the Xi generator of lindblad_solver.py
by hand, as sensitivity analyses outside the model:

  pure dephasing of 2p   L_phi = sqrt(2 gamma_phi) |3><3|            (Eq. G1)
                         damps rho13 and rho23 at gamma_phi, leaves the populations unchanged;
  population decay of 2p at the rate gamma_rel, either
      into a sink        d rho/dt |rel = -(gamma_rel/2) {P3, rho}     (Eq. G2; P3 = |3><3|)
                         (non-radiating states, e.g. momentum-dark excitons); P_sink = 1 - Tr rho;
      or back to 1s      jump operator sqrt(gamma_rel) |2><3|          (trace preserving).
  Width split at the measured 2p width:  hbar gamma_rel/2 + hbar gamma_phi = 3.955 meV
  (half of the 7.91 meV FWHM at T -> 0), with the share f = hbar gamma_rel / (2 x 3.955 meV).

Figures of merit (at the end of the run t_f unless stated):
  eta      = rho33 / (rho33 + N_emit)                     conversion efficiency (Eq. 76)
  eta_2p   = rho33 / (rho33 + P_sink + N_emit)            fraction of the excitation left in 2p
  eta_dark = (rho33 + P_sink) / (rho33 + P_sink + N_emit) dark-sector fraction
  C(t)     = |rho13|^2 / (rho11 rho33)                    coherence fraction, quoted at t_c = t12 + 3 sigma
  vertex suppression: max_t rho22 / rho33(t_f) (dephasing tasks, as summary() of lindblad_solver.py) and
                      max_t rho22 / max_t rho33 (task relax, where rho33 decays after the pulses)
  Weak-probe window centre: residual absorption Gamma12 (Gamma13 + gamma_phi) / (Gamma12 (Gamma13 + gamma_phi)
  + Omega23^2/4), normalized to the bare 1s line, checked against the Liouvillian null space.

Protocol: the Xi Gaussian pair of xi_conversion() in lindblad_solver.py (4 K, photon dissipator, n_X = 0,
two-photon resonance, tau_d = 1.25 sigma), hbar Omega12 = 0.05 min(hbar Omega23, hbar gamma_1s/2), DOP853
with max_step sigma/20, rtol 1e-8, atol 1e-13.

Tasks (each writes into the output folder, default ./appG_output):
  sweep             gamma_phi from 0 to 4 meV, sigma = 1.5 ps, hbar Omega23 = 1, 2.5, 5 meV, both pulse
                    orders; window centre against the closed formula          -> gamma_phi_sweep.csv
  sigma             sigma = 1.1, 1.5, 1.9 ps, 37 values of gamma_phi, counterintuitive order;
                    half point of eta in gamma_phi sigma                          -> gamma_phi_sigma.csv
  coherence_time    C at t12, t12 + 2 sigma, t12 + 3 sigma and t_f (basis of t_c)  -> coherence_time.csv
  halfpoint         dense gamma_phi grid in 0.005-0.3 meV, half point of C(t_c)
                    (needs the output of task sigma)                             -> coherence_halfpoint.csv
  relax             population decay of 2p: block A (measured width, f = 0 ... 1, both destinations and
                    orders), block B (f = 1, sink, width scale 1e-3 ... 1, three sigma: half points in
                    gamma_rel sigma), block C (zero width)                       -> relax_2p.csv
  independent       re-integration of five dephasing points with an own density-matrix right-hand side
                    (no lindblad_solver engine), compared with task sweep
  figure            Fig. S1 / Fig. 9 from the outputs of sigma, halfpoint and relax -> fig_dephasing.pdf/png
  all               every task above, in this order (about 5-10 min on a desktop PC)

Usage:  python appG_dephasing_relaxation.py [task ...] [--out DIR]     (reads params.yaml; default: all)
Units: energies and hbar*Omega, hbar*gamma in meV, times in ps; gamma_phi sigma and gamma_rel sigma are
dimensionless (gamma a rate, gamma sigma = hbar gamma [meV] x sigma [ps] / hbar).
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import lindblad_solver as ls

HW = 3.955                         # meV, half the measured 2p width (FWHM 7.91 meV at T -> 0)
XD = 1.25                          # tau_d / sigma
RTOL0, ATOL0 = 1e-8, 1e-13         # tolerances of xi_conversion() in lindblad_solver.py
W23S = [1.0, 2.5, 5.0]
SIGMAS = [1.1, 1.5, 1.9]
P3 = ls._op(2, 2)
I3 = np.eye(3)

# gamma_phi grids (meV)
G_SWEEP = sorted([0.0] + list(np.logspace(-3, np.log10(4.0), 15)) + [3.955])
G_SIGMA = sorted(G_SWEEP + list(np.logspace(-1, np.log10(2.0), 20)))
G_HALF = sorted(set([0.0] + list(np.logspace(np.log10(0.005), np.log10(0.3), 30))
                    + [g for g in G_SIGMA if 0.005 <= g <= 0.3]))
SCALES_RELAX = [0.0] + list(np.logspace(-3, 0, 25))


# =============================================================================
# generator and integration
# =============================================================================
def generator(P, g_phi=0.0, g_rel=0.0, dest="sink", D12=0.0, D23=0.0):
    """Xi generator of lindblad_solver.py plus pure dephasing (Eq. G1) and population decay (Eq. G2) of 2p."""
    gen = ls.Generator(P, ls.XI, T=4, frame_meV=(D12, D12 + D23))
    if g_phi > 0:
        gen.L0 = gen.L0 + (g_phi / ls.HB) * ls._diss(np.sqrt(2) * P3)
    g = g_rel / ls.HB
    if g > 0:
        if dest == "sink":                              # -(g/2){P3, rho}, column stacking
            gen.L0 = gen.L0 - 0.5 * g * (np.kron(I3, P3) + np.kron(P3.T, I3))
        elif dest == "1s":                              # jump |2><3| (state indices 1 <- 2)
            gen.L0 = gen.L0 + g * ls._diss(ls._op(1, 2))
        else:
            raise ValueError(dest)
    return gen


def conversion(P, W23, sigma, g_phi=0.0, g_rel=0.0, dest="sink", order="counterintuitive",
               rtol=RTOL0, atol=ATOL0, x=XD):
    """The Xi Gaussian pair of xi_conversion() (lindblad_solver.py) with the generator above."""
    W12 = P.weak(W23)
    first, second = 6 * sigma, 6 * sigma + x * sigma
    t23, t12 = (first, second) if order == "counterintuitive" else (second, first)
    t_end = second + 6 * sigma
    bps = [t12 - 6 * sigma, t12, t12 + 6 * sigma, t23 - 6 * sigma, t23, t23 + 6 * sigma]
    gen = generator(P, g_phi, g_rel, dest)
    r = ls.integrate(gen, (W12, W23),
                     (ls.Envelope("gauss", tc=t12, sigma=sigma), ls.Envelope("gauss", tc=t23, sigma=sigma)),
                     t_end, bps=bps, rtol=rtol, atol=atol, max_step=sigma / 20, dt_out=ls.OUTPUT_DT_PS)
    r.update(t12=t12, t23=t23, sigma=sigma, x=x, order=order)
    return r


def at(r, t):
    """State on the output grid point nearest to t: (t, rho11, rho33, |rho13|, 1 - C)."""
    k = int(np.argmin(np.abs(r["t"] - t)))
    rho = r["rho"][k]
    p11, p33, a13 = rho[0, 0].real, rho[2, 2].real, abs(rho[0, 2])
    return float(r["t"][k]), float(p11), float(p33), float(a13), float(1 - a13 ** 2 / (p11 * p33))


def C_at(r, t):
    """(t, C) on the output grid point nearest to t."""
    k = int(np.argmin(np.abs(r["t"] - t)))
    rho = r["rho"][k]
    return float(r["t"][k]), float(abs(rho[0, 2]) ** 2 / (rho[0, 0].real * rho[2, 2].real))


def write_csv(path, rows):
    keys = []
    for q in rows:
        keys += [k for k in q if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)


def read_csv(path):
    return list(csv.DictReader(open(path)))


def health(rows, trace_key="trace_err"):
    c = {"max_trace_err": max(q[trace_key] for q in rows), "min_min_eig": min(q["min_eig"] for q in rows),
         "max_one_exciton": max(q["max_one_exciton"] for q in rows)}
    c["pass"] = bool(c["max_trace_err"] <= 1e-8 and c["min_min_eig"] >= -1e-9)
    return c


def half_log(sub, xkey, ykey, linear_in_y=True):
    """Point where ykey falls to half its value at xkey = 0, interpolated in log10(xkey) between the
    bracketing grid points (linear in y, or in log y)."""
    sub = sorted(sub, key=lambda q: q[xkey]); y0 = sub[0][ykey]; tgt = y0 / 2
    pos = [q for q in sub if q[xkey] > 0]
    for a, b in zip(pos[:-1], pos[1:]):
        if a[ykey] >= tgt > b[ykey]:
            la, lb = np.log10(a[xkey]), np.log10(b[xkey])
            if linear_in_y:
                f = (tgt - a[ykey]) / (b[ykey] - a[ykey])
            else:
                f = (np.log(tgt) - np.log(a[ykey])) / (np.log(b[ykey]) - np.log(a[ykey]))
            return float(10 ** (la + f * (lb - la)))
    return None


# =============================================================================
# task sweep
# =============================================================================
def task_sweep(P, out):
    checks = {}
    # L_phi acts on a random density matrix as stated (rho13, rho23 damped at gamma_phi)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho_t = A @ A.conj().T; rho_t /= np.trace(rho_t)
    gtest = 2.0
    d = ls.unvec((gtest / ls.HB) * ls._diss(np.sqrt(2) * P3) @ ls.vec(rho_t))
    rate = gtest / ls.HB
    exp = np.zeros((3, 3), complex)
    for (a, b) in ((0, 2), (2, 0), (1, 2), (2, 1)):
        exp[a, b] = -rate * rho_t[a, b]
    err = float(np.abs(d - exp).max() / rate)
    checks["Lphi_action"] = {"max_abs_dev_over_rate": err, "pass": bool(err < 1e-12)}

    KEYS = ["eta", "rho33_f", "N_emit", "max_rho22", "vertex_suppression", "int_vertex_over_n2p_ps",
            "P_ion_per_2p", "P_D_min", "max_one_exciton", "trace_err", "min_eig", "purity_min"]
    rows = []
    for W23 in W23S:
        for g in G_SWEEP:
            for order, tag in (("counterintuitive", "ci"), ("intuitive", "seq")):
                r = conversion(P, W23, 1.5, g_phi=g, order=order)
                s = ls.summary(r)
                rho = r["rho"][-1]
                C = float(abs(rho[0, 2]) ** 2 / (rho[0, 0].real * rho[2, 2].real))
                row = {"hbar_gamma_phi_meV": g, "hbar_Omega23_meV": W23, "hbar_Omega12_meV": P.weak(W23),
                       "order": tag, **{k: s[k] for k in KEYS}, "C": C, "nfev": s["nfev"]}
                row["one_minus_C"] = 1 - row["C"]
                rows.append(row)
    for row in rows:
        other = [q for q in rows if q["hbar_Omega23_meV"] == row["hbar_Omega23_meV"]
                 and q["hbar_gamma_phi_meV"] == row["hbar_gamma_phi_meV"] and q["order"] != row["order"]][0]
        ci, sq = (row, other) if row["order"] == "ci" else (other, row)
        row["eta_ci_over_seq"] = ci["eta"] / sq["eta"]

    # gamma_phi = 0 reproduces xi_conversion() of lindblad_solver.py (same integration, no added term)
    dev = []
    for W23 in W23S:
        for order, tag in (("counterintuitive", "ci"), ("intuitive", "seq")):
            e = ls.summary(ls.xi_conversion(P, P.weak(W23), W23, 1.5, XD, order=order))["eta"]
            m = [q for q in rows if q["hbar_gamma_phi_meV"] == 0 and q["hbar_Omega23_meV"] == W23 and q["order"] == tag][0]
            dev.append(abs(m["eta"] / e - 1))
    checks["gamma_phi_0_equals_xi_conversion"] = {"max_rel_dev": max(dev), "pass": bool(max(dev) <= 1e-12)}

    # weak-probe window centre: closed formula against the Liouvillian null space (rates of the response runs)
    W12p = 1e-4
    R = P.rates(4, ls.RATES_TAU0)
    G12, G13 = R["gamma_1s_to_0"] / 2, R["gamma_2p_to_1s"] / 2   # hbar Gamma (meV) at 4 K

    def formula(W23, g):
        a = G12 * (G13 + g)
        return a / (a + W23 ** 2 / 4)

    def lindblad(W23, g):
        def im_chi(w):
            gen = ls.Generator(P, ls.XI, T=4, frame_meV=ls.xi_frame(0.0, 0.0), rate_block=ls.RATES_TAU0)
            if g > 0:
                gen.L0 = gen.L0 + (g / ls.HB) * ls._diss(np.sqrt(2) * P3)
            rho, info = ls.steady_state(gen, [W12p, w])
            return ls.chi(rho, (0, 1), W12p).imag
        return im_chi(W23) / im_chi(0.0)

    win = []
    for g, W23 in ((0.0, 1.0), (0.0, 5.0), (3.955, 1.0), (3.955, 5.0), (4.0, 1.0), (4.0, 5.0)):
        f_, l_ = formula(W23, g), lindblad(W23, g)
        win.append({"hbar_gamma_phi_meV": g, "hbar_Omega23_meV": W23, "formula": f_, "lindblad": l_,
                    "rel_dev": abs(l_ / f_ - 1)})
    tol = [{"hbar_Omega23_meV": W, "hbar_gamma_phi_depth_0.9_ueV": 1e3 * (W ** 2 / (36 * G12) - G13)}
           for W in (0.5, 1.0, 1.5)]
    checks["window_centre"] = {"rows": win, "max_rel_dev": max(q["rel_dev"] for q in win),
                               "pass": bool(max(q["rel_dev"] for q in win) <= 1e-3)}
    checks["health"] = health(rows)
    conv = []
    for W23 in W23S:
        e_t = ls.summary(conversion(P, W23, 1.5, g_phi=4.0, rtol=RTOL0 / 10, atol=ATOL0 / 10))["eta"]
        e0 = [q for q in rows if q["hbar_gamma_phi_meV"] == 4.0 and q["hbar_Omega23_meV"] == W23 and q["order"] == "ci"][0]["eta"]
        conv.append(abs(e_t / e0 - 1))
    checks["convergence"] = {"max_rel_change": max(conv), "pass": bool(max(conv) <= 1e-6)}

    half = {tag: {str(W): half_log([q for q in rows if q["hbar_Omega23_meV"] == W and q["order"] == tag],
                                   "hbar_gamma_phi_meV", "eta") for W in W23S} for tag in ("ci", "seq")}
    write_csv(out / "gamma_phi_sweep.csv", rows)
    return {"checks": checks, "half_eta_meV": half, "EIT_tolerance_depth_0.9": tol}


# =============================================================================
# task sigma
# =============================================================================
def task_sigma(P, out):
    KEYS = ["eta", "rho33_f", "N_emit", "max_rho22", "vertex_suppression", "P_D_min",
            "max_one_exciton", "trace_err", "min_eig"]
    rows = []
    for sigma in SIGMAS:
        for W23 in W23S:
            for g in G_SIGMA:
                r = conversion(P, W23, sigma, g_phi=g)
                s = ls.summary(r)
                (t1g, C1), (t2g, C2), (tfg, Cf) = (C_at(r, r["t12"]), C_at(r, r["t12"] + 2 * sigma),
                                                   C_at(r, r["t"][-1]))
                omc1, omc2, omcf = 1 - C1, 1 - C2, 1 - Cf
                row = {"sigma_ps": sigma, "hbar_Omega23_meV": W23, "hbar_Omega12_meV": P.weak(W23),
                       "hbar_gamma_phi_meV": g, **{k: s[k] for k in KEYS},
                       "t1_ps": t1g, "t2_ps": t2g, "tf_ps": tfg, "C_t1": C1, "C_t2": C2, "C_tf": Cf,
                       "one_minus_C_t1": omc1, "one_minus_C_t2": omc2, "one_minus_C_tf": omcf,
                       "R": Cf / (C2 * np.exp(-2 * g * (tfg - t2g) / ls.HB)) if C2 > 0 else np.nan,
                       "nfev": s["nfev"]}
                rows.append(row)
    checks = {"health": health(rows)}
    dev = []
    for sigma in SIGMAS:
        for W23 in W23S:
            e = ls.summary(ls.xi_conversion(P, P.weak(W23), W23, sigma, XD))["eta"]
            m = [q for q in rows if q["sigma_ps"] == sigma and q["hbar_Omega23_meV"] == W23 and q["hbar_gamma_phi_meV"] == 0][0]
            dev.append(abs(m["eta"] / e - 1))
    checks["gamma_phi_0_equals_xi_conversion"] = {"max_rel_dev": max(dev), "pass": bool(max(dev) <= 1e-12)}
    conv = []
    for W23 in W23S:
        et = ls.summary(conversion(P, W23, 1.1, g_phi=4.0, rtol=RTOL0 / 10, atol=ATOL0 / 10))["eta"]
        e0 = [q for q in rows if q["sigma_ps"] == 1.1 and q["hbar_Omega23_meV"] == W23 and q["hbar_gamma_phi_meV"] == 4.0][0]["eta"]
        conv.append(abs(et / e0 - 1))
    checks["convergence"] = {"max_rel_change": max(conv), "pass": bool(max(conv) <= 1e-6)}
    half = []
    for sigma in SIGMAS:
        for W23 in W23S:
            sub = [q for q in rows if q["sigma_ps"] == sigma and q["hbar_Omega23_meV"] == W23]
            g = half_log(sub, "hbar_gamma_phi_meV", "eta")
            e3955 = [q for q in sub if q["hbar_gamma_phi_meV"] == 3.955][0]["eta"] / sorted(sub, key=lambda q: q["hbar_gamma_phi_meV"])[0]["eta"]
            half.append({"sigma_ps": sigma, "hbar_Omega23_meV": W23, "g_half_meV": g,
                         "g_half_sigma_over_hbar": g * sigma / ls.HB, "eta_3955_over_eta0": e3955})
    v = [h["g_half_sigma_over_hbar"] for h in half]
    write_csv(out / "gamma_phi_sigma.csv", rows)
    return {"checks": checks, "half_eta": half,
            "half_eta_sigma_over_hbar": {"mean": float(np.mean(v)), "min": min(v), "max": max(v)}}


# =============================================================================
# task coherence_time
# =============================================================================
def task_coherence_time(P, out):
    def quantities(r):
        s = ls.summary(r)
        sig = r["sigma"]
        row = {k: s[k] for k in ("eta", "rho33_f", "N_emit", "vertex_suppression", "max_one_exciton",
                                 "trace_err", "min_eig")}
        for name, t in (("t1", r["t12"]), ("tc2", r["t12"] + 2 * sig), ("tc3", r["t12"] + 3 * sig),
                        ("tf", r["t"][-1])):
            tt, p11, p33, a13, omc = at(r, t)
            row[f"{name}_ps"] = tt
            row[f"one_minus_C_{name}"] = omc
            if name != "t1":
                row[f"rho11_{name}"] = p11; row[f"rho33_{name}"] = p33; row[f"abs_rho13_{name}"] = a13
        return row

    pr = P.proto
    base = pr["Xi_conversion"]["baseline"]
    S0, X0, W23b = base["sigma_ps"], base["delay_over_sigma"], base["hbar_Omega23_meV"]
    W12b = P.weak(W23b)
    c3 = pr["Part_A"]["case3"]["Xi"]
    rows = []
    # run A: cases 2 and 3 of fig3_R1.py at the baseline, gamma_phi = 0, exactly as there
    for case, (d12, d23) in ((2, (0, 0)), (3, (c3["Delta12_meV"], c3["Delta23_meV"]))):
        r = ls.xi_conversion(P, W12b, W23b, S0, X0, D12=d12, D23=d23, order="counterintuitive",
                             dissipator=True, rtol=1e-8, atol=1e-14)
        rows.append({"run": "A", "case": case, "sigma_ps": S0, "x": X0, "hbar_Omega23_meV": W23b,
                     "hbar_Omega12_meV": W12b, "D12_meV": d12, "D23_meV": d23, "hbar_gamma_phi_meV": 0.0,
                     "rtol": 1e-8, "atol": 1e-14, **quantities(r)})
    # run B: baseline with dephasing, the 37 gamma_phi values of task sigma
    for g in G_SIGMA:
        r = conversion(P, 5.0, 1.5, g_phi=g)
        rows.append({"run": "B", "case": "", "sigma_ps": 1.5, "x": 1.25, "hbar_Omega23_meV": 5.0,
                     "hbar_Omega12_meV": P.weak(5.0), "D12_meV": 0.0, "D23_meV": 0.0, "hbar_gamma_phi_meV": g,
                     "rtol": RTOL0, "atol": ATOL0, **quantities(r)})
    e0 = [q for q in rows if q["run"] == "B"][0]["eta"]
    for q in rows:
        if q["run"] == "B":
            q["eta_over_eta0"] = q["eta"] / e0
    A2, A3 = rows[0], rows[1]
    quoted = {"case2": {"eta": (A2["eta"], 0.479, 3), "one_minus_C_tf": (A2["one_minus_C_tf"], 1.1e-3, 2)},
              "case3": {"rho33_f": (A3["rho33_f"], 1.5e-4, 2), "N_emit": (A3["N_emit"], 4.7e-3, 2),
                        "eta": (A3["eta"], 0.030, 2), "vertex_suppression": (A3["vertex_suppression"], 3.4, 2),
                        "one_minus_C_tf": (A3["one_minus_C_tf"], 6.4e-3, 2)}}
    qc, ok = {}, True
    for case, dd in quoted.items():
        for k, (v, q, nd) in dd.items():
            good = bool(np.isclose(float(f"{v:.{nd}g}"), q, rtol=1e-9, atol=0))
            ok &= good
            qc[f"{case}.{k}"] = {"value": v, "quoted_in_paper": q, "match": good}
    checks = {"paper_values_cases_2_3": {"rows": qc, "pass": bool(ok)}, "health": health(rows)}
    analysis = {}
    for row in (A2, A3):
        v = {k: row[f"one_minus_C_{k}"] for k in ("t1", "tc2", "tc3", "tf")}
        analysis[f"case{row['case']}"] = {"one_minus_C": v, "rel_change_tc2_to_tc3": v["tc3"] / v["tc2"] - 1,
                                          "rel_change_tc3_to_tf": v["tf"] / v["tc3"] - 1}
    write_csv(out / "coherence_time.csv", rows)
    return {"checks": checks, "analysis": analysis}


# =============================================================================
# task halfpoint
# =============================================================================
def task_halfpoint(P, out):
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    sigf = out / "gamma_phi_sigma.csv"
    if not sigf.exists():
        sys.exit("task halfpoint needs the output of task sigma (gamma_phi_sigma.csv)")
    rows = []
    for sigma in SIGMAS:
        for W23 in W23S:
            for g in G_HALF:
                r = conversion(P, W23, sigma, g_phi=g)
                s = ls.summary(r)
                tc, p11, p33, a13, omc = at(r, r["t12"] + 3 * sigma)
                tf, _, _, _, omf = at(r, r["t"][-1])
                rows.append({"sigma_ps": sigma, "hbar_Omega23_meV": W23, "hbar_Omega12_meV": P.weak(W23),
                             "hbar_gamma_phi_meV": g, "tc_ps": tc, "tf_ps": tf,
                             "one_minus_C_tc": omc, "one_minus_C_tf": omf,
                             "rho11_tc": p11, "rho33_tc": p33, "abs_rho13_tc": a13,
                             "eta": s["eta"], "rho33_f": s["rho33_f"], "N_emit": s["N_emit"],
                             "max_one_exciton": s["max_one_exciton"], "trace_err": s["trace_err"],
                             "min_eig": s["min_eig"], "purity_min": s["purity_min"], "nfev": s["nfev"]})
            e0 = [q for q in rows if q["sigma_ps"] == sigma and q["hbar_Omega23_meV"] == W23
                  and q["hbar_gamma_phi_meV"] == 0][0]["eta"]
            for q in rows:
                if q["sigma_ps"] == sigma and q["hbar_Omega23_meV"] == W23:
                    q["eta_over_eta0"] = q["eta"] / e0

    def sub(sigma, W23):
        return sorted([q for q in rows if q["sigma_ps"] == sigma and q["hbar_Omega23_meV"] == W23],
                      key=lambda q: q["hbar_gamma_phi_meV"])

    mono = []
    for sigma in SIGMAS:
        for W in W23S:
            y = np.array([q["one_minus_C_tc"] for q in sub(sigma, W)])
            mono.append(bool(np.all(np.diff(y) > 0)))
    checks = {"monotonic_one_minus_C": {"pass": all(mono)}, "health": health(rows)}
    sig_rows = [{k: float(v) for k, v in q.items()} for q in read_csv(sigf)]
    ana = []
    for sigma in SIGMAS:
        for W in W23S:
            s_ = [q for q in sub(sigma, W) if q["hbar_gamma_phi_meV"] > 0]
            lg = np.log10([q["hbar_gamma_phi_meV"] for q in s_])
            y = np.array([q["one_minus_C_tc"] for q in s_])
            er = np.array([q["eta_over_eta0"] for q in s_])
            k = int(np.where((y[:-1] < 0.5) & (y[1:] >= 0.5))[0][0])
            f = (0.5 - y[k]) / (y[k + 1] - y[k])
            ga = 10 ** (lg[k] + f * (lg[k + 1] - lg[k]))
            py = PchipInterpolator(lg, y)
            gb = 10 ** brentq(lambda u: py(u) - 0.5, lg[k], lg[k + 1], xtol=1e-14)
            gh = half_log([q for q in sig_rows if q["sigma_ps"] == sigma and q["hbar_Omega23_meV"] == W],
                          "hbar_gamma_phi_meV", "eta")
            ana.append({"sigma_ps": sigma, "hbar_Omega23_meV": W, "gC_meV": ga, "gC_pchip_meV": gb,
                        "gC_sigma_over_hbar": ga * sigma / ls.HB, "g_half_eta_over_gC": gh / ga,
                        "eta_over_eta0_at_gC": float(er[k] + f * (er[k + 1] - er[k]))})
    v = [a["gC_sigma_over_hbar"] for a in ana]
    write_csv(out / "coherence_halfpoint.csv", rows)
    return {"checks": checks, "analysis": ana,
            "gC_sigma_over_hbar": {"mean": float(np.mean(v)), "min": min(v), "max": max(v)},
            "ln2_over_6": math.log(2) / 6}


# =============================================================================
# task relax
# =============================================================================
def relax_metrics(r, g_rel, dest):
    rho, t = r["rho"], r["t"]
    p11, p22, p33 = (rho[:, k, k].real for k in range(3))
    trace = np.einsum("tii->t", rho).real
    P_sink = float(1 - trace[-1]) if dest == "sink" else 0.0
    int33 = float(np.trapezoid(p33, t))       # gamma_rel * int rho33 dt must equal the lost trace (sink)
    book = abs(g_rel / ls.HB * int33 - P_sink) / max(P_sink, 1e-300) if (dest == "sink" and g_rel > 0) else 0.0
    N = float(r["N_emit"][-1]); n33 = float(p33[-1])
    tot = n33 + P_sink + N
    kmax = int(np.argmax(p33))
    tc = r["t12"] + 3 * r["sigma"]; kc = int(np.argmin(np.abs(t - tc)))
    C = float(abs(rho[kc, 0, 2]) ** 2 / (p11[kc] * p33[kc])) if p33[kc] > 0 else np.nan
    herm = 0.5 * (rho + np.conj(np.transpose(rho, (0, 2, 1))))
    lmin = float(np.linalg.eigvalsh(herm)[:, 0].min())
    return {"rho33_f": n33, "N_emit": N, "P_sink": P_sink,
            "eta": n33 / (n33 + N) if n33 + N > 0 else np.nan,
            "eta_2p": n33 / tot if tot > 0 else np.nan,
            "eta_dark": (n33 + P_sink) / tot if tot > 0 else np.nan,
            "rho33_max": float(p33[kmax]), "t_rho33_max_ps": float(t[kmax]),
            "max_rho22": float(p22.max()),
            "vertex_suppression": float(p22.max() / p33[kmax]) if p33[kmax] > 0 else np.nan,
            "C_tc": C, "t_c_ps": float(t[kc]), "max_one_exciton": float((1 - p11).max()),
            "trace_f": float(trace[-1]), "bookkeeping_rel": book, "min_eig": lmin, "nfev": r["nfev"]}


def task_relax(P, out):
    rows = []

    def do(block, W23, sigma, s, f, dest, order):
        g_rel = 2 * f * HW * s
        g_phi = (1 - f) * HW * s
        r = conversion(P, W23, sigma, g_phi=g_phi, g_rel=g_rel, dest=dest, order=order)
        rows.append({"block": block, "dest": dest, "order": order, "sigma_ps": sigma, "hbar_Omega23_meV": W23,
                     "hbar_Omega12_meV": P.weak(W23), "width_scale": s, "f_population": f,
                     "hbar_gamma_rel_meV": g_rel, "hbar_gamma_phi_meV": g_phi,
                     "extra_halfwidth_meV": g_rel / 2 + g_phi, **relax_metrics(r, g_rel, dest)})

    # block A: the measured width, split f = 0 ... 1, both destinations, both orders, sigma = 1.5 ps
    for dest in ("sink", "1s"):
        for order in ("counterintuitive", "sequential"):
            for W in W23S:
                for f in (0.0, 0.25, 0.5, 0.75, 1.0):
                    if f == 0.0 and dest == "1s":
                        continue                  # f = 0 has no decay: same run as dest = sink
                    do("A", W, 1.5, 1.0, f, dest, order)
    # block B: pure population decay into the sink (f = 1), width scale 1e-3 ... 1, three sigma
    for sigma in SIGMAS:
        for W in W23S:
            for s in SCALES_RELAX:
                do("B", W, sigma, s, 1.0, "sink", "counterintuitive")
    # block C: the zero-width reference, both orders, sigma = 1.5 ps
    for order in ("counterintuitive", "sequential"):
        for W in W23S:
            do("C", W, 1.5, 0.0, 0.0, "sink", order)

    checks = {}
    # f = 0 at the measured width is pure dephasing with hbar gamma_phi = 3.955 meV
    dev = []
    for W in W23S:
        m = [q for q in rows if q["block"] == "A" and q["order"] == "counterintuitive" and q["dest"] == "sink"
             and q["hbar_Omega23_meV"] == W and q["f_population"] == 0.0][0]
        e = ls.summary(conversion(P, W, 1.5, g_phi=HW))["eta"]
        dev.append(abs(m["eta"] / e - 1))
    checks["f0_equals_pure_dephasing"] = {"max_rel_dev": max(dev), "pass": bool(max(dev) <= 1e-12)}
    bk = [q["bookkeeping_rel"] for q in rows if q["dest"] == "sink" and q["hbar_gamma_rel_meV"] > 0]
    checks["bookkeeping"] = {"max_rel": float(max(bk)), "pass": bool(max(bk) <= 1e-3)}
    tr1 = [abs(q["trace_f"] - 1) for q in rows if q["dest"] == "1s"]
    checks["health"] = {"min_min_eig": float(min(q["min_eig"] for q in rows)), "max_trace_err_1s": float(max(tr1)),
                        "pass": bool(min(q["min_eig"] for q in rows) >= -1e-9 and max(tr1) <= 1e-8)}
    conv = []
    for W in W23S:
        gr = 2 * HW
        a = relax_metrics(conversion(P, W, 1.1, g_rel=gr), gr, "sink")
        b = relax_metrics(conversion(P, W, 1.1, g_rel=gr, rtol=RTOL0 / 10, atol=ATOL0 / 10), gr, "sink")
        conv += [abs(b[k] / a[k] - 1) for k in ("eta_dark", "rho33_max", "N_emit")]
    checks["convergence"] = {"max_rel_change": max(conv), "pass": bool(max(conv) <= 1e-5)}

    hp = []
    for sigma in SIGMAS:
        for W in W23S:
            sub = [q for q in rows if q["block"] == "B" and q["sigma_ps"] == sigma and q["hbar_Omega23_meV"] == W]
            for key in ("eta_2p", "eta_dark", "eta"):
                g = half_log(sub, "hbar_gamma_rel_meV", key, linear_in_y=False)
                hp.append({"sigma_ps": sigma, "hbar_Omega23_meV": W, "quantity": key,
                           "hbar_gamma_rel_half_meV": g, "gamma_rel_sigma_over_hbar": g * sigma / ls.HB if g else None})
    table = []
    for W in W23S:
        pick = lambda blk, order, f: [q for q in rows if q["block"] == blk and q["dest"] == "sink" and q["order"] == order
                                      and q["hbar_Omega23_meV"] == W and q["f_population"] == f][0]
        table.append({"hbar_Omega23_meV": W, "eta0": pick("C", "counterintuitive", 0.0)["eta"],
                      "eta_dark_ci_f0.25": pick("A", "counterintuitive", 0.25)["eta_dark"],
                      "eta_dark_ci_f1": pick("A", "counterintuitive", 1.0)["eta_dark"],
                      "eta_dark_seq_f1": pick("A", "sequential", 1.0)["eta_dark"],
                      "vertex_suppression_ci_f1": pick("A", "counterintuitive", 1.0)["vertex_suppression"]})
    eta_max = max(q["eta"] for q in rows if q["block"] == "A" and q["f_population"] >= 0.25)
    write_csv(out / "relax_2p.csv", rows)
    return {"checks": checks, "half_points": hp, "table_SVII_XI": table, "max_eta_f_ge_quarter": eta_max}


# =============================================================================
# task independent: own density-matrix right-hand side, no lindblad_solver engine
# =============================================================================
def task_independent(P, out):
    from scipy.integrate import solve_ivp
    HB = ls.HB
    R = P.rates(4)
    g1, g2 = R["gamma_1s_to_0"] / HB, R["gamma_2p_to_1s"] / HB
    sig, x = 1.5, XD

    def eta(W23, gphi, order):
        W12 = P.weak(W23)
        first, second = 6 * sig, 6 * sig + x * sig
        t23, t12 = (first, second) if order == "ci" else (second, first)
        tend = second + 6 * sig
        E = lambda t, tc: np.exp(-0.5 * ((t - tc) / sig) ** 2)
        s12 = np.zeros((3, 3), complex); s12[0, 1] = s12[1, 0] = 0.5
        s23 = np.zeros((3, 3), complex); s23[1, 2] = s23[2, 1] = 0.5
        L1 = np.zeros((3, 3), complex); L1[0, 1] = 1        # 1s -> 0
        L2 = np.zeros((3, 3), complex); L2[1, 2] = 1        # 2p -> 1s
        Lp = np.sqrt(2) * np.diag([0, 0, 1.0]).astype(complex)
        jumps = [(g1, L1), (g2, L2), (gphi / HB, Lp)]

        def rhs(t, y):
            r = y[:9].reshape(3, 3)
            H = (W12 * E(t, t12) * s12 + W23 * E(t, t23) * s23) / HB
            d = -1j * (H @ r - r @ H)
            for g, L in jumps:
                if g > 0:
                    d += g * (L @ r @ L.conj().T - 0.5 * (L.conj().T @ L @ r + r @ L.conj().T @ L))
            return np.concatenate([d.ravel(), [g1 * r[1, 1]]])

        y = np.zeros(10, complex); y[0] = 1
        bps = sorted({0, t12 - 6 * sig, t12, t12 + 6 * sig, t23 - 6 * sig, t23, t23 + 6 * sig, tend})
        for a, b in zip(bps[:-1], bps[1:]):
            if b > a:
                y = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=1e-11, atol=1e-15, max_step=sig / 20).y[:, -1]
        r = y[:9].reshape(3, 3); n33 = r[2, 2].real; Ne = y[9].real
        return n33 / (n33 + Ne)

    sw = out / "gamma_phi_sweep.csv"
    ref = [{k: (float(v) if k != "order" else v) for k, v in q.items()} for q in read_csv(sw)] if sw.exists() else []
    pts = [(5.0, 0.0, "ci"), (5.0, 4.0, "ci"), (5.0, 4.0, "seq"), (1.0, 4.0, "seq"), (2.5, G_SWEEP[10], "ci")]
    res = []
    for W, g, o in pts:
        e = eta(W, g, o)
        m = [q for q in ref if q["hbar_Omega23_meV"] == W and q["order"] == o and math.isclose(q["hbar_gamma_phi_meV"], g, rel_tol=1e-12, abs_tol=0)]
        res.append({"hbar_Omega23_meV": W, "hbar_gamma_phi_meV": g, "order": o, "eta_independent": e,
                    "eta_lindblad_solver": m[0]["eta"] if m else None,
                    "rel_dev": abs(e / m[0]["eta"] - 1) if m else None})
    devs = [q["rel_dev"] for q in res if q["rel_dev"] is not None]
    return {"checks": {"independent_integrator": {"rows": res, "max_rel_dev": max(devs) if devs else None,
                                                  "pass": bool(devs and max(devs) <= 1e-4)}}}


# =============================================================================
# task figure: Fig. S1 (NJP SM) / Fig. 9 (arXiv)
# =============================================================================
def task_figure(P, out):
    from collections import defaultdict
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HB = 0.658211957
    for f in ("gamma_phi_sigma.csv", "coherence_halfpoint.csv", "relax_2p.csv"):
        if not (out / f).exists():
            sys.exit(f"task figure needs {f} (tasks sigma, halfpoint, relax)")

    def load(p, col, norm=False):
        D = defaultdict(list)
        for r in csv.DictReader(open(p)):
            D[(float(r["sigma_ps"]), float(r["hbar_Omega23_meV"]))].append((float(r["hbar_gamma_phi_meV"]), float(r[col])))
        res = {}
        for k, v in D.items():
            v = sorted(v); y0 = v[0][1]
            res[k] = [(g * k[0] / HB, (y / y0 if norm else y)) for g, y in v if g > 0]
        return res

    def load_relax(p, col):
        D = defaultdict(list)
        for r in csv.DictReader(open(p)):
            if r["block"] != "B" or r["dest"] != "sink" or r["order"] != "counterintuitive":
                continue
            D[(float(r["sigma_ps"]), float(r["hbar_Omega23_meV"]))].append(
                (float(r["hbar_gamma_rel_meV"]), float(r[col]), float(r["eta"])))
        res = {}
        for k, v in D.items():
            v = sorted(v); e0 = v[0][2]
            res[k] = [(g * k[0] / HB, y / e0) for g, y, _ in v if g > 0]
        return res

    E = load(out / "gamma_phi_sigma.csv", "eta", norm=True)
    C = load(out / "coherence_halfpoint.csv", "one_minus_C_tc")
    P2 = load_relax(out / "relax_2p.csv", "eta_2p")
    PD = load_relax(out / "relax_2p.csv", "eta_dark")
    hue = {1.1: "#c2185b", 1.5: "#1f6fd1", 1.9: "#1a9e77"}
    lst = {1.0: ":", 2.5: "--", 5.0: "-"}
    alpha = {1.0: 0.55, 2.5: 0.75, 5.0: 1.0}
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))
    ax = ax.ravel()
    for D, a in ((E, ax[0]), (C, ax[1]), (P2, ax[2]), (PD, ax[3])):
        for (s, W), pts in sorted(D.items()):
            xx, yy = zip(*pts)
            a.plot(xx, yy, color=hue[s], ls=lst[W], lw=1.3, alpha=alpha[W])
        a.set_xscale("log"); a.axhline(0.5, color="0.6", lw=0.7)
        a.set_ylim(0, 1.05)
    ax[0].set_xlabel(r"$\gamma_\phi\sigma$"); ax[1].set_xlabel(r"$\gamma_\phi\sigma$")
    ax[2].set_xlabel(r"$\gamma_{\rm rel}\sigma$"); ax[3].set_xlabel(r"$\gamma_{\rm rel}\sigma$")
    ax[0].axvline(1.0, color="0.3", lw=0.7)
    ax[0].set_ylabel(r"$\eta/\eta(\gamma_\phi=0)$")
    ax[1].axvline(math.log(2) / 6, color="0.3", lw=0.7, ls="-.")
    ax[1].text(math.log(2) / 6 * 1.08, 0.06, r"$\ln 2/6$", fontsize=8, color="0.3")
    ax[1].set_ylabel(r"$1-C(t_c)$")
    ax[2].axvline(0.11, color="0.3", lw=0.7)
    ax[2].set_ylabel(r"$\eta_{2p}/\eta(0)$")
    ax[3].set_ylabel(r"$\eta_{\rm dark}/\eta(0)$")
    for a, lab in zip(ax, ["(a)", "(b)", "(c)", "(d)"]):
        a.set_title(lab, fontsize=9)
    h = [Line2D([], [], color=hue[s], lw=1.5, label=rf"$\sigma={s}$ ps") for s in hue] + \
        [Line2D([], [], color="k", ls=lst[W], lw=1.3, label=rf"$\hbar\Omega_{{23}}={W:g}$ meV") for W in lst]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "fig_dephasing.pdf"); fig.savefig(out / "fig_dephasing.png", dpi=150)
    plt.close(fig)
    return {"written": ["fig_dephasing.pdf", "fig_dephasing.png"]}


TASKS = {"sweep": task_sweep, "sigma": task_sigma, "coherence_time": task_coherence_time,
         "halfpoint": task_halfpoint, "relax": task_relax, "independent": task_independent,
         "figure": task_figure}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("tasks", nargs="*", default=["all"], choices=list(TASKS) + ["all"])
    ap.add_argument("--out", default="appG_output")
    a = ap.parse_args()
    tasks = list(TASKS) if "all" in a.tasks else a.tasks
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = ls.Params("params.yaml")
    sf = out / "appG_summary.json"
    summ = json.load(open(sf)) if sf.exists() else {}
    for name in tasks:
        t0 = time.time()
        res = TASKS[name](P, out)
        res["run_time_s"] = round(time.time() - t0, 1)
        summ[name] = res
        ok = all(c["pass"] for c in res.get("checks", {}).values())
        print(f"{name}: {'all checks pass' if ok else 'SOME CHECK FAILED'} ({res['run_time_s']} s)", flush=True)
        for k, c in res.get("checks", {}).items():
            print(f"   {k}: {'pass' if c['pass'] else 'FAIL'}")
        json.dump(summ, open(sf, "w"), indent=1, default=float)
    allok = all(c["pass"] for r in summ.values() for c in r.get("checks", {}).values())
    print("ALL CHECKS PASS" if allok else "SOME CHECK FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
