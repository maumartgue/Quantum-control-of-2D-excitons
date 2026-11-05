"""
fig7_R4.py -- Fig. 7 (R4) of the paper "Dark-State Coherent Control of Excitons in TMD Monolayers:
A Reduced Intravalley Lindblad Framework for WSe2": CPT plateau and cascade response of the Xi
manifold, Sect. VI D (4 K, photon dissipator).

(a) CPT plateau: weak optical leg hbar*Omega12 = 0.05 meV, hbar*Omega23 = 1 meV, 50 ps plateau
    (tau_on = tau_off = 0.5 ps), case 2 (resonant) and case 3 (bare-tuned lasers at n_X = 1e11 cm^-2).
    Also printed: the population outside the dark state at t_off with and without the photon
    dissipator (case 1 keeps the bright fraction of the initial state).
(b)-(d) weak-probe cascade spectra, Liouvillian null space, Delta~23 = 0, probe
    hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2):
      chi_hat = 2 Gamma12 rho_21 / Omega12 (Im chi_hat = 1 at the two-level peak).
(e)-(f) EIT against Autler-Townes, computed here from the same steady state:
      chi_hat(Delta) = 2 Gamma12 rho12 / Omega12,   Gamma12 = (gamma_0->1s + gamma_1s->0 + gamma_1s->2p)/2,
    on the 481-point probe grid (+-15 meV) for hbar*Omega23 in {0.25, ..., 5} meV;
      * window width = FWHM of the hole h = A0 - A; absorption-peak splitting;
      * exact regime from the pole structure: AT (real splitting) iff Omega23 > Gamma12 - Gamma13;
      * Anisimov test (Anisimov, Dowling, Sanders, PRL 107, 163604 (2011)):
            EIT model  A = C+ G+^2/(D^2+G+^2) - C- G-^2/(D^2+G-^2)
            AT  model  A = C [G^2/((D-d0)^2+G^2) + G^2/((D+d0)^2+G^2)]
        least-squares fits on a uniform 0.02 meV grid over +-15 meV (1501 points, null space),
        Akaike information I = N ln(RSS/N) + 2k; per-point weights
        w_EIT = exp(-I_EIT/2N) / [exp(-I_EIT/2N) + exp(-I_AT/2N)];
        a fine sweep hbar*Omega23 = 1.4-4.0 meV (step 0.1) locates the crossing w_EIT = 0.5.
    Per Omega23 also: transparency depth, dispersion slope at the centre, |rho13| at the centre,
    probe nonlinearity (half probe, Richardson).

Outputs: figR4.pdf/png; fig7_R4.json (blocks "plateau", "partB_eit": rows, fine_anisimov,
anisimov_crossing_meV, Gamma12_meV, Gamma13_meV, pole_threshold_meV; "spectra_check");
fig7_R4_runs.npz (plateau trajectories); fig7_R4_spectra.npz (spectra of (b)-(d));
fig7_R4.h5 (EIT spectra on the probe grid).
Usage:  python fig7_R4.py [--nproc 2]   (reads params.yaml)
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import lindblad_solver as X

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PULSE = "#a9a8a3"
HB = 0.658211957

_P = None


# ==================================================================== EIT / AT analysis
def _init(path):
    global _P
    _P = X.Params(path)


def gammas(P):
    g = P.rates(4)
    G12 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_1s_to_0"] + g["gamma_1s_to_2p"])
    G13 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_2p_to_1s"])
    return G12, G13


def ss_rho(P, D12, D23, W12, W23):
    # EIT/AT steady states: rates from the 'dimensionless_gamma_tau0' block (x E0)
    gen = X.Generator(P, X.XI, T=4, frame_meV=X.xi_frame(D12, D23), rate_block=X.RATES_TAU0)
    rho, info = X.steady_state(gen, [W12, W23])
    return rho, info


def chi_hat(P, rho, W12_meV):
    """2 Gamma12 rho12 / Omega12 with Gamma12 and Omega12 both in meV (dimensionless)."""
    G12, _ = gammas(P)
    return 2 * G12 * rho[0, 1] / W12_meV


def spectrum(args):
    """Null-space spectrum on a grid for one (W23, W12)."""
    W23, W12, grid = args
    P = _P
    chi = np.empty(len(grid), complex); r13 = np.empty(len(grid), complex); uniq = True
    for k, D in enumerate(grid):
        rho, info = ss_rho(P, D, 0.0, W12, W23)
        chi[k] = chi_hat(P, rho, W12); r13[k] = rho[0, 2]; uniq &= info["unique"]
    return chi, r13, uniq


def fwhm_centered(x, y):
    """FWHM of a peak centred near x = 0 (linear interpolation)."""
    k = int(np.argmax(y)); half = y[k] / 2
    r = next((i for i in range(k, len(y)) if y[i] < half), None)
    l = next((i for i in range(k, -1, -1) if y[i] < half), None)
    if r is None or l is None:
        return np.nan
    xr = x[r - 1] + (half - y[r - 1]) * (x[r] - x[r - 1]) / (y[r] - y[r - 1])
    xl = x[l + 1] + (half - y[l + 1]) * (x[l] - x[l + 1]) / (y[l] - y[l + 1])
    return xr - xl


def peak_split(x, A):
    """Distance between the two absorption maxima on either side of x = 0."""
    m = x > 0
    if not np.any(m):
        return np.nan
    xp = x[m][int(np.argmax(A[m]))]; xm = x[~m][int(np.argmax(A[~m]))]
    return xp - xm


def eit_model(D, Cp, Gp, Cm, Gm):
    return Cp * Gp**2 / (D**2 + Gp**2) - Cm * Gm**2 / (D**2 + Gm**2)


def at_model(D, C, G, d0):
    return C * (G**2 / ((D - d0)**2 + G**2) + G**2 / ((D + d0)**2 + G**2))


def anisimov(x, A, G12, W23):
    N = len(x)
    best = {}
    trials_eit = [(1.0, G12, 1.0, s) for s in (W23**2 / (4 * G12), W23**2 / (2 * G12), W23 / 2, 0.05, 0.3)]
    trials_at = [(0.5, G12 / 2, max(W23 / 2, 0.05)), (0.5, G12, W23 / 2), (0.5, G12 / 2, 0.5),
                 (0.5, G12 / 4, W23 / 2)]
    bounds = {"EIT": ([0, 1e-6, 0, 1e-6], [1e4, 50, 1e4, 50]), "AT": ([0, 1e-6, 0], [1e4, 50, 20])}
    for name, f, trials, k in (("EIT", eit_model, trials_eit, 4), ("AT", at_model, trials_at, 3)):
        rss_best, p_best = np.inf, None
        for p0 in trials:
            try:
                p, _ = curve_fit(f, x, A, p0=p0, bounds=bounds[name], maxfev=40000)
                rss = float(np.sum((f(x, *p) - A) ** 2))
                if rss < rss_best:
                    rss_best, p_best = rss, p
            except RuntimeError:
                continue
        rss_best = max(rss_best, 1e-300)
        best[name] = {"rss": rss_best, "params": list(map(float, p_best)) if p_best is not None else None,
                      "I": N * np.log(rss_best / N) + 2 * k}
    a, b = -best["EIT"]["I"] / (2 * N), -best["AT"]["I"] / (2 * N)
    m = max(a, b)
    wE = np.exp(a - m) / (np.exp(a - m) + np.exp(b - m))
    return {"w_EIT": float(wE), "w_AT": float(1 - wE), "fits": best}


def eit_analysis(P, path, nproc):
    """Spectra, widths, splittings and Anisimov test for every Omega23 of the response grid."""
    xr = P.proto["Xi_response"]
    grid = np.array(xr["probe_Delta12_grid"]["values_meV"])
    fitgrid = np.round(np.arange(-15.0, 15.0001, 0.02), 6)
    Ws, W12s = xr["hbar_Omega23_meV"], xr["hbar_Omega12_meV_derived"]
    G12, G13 = gammas(P)
    with Pool(nproc, initializer=_init, initargs=(path,)) as pool:
        main_ = pool.map(spectrum, [(W, w, grid) for W, w in zip(Ws, W12s)])
        half_ = pool.map(spectrum, [(W, 0.5 * w, grid) for W, w in zip(Ws, W12s)])
        fit_ = pool.map(spectrum, [(W, w, fitgrid) for W, w in zip(Ws, W12s)])
        fineW = [float(w) for w in np.round(np.arange(1.4, 4.01, 0.1), 3)]
        fine_ = pool.map(spectrum, [(W, xr["weak_probe_rule"] and 0.05 * min(W, gammas(P)[0]), fitgrid)
                                    for W in fineW])
    _init(path)
    ref2 = spectrum((0.0, W12s[0], grid))[0]          # Omega23 = 0 reference (numerical)
    S = {"grid": grid, "W23": Ws, "spec": {}, "rows": []}
    for W, w, (chi, r13, u), (chih, _, _), (chif, _, _) in zip(Ws, W12s, main_, half_, fit_):
        A, R = chi.imag, chi.real
        extrap = (4 * chih - chi) / 3
        k0 = int(np.argmin(np.abs(grid)))
        h = ref2.imag - A
        slope = (R[k0 + 1] - R[k0 - 1]) / (grid[k0 + 1] - grid[k0 - 1])
        ani = anisimov(fitgrid, chif.imag, G12, W)
        disc = W**2 - (G12 - G13)**2
        row = {"W23": W, "W12": w, "unique": bool(u),
               "depth": float(1 - A[k0] / ref2.imag[k0]), "A_centre": float(A[k0]),
               "hole_fwhm": float(fwhm_centered(grid, h)), "peak_split": float(peak_split(grid, A)),
               "disp_slope": float(slope), "rho13_centre": float(abs(r13[k0])),
               "nonlin_probe": float(np.abs(chi - extrap).max() / np.abs(extrap).max()),
               "regime_poles": "AT" if disc > 0 else "EIT",
               "pole_split_meV": float(np.sqrt(disc)) if disc > 0 else 0.0,
               "anisimov": ani, "eit_claim_allowed": W <= xr["eit_claims_only_for_hbar_Omega23_le_meV"]}
        S["rows"].append(row)
        S["spec"][W] = {"A": A, "R": R, "A_lin": extrap.imag, "R_lin": extrap.real, "rho13": r13}
    S["fine"] = []
    for W, (c, _, _) in zip(fineW, fine_):
        a_ = anisimov(fitgrid, c.imag, G12, W)
        S["fine"].append({"W23": W, "w_EIT": a_["w_EIT"], "w_AT": a_["w_AT"],
                          "rss_EIT": a_["fits"]["EIT"]["rss"], "rss_AT": a_["fits"]["AT"]["rss"]})
    fw = [f["w_EIT"] for f in S["fine"]]
    cross = next((0.5 * (fineW[i] + fineW[i + 1]) for i in range(len(fw) - 1)
                  if fw[i] >= 0.5 > fw[i + 1]), np.nan)
    S["anisimov_crossing_meV"] = cross
    S["ref2"], S["fitgrid"] = ref2, fitgrid
    js = {"rows": S["rows"], "fine_anisimov": S["fine"],
          "anisimov_crossing_meV": S["anisimov_crossing_meV"], "Gamma12_meV": G12,
          "Gamma13_meV": G13, "pole_threshold_meV": G12 - G13}
    return S, js


# ==================================================================== spectra of (b)-(d)
def r4_spectra(P):
    """Weak-probe cascade spectra of panels (b)-(d) (null space, Delta~23 = 0, 4 K)."""
    R = P.rates(4); HB = X.HB
    G12 = 0.5 * R["gamma_1s_to_0"]
    W_ALL = (0.5, 1.0, 1.5, 1.6, 2.0, 3.0, 4.0, 5.0, 0.25, 0.75, 1.25)
    D = np.unique(np.r_[np.arange(-6, -2, 0.01), np.arange(-2, 2, 0.001), np.arange(2, 6.0001, 0.01)])

    def L_of(h2, h3, W12, W23):
        H = np.diag([0, h2, h3]).astype(complex); H[1, 0] = H[0, 1] = W12 / 2; H[2, 1] = H[1, 2] = W23 / 2
        L = X._comm(H / HB)
        for src, dst, key in X.XI.jumps:
            g = R.get(key, 0.0) / HB
            if g > 0:
                L = L + g * X._diss(X._op(dst, src))
        return L

    def chi(W23, W12):
        out = np.empty(len(D), complex)
        for k, d in enumerate(D):
            L = L_of(d, d, W12, W23)               # Delta23 = 0: h3 = Delta12
            A = np.vstack([L, np.eye(3).reshape(1, 9, order="F")]); b = np.zeros(10, complex); b[-1] = 1
            rho = np.linalg.lstsq(A, b, rcond=None)[0].reshape(3, 3, order="F")
            out[k] = 2 * G12 * rho[1, 0] / W12
        return out

    res = {"Delta12_meV": D}
    two = chi(0.0, 0.01)
    if two.imag[np.argmin(abs(D))] < 0:
        sign = -1
    else:
        sign = 1
    two = np.conj(two) if sign < 0 else two
    res["chi_hat_two_level"] = two
    check = {}
    for W in W_ALL:
        c = chi(W, P.weak(W)); c = np.conj(c) if sign < 0 else c
        res[f"W23_{W:g}meV"] = c
        A = c.imag; k0 = np.argmin(abs(D)); m = D >= 0
        kpk = np.argmax(np.where(m, A, -1)); peak = D[kpk]
        half = 0.5 * (A[k0] + A[kpk])                    # half depth of the hole
        kk = np.where(m & (D < peak) & (A >= half))[0]
        fwhm = 2 * D[kk[0]] if len(kk) else np.nan
        check[W] = dict(hole_fwhm=float(fwhm), peak_split=float(2 * peak), A_centre=float(A[k0]))
    print("sign convention:", sign, " two-level Im at 0:", two.imag[np.argmin(abs(D))],
          " Re slope:", np.polyfit(D[abs(D) < 0.05], two.real[abs(D) < 0.05], 1)[0])
    return res, check


# ==================================================================== figure
def fig_R4(S, Z, Sp, R):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "legend.fontsize": 6.6, "legend.handlelength": 2.4,
                         "legend.labelspacing": 0.3})

    def run(name):
        d = {k.split("__")[1]: Z[k] for k in Z if k.startswith(name + "__")}
        d["rho11"], d["rho22"], d["rho33"] = d["rho11"].real, d["rho22"].real, d["rho33"].real
        return d

    G12, thr, cross = R["Gamma12_meV"], R["pole_threshold_meV"], R["anisimov_crossing_meV"]
    W_ALL = (0.5, 1.0, 1.5, 1.6, 3.0, 5.0)
    HUES = dict(zip(W_ALL, ("#d6334a", "#1baf7a", "#2a78d6", "#8a4fd1", "#b8860b", "#009c9c")))
    ALPHA = dict(zip(W_ALL, np.linspace(0.45, 1.0, len(W_ALL))))
    COL = {W: matplotlib.colors.to_rgba(HUES[W], ALPHA[W]) for W in W_ALL}
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.8), constrained_layout=True)
    (a, b, c), (d, e, f) = ax
    # (a) CPT at weak optical leg
    p2, p3 = run("Xi_plateau_case2"), run("Xi_plateau_case3")
    t_on, t_off = float(Z["Xi_plateau_case2__t_on"]), float(Z["Xi_plateau_case2__t_off"])
    pl = S["plateau"]["Xi_plateau_case2"]
    a.axvspan(t_on, t_off, color=GRID, alpha=0.8, lw=0)
    a.semilogy(p2["t"], np.maximum(p2["rho33"], 1e-14), color=C3, lw=1.3)
    a.semilogy(p2["t"], np.maximum(np.abs(p2["rho13"]), 1e-14), color=INK, lw=1.3)
    a.semilogy(p2["t"], np.maximum(p2["rho22"], 1e-14), color=C2, lw=1.3)
    a.semilogy(p3["t"], np.maximum(p3["rho22"], 1e-14), color=C2, lw=1.2, ls=":")
    W12a, W23a = 0.05, 1.0
    a.axhline((W12a / W23a) ** 2, color=C3, lw=0.8, ls="--")
    a.axhline(W12a / W23a, color=INK2, lw=0.8, ls="--")
    a.set_xlim(0, p2["t"][-1]); a.set_ylim(1e-10, 0.3)
    a.set_xlabel(r"$t$ (ps)"); a.set_ylabel("population, coherence")
    HA = [Line2D([], [], color=INK, lw=1.3, label=r"(a) $|\bar\rho_{13}|$"),
          Line2D([], [], color=C3, lw=1.3, label=r"(a) $\bar\rho_{33}$"),
          Line2D([], [], color=C2, lw=1.3, label=r"(a) $\bar\rho_{22}$, case 2"),
          Line2D([], [], color=C2, lw=1.2, ls=":", label=r"(a) $\bar\rho_{22}$, case 3"),
          Line2D([], [], color=INK2, lw=0.8, ls="--", label=r"(a) $\Omega_{12}/\Omega_{23}$, $(\Omega_{12}/\Omega_{23})^2$")]
    # (b)-(d) spectra
    D = Sp["Delta12_meV"]; two = Sp["chi_hat_two_level"]
    chi = {W: Sp[f"W23_{W:g}meV"] for W in W_ALL}
    for p, Ws, lim in ((b, (0.5, 1.0, 1.5), 2.0), (c, (1.6, 3.0, 5.0), 6.0)):
        mm = np.abs(D) <= lim
        p.plot(D[mm], two.imag[mm], color=INK2, ls="--", lw=1.0)
        for W in Ws:
            p.plot(D[mm], chi[W].imag[mm], color=COL[W], lw=1.3)
        p.set_xlim(-lim, lim); p.set_ylim(0, 1.08)
        p.set_xlabel(r"$\hbar\breve\Delta_{12}$ (meV)"); p.set_ylabel(r"$\mathrm{Im}\,\hat\chi$")
    b.set_xticks([-2, -1, 0, 1, 2])
    lim = 0.4; mm = np.abs(D) <= lim
    d.axhline(0, color=GRID, lw=0.8); d.axvline(0, color=GRID, lw=0.8)
    d.plot(D[mm], two.real[mm], color=INK2, ls="--", lw=1.0)
    for W in (0.5, 1.0, 1.5):
        d.plot(D[mm], chi[W].real[mm], color=COL[W], lw=1.3)
    d.set_xlim(-lim, lim)
    d.set_xlabel(r"$\hbar\breve\Delta_{12}$ (meV)"); d.set_ylabel(r"$\mathrm{Re}\,\hat\chi$")
    # (e) widths
    rows = R["rows"]; Wr = np.array([r["W23"] for r in rows]); Wl = np.geomspace(0.2, 6, 200)
    e.vlines(thr, 0.012, 8, color=INK2, ls="-.", lw=0.9)
    e.loglog(Wl, Wl ** 2 / (2 * G12), color=INK, ls="--", lw=0.9)
    e.loglog(Wl, Wl, color=INK2, ls=":", lw=1.0)
    e.loglog(Wr, [r["hole_fwhm"] for r in rows], "o", color=INK, ms=4)
    e.loglog(Wr, [r["peak_split"] for r in rows], "s", mfc="none", mec=INK2, ms=4.5, mew=0.9)
    e.set_xlim(0.2, 6); e.set_ylim(0.012, 400)
    e.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    e.set_xticks([0.25, 0.5, 1, 2, 5]); e.set_xticklabels(["0.25", "0.5", "1", "2", "5"])
    e.set_yticks([0.1, 1, 10, 100]); e.set_yticklabels(["0.1", "1", "10", "100"])
    e.set_xlabel(r"$\hbar\Omega_{23}$ (meV)"); e.set_ylabel("width (meV)")
    e.legend(handles=[Line2D([], [], color=INK, ls="", marker="o", ms=4, label="window FWHM"),
                      Line2D([], [], color=INK2, ls="", marker="s", mfc="none", ms=4.5, label="separation of maxima"),
                      Line2D([], [], color=INK, ls="--", lw=0.9, label=r"$\hbar\Omega_{23}^2/2\hbar\Gamma_{12}$"),
                      Line2D([], [], color=INK2, ls=":", lw=1.0, label=r"$\hbar\Omega_{23}$")],
             loc="upper left", borderpad=0.3)
    # (f) Akaike weights
    fs = R["fine_anisimov"]
    f.axvspan(0, 1.5, color=GRID, alpha=0.8, lw=0)
    f.axvline(thr, color=INK2, ls="-.", lw=0.9); f.axhline(0.5, color=GRID, lw=0.8)
    f.plot([x["W23"] for x in fs], [x["w_EIT"] for x in fs], color=INK, lw=1.2)
    f.plot([x["W23"] for x in fs], [x["w_AT"] for x in fs], color=INK2, lw=1.1, ls="--")
    f.plot(Wr, [r["w_EIT"] for r in rows], "o", color=INK, ms=4)
    f.plot(Wr, [r["w_AT"] for r in rows], "o", mfc="none", mec=INK2, ms=4, mew=0.9)
    f.annotate(rf"crossing, {cross:.2f} meV", (cross, 0.5), xytext=(4.1, 0.13), ha="center", fontsize=6.5,
               color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))
    f.text(0.75, 0.5, "EIT claims", ha="center", va="center", rotation=90, fontsize=6.8, color=INK2)
    f.set_xlim(0, 5.2); f.set_ylim(-0.03, 1.05)
    f.set_xlabel(r"$\hbar\Omega_{23}$ (meV)"); f.set_ylabel("Akaike weight")
    f.legend(handles=[Line2D([], [], color=INK, lw=1.2, marker="o", ms=4, label=r"$w_{\rm EIT}$"),
                      Line2D([], [], color=INK2, lw=1.1, ls="--", marker="o", mfc="none", ms=4, label=r"$w_{\rm AT}$")],
             loc="upper right")
    for p, lab in zip(ax.flat, "abcdef"):
        p.set_title(f"({lab})", fontsize=9, color=INK, pad=4)
    for p in (a, b, c, d, e, f):
        p.grid(color=GRID, lw=0.6, which="major")
    h = HA + [Patch(color=GRID, label=r"(a) fields on, $\hbar\Omega_{12}=0.05$ meV, $\hbar\Omega_{23}=1$ meV"),
         Line2D([], [], color=INK2, ls="--", lw=1.0, label=r"(b)-(d) $\Omega_{23}=0$")]
    h += [Line2D([], [], color=COL[W], lw=1.6, label=rf"$\hbar\Omega_{{23}}$ = {W:g} meV") for W in W_ALL]
    h += [Line2D([], [], color=INK2, ls="-.", lw=0.9, label=r"(e), (f) $\hbar(\Gamma_{12}-\Gamma_{13})$: poles split")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, columnspacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"figR4.{ext}", dpi=220)
    plt.close(fig)


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=2)
    args = ap.parse_args()
    path = "params.yaml"
    P = X.Params(path)
    pr = P.proto
    c3 = pr["Part_A"]["case3"]["Xi"]            # case 3: bare-tuned lasers at n_X = 1e11 cm^-2
    D12c3, D23c3 = c3["Delta12_meV"], c3["Delta23_meV"]
    t0 = time.time()
    summ, traj = {}, {}

    def keep(name, r, extra=()):
        traj[f"{name}__t"] = r["t"]
        for a in range(3):
            for b in range(3):
                if a <= b:
                    traj[f"{name}__rho{a+1}{b+1}"] = r["rho"][:, a, b]
        traj[f"{name}__W12"] = r["W"][:, 0]; traj[f"{name}__W23"] = r["W"][:, 1]
        traj[f"{name}__N_emit"] = r["N_emit"]

    def S(r):
        return X.summary(r)

    # ---- CPT plateau ---------------------------------------------------------------------
    summ["plateau"] = {}
    A4 = pr["Part_A"]["A4_Xi_plateau_CPT"]
    for case, (D12, D23) in ((2, (0, 0)), (3, (D12c3, D23c3))):
        r = X.xi_plateau(P, A4["hbar_Omega12_meV"], A4["hbar_Omega23_meV"], tau=A4["tau_on_off_ps"],
                         T_pulse=A4["T_pulse_ps"], D12=D12, D23=D23, atol=1e-14)
        name = f"Xi_plateau_case{case}"; keep(name, r)
        s = S(r); k = int(np.argmin(abs(r["t"] - r["t_off"])))
        s.update(rho22_t_off=float(r["rho"][k, 1, 1].real), rho33_t_off=float(r["rho"][k, 2, 2].real),
                 abs_rho13_t_off=float(abs(r["rho"][k, 0, 2])), t_on=r["t_on"], t_off=r["t_off"],
                 target_rho33=(A4["hbar_Omega12_meV"] / A4["hbar_Omega23_meV"]) ** 2)
        # time at which rho22 has fallen to 1e-3 of its peak after switch-on (formation time)
        p22 = r["rho"][:, 1, 1].real; kpk = int(np.argmax(p22))
        after = np.where((r["t"] > r["t"][kpk]) & (p22 < 1e-2 * p22[kpk]))[0]
        s["t_form_1pct_ps"] = float(r["t"][after[0]] - r["t_on"]) if len(after) else None
        summ["plateau"][name] = s
        traj[f"{name}__t_on"] = np.array(r["t_on"]); traj[f"{name}__t_off"] = np.array(r["t_off"])
    print(f"plateau done {time.time()-t0:.0f}s")

    # ---- weak CPT plateau without the photon dissipator (case 1): bright fraction kept ------
    for diss in (False, True):
        r = X.xi_plateau(P, 0.05, 1.0, tau=0.5, T_pulse=50, dissipator=diss, rtol=1e-10, atol=1e-14)
        k = int(np.argmin(abs(r["t"] - r["t_off"]))); rho = r["rho"][k]
        th = np.arctan(0.05 / 1.0); Dk = np.array([np.cos(th), 0, -np.sin(th)])
        PD = float(np.real(Dk.conj() @ rho @ Dk))
        print(f"dissipator={diss}: 1-P_D at t_off = {1-PD:.3e}, rho22 = {rho[1,1].real:.2e}, rho33 = {rho[2,2].real:.3e}, sin^2(theta) = {np.sin(th)**2:.3e}")

    # ---- spectra (b)-(d) and EIT / AT analysis (e)-(f) ------------------------------------
    Sp, check = r4_spectra(P)
    E, js = eit_analysis(P, path, args.nproc)
    print(f"spectra done {time.time()-t0:.0f}s")
    print("W23   FWHM (b)-(d) grid / probe grid    peak split (b)-(d) grid / probe grid")
    for r in js["rows"]:
        if r["W23"] in check:
            c = check[r["W23"]]
            print(f"{r['W23']:5.2f}  {c['hole_fwhm']:.4f} / {r['hole_fwhm']:.4f}   {c['peak_split']:.3f} / {r['peak_split']:.3f}   A0 {c['A_centre']:.2e}")
    print(f"Akaike crossing w_EIT = 0.5 at hbar*Omega23 = {js['anisimov_crossing_meV']:.2f} meV; "
          f"pole threshold {js['pole_threshold_meV']:.3f} meV")

    # ---- outputs ---------------------------------------------------------------------------
    json.dump({"plateau": summ["plateau"], "partB_eit": js, "spectra_check": {str(k): v for k, v in check.items()}},
              open("fig7_R4.json", "w"), indent=1, default=float)
    np.savez_compressed("fig7_R4_runs.npz", **traj)
    np.savez_compressed("fig7_R4_spectra.npz", **Sp)
    import h5py
    X.write_h5("fig7_R4.h5", P, meta={"purpose": "Xi weak-probe response, EIT/AT (Fig. 7)"})
    with h5py.File("fig7_R4.h5", "a") as f:
        g = f["scans"].create_group("Xi_EIT")
        g.create_dataset("Delta12_meV", data=E["grid"]); g.create_dataset("fit_grid_meV", data=E["fitgrid"])
        g.create_dataset("chi_hat_two_level", data=E["ref2"])
        for W in E["W23"]:
            h = g.create_group(f"W23_{W:g}meV")
            for k, v in E["spec"][W].items():
                h.create_dataset(k, data=v)

    R = {"Gamma12_meV": js["Gamma12_meV"], "pole_threshold_meV": js["pole_threshold_meV"],
         "anisimov_crossing_meV": js["anisimov_crossing_meV"], "fine_anisimov": js["fine_anisimov"],
         "rows": [{"W23": r["W23"], "hole_fwhm": r["hole_fwhm"], "peak_split": r["peak_split"],
                   "w_EIT": r["anisimov"]["w_EIT"], "w_AT": r["anisimov"]["w_AT"]} for r in js["rows"]]}
    fig_R4(summ, traj, Sp, R)
    print(f"figR4 written; all done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
