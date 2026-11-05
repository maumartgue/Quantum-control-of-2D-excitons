"""
fig6_R2.py -- Fig. 6 (R2) of the paper "Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley
Lindblad Framework for WSe2": two-photon detuning and density calibration of the Xi
conversion, Sect. VI C. (a) eta over delta~Xi (on the THz leg, Delta~12 = 0) x hbar*Omega23 with the
contour eta = 0.9 eta(0); (b) cuts at 5 and 3 meV, the control with the detuning on the optical leg,
and lasers calibrated at n_cal with the actual density off by -10 % ... +10 %; (c) bare-tuned lasers
(n_cal = 0) against n_X. Also the absolute density-error scan at n_cal = 1e11 cm^-2 (block "dn_scan",
used for the tolerance band of (b)). Detunings in the sign convention of Eqs. (65): hbar*Delta~ =
Sigma(n_X) - Sigma(n_cal).

Runs from params.yaml with lindblad_solver.py (4 K, photon dissipator unless stated): Xi conversion with
the weak optical leg hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2), Gaussian pulses, baseline
sigma = 1.5 ps, tau_d = 1.25 sigma, hbar*Omega23 = 5 meV. Every run records max_t (1 - rho11), the
validity of the one-exciton truncation.

Outputs: figR2.pdf/png, fig6_R2.json (blocks "map", "map_optical", "mismatch", "bare_scan", "dn_scan").
Usage:  python fig6_R2.py   (reads params.yaml)
"""
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

import lindblad_solver as X

P = X.Params("params.yaml")
pr = P.proto
xc = pr["Xi_conversion"]; base = xc["baseline"]; G = xc["grids"]
S0, X0, W23b = base["sigma_ps"], base["delay_over_sigma"], base["hbar_Omega23_meV"]
W12b = P.weak(W23b)
c3 = pr["Part_A"]["case3"]["Xi"]            # case 3: bare-tuned lasers at n_X = 1e11 cm^-2
D12c3, D23c3 = c3["Delta12_meV"], c3["Delta23_meV"]
Sig = P.Sig1e11
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


# ---- baseline case 2 (reference efficiency of panel (c)) ----------------------------------
summ["partA"] = {"Xi_ci_case2": S(X.xi_conversion(P, W12b, W23b, S0, X0, order="counterintuitive",
                                                  dissipator=True, rtol=1e-8, atol=1e-14))}

# ---- robustness map ----------------------------------------------------------------------
summ["map"] = []
for W23 in G["hbar_Omega23_meV"]:
    for d in G["delta_Xi_meV"]:
        s = S(X.xi_conversion(P, P.weak(W23), W23, S0, X0, D12=0.0, D23=d))
        summ["map"].append(dict(W23=W23, delta=d, **s))
summ["map_optical"] = []
for d in G["delta_Xi_meV"]:
    s = S(X.xi_conversion(P, W12b, W23b, S0, X0, D12=d, D23=-0.0 + 0.0))
    summ["map_optical"].append(dict(W23=W23b, delta=d, **s))
print(f"map done {time.time()-t0:.0f}s")

# ---- density calibration -----------------------------------------------------------------
summ["mismatch"] = []
for c in P.raw["calibration"]["mismatch_scan"]:
    for row in c["rows"]:
        s = S(X.xi_conversion(P, W12b, W23b, S0, X0, D12=row["Xi_Delta12"], D23=row["Xi_Delta23"]))
        summ["mismatch"].append(dict(n_nominal=c["n_X_nominal_cm2"], n_cal=row["n_cal"],
                                     n_act=row["n_act"], label=row["label"], delta=row["Xi_delta"],
                                     D12=row["Xi_Delta12"], D23=row["Xi_Delta23"], **s))
summ["bare_scan"] = []
for n in (2.5e9, 5e9, 1e10, 1.5e10, 2e10, 2.5e10, 3e10, 4e10, 5e10, 6e10, 8e10, 1e11, 1.25e11,
          1.5e11, 2e11):
    D12, D23 = Sig["1s"] * n / 1e11, (Sig["2p"] - Sig["1s"]) * n / 1e11
    s = S(X.xi_conversion(P, W12b, W23b, S0, X0, D12=D12, D23=D23))
    summ["bare_scan"].append(dict(n_X=n, D12=D12, D23=D23, delta=D12 + D23, **s))
summ["dn_scan"] = []
for dn in np.r_[-4e10, -3e10, -2e10, -1.5e10, -1.25e10, -1e10, -7.5e9, -5e9, -2.5e9, 0,
                2.5e9, 5e9, 7.5e9, 1e10, 1.25e10, 1.5e10, 2e10, 3e10, 4e10]:
    D12, D23 = Sig["1s"] * dn / 1e11, (Sig["2p"] - Sig["1s"]) * dn / 1e11
    s = S(X.xi_conversion(P, W12b, W23b, S0, X0, D12=D12, D23=D23))
    summ["dn_scan"].append(dict(dn=float(dn), delta=D12 + D23, **s))
print(f"mismatch done {time.time()-t0:.0f}s")

json.dump({k: summ[k] for k in ('map', 'map_optical', 'mismatch', 'bare_scan', 'dn_scan')}, open("fig6_R2.json", "w"), indent=1, default=float)

# ---- figure ------------------------------------------------------------------------------
S = summ
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PULSE = "#a9a8a3"
COLS = {1.1: "#c2257d", 1.5: "#5b4fd6", 1.9: "#8a7a00"}           # sigma colours (plot_R3)
NCOL = {2.5e10: "#2a78d6", 5e10: "#eb6834", 1e11: "#1baf7a"}      # density colours (plot_R2)
HB = 0.658211957
G1S = 3.20559
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 6.6, "legend.handlelength": 2.4,
                     "legend.labelspacing": 0.3})
CASES = {1: "case 1: coherent only", 2: "case 2: + photon dissipator", 3: "case 3: + mean field"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=220)
    plt.close(fig)


def fig_R2():
    m = S["map"]; W = sorted(set(r["W23"] for r in m)); D = sorted(set(r["delta"] for r in m))
    M = np.array([[next(r for r in m if r["W23"] == w and r["delta"] == d)["eta"] for d in D] for w in W])
    rel = M / M[:, [D.index(0.0)]]
    dn = sorted([r for r in S["dn_scan"] if r["dn"] >= 0], key=lambda r: r["dn"])
    e0 = dn[0]["eta"]
    tol = float(np.interp(0.9, [r["eta"] / e0 for r in dn][::-1], [r["delta"] for r in dn][::-1]))
    slope = 1.84081                                                    # d delta~Xi / d n_X = Sigma_2p(1e11), meV per 1e11 cm^-2
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.85), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]})
    a, b, c = ax
    dD, dW = D[1] - D[0], W[1] - W[0]
    im = a.pcolormesh(np.r_[np.array(D) - dD / 2, D[-1] + dD / 2], np.r_[np.array(W) - dW / 2, W[-1] + dW / 2],
                      M, cmap="Blues", shading="flat", edgecolors="white", linewidth=0.5, vmin=0, vmax=0.5)
    cb = fig.colorbar(im, ax=a, pad=0.02); cb.set_label(r"$\eta$"); cb.outline.set_visible(False)
    Dfine = np.array(D)
    a.contour(Dfine, W, rel, levels=[0.9], colors=INK, linewidths=1.2)
    for s_ in (-1, 1):
        a.axvline(s_ * HB / 1.5, color=INK2, ls="-.", lw=0.9)
    a.set_xlabel(r"$\hbar\breve\delta_\Xi$ (meV)"); a.set_ylabel(r"$\hbar\Omega_{23}$ (meV)")
    a.grid(False)
    # (b) cuts + calibration
    b.axvspan(-tol, tol, color=GRID, alpha=0.8, lw=0)
    for w, ls, mk in ((5.0, "-", "o"), (3.0, "--", None)):
        rr = [r["eta"] for r in sorted([r for r in m if r["W23"] == w], key=lambda r: r["delta"])]
        b.plot(D, rr, ls=ls, marker=mk, ms=3, color=INK, lw=1.2)
    ro = sorted(S["map_optical"], key=lambda r: r["delta"])
    b.plot([r["delta"] for r in ro], [r["eta"] for r in ro], "s", mfc="none", mec=INK2, ms=4.5, mew=0.9)
    for n0, col in NCOL.items():
        rr = [r for r in S["mismatch"] if r["n_nominal"] == n0 and r["n_cal"] > 0]
        b.scatter([r["delta"] for r in rr], [r["eta"] for r in rr], s=10 + 22 * (n0 / 1e11),
                  facecolors="none", edgecolors=col, linewidths=1.1, zorder=4)
    b.set_xlim(-1.05, 1.05); b.set_ylim(0, 0.55)
    b.set_xlabel(r"$\hbar\breve\delta_\Xi$ (meV)"); b.set_ylabel(r"$\eta$")
    top = b.secondary_xaxis("top", functions=(lambda x: x / slope * 10, lambda y: y * slope / 10))
    top.set_xlabel(r"$n_X-n_{\rm cal}$ ($10^{10}$ cm$^{-2}$)", fontsize=7)
    b.grid(color=GRID, lw=0.6)
    # (c) bare lasers
    bs = sorted(S["bare_scan"], key=lambda r: r["n_X"]); DEL = {}
    e0 = S["partA"]["Xi_ci_case2"]["eta"]
    c.semilogy([r["n_X"] / 1e10 for r in bs], [r["eta"] for r in bs], "-", color=INK2, lw=1.1)
    c.axhline(e0, color=INK, ls=":", lw=1.0)
    for n0, col in NCOL.items():
        r = next(r for r in bs if r["n_X"] == n0)
        c.semilogy(n0 / 1e10, r["eta"], "D", color=col, ms=5)
        DEL[n0] = r["delta"]
    c.set_xlim(0, 21); c.set_ylim(1e-4, 1)
    c.set_xlabel(r"$n_X$ ($10^{10}$ cm$^{-2}$)"); c.set_ylabel(r"$\eta$")
    c.grid(color=GRID, lw=0.6)
    for p, lab in zip(ax, "abc"):
        p.set_title(f"({lab})", fontsize=9, color=INK, pad=4)
    h = [Line2D([], [], color=INK, marker="o", ms=3, lw=1.2, label=r"cut, $\hbar\Omega_{23}$ = 5 meV"),
         Line2D([], [], color=INK, ls="--", lw=1.2, label=r"cut, 3 meV"),
         Line2D([], [], color=INK2, ls="", marker="s", mfc="none", ms=4.5, label=r"5 meV, $\breve\delta_\Xi$ on optical leg"),
         Line2D([], [], color=INK, lw=1.2, label=r"(a) contour $\eta=0.9\,\eta(\breve\delta_\Xi=0)$"),
         Line2D([], [], color=INK2, ls="-.", lw=0.9, label=r"(a) $|\breve\delta_\Xi|=1/\sigma$"),
         Patch(color=GRID, label=rf"(b) $|n_X-n_{{\rm cal}}|\leq{tol/slope*10:.1f}\times10^{{10}}$ cm$^{{-2}}$, $\eta\geq0.9\,\eta(0)$"),
         Line2D([], [], color=INK, ls=":", lw=1.0, label="(c) calibrated lasers"),
         Line2D([], [], color=INK2, lw=1.1, label=r"(c) bare-tuned lasers ($n_{\rm cal}=0$)")]
    h += [(Line2D([], [], ls="", marker="o", mfc="none", mec=col, ms=4 + 2 * (n0 / 1e11), mew=1.1),
           Line2D([], [], ls="", marker="D", color=col, ms=4.5)) for n0, col in NCOL.items()]
    labs = [x.get_label() if not isinstance(x, tuple) else None for x in h]
    for k, (n0, col) in enumerate(NCOL.items()):
        labs[len(h) - 3 + k] = (rf"(b) $n_{{\rm cal}}$, (c) $n_X$ = {n0/1e10:g}$\times10^{{10}}$ cm$^{{-2}}$"
                                rf" [(c): $\hbar\breve\delta_\Xi$ = {DEL[n0]:.2f} meV]")
    fig.legend(h, labs, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, loc="outside lower center",
               ncol=3, fontsize=6.4, columnspacing=1.2)
    save(fig, "figR2")


fig_R2()
print("figR2 written")
