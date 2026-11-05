"""
fig5_R3.py -- Fig. 5 (R3) of the paper "Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley
Lindblad Framework for WSe2": adiabaticity and loss channels of the Xi conversion,
Sect. VI B. Grid hbar*Omega23 x sigma x tau_d/sigma, both pulse orders, delta~Xi = 0: conversion
efficiency eta (Eq. 76) against hbar*Omega23 and the adiabaticity parameter A (Eq. 77), vertex
suppression max rho22 / rho33(t_f), and the 2p ionization P_ion / rho33(t_f) (Eq. 57).

Runs from params.yaml with lindblad_solver.py (4 K, photon dissipator unless stated): Xi conversion with
the weak optical leg hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2), Gaussian pulses, baseline
sigma = 1.5 ps, tau_d = 1.25 sigma, hbar*Omega23 = 5 meV. Every run records max_t (1 - rho11), the
validity of the one-exciton truncation.

Outputs: figR3.pdf/png, fig5_R3.json (block "grid").
Usage:  python fig5_R3.py   (reads params.yaml)
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


# ---- adiabaticity grid -------------------------------------------------------------------
summ["grid"] = []
for W23 in G["hbar_Omega23_meV"]:
    for s_ in G["sigma_ps"]:
        for x in G["delay_over_sigma"]:
            for order in ("counterintuitive", "intuitive"):
                s = S(X.xi_conversion(P, P.weak(W23), W23, s_, x, order=order))
                summ["grid"].append(dict(W23=W23, sigma=s_, x=x, order=order, **s))
print(f"grid done {time.time()-t0:.0f}s")

json.dump({k: summ[k] for k in ('grid',)}, open("fig5_R3.json", "w"), indent=1, default=float)

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


def fig_R3():
    G = S["grid"]
    ci = [r for r in G if r["order"] == "counterintuitive"]
    sq = [r for r in G if r["order"] == "intuitive"]
    Aof = lambda r: r["W23"] ** 2 * r["sigma"] / (HB * G1S)
    A_lo, A_hi = 3.0 ** 2 * 1.1 / (HB * G1S), 5.0 ** 2 * 1.9 / (HB * G1S)
    HT = {tuple: HandlerTuple(ndivide=None, pad=0.0)}
    key = lambda sig, **kw: tuple(Line2D([], [], color=COLS[s], **kw) for s in sig)
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.3), constrained_layout=True)
    a, b, c, d = ax.ravel()
    styles = {1.0: ("--", None), 1.25: ("-", "o"), 1.5: (":", None)}
    for s in (1.1, 1.5, 1.9):
        for x, (ls, mk) in styles.items():
            rr = sorted([r for r in ci if r["sigma"] == s and r["x"] == x], key=lambda r: r["W23"])
            a.plot([r["W23"] for r in rr], [r["eta"] for r in rr], ls=ls, marker=mk, ms=3, color=COLS[s],
                   lw=1.4 if x == 1.25 else 1.1)
    a.set_xlabel(r"$\hbar\Omega_{23}$ (meV)"); a.set_ylabel(r"$\eta$"); a.set_ylim(0, 0.7)
    a.legend([key((1.1, 1.5, 1.9), ls=ls, marker=mk, ms=3, lw=1.2) for x, (ls, mk) in styles.items()],
             [rf"$\tau_d/\sigma$ = {x:g}" for x in styles], handler_map=HT, loc="lower right", handlelength=4.2)
    for s in (1.1, 1.5, 1.9):
        rr = sorted([r for r in ci if r["sigma"] == s and r["x"] == 1.25], key=Aof)
        b.plot([Aof(r) for r in rr], [r["eta"] for r in rr], "-o", ms=3, color=COLS[s], lw=1.4)
        rq = sorted([r for r in sq if r["sigma"] == s and r["x"] == 1.25], key=Aof)
        b.plot([Aof(r) for r in rq], [r["eta"] for r in rq], ":", color=COLS[s], lw=1.3)
        c.loglog([Aof(r) for r in rr], [r["vertex_suppression"] for r in rr], "-o", ms=3, color=COLS[s], lw=1.4)
        d.plot([r["W23"] for r in rr], [100 * r["P_ion_per_2p"] for r in rr], "-o", ms=3, color=COLS[s], lw=1.4)
    b.set_xscale("log"); b.set_ylim(0, 0.6); b.set_ylabel(r"$\eta$")
    b.legend([key((1.1, 1.5, 1.9), ls="-", marker="o", ms=3, lw=1.4), key((1.1, 1.5, 1.9), ls=":", lw=1.3)],
             ["counterintuitive order", "sequential order"], handler_map=HT, loc="upper left", handlelength=3.4)
    c.set_ylabel(r"vertex suppression, $\max_t\bar\rho_{22}/\bar\rho_{33}(t_f)$")
    for p in (b, c):
        p.axvspan(A_lo, A_hi, color=GRID, alpha=0.6, lw=0)
        p.set_xlabel(r"$\mathcal{A}=\Omega_{23}^2\sigma/\gamma_{1s\to0}$")
    d.set_xlabel(r"$\hbar\Omega_{23}$ (meV)"); d.set_ylabel(r"$P_{\rm ion}/\bar\rho_{33}(t_f)$ (%)")
    d.set_ylim(0, None)
    for p, lab in zip((a, b, c, d), "abcd"):
        p.grid(color=GRID, lw=0.6, which="major")
        p.set_title(f"({lab})", fontsize=10, color=INK, pad=6)
    fig.legend(handles=[Line2D([], [], color=COLS[s_], lw=1.8, label=f"σ = {s_} ps") for s_ in (1.1, 1.5, 1.9)]
               + [Patch(color=GRID, alpha=0.6, label=r"(b), (c) $\hbar\Omega_{23}$ = 3–5 meV, σ = 1.1–1.9 ps")],
               loc="outside lower center", ncol=4, columnspacing=2.0, fontsize=7.2)
    save(fig, "figR3")


fig_R3()
print("figR3 written")
