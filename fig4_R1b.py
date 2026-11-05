"""
fig4_R1b.py -- Fig. 4 (R1b) of the paper "Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley
Lindblad Framework for WSe2": pulse-order control of the Xi conversion, sequential
(t12 < t23) against counterintuitive (t23 < t12) order, cases 1-3 at the baseline, Sect. VI B.

Runs from params.yaml with lindblad_solver.py (4 K, photon dissipator unless stated): Xi conversion with
the weak optical leg hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2), Gaussian pulses, baseline
sigma = 1.5 ps, tau_d = 1.25 sigma, hbar*Omega23 = 5 meV. Every run records max_t (1 - rho11), the
validity of the one-exciton truncation.

Outputs: figR1b.pdf/png, fig4_R1b.json (block "partA", both orders), fig4_R1b_runs.npz (trajectories).
Usage:  python fig4_R1b.py   (reads params.yaml)
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


# ---- cases 1-3 at the baseline ----------------------------------------------------------
summ["partA"] = {}
for case, diss, (D12, D23) in ((1, False, (0, 0)), (2, True, (0, 0)), (3, True, (D12c3, D23c3))):
    for order, tag in (("counterintuitive", "ci"), ("intuitive", "seq")):
        r = X.xi_conversion(P, W12b, W23b, S0, X0, D12=D12, D23=D23, order=order,
                            dissipator=diss, rtol=1e-10 if not diss else 1e-8, atol=1e-14)
        name = f"Xi_{tag}_case{case}"
        keep(name, r); summ["partA"][name] = S(r)
        traj[f"{name}__t12"] = np.array(r["t12"]); traj[f"{name}__t23"] = np.array(r["t23"])
print(f"cases 1-3 done {time.time()-t0:.0f}s")

json.dump({k: summ[k] for k in ('partA',)}, open("fig4_R1b.json", "w"), indent=1, default=float)
np.savez_compressed("fig4_R1b_runs.npz", **traj)

# ---- figure ------------------------------------------------------------------------------
Z = traj
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


def run(name):
    d = {k.split("__")[1]: Z[k] for k in Z if k.startswith(name + "__")}
    d["rho11"], d["rho22"], d["rho33"] = d["rho11"].real, d["rho22"].real, d["rho33"].real
    return d


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=220)
    plt.close(fig)


def pulses_twin(ax, d, show_labels):
    tw = ax.twinx()
    tw.plot(d["t"], d["W12"] / d["W12"].max(), color=PULSE, lw=0.9, ls="-", zorder=0)
    tw.plot(d["t"], d["W23"] / d["W23"].max(), color=PULSE, lw=0.9, ls="--", zorder=0)
    tw.set_ylim(0, 1.05); tw.spines["right"].set_visible(True); tw.spines["top"].set_visible(False)
    tw.tick_params(axis="y", colors=PULSE, labelsize=6.5)
    tw.set_yticks([0, 0.5, 1])
    if show_labels:
        tw.set_ylabel(r"$\Omega_\ell(t)/\Omega_\ell^{\max}$", color=PULSE, fontsize=7)
    else:
        tw.set_yticklabels([])
    ax.set_zorder(tw.get_zorder() + 1); ax.patch.set_visible(False)


def fig_R1b():
    fig, ax = plt.subplots(3, 3, figsize=(7.0, 5.2), sharex=True, constrained_layout=True)
    FIN = {}
    rows = (("1-rho11", C1, r"$1-\bar\rho_{11}$"), ("rho22", C2, r"$\bar\rho_{22}$ ($|2\rangle$, 1s)"),
            ("rho33", C3, r"$\bar\rho_{33}$ ($|3\rangle$, 2p)"))
    for j, c in enumerate((1, 2, 3)):
        seq, ci = run(f"Xi_seq_case{c}"), run(f"Xi_ci_case{c}")
        for i, (key, col, lab) in enumerate(rows):
            p = ax[i, j]
            f = (lambda d: 1 - d["rho11"]) if key == "1-rho11" else (lambda d, k=key: d[k])
            p.semilogy(seq["t"], np.maximum(f(seq), 1e-12), color=col, lw=1.4)
            p.semilogy(ci["t"], np.maximum(f(ci), 1e-12), color=col, lw=1.6, ls=":")
            p.set_ylim(1e-7, 3e-2); p.set_yticks([1e-6, 1e-4, 1e-2])
            fs, fc = f(seq)[-1], f(ci)[-1]
            FIN[(i, j)] = rf"$t_f$: seq. {fs:.1e}, c.i. {fc:.1e}"
            tw = p.twinx()
            tw.plot(seq["t"], seq["W12"] / seq["W12"].max(), color=PULSE, lw=0.9)
            tw.plot(seq["t"], seq["W23"] / seq["W23"].max(), color=PULSE, lw=0.9, ls="--")
            tw.set_ylim(0, 1.05); tw.set_yticks([]); tw.spines["right"].set_visible(False)
            p.set_zorder(tw.get_zorder() + 1); p.patch.set_visible(False)
            if j == 0:
                p.set_ylabel(lab)
    for j in range(3):
        ax[2, j].set_xlabel(r"$t$ (ps)")
    for p in ax.ravel():
        p.set_xlim(4, 20); p.grid(color=GRID, lw=0.6)
    letters = "abcdefghi"
    for k, p in enumerate(ax.ravel()):
        head = f"{CASES[k + 1]}\n" if k < 3 else ""
        p.set_title(f"{head}({letters[k]})  " + FIN[divmod(k, 3)], fontsize=7.6, color=INK, pad=4)
    h = [Line2D([], [], color=INK2, lw=1.4, label=r"sequential order ($t_{12}<t_{23}$)"),
         Line2D([], [], color=INK2, lw=1.6, ls=":", label=r"counterintuitive order ($t_{23}<t_{12}$)"),
         Line2D([], [], color=PULSE, lw=0.9, label=r"optical pulse $\Omega_{12}$ (sequential order: first)"),
         Line2D([], [], color=PULSE, lw=0.9, ls="--", label=r"THz pulse $\Omega_{23}$ (sequential order: second)")]
    fig.legend(handles=h, loc="outside lower center", ncol=2, fontsize=6.8, columnspacing=2.0)
    save(fig, "figR1b")


fig_R1b()
print("figR1b written")
