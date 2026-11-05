"""
fig3_R1.py -- Fig. 3 (R1) of the paper "Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley
Lindblad Framework for WSe2": time evolution of the Xi conversion at the baseline,
counterintuitive order, cases 1 (coherent only), 2 (+ photon dissipator) and 3 (+ exchange mean field
with bare-tuned lasers), Sect. VI B.

Runs from params.yaml with lindblad_solver.py (4 K, photon dissipator unless stated): Xi conversion with
the weak optical leg hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2), Gaussian pulses, baseline
sigma = 1.5 ps, tau_d = 1.25 sigma, hbar*Omega23 = 5 meV. Every run records max_t (1 - rho11), the
validity of the one-exciton truncation.

Prints, for each case, the coherence fraction C = |rho13|^2/(rho11 rho33) at t_f, the 10-90 % window of
N_emit, the time at which |rho13| departs from the dark-state value by 10 %, and |rho13|(t_f).
Outputs: figR1.pdf/png, fig3_R1.json (block "partA"), fig3_R1_runs.npz (trajectories).
Usage:  python fig3_R1.py   (reads params.yaml)
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
    for order, tag in (("counterintuitive", "ci"),):
        r = X.xi_conversion(P, W12b, W23b, S0, X0, D12=D12, D23=D23, order=order,
                            dissipator=diss, rtol=1e-10 if not diss else 1e-8, atol=1e-14)
        name = f"Xi_{tag}_case{case}"
        keep(name, r); summ["partA"][name] = S(r)
        traj[f"{name}__t12"] = np.array(r["t12"]); traj[f"{name}__t23"] = np.array(r["t23"])
print(f"cases 1-3 done {time.time()-t0:.0f}s")

json.dump({k: summ[k] for k in ('partA',)}, open("fig3_R1.json", "w"), indent=1, default=float)
np.savez_compressed("fig3_R1_runs.npz", **traj)

# ---- numbers of Fig. 3 read off the trajectories ---------------------------------------
Z = traj
for c in (1, 2, 3):
    n = f"Xi_ci_case{c}"; t = Z[n + "__t"]
    r11, r33, r13 = Z[n + "__rho11"].real, Z[n + "__rho33"].real, np.abs(Z[n + "__rho13"])
    C = r13[-1] ** 2 / (r11[-1] * r33[-1])
    N = Z[n + "__N_emit"]
    w = (t[np.searchsorted(N, 0.1 * N[-1])], t[np.searchsorted(N, 0.9 * N[-1])]) if N[-1] > 0 else None
    W12, W23 = Z[n + "__W12"], Z[n + "__W23"]
    on = W23 >= 0.1 * W23.max(); pred = np.where(on, W12 / np.where(on, W23, 1) * r11, np.nan)
    k = np.where(on & (t > Z[n + "__t23"]) & (np.abs(r13 / pred - 1) > 0.1))[0]
    print(f"case {c}: 1-C(t_f) = {1-C:.2e}; N_emit 10-90% window {w}; |rho13| departs 10% from dark value at t = {t[k[0]] if len(k) else None}; "
          f"t23 = {float(Z[n+'__t23'])}, t12 = {float(Z[n+'__t12'])}; |rho13|_f = {r13[-1]:.4f}")


# ---- figure ------------------------------------------------------------------------------
S = summ                 # run summaries, as read by the plotting functions
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


def fig_R1():
    fig, ax = plt.subplots(4, 3, figsize=(7.0, 6.8), sharex=True, constrained_layout=True)
    eta = {c: S["partA"][f"Xi_ci_case{c}"]["eta"] for c in (1, 2, 3)}
    for j, c in enumerate((1, 2, 3)):
        d = run(f"Xi_ci_case{c}"); t = d["t"]
        a = ax[0, j]                                                     # populations
        a.semilogy(t, np.maximum(1 - d["rho11"], 1e-12), color=C1, lw=1.4)
        a.semilogy(t, np.maximum(d["rho22"], 1e-12), color=C2, lw=1.4)
        a.semilogy(t, np.maximum(d["rho33"], 1e-12), color=C3, lw=1.4)
        a.set_ylim(1e-7, 3e-2); a.set_yticks([1e-6, 1e-4, 1e-2])
        pulses_twin(a, d, j == 2)
        b = ax[1, j]                                                     # vertex, emission, 2p
        b.semilogy(t, np.maximum(d["rho22"], 1e-12), color=C2, lw=1.4)
        b.semilogy(t, np.maximum(d["rho33"], 1e-12), color=C3, lw=1.1)
        if c > 1:
            b.semilogy(t, np.maximum(d["N_emit"], 1e-12), color=INK2, lw=1.2, ls="--")
            b.text(0.04, 0.90, rf"$\eta={eta[c]:.3f}$", transform=b.transAxes, ha="left", va="top",
                   fontsize=7.5, color=INK)
        b.set_ylim(1e-7, 3e-2); b.set_yticks([1e-6, 1e-4, 1e-2])
        e = ax[2, j]                                                     # dark-state following
        r13 = np.abs(d["rho13"])
        on = d["W23"] >= 0.1 * d["W23"].max()                         # plot only while the THz leg is on
        pred = np.where(on, d["W12"] / np.where(on, d["W23"], 1.0) * d["rho11"], np.nan)
        e.semilogy(t, np.maximum(r13, 1e-12), color=INK, lw=1.4)
        e.semilogy(t, pred, color=INK2, lw=1.1, ls="--")
        e.set_ylim(1e-4, 1.0)
        g = ax[3, j]                                                     # incoherent fraction
        g.set_yscale("log")
        C = r13 ** 2 / (d["rho11"] * np.maximum(d["rho33"], 1e-300))
        inc = np.where(d["rho33"] > 1e-8, 1 - C, np.nan)
        if c == 1:
            g.text(0.5, 0.5, "no dissipator:\nfully coherent ($1-C<10^{-12}$)", transform=g.transAxes,
                   ha="center", va="center", fontsize=7, color=INK2)
        else:
            g.semilogy(t, np.maximum(inc, 1e-12), color=INK, lw=1.4)
        g.set_ylim(1e-6, 1e-1); g.set_yticks([1e-5, 1e-3, 1e-1])
    ax[0, 0].set_ylabel("population")
    ax[1, 0].set_ylabel(r"$\bar\rho_{22}$, $\bar\rho_{33}$, $N_{\rm emit}$")
    ax[2, 0].set_ylabel(r"$|\bar\rho_{13}|$")
    ax[3, 0].set_ylabel(r"$1-|\bar\rho_{13}|^2/\bar\rho_{11}\bar\rho_{33}$")
    for j in range(3):
        ax[3, j].set_xlabel(r"$t$ (ps)")
    for p in ax.ravel():
        p.set_xlim(4, 20); p.grid(color=GRID, lw=0.6)
    letters = "abcdefghijkl"
    for k, p in enumerate(ax.ravel()):
        head = f"{CASES[k + 1]}\n" if k < 3 else ""
        p.set_title(f"{head}({letters[k]})", fontsize=8.5, color=INK, pad=4)
    h = [Line2D([], [], color=C1, lw=1.4, label=r"$1-\bar\rho_{11}$ (one-exciton sector)"),
         Line2D([], [], color=C2, lw=1.4, label=r"$\bar\rho_{22}$ ($|2\rangle$, 1s)"),
         Line2D([], [], color=C3, lw=1.4, label=r"$\bar\rho_{33}$ ($|3\rangle$, 2p)"),
         Line2D([], [], color=INK2, lw=1.2, ls="--", label=r"$N_{\rm emit}(t)$"),
         Line2D([], [], color=INK, lw=1.4, label=r"$|\bar\rho_{13}|$"),
         Line2D([], [], color=INK2, lw=1.1, ls="--", label=r"dark state, $(\Omega_{12}/\Omega_{23})\,\bar\rho_{11}$ ($\Omega_{23}\geq0.1\,\Omega_{23}^{\max}$)"),
         Line2D([], [], color=PULSE, lw=0.9, ls="-", label=r"$\Omega_{12}/\Omega_{12}^{\max}$"),
         Line2D([], [], color=PULSE, lw=0.9, ls="--", label=r"$\Omega_{23}/\Omega_{23}^{\max}$")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=6.8, columnspacing=1.4)
    save(fig, "figR1")


fig_R1()
print("figR1 written")
