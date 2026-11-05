"""
fig8_R5.py -- Fig. 8 (R5) of the paper "Dark-State Coherent Control of Excitons in TMD Monolayers:
A Reduced Intravalley Lindblad Framework for WSe2": the V manifold (|0>, 1s, 2s) as the unprotected
comparison, Sect. VI E (4 K, photon dissipator).

V response (Liouvillian null space): Delta~12 = 0, delta~V on the 281-point grid (+-5 meV),
  drive hbar*Omega12 = 0.1 meV, hbar*Omega13 = 0.01 meV, half-drive linearity check. Normalized
  susceptibilities (two-level peak = 1):
      chi12 = 2 G12 rho12/Omega12,  chi13 = 2 G13 rho13/Omega13,
      G12 = (g0->1s + g0->2s + g1s->0)/2,  G13 = (g0->1s + g0->2s + g2s->0)/2.
  Checks: rho23 against the lowest-order rho21 rho13 = Omega12 Omega13 / [4 G12 (G13 - i delta)];
  HWHM of |rho23|^2 (-> G13) and |rho23| (-> sqrt3 G13); Im chi13 is a single Lorentzian (no
  transparency window); Im chi12 flat in delta~V. Quasi-stationary diagnostic: plateaus of 20 ps at
  delta~V = 0, 0.18, 1, 5 meV against the null space.
V beat (time domain): synchronous Gaussian pair, sigma in {0.093, 0.12, 0.15, 0.18} ps,
  hbar*Omega = 1 meV each (amplitude check 0.5 meV), Delta~ = 0; plus the coherent-only reference.
  Observables: T2 from the free decay of |rho23|, lab-frame period (bare calibration carriers),
  |rho23| at the end of the pulses (t_c + 3 sigma), lowest-order prediction
      |rho23(t)| = (W12 W13/4)(2 pi sigma^2) exp[(g2^2 + g3^2) sigma^2/8] e^{-G23 (t - t_c)}.
Cross-topology: outer-coherence damping Xi (rho13: G13^Xi) against V (rho23: G23^V); panel (f)
  overlays the V |rho23| after the pulses with the Xi |rho13| after the CPT plateau of Fig. 7(a).
Panels
  (a) V populations rho11, rho22 = 1s, rho33 = 2s (sigma = 0.12 ps); grey: pulse envelope
  (b) |rho23(t)| for the four sigma; dashed: e^{-(t - t_c)/T2} fits
  (c) lab-frame beat Re rho23^lab at sigma = 0.12 ps, with the envelope |rho23|
  (d) |rho23| at t_c + 3 sigma vs sigma, with and without dissipator; surviving fraction
  (e) stationary response vs delta_V: Im chi13 (0 -> 2s), Im chi12 (0 -> 1s), |rho23|^2 normalized
  (f) outer coherence after the pulses, normalized: V |rho23| (sigma = 0.12 ps) vs Xi |rho13|

Outputs: figR5.pdf/png; fig8_R5.json (blocks "response", "plateau_diag", "beat", "cross_topology");
fig8_R5.h5 (beat trajectories, V response scan).
Usage:  python fig8_R5.py   (reads params.yaml)
"""
import json
import time

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import lindblad_solver as ls

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SIGMAS = (0.093, 0.12, 0.15, 0.18)
ALPHA = dict(zip(SIGMAS, np.linspace(0.45, 1.0, len(SIGMAS))))
HUE = dict(zip(SIGMAS, ("#d6334a", "#8a4fd1", "#b8860b", "#009c9c")))
SCOL = {s: matplotlib.colors.to_rgba(HUE[s], ALPHA[s]) for s in SIGMAS}


# ==================================================================== V response and beat
def rates(P):
    g = P.rates(4)
    G12 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_0_to_2s"] + g["gamma_1s_to_0"])
    G13 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_0_to_2s"] + g["gamma_2s_to_0"])
    G23 = 0.5 * (g["gamma_1s_to_0"] + g["gamma_2s_to_0"])
    return g, G12, G13, G23


def hwhm(x, y):
    k = int(np.argmax(y)); half = y[k] / 2
    r = next(i for i in range(k, len(y)) if y[i] < half)
    l = next(i for i in range(k, -1, -1) if y[i] < half)
    xr = x[r - 1] + (half - y[r - 1]) * (x[r] - x[r - 1]) / (y[r] - y[r - 1])
    xl = x[l + 1] + (half - y[l + 1]) * (x[l] - x[l + 1]) / (y[l] - y[l + 1])
    return 0.5 * (xr - xl)


# V runs (response, plateau, beat): generator rates from the 'dimensionless_gamma_tau0'
# block of params.yaml (x E0); the reference widths G12, G13, G23 from 'hbar_gamma_meV'.
def v_response(P, grid, W12, W13):
    g, G12, G13, _ = rates(P)
    out = {k: np.empty(len(grid), complex) for k in ("r12", "r13", "r23")}
    uniq = True
    for k, d in enumerate(grid):
        gen = ls.Generator(P, ls.V, T=4, frame_meV=ls.v_frame(0.0, d), rate_block=ls.RATES_TAU0)
        rho, info = ls.steady_state(gen, [W12, W13])
        out["r12"][k], out["r13"][k], out["r23"][k] = rho[0, 1], rho[0, 2], rho[1, 2]
        uniq &= info["unique"]
    out["chi12"] = 2 * G12 * out["r12"] / W12
    out["chi13"] = 2 * G13 * out["r13"] / W13
    out["r23_lo"] = W12 * W13 / (4 * G12 * (G13 - 1j * grid))
    out["unique"] = uniq
    return out


def plateau_diag(P, W12, W13, T_ps, d):
    vr = P.proto["V_response"]
    gen = ls.Generator(P, ls.V, T=4, frame_meV=ls.v_frame(0.0, d), rate_block=ls.RATES_TAU0)
    pro = ls.plateau_protocol(P, [W12, W13], vr["tau_on_off_ps"], T_ps)
    tr = ls.propagate(gen, pro)
    k = int(np.argmin(np.abs(tr.t - (pro.meta["t_off_ps"] - 10 * vr["tau_on_off_ps"]))))
    rss, _ = ls.steady_state(gen, [W12, W13])
    _, G12, G13, _ = rates(P)
    dchi13 = abs(2 * G13 * (tr.rho[k, 0, 2] - rss[0, 2]) / W13)
    dchi12 = abs(2 * G12 * (tr.rho[k, 0, 1] - rss[0, 1]) / W12)
    dr23 = abs(tr.rho[k, 1, 2] - rss[1, 2]) / abs(rss[1, 2])
    return {"T_pulse_ps": T_ps, "deltaV": d, "abs_dchi12": float(dchi12), "abs_dchi13": float(dchi13),
            "rel_drho23": float(dr23)}


def v_beat(P, sigma, W, dissipator=True):
    gen = ls.Generator(P, ls.V, T=4, frame_meV=(0.0, 0.0), dissipator=dissipator,
                       rate_block=ls.RATES_TAU0)
    pro = ls.v_beat_protocol(P, sigma, W, W)
    tr = ls.propagate(gen, pro)
    tps = tr.t; tc = pro.meta["t_c_ps"]
    r = np.abs(tr.rho[:, 1, 2])
    res = {"sigma_ps": sigma, "W_meV": W, "dissipator": dissipator,
           "pulse_area_rad": float(W * sigma * np.sqrt(2 * np.pi) / P.hbar_meV_ps)}
    kend = int(np.argmin(np.abs(tps - (tc + 3 * sigma))))
    res["rho23_end_of_pulse"] = float(r[kend])
    g, G12, G13, G23 = rates(P)
    if dissipator:
        sel = tps > pro.meta["free_decay_from_ps"]
        slope = np.polyfit(tps[sel], np.log(r[sel]), 1)[0]
        res["Gamma_fit_meV"] = float(-slope * P.hbar_meV_ps)
        res["T2_ps"] = float(-1 / slope)
        g2, g3 = g["gamma_1s_to_0"] / P.hbar_meV_ps, g["gamma_2s_to_0"] / P.hbar_meV_ps   # ps^-1
        Wr = W / P.hbar_meV_ps
        pred = (Wr**2 / 4) * 2 * np.pi * sigma**2 * np.exp((g2**2 + g3**2) * sigma**2 / 8)
        extrap = r[sel] * np.exp(G23 / P.hbar_meV_ps * (tps[sel] - tc))
        res["amp_extrap_tc"] = float(np.mean(extrap))
        res["amp_pred_lowest_order"] = float(pred)
        res["amp_ratio_num_over_pred"] = float(np.mean(extrap) / pred)
        res["n_periods_to_1e"] = float(res["T2_ps"] / (2 * np.pi * P.hbar_meV_ps /
                                                        (P.laser_energies_meV(0)["V_opt_0-2s"] -
                                                         P.laser_energies_meV(0)["V_opt_0-1s"])))
        EL = P.laser_energies_meV(0.0)
        lab = ls.lab_frame_coherence(tr, EL["V_opt_0-1s"], EL["V_opt_0-2s"])
        ph = np.unwrap(np.angle(lab[sel])); w = np.polyfit(tps[sel], ph, 1)[0]
        res["period_fs"] = float(2 * np.pi / abs(w) * 1e3)
    res["min_eig"] = float(ls.diagnostics(tr.rho)["min_eig"].min())
    return res, tr


# ==================================================================== figure
def fig_R5(P, S, trs, grid, R, xi):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "legend.fontsize": 6.6, "legend.handlelength": 2.2,
                         "legend.labelspacing": 0.3})
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 7.4), constrained_layout=True)
    (a, b), (c, d), (e, f) = ax
    OUT = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # (a) populations, case 2, sigma = 0.12 ps
    g = trs[0.12]; t = g.t; tc = g.protocol.meta["t_c_ps"]
    r11, r22, r33, W = g.rho[:, 0, 0].real, g.rho[:, 1, 1].real, g.rho[:, 2, 2].real, g.omegas[:, 0]
    # Xi CPT plateau, the run of Fig. 7(a)
    tx = xi["t"]; r13x = np.abs(xi["rho"][:, 0, 2])
    tau_x = float(P.proto["Part_A"]["A4_Xi_plateau_CPT"]["tau_on_off_ps"])
    pro_x = {"t_off_ps": float(xi["t_off"]), "tau_on_ps": tau_x}
    a.plot(t - tc, W / W.max(), color=INK2, lw=0.9, alpha=0.6)
    for y, col in ((r11, C1), (r22, C2), (r33, C3)):
        a.semilogy(t - tc, np.maximum(y, 1e-12), color=col, lw=1.3)
    a.set_xlim(-0.5, 2.5); a.set_ylim(1e-4, 2)
    a.set_xlabel(r"$t-t_c$ (ps)"); a.set_ylabel("population")
    a.legend(handles=[Line2D([], [], color=C1, lw=1.3, label=r"$\bar\rho_{11}$ (ground)"),
                      Line2D([], [], color=C2, lw=1.3, label=r"$\bar\rho_{22}$ ($1s$)"),
                      Line2D([], [], color=C3, lw=1.3, label=r"$\bar\rho_{33}$ ($2s$)"),
                      Line2D([], [], color=INK2, lw=0.9, alpha=0.6, label="pulse envelope\n" r"$\Omega(t)/\Omega_0$")],
             **OUT)

    # (b) |rho23| for the four sigma, with T2 fits
    B = {(r["sigma_ps"], r["W_meV"], r["dissipator"]): r for r in S["beat"]}
    tr = {}
    for s in SIGMAS:
        meta = trs[s].protocol.meta
        tr[s] = (trs[s].t - meta["t_c_ps"], trs[s].rho[:, 1, 2], meta)
    dV = grid
    chi12, chi13, r23s = R["chi12"], R["chi13"], R["r23"]
    for s in SIGMAS:
        tt, r23, meta = tr[s]
        b.semilogy(tt, np.maximum(np.abs(r23), 1e-12), color=SCOL[s], lw=1.3)
        bb = B[(s, 1.0, True)]
        tf = np.linspace(meta["free_decay_from_ps"] - meta["t_c_ps"], tt[-1], 50)
        b.semilogy(tf, bb["amp_extrap_tc"] * np.exp(-tf / bb["T2_ps"]), color=INK, ls="--", lw=0.8)
    b.set_xlim(-0.5, 2.5); b.set_ylim(1e-4, 0.2)
    b.set_xlabel(r"$t-t_c$ (ps)"); b.set_ylabel(r"$|\bar\rho_{23}|$")
    b.legend(handles=[Line2D([], [], color=INK, ls="--", lw=0.8, label="fit " r"$e^{-(t-t_c)/T_2}$" "\n" r"$T_2=0.369$ ps")],
             **OUT)

    # (c) lab-frame beat, sigma = 0.12 ps
    tt, r23, meta = tr[0.12]
    EL = P.laser_energies_meV(0.0)
    tfine = np.arange(-0.3, 0.9, 2e-4)
    rb = np.interp(tfine, tt, r23.real) + 1j * np.interp(tfine, tt, r23.imag)
    lab = np.exp(1j * (EL["V_opt_0-2s"] - EL["V_opt_0-1s"]) / P.hbar_meV_ps * (tfine + meta["t_c_ps"])) * rb
    c.plot(tfine, lab.real, color=SCOL[0.12], lw=0.6)
    c.plot(tfine, np.abs(lab), color=INK, ls="--", lw=0.9)
    c.plot(tfine, -np.abs(lab), color=INK, ls="--", lw=0.9)
    m = np.abs(lab).max(); c.set_ylim(-1.15 * m, 2.1 * m); c.set_xlim(-0.3, 0.9)
    c.set_xlabel(r"$t-t_c$ (ps)"); c.set_ylabel(r"$\bar\rho_{23}$, laboratory frame")
    c.legend(handles=[Line2D([], [], color=SCOL[0.12], lw=1.0, label=r"Re $\rho^{\rm lab}_{23}$, $\sigma=0.12$ ps"),
                      Line2D([], [], color=INK, ls="--", lw=0.9, label=r"$\pm|\bar\rho_{23}|$")],
             loc="upper right", ncol=1)
    c.text(0.97, 0.06, f"period {B[(0.12, 1.0, True)]['period_fs']:.1f} fs", transform=c.transAxes,
           ha="right", color=INK2, fontsize=6.6)

    # (d) amplitude at the end of the pulses vs sigma
    dv = [B[(s, 1.0, True)]["rho23_end_of_pulse"] for s in SIGMAS]
    cv = [B[(s, 1.0, False)]["rho23_end_of_pulse"] for s in SIGMAS]
    d.plot(SIGMAS, cv, color=INK2, ls="--", lw=0.9, marker="s", mfc="none", ms=4.5, mew=0.9)
    d.plot(SIGMAS, dv, color=INK2, lw=0.8, zorder=1)
    for s, y in zip(SIGMAS, dv):
        d.plot([s], [y], "o", color=SCOL[s], ms=6, zorder=3)
    for s, y, yc in zip(SIGMAS, dv, cv):
        d.text(s, y - 0.015, f"×{y/yc:.2f}", ha="center", color=INK2, fontsize=6.5)
    d.set_xlim(0.08, 0.19); d.set_ylim(-0.012, 0.145)
    d.set_xticks(SIGMAS); d.set_xticklabels([f"{s:g}" for s in SIGMAS])
    d.set_xlabel(r"$\sigma$ (ps)"); d.set_ylabel(r"$|\bar\rho_{23}|$ at $t_c+3\sigma$")
    d.legend(handles=[Line2D([], [], color=INK2, ls="", marker="o", ms=6, label="with dissipator (colour = $\\sigma$)"),
                      Line2D([], [], color=INK2, ls="--", marker="s", mfc="none", ms=4.5, label="coherent only"),
                      Line2D([], [], ls="", label=r"$\times$: surviving fraction")],
             loc="upper left")

    # (e) stationary response
    mE = np.abs(dV) <= 2.0
    a2 = np.abs(r23s) ** 2
    e.plot(dV[mE], chi13.imag[mE], color=C3, lw=1.3)
    e.plot(dV[mE], chi12.imag[mE], color=C2, lw=1.3)
    e.plot(dV[mE], (a2 / a2.max())[mE], color=INK, ls=":", lw=1.1)
    e.set_xlim(-2, 2); e.set_ylim(0, 1.15)
    e.set_xlabel(r"$\hbar\breve\delta_V$ (meV)"); e.set_ylabel("normalized response")
    e.legend(handles=[Line2D([], [], color=C3, lw=1.3, label=r"Im $\hat\chi_{13}$ ($0\to2s$)"),
                      Line2D([], [], color=C2, lw=1.3, label=r"Im $\hat\chi_{12}$ ($0\to1s$)"),
                      Line2D([], [], color=INK, ls=":", lw=1.1, label=r"$|\bar\rho_{23}|^2$ (normalized)")],
             **OUT)

    # (f) outer coherence after the pulses, normalized
    sV = 0.12; tt, r23, meta = tr[sV]
    t0V = 3 * sV                                        # end of the pulses, t_c + 3 sigma
    mV = tt >= t0V
    yV = np.abs(r23[mV]); f.semilogy(tt[mV] - t0V, yV / yV[0], color=SCOL[sV], lw=1.4)
    t0X = pro_x["t_off_ps"] + 3 * pro_x["tau_on_ps"]     # end of the switch-off ramp
    mX = tx >= t0X
    f.semilogy(tx[mX] - t0X, r13x[mX] / r13x[mX][0], color=INK, lw=1.4)
    tl = np.linspace(0, 1.5, 50)
    f.semilogy(tl, np.exp(-tl / B[(sV, 1.0, True)]["T2_ps"]), color=INK2, ls="--", lw=0.8)
    f.set_xlim(0, 1.5); f.set_ylim(1e-2, 2)
    f.set_xlabel(r"time after the pulses (ps)"); f.set_ylabel("outer coherence (normalized)")
    f.legend(handles=[Line2D([], [], color=INK, lw=1.4, label=r"$\Xi$: $|\bar\rho_{13}|$" "\nafter CPT"),
                      Line2D([], [], color=SCOL[sV], lw=1.4, label=r"$V$: $|\bar\rho_{23}|$," "\n" r"$\sigma=0.12$ ps"),
                      Line2D([], [], color=INK2, ls="--", lw=0.8, label=r"$e^{-t/T_2}$")],
             **OUT)
    f.text(0.04, 0.06, r"$\Xi$: $\hbar/\Gamma_{13}=1.2\,\mu$s" "\n" r"$V$: $T_2=0.369$ ps",
           transform=f.transAxes, ha="left", va="bottom", color=INK2, fontsize=6.6)

    for p, lab_ in zip(ax.flat, "abcdef"):
        p.set_title(f"({lab_})", fontsize=9, color=INK, pad=4)
        p.grid(color=GRID, lw=0.6, which="major")
    h = [Line2D([], [], color=SCOL[s_], lw=1.6, label=rf"$\sigma$ = {s_:g} ps") for s_ in SIGMAS]
    fig.legend(handles=h, loc="outside lower center", ncol=4, columnspacing=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(f"figR5.{ext}", dpi=220)
    plt.close(fig)


# ==================================================================== main
def main():
    P = ls.Params("params.yaml")
    vr, vb = P.proto["V_response"], P.proto["V_beat"]
    grid = np.array(vr["delta_V_grid"]["values_meV"])
    W12, W13 = vr["hbar_Omega12_meV"], vr["hbar_Omega13_meV"]
    g, G12, G13, G23 = rates(P)
    t0 = time.time()
    R = v_response(P, grid, W12, W13)
    Rh = v_response(P, grid, 0.5 * W12, 0.5 * W13)
    R["hwhm2"] = hwhm(grid, np.abs(R["r23"])**2); R["hwhm1"] = hwhm(grid, np.abs(R["r23"]))
    R["hwhm_re"] = hwhm(grid, R["r23"].real)
    lor = lambda x, a, G, x0: a * G**2 / ((x - x0)**2 + G**2)
    p, _ = curve_fit(lor, grid, R["chi13"].imag, p0=(1, G13, 0))
    lor_res = float(np.abs(lor(grid, *p) - R["chi13"].imag).max())
    resp = {"W12": W12, "W13": W13, "unique": bool(R["unique"]),
            "hwhm_abs2_meV": R["hwhm2"], "hwhm_abs_meV": R["hwhm1"], "hwhm_re_meV": R["hwhm_re"],
            "G13_meV": G13, "sqrt3_G13_meV": float(np.sqrt(3) * G13),
            "rel_dev_vs_lowest_order": float(np.abs(R["r23"] - R["r23_lo"]).max() / np.abs(R["r23_lo"]).max()),
            "chi12_im_variation": float(np.ptp(R["chi12"].imag) / np.abs(R["chi12"].imag).max()),
            "chi13_peak": float(R["chi13"].imag.max()),
            "chi13_lorentz_fit": {"amp": float(p[0]), "hwhm_meV": float(abs(p[1])), "centre_meV": float(p[2]),
                                  "max_residual": lor_res},
            "linearity_chi13": float(np.abs(Rh["chi13"] - R["chi13"]).max()),
            "linearity_chi12": float(np.abs(Rh["chi12"] - R["chi12"]).max())}
    diag = [plateau_diag(P, W12, W13, T, d) for T in (vr["T_pulse_ps_min"], 20.0) for d in (0.0, 0.18, 1.0, 5.0)]
    B, trs = [], {}
    for s in vb["sigma_ps_grid"]:
        for W in (vb["hbar_Omega12_meV"], vb["amplitude_check_hbar_Omega_meV"]):
            b, tr = v_beat(P, s, W); B.append(b)
            if W == vb["hbar_Omega12_meV"]:
                trs[s] = tr
        b, _ = v_beat(P, s, vb["hbar_Omega12_meV"], dissipator=False); B.append(b)
    Gxi = 0.5 * (g["gamma_0_to_1s"] + g["gamma_2p_to_1s"])
    cross = {"Gamma13_Xi_meV": Gxi, "T_Xi_ps": P.hbar_meV_ps / Gxi,
             "Gamma23_V_meV": G23, "T_V_ps": P.hbar_meV_ps / G23, "ratio": G23 / Gxi}
    # Xi CPT plateau, case 2 (the run of Fig. 7(a)), for panel (f)
    A4 = P.proto["Part_A"]["A4_Xi_plateau_CPT"]
    xi = ls.xi_plateau(P, A4["hbar_Omega12_meV"], A4["hbar_Omega23_meV"], tau=A4["tau_on_off_ps"],
                       T_pulse=A4["T_pulse_ps"], atol=1e-14)
    print(f"runs done in {time.time()-t0:.0f} s")

    js = {"response": resp, "plateau_diag": diag, "beat": B, "cross_topology": cross}
    json.dump(js, open("fig8_R5.json", "w"), indent=1, default=float)
    import h5py
    ls.write_h5("fig8_R5.h5", P, runs={f"V_beat_sigma{s}": (tr, ls.Generator(P, ls.V, T=4, rate_block=ls.RATES_TAU0))
                                       for s, tr in trs.items()},
                meta={"purpose": "V response and beat (Fig. 8)"})
    with h5py.File("fig8_R5.h5", "a") as f:
        h = f["scans"].create_group("V_response")
        h.create_dataset("deltaV_meV", data=grid)
        for k in ("r12", "r13", "r23", "chi12", "chi13", "r23_lo"):
            h.create_dataset(k, data=R[k]); h.create_dataset(k + "_half_drive", data=Rh[k])

    print(f"V response: HWHM |rho23|^2 {resp['hwhm_abs2_meV']:.4f} meV (Gamma13 = {G13:.4f}), HWHM |rho23| "
          f"{resp['hwhm_abs_meV']:.4f} meV (sqrt3 Gamma13 = {np.sqrt(3)*G13:.4f}), lowest-order deviation "
          f"{resp['rel_dev_vs_lowest_order']:.2e}, Im chi13 Lorentzian HWHM {resp['chi13_lorentz_fit']['hwhm_meV']:.4f} meV")
    for b in B:
        if b["dissipator"]:
            print(f"V beat sigma {b['sigma_ps']:g} ps, {b['W_meV']:g} meV: |rho23|(t_c+3 sigma) {b['rho23_end_of_pulse']:.4e}, "
                  f"T2 {b['T2_ps']:.4f} ps, period {b['period_fs']:.2f} fs")
        else:
            print(f"V beat sigma {b['sigma_ps']:g} ps, {b['W_meV']:g} meV, coherent only: |rho23|(t_c+3 sigma) {b['rho23_end_of_pulse']:.4e}")
    print(f"outer coherence: Xi Gamma13 = {Gxi:.2e} meV (lifetime {cross['T_Xi_ps']:.3g} ps); "
          f"V Gamma23 = {G23:.4f} meV (lifetime {cross['T_V_ps']:.3f} ps)")
    fig_R5(P, js, trs, grid, R, xi)
    print("figR5 written")


if __name__ == "__main__":
    main()
