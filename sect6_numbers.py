"""
sect6_numbers.py -- numbers of Sect. VI of the paper "Dark-State Coherent Control of Excitons in TMD
Monolayers: A Reduced Intravalley Lindblad Framework for WSe2" that are not read off a figure.

  conversion     Xi conversion at the baseline (weak optical leg, sigma 1.5 ps, tau_d 1.25 sigma,
                 hbar*Omega23 5 meV): linearity in hbar*Omega12 (x 0.25, 0.5, 1, 2) and the
                 efficiency at T_gamma = 4, 77, 300 K (Sects. VI B, V item 6); V beat at every sigma of
                 the grid, cases 1 and 2, at 1 and 0.5 meV: max_t (1 - rho11), the validity of the
                 one-exciton truncation (Sect. VI A).
  vbeat_extra    populations, purity and the case-3 coherence of the V beat at the Fig. 8 baseline
                 (sigma = 0.12 ps, 1 meV on both legs, 4 K). Case 2: both legs resonant. Case 3:
                 bare-tuned lasers at n_X = 1e11 cm^-2 (Delta~12 = Sigma_1s, Delta~13 = Sigma_2s).
  eit_checks     Xi cascade response (Sect. VI D, weak probe 0.05 min(hbar*Omega23, hbar*gamma_1s/2),
                 Delta~23 = 0, 4 K): window centre against the n_X mismatch (Omega23 = 0.5 and 1 meV,
                 lasers calibrated at n_cal, THz tone fixed, probe scanned in laser detuning; centre
                 located on a local 0.002 meV grid and compared with the prediction -delta~Xi), and a
                 quasi-stationary diagnostic (50 ps plateaus at Omega23 = 0.5, 1.5, 5 meV and
                 Delta~12 = 0, 0.1, 1, 5, 15 meV: chi(t_off - 10 tau) against the null space).
  elevated_T     77 K and 300 K against 4 K, the photon rates being the only T-dependent input of the
                 generator (upward rates and n_gamma(omega) + 1): Xi EIT spectrum (hbar*Omega23 = 1 meV,
                 null space on the 481-point grid), V response ((0.1, 0.01) meV, null space on the
                 281-point grid), V beat (sigma 0.12 ps, 1 meV, T2 from the free decay); also the
                 thermal rates and the omitted 2p<->2s channel of Sect. V item 6 (ratio x gamma_2p->1s).
  window_centre  centre of the Xi transparency window as a density calibration (Sect. VI D), from an
                 independent null-space solver of the Xi generator, Eqs. (64)-(67): control on the
                 1s-2p leg (0.5 or 1 meV), probe on the 0-1s leg with the weak-probe rule; lasers
                 calibrated at n_cal, gas at n_X: Delta~12 = Sigma_1s(n_X)-Sigma_1s(n_cal) + p,
                 Delta~23 = [Sigma_2p - Sigma_1s](n_X) - [..](n_cal), p the probe scan. Absorption
                 A(p) = Im rho_21 / Omega12; the centre is the minimum of A(p) near p = -delta~Xi.
                 n_cal in {2.5e10, 5e10, 1e11} x {bare (n_cal = 0), -10, -5, 0, +5, +10 %}. Also the
                 dip FWHM and the density error it corresponds to.
  case3          the case-3 numbers of Sects. VI D-E with bare-tuned lasers, from an independent
                 integrator: (a) Xi CPT on the plateau, hbar*Omega12 = hbar*Omega23 = 1 meV, plateau
                 tau_on = tau_off = 0.5 ps, T_pulse = 50 ps (Eq. 42); (b) V quantum beat, synchronous
                 Gaussians sigma = 0.12 ps, 1 meV on both legs: reduction of |rho23| at the end of the
                 pulses, at n_X = 2.5e10 and 1e11 cm^-2.

Inputs (current directory): params.yaml; vertices_keldysh_200_8000_250_100.json (exchange_mean_field.py)
for window_centre and case3. Output: sect6_numbers.json (blocks "linearity", "elevated_T" (conversion at
77/300 K), "vbeat", "vbeat_extra", "partB_eit" (mismatch, plateau_diag), "elevated_T_response",
"window_centre", "case3_cpt_vbeat").
Usage:  python sect6_numbers.py [conversion|vbeat_extra|eit_checks|elevated_T|window_centre|case3|all]
"""
import json
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq

import lindblad_solver as X

VERTEX_FILE = "vertices_keldysh_200_8000_250_100.json"


# ==================================================================== conversion
def conversion(P, summ):
    pr = P.proto
    xc = pr["Xi_conversion"]; base = xc["baseline"]
    S0, X0, W23b = base["sigma_ps"], base["delay_over_sigma"], base["hbar_Omega23_meV"]
    W12b = P.weak(W23b)
    t0 = time.time()
    S = X.summary
    # ---- linearity and temperature -----------------------------------------------------------
    summ["linearity"] = []
    for f in (0.25, 0.5, 1.0, 2.0):
        s = S(X.xi_conversion(P, f * W12b, W23b, S0, X0))
        summ["linearity"].append(dict(factor=f, W12=f * W12b, **s))
    summ["elevated_T"] = {}
    for T in (4, 77, 300):
        s = S(X.xi_conversion(P, W12b, W23b, S0, X0, T=T))
        summ["elevated_T"][str(T)] = s
    print(f"lin/T done {time.time()-t0:.0f}s")
    # ---- V beat validity check ---------------------------------------------------------------
    summ["vbeat"] = []
    vb = pr["V_beat"]
    for W in (vb["hbar_Omega12_meV"], vb["amplitude_check_hbar_Omega_meV"]):
        for s_ in vb["sigma_ps_grid"]:
            for case, diss in ((1, False), (2, True)):
                r = X.v_beat(P, W, W, s_, dissipator=diss, rtol=1e-10 if not diss else 1e-8)
                s = S(r); k = int(np.argmin(abs(r["t"] - (r["tc"] + 6 * s_))))
                summ["vbeat"].append(dict(W=W, sigma=s_, case=case, max_one_exciton=s["max_one_exciton"],
                                          rho23_after=float(abs(r["rho"][k, 1, 2])),
                                          area_rad=W * s_ * np.sqrt(2 * np.pi) / X.HB))
    print(f"vbeat done {time.time()-t0:.0f}s")
    for r in summ["linearity"]:
        print(f"  linearity x{r['factor']:g}: eta = {r['eta']:.6f}")
    for T, s in summ["elevated_T"].items():
        print(f"  {T} K: eta = {s['eta']:.8f}, rho33 = {s['rho33_f']:.6e}")
    print(f"  V beat: max(1 - rho11) over the grid = {max(r['max_one_exciton'] for r in summ['vbeat']):.3e}")


# ==================================================================== vbeat_extra
def vbeat_extra(P, summ):
    out = {}
    for case, (D12, D13) in (("case2", (0.0, 0.0)), ("case3", (P.Sig1e11["1s"], P.Sig1e11["2s"]))):
        r = X.v_beat(P, 1.0, 1.0, 0.12, D12=D12, D13=D13)
        t, rho = r["t"], r["rho"]
        k = int(np.argmin(abs(t - (r["tc"] + 3 * 0.12))))          # end of the pulses, t_c + 3 sigma
        pur = np.einsum("tij,tji->t", rho, rho).real
        out[case] = dict(D12=D12, D13=D13, rho22_end=float(rho[k, 1, 1].real), rho33_end=float(rho[k, 2, 2].real),
                         max_rho22=float(rho[:, 1, 1].real.max()), max_rho33=float(rho[:, 2, 2].real.max()),
                         rho23_end=float(abs(rho[k, 1, 2])), purity_min=float(pur.min()))
    summ["vbeat_extra"] = out
    print(json.dumps(out, indent=1))


# ==================================================================== eit_checks
def gammas(P):
    g = P.rates(4)
    G12 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_1s_to_0"] + g["gamma_1s_to_2p"])
    G13 = 0.5 * (g["gamma_0_to_1s"] + g["gamma_2p_to_1s"])
    return G12, G13


def ss_rho(P, D12, D23, W12, W23):
    # EIT steady states: rates from the 'dimensionless_gamma_tau0' block (x E0)
    gen = X.Generator(P, X.XI, T=4, frame_meV=X.xi_frame(D12, D23), rate_block=X.RATES_TAU0)
    rho, info = X.steady_state(gen, [W12, W23])
    return rho, info


def chi_hat(P, rho, W12_meV):
    """2 Gamma12 rho12 / Omega12 with Gamma12 and Omega12 both in meV (dimensionless)."""
    G12, _ = gammas(P)
    return 2 * G12 * rho[0, 1] / W12_meV


def window_centre(P, W23, W12, row):
    """Hole centre (probe laser detuning relative to the calibrated resonance)."""
    d1s, d23 = row["Xi_Delta12"], row["Xi_Delta23"]
    pred = -row["Xi_delta"]
    x = pred + np.arange(-0.2, 0.2001, 0.002)
    A = np.array([chi_hat(P, ss_rho(P, xx + d1s, d23, W12, W23)[0], W12).imag for xx in x])
    k = int(np.argmin(A))
    if 0 < k < len(x) - 1:
        a, b, c = A[k - 1], A[k], A[k + 1]
        xc = x[k] + 0.5 * (a - c) / (a - 2 * b + c) * (x[1] - x[0])
    else:
        xc = x[k]
    return float(xc), float(pred), float(A[k])


def plateau_diag(P, W23, W12, D):
    gen = X.Generator(P, X.XI, T=4, frame_meV=X.xi_frame(D, 0.0), rate_block=X.RATES_TAU0)  # rates: dimensionless_gamma_tau0 x E0
    xr = P.proto["Xi_response"]
    pro = X.plateau_protocol(P, [W12, W23], xr["tau_on_off_ps"], xr["T_pulse_ps"])
    tr = X.propagate(gen, pro)
    tchk = pro.meta["t_off_ps"] - 10 * xr["tau_on_off_ps"]
    k = int(np.argmin(np.abs(tr.t - tchk)))
    rss, _ = ss_rho(P, D, 0.0, W12, W23)
    c_t = chi_hat(P, tr.rho[k], W12); c_s = chi_hat(P, rss, W12)
    return {"W23": W23, "D12": D, "chi_plateau": [c_t.real, c_t.imag], "chi_ss": [c_s.real, c_s.imag],
            "abs_diff": float(abs(c_t - c_s)), "rel_diff": float(abs(c_t - c_s) / abs(c_s)),
            "t_check_ps": float(tr.t[k])}


def eit_checks(P, summ):
    xr = P.proto["Xi_response"]
    Ws, W12s = xr["hbar_Omega23_meV"], xr["hbar_Omega12_meV_derived"]
    t0 = time.time()
    diag = [plateau_diag(P, W, w, D) for W, w in zip(Ws, W12s) if W in (0.5, 1.5, 5.0)
            for D in (0.0, 0.1, 1.0, 5.0, 15.0)]
    mm = []
    for W23 in (0.5, 1.0):
        w12 = [w for W, w in zip(Ws, W12s) if W == W23][0]
        for c in P.raw["calibration"]["mismatch_scan"]:
            for r in c["rows"]:
                xc, pred, amin = window_centre(P, W23, w12, r)
                mm.append({"W23": W23, "n_cal": r["n_cal"], "n_act": r["n_act"], "label": r["label"],
                           "centre": xc, "pred": pred, "A_min": amin})
    summ["partB_eit"] = {"mismatch": mm, "plateau_diag": diag}
    print(f"EIT checks done in {time.time()-t0:.0f} s")
    print("window centre vs n_X mismatch: max |numerical - predicted| = "
          f"{max(abs(m['centre'] - m['pred']) for m in mm):.1e} meV; max |delta~Xi| = {max(abs(m['pred']) for m in mm):.3f} meV")
    print("plateau vs null space: max abs. difference of chi_hat = "
          f"{max(d['abs_diff'] for d in diag):.1e}")


# ==================================================================== elevated_T
def elevated_T(P, summ):
    Ts = (4, 77, 300)
    xr, vr, vb = P.proto["Xi_response"], P.proto["V_response"], P.proto["V_beat"]
    gx = np.array(xr["probe_Delta12_grid"]["values_meV"]); gv = np.array(vr["delta_V_grid"]["values_meV"])
    W12e = [w for W, w in zip(xr["hbar_Omega23_meV"], xr["hbar_Omega12_meV_derived"]) if W == 1.0][0]
    res = {}
    # EIT, V-response and V-beat runs: generator rates from the 'dimensionless_gamma_tau0'
    # block (x E0); the reported thermal rates are the 'hbar_gamma_meV' block
    for T in Ts:
        r = {"rates_meV": P.rates(T), "n_gamma": P.raw["rates"][f"T_{T}K"]["n_gamma"],
             "omitted_2p_2s_over_retained": P.raw["rates"][f"T_{T}K"]["omitted_2p_2s_over_retained"]}
        # omitted 2p -> 2s upward channel: rate = ratio * gamma_2p->1s
        g_om = r["omitted_2p_2s_over_retained"] * r["rates_meV"]["gamma_2p_to_1s"] / P.hbar_meV_ps   # ps^-1
        r["omitted_leak_per_ns_in_2p"] = float(g_om * 1e3)
        chi = []
        for D in gx:
            g = X.Generator(P, X.XI, T=T, frame_meV=X.xi_frame(D, 0.0), rate_block=X.RATES_TAU0)
            rho, _ = X.steady_state(g, [W12e, 1.0])
            chi.append(rho[0, 1])
        r["_xi_chi"] = np.array(chi)
        r23 = []
        for d in gv:
            g = X.Generator(P, X.V, T=T, frame_meV=X.v_frame(0.0, d), rate_block=X.RATES_TAU0)
            rho, _ = X.steady_state(g, [vr["hbar_Omega12_meV"], vr["hbar_Omega13_meV"]])
            r23.append(rho[1, 2])
        r["_v_r23"] = np.array(r23)
        g = X.Generator(P, X.V, T=T, rate_block=X.RATES_TAU0)
        pro = X.v_beat_protocol(P, vb["sigma_ps_baseline"], vb["hbar_Omega12_meV"], vb["hbar_Omega13_meV"])
        trb = X.propagate(g, pro)
        tps = trb.t; sel = tps > pro.meta["free_decay_from_ps"]
        r["V_beat_T2_ps"] = float(-1 / np.polyfit(tps[sel], np.log(np.abs(trb.rho[sel, 1, 2])), 1)[0])
        res[T] = r
    ref = res[4]
    for T in Ts:
        r = res[T]
        r["Xi_EIT_max_rel_diff_vs_4K"] = float(np.abs(r["_xi_chi"] - ref["_xi_chi"]).max() / np.abs(ref["_xi_chi"]).max())
        r["V_resp_max_rel_diff_vs_4K"] = float(np.abs(r["_v_r23"] - ref["_v_r23"]).max() / np.abs(ref["_v_r23"]).max())
    for T in Ts:
        res[T].pop("_xi_chi"); res[T].pop("_v_r23")
    summ["elevated_T_response"] = {str(T): v for T, v in res.items()}
    print("| T | Xi EIT chi max rel. diff vs 4 K | V resp. rho23 max rel. diff vs 4 K | V T2 (ps) | omitted channel (per ns in 2p) |")
    for T in Ts:
        r = res[T]
        print(f"| {T} K | {r['Xi_EIT_max_rel_diff_vs_4K']:.1e} | {r['V_resp_max_rel_diff_vs_4K']:.1e} | "
              f"{r['V_beat_T2_ps']:.6f} | {r['omitted_leak_per_ns_in_2p']:.1e} |")


# ==================================================================== window_centre (independent solver)
def window_centre_calibration(summ):
    HB = 0.658211957
    E0 = 269.0
    G10, G32 = 3.20559, 1.1139e-6          # hbar*gamma in meV; work in meV units (hbar = 1)
    v = json.load(open(VERTEX_FILE))["vertices"]
    EaB2 = E0 * (1.19e-7) ** 2 * 1e11
    SIG = {a: EaB2 * (v[f"{a},{a}"]["ccr"] + v[f"{a},{a}"]["cca_herm"] + v[f"{a},{a}"]["eh"])
           for a in ("1s", "2p")}
    I3 = np.eye(3)

    def op(a, b):
        m = np.zeros((3, 3)); m[a, b] = 1.0
        return m

    def liouv(h2, h3, W12, W23):
        H = np.diag([0.0, h2, h3]).astype(complex)
        H[1, 0] = H[0, 1] = W12 / 2
        H[2, 1] = H[1, 2] = W23 / 2
        L = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
        for g, Lm in ((G10, op(0, 1)), (G32, op(1, 2))):
            LdL = Lm.T @ Lm
            L = L + g * (np.kron(Lm, Lm) - 0.5 * np.kron(I3, LdL) - 0.5 * np.kron(LdL.T, I3))
        return L

    def absorb(p, D12, D23, W12, W23):
        h2 = D12 + p
        L = liouv(h2, h2 + D23, W12, W23)
        A = np.vstack([L, I3.reshape(1, 9, order="F")])
        b = np.zeros(10, complex); b[-1] = 1
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        rho = x.reshape(3, 3, order="F")
        return float(np.imag(rho[1, 0]) / W12)

    def centre(D12, D23, dXi, W23):
        W12 = 0.05 * min(W23, G10 / 2)
        f = lambda p: abs(absorb(p, D12, D23, W12, W23))
        r = minimize_scalar(f, bracket=(-dXi - 0.02, -dXi, -dXi + 0.02), tol=1e-12)
        return r.x, W12

    def fwhm(W23):
        """FWHM of the transparency dip at two-photon resonance (no mean-field offset)."""
        W12 = 0.05 * min(W23, G10 / 2)
        a = lambda p: abs(absorb(p, 0.0, 0.0, W12, W23))
        abg = a(0.0); apk = max(a(x) for x in np.linspace(0.0, W23, 400))
        half = 0.5 * (abg + apk)
        pk = np.linspace(0.0, W23, 400)[np.argmax([a(x) for x in np.linspace(0.0, W23, 400)])]
        hw = brentq(lambda p: a(p) - half, 1e-9, pk)
        return 2 * hw

    def settings(S):
        rows = []
        for nc in (2.5e10, 5e10, 1e11):
            for lab, na, ncal in [("bare", nc, 0.0)] + [(f"{f:+.0%}", nc * (1 + f), nc)
                                                         for f in (-0.10, -0.05, 0.0, 0.05, 0.10)]:
                d = (na - ncal) / 1e11
                D12 = S["1s"] * d
                D23 = (S["2p"] - S["1s"]) * d
                rows.append(dict(n_nom=nc, label=lab, D12=D12, D23=D23, dXi=D12 + D23))
        return rows

    rows = settings(SIG)
    worst = {}
    for W23 in (0.5, 1.0):
        errs = []
        for r in rows:
            c, W12 = centre(r["D12"], r["D23"], r["dXi"], W23)
            r[f"centre_{W23}"] = c
            errs.append(abs(c + r["dXi"]))
        worst[W23] = max(errs)
    out = dict(rows=rows, worst=worst, max_abs_dXi=max(abs(r["dXi"]) for r in rows))
    print(f"max |delta_Xi| = {out['max_abs_dXi']:.3f} meV;  "
          f"max |centre + delta_Xi| = {worst[0.5]:.1e} meV (0.5 meV), {worst[1.0]:.1e} meV (1 meV)")
    slope = abs(SIG["2p"]) / 1e11
    for W23 in (0.5, 1.0):
        w = fwhm(W23)
        out[f"fwhm_{W23}"] = w
        print(f"dip FWHM at hbar*Omega23 = {W23} meV: {w:.4f} meV  -> density error {w/slope:.2e} cm^-2")
    summ["window_centre"] = out


# ==================================================================== case3 (independent integrator)
def case3_cpt_vbeat(summ):
    HB, E0 = 0.658211957, 269.0
    v = json.load(open(VERTEX_FILE))["vertices"]
    EaB2 = E0 * (1.19e-7) ** 2 * 1e11
    SIG = {a: EaB2 * (v[f"{a},{a}"]["ccr"] + v[f"{a},{a}"]["cca_herm"] + v[f"{a},{a}"]["eh"])
           for a in ("1s", "2p", "2s")}

    def evolve(h, legs, jumps, env_list, amps, tend, maxstep, bps, teval):
        h = np.asarray(h) / HB

        def rhs(t, y):
            r = y.reshape(3, 3)
            H = np.diag(h).astype(complex)
            for (i, j), e, a in zip(legs, env_list, amps):
                H[j, i] += a / HB * e(t) / 2; H[i, j] += a / HB * e(t) / 2
            d = -1j * (H @ r - r @ H)
            for g, a_, b in jumps:                          # L = |a_><b| at rate g (meV)
                g = g / HB
                d[a_, a_] += g * r[b, b]; d[b, b] -= g * r[b, b]
                for k in range(3):
                    if k != b:
                        d[b, k] -= 0.5 * g * r[b, k]; d[k, b] -= 0.5 * g * r[k, b]
            return d.ravel()

        y = np.zeros(9, complex); y[0] = 1
        T, R = [], []
        for a, b in zip(bps[:-1], bps[1:]):
            te = teval[(teval >= a) & (teval <= b)]
            s = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=1e-8, atol=1e-10, max_step=maxstep,
                          t_eval=te)
            T.append(s.t); R.append(s.y.T)
            s2 = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=1e-8, atol=1e-10, max_step=maxstep)
            y = s2.y[:, -1]
        return np.concatenate(T), np.concatenate(R).reshape(-1, 3, 3)

    def cpt(S, n):
        d = n / 1e11
        D12, D23 = S["1s"] * d, (S["2p"] - S["1s"]) * d
        ton, Tp = 0.5, 50.0
        t_on, t_off = 6 * ton, 6 * ton + Tp
        tend = t_off + 6 * ton
        env = lambda t: 0.25 * (1 + np.tanh((t - t_on) / ton)) * (1 - np.tanh((t - t_off) / ton))
        te = np.linspace(0, tend, 6001)
        t, r = evolve([0, D12, D12 + D23], [(0, 1), (1, 2)], [(3.20559, 0, 1), (1.1139e-6, 1, 2)],
                      [env, env], [1.0, 1.0], tend, ton / 20,
                      sorted({0, t_on - 6 * ton, t_on, t_on + 6 * ton, t_off - 6 * ton, t_off, tend}), te)
        PD = 0.5 * (r[:, 0, 0].real + r[:, 2, 2].real) - r[:, 0, 2].real    # theta = 45 deg
        on = env(t) > 1e-6
        return dict(D12=D12, D23=D23, dXi=D12 + D23, PD_max=float(PD[on].max()), PD_end=float(PD[on][-1]))

    def vbeat(S, n):
        d = n / 1e11
        D12, D13 = S["1s"] * d, S["2s"] * d
        sg = 0.12; tc = 6 * sg
        env = lambda t: np.exp(-0.5 * ((t - tc) / sg) ** 2)
        tend = tc + 6 * sg
        te = np.linspace(0, tend, 2001)
        t, r = evolve([0, D12, D13], [(0, 1), (0, 2)], [(3.20559, 0, 1), (0.360901, 0, 2)],
                      [env, env], [1.0, 1.0], tend, sg / 20, [0, tc, tend], te)
        return dict(D12=D12, D13=D13, rho23_end=float(abs(r[-1, 1, 2])))

    out = {}
    ref = vbeat(SIG, 0.0)["rho23_end"]
    print(f"V beat, calibrated lasers: |rho23| at end of pulses = {ref:.5f}")
    for tag, n in (("2.5e10", 2.5e10), ("1e11", 1e11)):
        c = cpt(SIG, n); vb = vbeat(SIG, n)
        vb["reduction"] = 1 - vb["rho23_end"] / ref
        out[tag] = dict(cpt=c, vbeat=vb)
        print(f"{tag:7s} CPT: dXi {c['dXi']:+.3f} meV, max P_D {c['PD_max']:.3f}, P_D(end) {c['PD_end']:.3f} | "
              f"V beat: D12 {vb['D12']:+.2f}, D13 {vb['D13']:+.2f} meV, |rho23| {vb['rho23_end']:.5f} "
              f"(reduction {100*vb['reduction']:.1f}%)")
    summ["case3_cpt_vbeat"] = dict(ref_rho23=ref, **out)


# ==================================================================== main
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    P = X.Params("params.yaml")
    summ = {}
    parts = (("conversion", lambda: conversion(P, summ)), ("vbeat_extra", lambda: vbeat_extra(P, summ)),
             ("eit_checks", lambda: eit_checks(P, summ)), ("elevated_T", lambda: elevated_T(P, summ)),
             ("window_centre", lambda: window_centre_calibration(summ)), ("case3", lambda: case3_cpt_vbeat(summ)))
    for name, fn in parts:
        if which in (name, "all"):
            print(f"\n######## {name}")
            fn()
    json.dump(summ, open("sect6_numbers.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
