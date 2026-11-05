"""
exchange_mean_field.py  --  exciton-gas exchange vertices and the exchange mean field.

Part 1, vertices (App. D 6, Eqs. D8-D10, Table IX). The reduced exchange vertices
zeta_{n'n} of a tagged exciton n -> n' with a gas exciton in 1s, in E_0 a_B^2:

  zeta^{cc,r}  like-carrier lines of the carrier exchange, Eq. (D8);
  zeta^{cc,a}  electron-hole Coulomb lines of the carrier exchange, Eq. (D9):

    zeta^{cc,a}_{n'n} = 2 (2pi) delta_{m',m} INT dk k R_n'(k) R_n(k) R_1s(k)^2
                        [ k^2 + E_b(1s) + (E_b(n) + E_b(n'))/2 ]      (Hermitian)

    with the prior ("in") form  [k^2 + E_b(1s) + E_b(n')]  and the post ("out") form
    [k^2 + E_b(1s) + E_b(n)] also reported;
  zeta^{eh}    interband electron-hole exchange, Eq. (D10).

The Wannier identity used to write zeta^{cc,a} as a single momentum sum is checked
against direct quadrature:

    H_|m|[ -V(r) psi_a(r) ](k)  =?=  (k^2/2 - E_a) R_a(k)                     (W)

and the direct (kernel) form of zeta^{cc,a} is evaluated from the left side of (W).

Part 2, mean field (keldysh model only). The vertices are propagated to the numbers
quoted in Sect. V A (Eq. 62 and the detuning shifts), Sect. VI B (case-3 detunings)
and App. D 6 (Tables IX and X, chi threshold, admixture estimates), with the
formulas of the paper:
  Sigma_a = n_X a_B^2 E0 zeta_aa ;  Delta12 shift = Sigma_1s ; Delta23 shift = Sigma_2p - Sigma_1s ;
  delta_Xi shift = Sigma_2p ; delta_V shift = Sigma_2s - Sigma_1s  (Sigma_0 = 0)
  chi_1s,2s = |Sigma_1s,2s| / (eps_2s - eps_1s)                (Eq. D6, leading order Eq. D11)

Units: solver natural units (a_B = 1, E_0 = 1). zeta in E_0 a_B^2.
Outputs: vertices_<model>_<R_CUT>_<N_R>_<NK_SEG>_<K_MAX>.json (read by
         solver_inputs_from_wannier.py) and, for keldysh, propagated.json.
Usage:  python exchange_mean_field.py keldysh      (WSe2/hBN, r0_eff = 0.84; ~35 s)
        python exchange_mean_field.py coulomb      (2D hydrogen benchmark)
        optional quadrature arguments: R_CUT N_R NK_SEG K_MAX (defaults 200 8000 250 100);
        environment RR_NPS, RR_NSC change the Rayleigh-Ritz basis (default 6 x 6).
"""
import sys
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jv
from rayleigh_ritz_2d import (RayleighRitzSolver2D, keldysh_potential,
                              coulomb_potential, make_rr_grid, envelope_at_origin)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "keldysh"
E0_meV, aB_nm = 269.0, 1.19
r0 = 0.84
Eg_nat = 1890.0 / E0_meV
# default quadrature (converged); overridable for convergence runs
R_CUT = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
N_R = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
NK_SEG = int(sys.argv[4]) if len(sys.argv) > 4 else 250
K_MAX = float(sys.argv[5]) if len(sys.argv) > 5 else 100.0
NU_MAX = 12.0

if MODEL == "keldysh":
    Vfun = lambda rr: keldysh_potential(rr, 1.0, r0)
    solve_kw = {}
elif MODEL == "coulomb":
    Vfun = lambda rr: coulomb_potential(rr, 1.0)
    solve_kw = {"scales": np.unique(np.concatenate([np.geomspace(0.3, 3.0, 6), [1.0]]))}
else:
    raise SystemExit("model must be keldysh or coulomb")

import os
_NPS = int(os.environ.get("RR_NPS", 6)); _NSC = int(os.environ.get("RR_NSC", 6))
res = {m_: RayleighRitzSolver2D(Vfun, m=m_, n_per_scale=_NPS, n_scales=_NSC).solve(n_states=5, verbose=False, **solve_kw)
       for m_ in (0, 1)}
STATE = {'1s': (0, 0), '2s': (0, 1), '3s': (0, 2), '2p': (1, 0), '3p': (1, 1), '4p': (1, 2)}
mval = lambda l: STATE[l][0]
E = lambda l: res[mval(l)].levels[STATE[l][1]].energy          # signed, E0 units


def _gl(edges, n):
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        t, w = leggauss(n)
        xs.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo)); ws.append(0.5 * (hi - lo) * w)
    return np.concatenate(xs), np.concatenate(ws)


_LAM = max(res[0].scales.max(), res[1].scales.max())
r, wr = make_rr_grid(N_R, _LAM, NU_MAX, 0)
kg, wk = _gl([1e-12, 1.0, 3.0, 10.0, K_MAX], NK_SEG)
kdk = kg * wk
_J = {m: jv(m, np.outer(r, kg)) for m in (0, 1)}
psi = {l: res[mval(l)].wavefunction(STATE[l][1], r) for l in STATE}
# order-|m| Hankel transforms, sign convention R_n(k) > 0 as k -> 0 (App. D)
R = {}
for l in STATE:
    Rl = (wr * r * psi[l]) @ _J[abs(mval(l))]
    i0 = np.argmax(np.abs(Rl[kg < 0.3])) if abs(mval(l)) else 0
    if Rl[i0] < 0:
        Rl, psi[l] = -Rl, -psi[l]
    R[l] = Rl

_Vr = Vfun(r)                                   # attractive, V < 0
_mask = r <= R_CUT
_wV = (wr * r * _Vr) * _mask


def Pair(x, y, m):
    return _J[m] @ (kdk * R[x] * R[y])


def zeta_ccr(a, b):
    """Like-carrier lines of the carrier exchange, Eq. (D8)."""
    if mval(a) != mval(b):
        return 0.0
    m = abs(mval(a))
    return 2.0 * (2 * np.pi) * np.sum(_wV * (Pair(a, '1s', m) * Pair('1s', b, m)
                                            + Pair(a, b, 0) * Pair('1s', '1s', 0)))


# Wannier check (W): D_a(k) = H_|m|[-V psi_a](k) vs (k^2/2 - E_a) R_a(k)
D = {l: (wr * r * (-_Vr) * psi[l]) @ _J[abs(mval(l))] for l in STATE}
W = {l: (0.5 * kg ** 2 - E(l)) * R[l] for l in STATE}


def zeta_cca(a, b, form="herm", kernel=False):
    """Electron-hole lines of the carrier exchange, gas in 1s, Eq. (D9).
    a = outgoing tagged state n', b = incoming n.  form: 'prior' | 'post' | 'herm'.
    kernel=True evaluates the prior form with the direct kernel sums D instead of
    the Wannier identity (only 'prior' is defined that way)."""
    if mval(a) != mval(b):
        return 0.0
    g = R['1s'] ** 2
    if kernel:
        if form != "prior":
            raise ValueError
        val = np.sum(kdk * (D[a] * R[b] * g + R[a] * R[b] * R['1s'] * D['1s']))
        return 2.0 * (2 * np.pi) * val
    Eb1, Ebn, Ebnp = -E('1s'), -E(b), -E(a)
    extra = {"prior": Ebnp, "post": Ebn, "herm": 0.5 * (Ebn + Ebnp)}[form]
    return 2.0 * (2 * np.pi) * np.sum(kdk * R[a] * R[b] * g * (kg ** 2 + Eb1 + extra))


_NPHI = 600
_phi = np.linspace(0, 2 * np.pi, _NPHI, endpoint=False)


def _w0():
    K = kg[:, None]; Kp = kg[None, :]
    out = np.zeros((len(kg), len(kg)))
    for c in np.cos(_phi):
        q = np.maximum(np.sqrt(np.maximum(K * K + Kp * Kp - 2 * K * Kp * c, 0.0)), 1e-12)
        out += (2 * np.pi / (4 * Eg_nat)) * q / (1 + r0 * q)
    return out * (2 * np.pi / _NPHI) / (2 * np.pi)


def zeta_eh(a, b):
    """Interband electron-hole exchange, Eq. (D10)."""
    if mval(a) != mval(b):
        return 0.0
    return 2.0 * np.sum(R[a] * R[b] * kdk * (_W0 @ (R['1s'] ** 2 * kdk)))


def propagate(vertex_file):
    """Part 2: exchange mean field from the vertex file; prints the table and writes propagated.json."""
    E0, aB_nm = 269.0, 1.19
    aB2_cm2 = (aB_nm * 1e-7) ** 2
    aB2_um2 = (aB_nm * 1e-3) ** 2
    d = json.load(open(vertex_file))
    V = d["vertices"]
    En = d["E_nat"]
    gamma1s = 3.21                        # meV, Eq. (C26)
    dE12 = (En["2s"] - En["1s"]) * E0     # eps_2s - eps_1s  [meV]

    def z(a, b, part="total", form="herm"):
        v = V[f"{a},{b}"]
        if part == "total":
            cca = {"herm": v["cca_herm"], "prior": v["cca_prior"], "post": v["cca_post"]}[form]
            return v["ccr"] + cca + v["eh"]
        return v[part]

    S = lambda a, n, form="herm": E0 * n * aB2_cm2 * z(a, a, form=form)      # meV
    S12 = lambda n, form="herm": E0 * n * aB2_cm2 * z("1s", "2s", form=form)

    out = {}
    g = E0 * aB2_um2 * z("1s", "1s") * 1e3         # ueV um^2
    print(f"Eq. (62): Sigma_1s/n_X = {g:+.2f} ueV um^2")
    out["gXX_ueV_um2"] = g
    print(f"eps_2s - eps_1s = {dE12:.2f} meV")

    print("\nTable IX (E0 aB^2), Hermitian prescription for zeta^{cc,a}:")
    print(f"{'(a,b)':9s} {'cc,r':>8s} {'cc,a':>8s} {'eh':>7s} {'zeta':>8s}   prior/post total")
    tab = {}
    for a, b in [("1s", "1s"), ("2s", "2s"), ("2p", "2p"), ("3p", "3p"), ("1s", "2s"), ("2p", "3p")]:
        v = V[f"{a},{b}"]
        zp = v["ccr"] + v["cca_prior"] + v["eh"]; zo = v["ccr"] + v["cca_post"] + v["eh"]
        tab[f"{a},{b}"] = dict(ccr=v["ccr"], cca=v["cca_herm"], eh=v["eh"], zeta=z(a, b), prior=zp, post=zo)
        print(f"({a},{b}) {v['ccr']:8.2f} {v['cca_herm']:8.2f} {v['eh']:7.3f} {z(a, b):8.2f}   {zp:6.2f} / {zo:6.2f}")
    out["tableIX"] = tab

    for n in (1e11, 2.5e10):
        s1, s2p, s2s = S("1s", n), S("2p", n), S("2s", n)
        print(f"\nn_X = {n:.1e} cm^-2: Sigma_1s = {s1:+.3f}, Sigma_2p = {s2p:+.3f}, Sigma_2s = {s2s:+.3f} meV")
        print(f"  Xi: Delta12 {s1:+.3f}  Delta23 {s2p - s1:+.3f}  delta_Xi {s2p:+.3f} meV;   V: delta_V {s2s - s1:+.3f} meV")
        out[f"shifts_{n:.1e}"] = dict(S1s=s1, S2p=s2p, S2s=s2s, D12=s1, D23=s2p - s1, dXi=s2p, dV=s2s - s1)

    n1 = gamma1s / abs(S("2p", 1.0))
    print(f"\n|delta_Xi| equals the 3.21 meV interband linewidth at n_X = {n1:.2e} cm^-2 "
          f"(n_X aB^2 = {n1 * aB2_cm2:.4f})")
    out["n_dXi_equals_gamma"] = n1

    # V-block admixture estimates at 1e11 (Sect. V A), Hermitian and the in/out range
    n = 1e11
    for form in ("herm", "prior", "post"):
        s12 = S12(n, form); chi = abs(s12) / dE12
        second = 2 * s12 ** 2 / dE12
        first = S("2s", n) - S("1s", n)
        print(f"[{form:5s}] Sigma_1s,2s = {s12:+.3f} meV  chi = {chi:.4f}  2|S12|^2/dE = {second:.4f} meV "
              f"= {100 * second / first:.1f}% of delta_V shift")
        out[f"V_{form}"] = dict(S12=s12, chi=chi, second=second, ratio=second / first)
    # envelope admixture of the 1s-2s mixing, from the solver envelopes:
    # |phi_1s(0)/phi_2s(0)| = |psi_1s(0)/psi_2s(0)|
    res = RayleighRitzSolver2D(lambda r: keldysh_potential(r, 1.0, 0.84), m=0).solve(n_states=3, verbose=False)
    ratio = abs(envelope_at_origin(res, 0) / envelope_at_origin(res, 1))
    chi = abs(S12(n)) / dE12
    print(f"|phi_1s(0)/phi_2s(0)| = {ratio:.3f};  2s envelope change {100 * chi * ratio:.1f}% "
          f"(rate/Rabi ~{200 * chi * ratio:.1f}%), 1s {100 * chi / ratio:.2f}% ({200 * chi / ratio:.2f}%)")
    out["admix"] = dict(ratio=ratio, d2s=chi * ratio, d1s=chi / ratio)
    # density below which the 2s rate correction (2 chi ratio) stays below 10 %
    n10 = 0.10 / (2 * ratio * abs(S12(1.0)) / dE12)
    nsec = 0.10 / (2 * abs(S12(1.0)) ** 2 / dE12 / (S("2s", 1.0) - S("1s", 1.0)))
    print(f"2s-rate correction < 10% for n_X < {n10:.2e};  second/first < 10% for n_X < {nsec:.2e} cm^-2")
    out["n_2s_rate_10pc"] = n10; out["n_second_10pc"] = nsec

    # Table X against n_X, with |Sigma_1s| and the in/out range of chi
    print("\nTable X:")
    rowsX = []
    for n in (1e10, 2.5e10, 5e10, 1e11, 2e11):
        rowsX.append(dict(n=n, S1s=S("1s", n), chi=abs(S12(n)) / dE12,
                          chi_prior=abs(S12(n, "prior")) / dE12, chi_post=abs(S12(n, "post")) / dE12))
        r_ = rowsX[-1]
        print(f"  n_X {n:.1e}  Sigma_1s {r_['S1s']:+.3f} meV  chi {r_['chi']:.4f}  (in/out {r_['chi_post']:.4f}-{r_['chi_prior']:.4f})")
    out["tableX"] = rowsX
    nchi = 0.1 * dE12 / abs(S12(1.0))
    print(f"chi = 0.1 at n_X = {nchi:.2e} cm^-2 (n_X aB^2 = {nchi * aB2_cm2:.4f}), Sigma_1s = {S('1s', nchi):.1f} meV;"
          f"  prior form: {0.1 * dE12 / abs(S12(1.0, 'prior')):.2e}")
    out["n_chi_0p1"] = nchi
    json.dump(out, open("propagated.json", "w"), indent=1)


if __name__ == "__main__":
    out = {"model": MODEL, "quad": dict(R_CUT=R_CUT, N_R=N_R, NK_SEG=NK_SEG, K_MAX=K_MAX)}
    print(f"== model: {MODEL}   R_CUT={R_CUT} N_R={N_R} NK_SEG={NK_SEG} K_MAX={K_MAX}")
    print("binding energies [E0]:", {l: round(-E(l), 6) for l in STATE})
    print("Parseval:", {l: round(float(np.sum(R[l] ** 2 * kdk)), 7) for l in STATE})
    # Wannier identity residual, relative L2 on the k grid
    wres = {}
    for l in ['1s', '2s', '2p', '3p']:
        wres[l] = float(np.sqrt(np.sum(kdk * (D[l] - W[l]) ** 2) / np.sum(kdk * W[l] ** 2)))
    print("Wannier identity, rel. L2 residual:", {k: f"{v:.1e}" for k, v in wres.items()})
    out["wannier_residual"] = wres

    pairs = [('1s', '1s'), ('2s', '2s'), ('2p', '2p'), ('3p', '3p'),
             ('1s', '2s'), ('2s', '1s'), ('2p', '3p'), ('3p', '2p'), ('1s', '2p')]
    _W0 = _w0() if MODEL == "keldysh" else None
    rows = {}
    print(f"\n{'(a,b)':9s} {'cc,r':>9s} {'cc,a herm':>10s} {'cc,a prior':>11s} {'cc,a post':>10s}"
          f" {'cc,a kern':>10s} {'eh':>8s} {'total':>9s}")
    for a, b in pairs:
        ccr = zeta_ccr(a, b)
        cah, cap, capo = zeta_cca(a, b, "herm"), zeta_cca(a, b, "prior"), zeta_cca(a, b, "post")
        cak = zeta_cca(a, b, "prior", kernel=True) if mval(a) == mval(b) else 0.0
        eh = zeta_eh(a, b) if MODEL == "keldysh" else 0.0
        tot = ccr + cah + eh
        rows[f"{a},{b}"] = dict(ccr=ccr, cca_herm=cah, cca_prior=cap, cca_post=capo,
                                cca_prior_kernel=cak, eh=eh, total=tot)
        print(f"({a},{b}) {ccr:9.4f} {cah:10.4f} {cap:11.4f} {capo:10.4f} {cak:10.4f} {eh:8.4f} {tot:9.4f}")
    out["vertices"] = rows
    out["E_nat"] = {l: E(l) for l in STATE}
    if MODEL == "coulomb":
        # 2D hydrogen in solver units: E_b(1s) = 2 E0, lambda_2D = a_B/2 -> E_b lambda^2 = 0.5 E0 a_B^2
        u = 0.5
        v = rows["1s,1s"]
        print("\nHydrogenic benchmark in E_b lambda_2D^2:")
        print(f"  cc,r = {v['ccr']/u:.4f}  (expected -38.15)   cc,a = {v['cca_herm']/u:.4f}  (expected +50.27)"
              f"   total = {(v['ccr']+v['cca_herm'])/u:.4f}  (expected +12.11 = 2 x Ciuti 6.06)")
    vertex_file = (f"vertices_{MODEL}_{int(R_CUT)}_{N_R}_{NK_SEG}_{int(K_MAX)}"
                   + (f"_b{_NPS}x{_NSC}" if (_NPS, _NSC) != (6, 6) else "") + ".json")
    json.dump(out, open(vertex_file, "w"), indent=1)
    if MODEL == "keldysh":
        print("\n== exchange mean field (vertex file " + vertex_file + ")")
        propagate(vertex_file)
