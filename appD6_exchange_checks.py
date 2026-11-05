"""
appD6_exchange_checks.py -- numerical checks of the exciton-gas exchange vertices of App. D 6 of
the paper "Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley
Lindblad Framework for WSe2".

Checks (i)-(iii) of App. D 6 (unitarity of the Hankel transforms, Sigma_ab = Sigma_ba, vanishing of
Sigma_1s,2p) are printed by exchange_mean_field.py itself, and check (iv) is its convergence run with
the quadrature arguments. This script adds:

  (a) wannier   Wannier identity used for zeta^{cc,a}, Eqs. (D9), (A14), weighted by the combination
                entering the diagonal vertex (WSe2/hBN, Keldysh): for each state,
                INT k dk R_1s(k)^2 R_a(k) [D_a(k) - W_a(k)] against the same integral of W_a.
  (b) sign      sign of the zero-transfer exchange vertex for the 2D hydrogenic 1s exciton,
                units lambda_2D = 1, E_b = e^2/(eps lambda) = 1, V(q) = 2 pi/q:
                like-carrier lines only (-4S) against all four Coulomb lines (-4S+4T), Eqs. (A11)-(A13).
  (c) offdiag   Wannier-equation form of the electron-hole lines of the carrier exchange and the
                prior/post (off-shell) asymmetry of the off-diagonal element; 2D hydrogen, gas in 1s.
  (d) hydro     check (v): closed forms S = 315 pi^3/1024, T = 4 pi, vertex 8 pi - 315 pi^3/512 =
                6.056614 E0 a_B^2, against independent quadrature, against the solver run
                vertices_coulomb_200_8000_250_100.json (python exchange_mean_field.py coulomb) and
                against the q = 0 value of Fig. 2 and Eq. (22) of Ciuti et al., PRB 58, 7926 (1998).
  (e) lattice   brute-force check of the general-Delta exchange vertex (like-carrier and electron-hole
                lines) on a 1D periodic lattice: the real-space four-carrier matrix element is summed
                directly and compared with the momentum-space formulas, Eqs. (A10)-(A12).

Units (d): paper units E0 (= e^2/(eps a_B)), a_B. 2D hydrogen: E_b(1s) = 2 E0, lambda_2D = a_B/2,
  so E_b lambda_2D^2 = E0 a_B^2 / 2. S and T are computed with lambda_2D = 1, E_b(1s) = 1, V(q) = 2 pi/q:
    S = int d2k d2k'/(2pi)^4 V(k-k') phi(k)^2 phi(k')^2          (one like-carrier line)
    T = int d2k d2k'/(2pi)^4 V(k-k') phi(k)^3 phi(k')            (one electron-hole line)
  Both carriers exchanged, four lines each:  zeta^{cc,r} = -4S, zeta^{cc,a} = +4T  [E_b lambda^2]
  Per exchanged carrier species (Ciuti's H_e^exch, their Eq. (22)): 2(T - S)          [E_b lambda^2]
  In paper units the full vertex is (4T - 4S) * (E_b lambda^2)/(E0 a_B^2) = 2(T - S)  [E0 a_B^2]
  i.e. numerically the same number.
  Closed forms: phi(k)^2 = 8 pi /(1+k^2)^3 -> g(r) = FT[phi^2](r) = r^2 K_2(r)/2,
  S = 2 pi int_0^inf g(r)^2 dr = (pi/2) int r^4 K_2(r)^2 dr = 315 pi^3 / 1024, T = 4 pi.

Usage:  python appD6_exchange_checks.py [wannier|sign|offdiag|hydro|lattice|all] [--json PATH]
        (default all; --json: solver run for (d), default vertices_coulomb_200_8000_250_100.json)
"""
import json
import math
import os
import sys

import numpy as np
from scipy.special import ellipk, kv
from scipy.integrate import quad

PI = math.pi


def check_wannier():
    """(a) Wannier identity for zeta^{cc,a}, WSe2/hBN (setup of exchange_mean_field.py, keldysh)."""
    argv = sys.argv; sys.argv = [argv[0], "keldysh"]
    try:
        import exchange_mean_field as X
    finally:
        sys.argv = argv
    for l in ['1s', '2s', '2p', '3p']:
        g = X.R['1s']**2 * X.R[l]
        num = np.sum(X.kdk * g * (X.D[l] - X.W[l])); den = np.sum(X.kdk * g * X.W[l])
        lo = X.kg < 5
        rl = np.sqrt(np.sum((X.kdk*(X.D[l]-X.W[l])**2)[lo]) / np.sum((X.kdk*X.W[l]**2)[lo]))
        print(f"{l}: weighted rel. diff = {num/den:.2e}   L2 residual on k<5 a_B^-1 = {rl:.1e}")


def check_sign():
    """(b) Sign of the zero-transfer exchange vertex, 2D hydrogenic 1s exciton."""

    phi = lambda k: 2*np.sqrt(2*np.pi)/(1+k*k)**1.5        # int d^2k/(2pi)^2 |phi|^2 = 1
    print("norm:", quad(lambda k: k*phi(k)**2/(2*np.pi), 0, np.inf)[0])

    def ang(k, kp):
        # int_0^{2pi} dth V(|k-k'|) = 2pi * 4K(m)/(k+k'),  m = 4kk'/(k+k')^2 ; log-singular at k = k'
        m = 4*k*kp/(k+kp)**2
        return 2*np.pi*4*ellipk(min(m, 1-1e-16))/(k+kp)

    def I(fk, fkp):
        # int d^2k d^2k'/(2pi)^4 V(k-k') fk(k) fkp(k')  (both radial)
        def inner(k):
            g = lambda kp: kp*ang(k, kp)*fkp(kp)
            a = quad(g, 0, k, limit=400)[0]; b = quad(g, k, 2*k+5, limit=400)[0]; c = quad(g, 2*k+5, np.inf, limit=400)[0]
            return (a+b+c)
        return quad(lambda k: k*fk(k)*inner(k), 0, np.inf, limit=400)[0]*(2*np.pi)/(2*np.pi)**4

    S = I(lambda k: phi(k)**2, lambda k: phi(k)**2)    # e-e or h-h line (self-paired = cross-paired for 1s-1s)
    T = I(lambda k: phi(k)**3, lambda k: phi(k))       # e-h attraction line of the exchanged pair
    print(f"S = {S:.4f}   T = {T:.4f}   (units E_b lambda^2)")
    print(f"like-carrier lines only (Eq. A11 at zero transfer):      -4S     = {-4*S:+.4f}")
    print(f"all four Coulomb lines (Eqs. A11-A13):                    -4S+4T = {4*(T-S):+.4f}")


def check_offdiag():
    """(c) Electron-hole lines in Wannier form and prior/post asymmetry, 2D hydrogen."""

    phi1 = lambda k: 2*np.sqrt(2*np.pi)/(1+k*k)**1.5                     # exact 1s, E = -1
    a_ = 1/3.
    _u = lambda k: 2*np.pi*a_/(a_*a_+k*k)**1.5 - (2/3.)*2*np.pi*(2*a_*a_-k*k)/(a_*a_+k*k)**2.5  # FT of (1-2r/3)e^{-r/3}
    _n = quad(lambda k: k*_u(k)**2/(2*np.pi), 0, np.inf, limit=400)[0]
    phi2 = lambda k: _u(k)/np.sqrt(_n)                                    # exact 2s, E = -1/9
    E1, E2 = -1.0, -1/9.

    print("norm1 %.8f  norm2 %.8f  <1s|2s> %.1e" % (
        quad(lambda k: k*phi1(k)**2/(2*np.pi), 0, np.inf)[0],
        quad(lambda k: k*phi2(k)**2/(2*np.pi), 0, np.inf, limit=400)[0],
        quad(lambda k: k*phi1(k)*phi2(k)/(2*np.pi), 0, np.inf, limit=400)[0]))

    def ang(k, kp):
        m = 4*k*kp/(k+kp)**2
        return 2*np.pi*4*ellipk(min(m, 1-1e-16))/(k+kp)

    def I(fk, fkp):
        """sum_{k,k'} V(k-k') fk(k) fkp(k')  for radial fk, fkp."""
        def inner(k):
            g = lambda kp: kp*ang(k, kp)*fkp(kp)
            return quad(g, 0, k, limit=200)[0] + quad(g, k, 2*k+3, limit=200)[0] + quad(g, 2*k+3, np.inf, limit=200)[0]
        return quad(lambda k: k*fk(k)*inner(k), 0, np.inf, limit=200)[0]*(2*np.pi)/(2*np.pi)**4

    def S1(f):
        return quad(lambda k: k*f(k)/(2*np.pi), 0, np.inf, limit=400)[0]

    P = {"1s": (phi1, E1), "2s": (phi2, E2)}
    for a, b in (("1s", "1s"), ("1s", "2s"), ("2s", "1s"), ("2s", "2s")):
        pa, Ea = P[a]; pb, Eb = P[b]
        # cv lines, gas l = l' = 1s, outgoing tagged n' = a, incoming n = b
        T1 = I(lambda k: pa(k)*pb(k)*phi1(k), phi1)        # split envelope: outgoing gas 1s
        T2 = I(lambda k: pb(k)*phi1(k)**2, pa)             # split envelope: outgoing tagged n'
        prior = S1(lambda k: pa(k)*pb(k)*phi1(k)**2*(2*k*k - Ea - E1))
        post  = S1(lambda k: pa(k)*pb(k)*phi1(k)**2*(2*k*k - Eb - E1))
        # cc lines for the same element
        Sself = I(lambda k: pa(k)*pb(k), lambda k: phi1(k)**2)
        Scros = I(lambda k: pa(k)*phi1(k), lambda k: phi1(k)*pb(k))
        full = -2*(Sself+Scros) + 2*(T1+T2)
        print(f"(n'={a}, n={b}): T1+T2 kernel = {T1+T2:8.4f}  single-k prior = {prior:8.4f}  post = {post:8.4f}"
              f" | cc lines -2(S+C) = {-2*(Sself+Scros):8.4f}  full = {full:8.4f}")


def check_hydro(json_path="vertices_coulomb_200_8000_250_100.json"):
    """(d) Hydrogenic benchmark: closed forms, quadrature, solver run and Ciuti et al."""
    PI = math.pi

    # ---------- 1. closed forms ----------
    S_closed = 315 * PI**3 / 1024
    T_closed = 4 * PI
    zeta_closed = 8 * PI - 315 * PI**3 / 512          # = 2(T - S)

    # ---------- 2. independent numerics ----------
    # (a) S in real space via the Bessel form
    S_bessel = (PI / 2) * quad(lambda r: r**4 * kv(2, r)**2, 0, np.inf, limit=400, epsabs=1e-13)[0]

    # (b) S and T in momentum space with the angular integral done by an elliptic K (as in check (b))
    phi = lambda k: 2 * np.sqrt(2 * PI) / (1 + k * k) ** 1.5      # int d2k/(2pi)^2 phi^2 = 1

    def ang(k, kp):
        m = 4 * k * kp / (k + kp) ** 2
        return 2 * PI * 4 * ellipk(min(m, 1 - 1e-16)) / (k + kp)

    def I(fk, fkp):
        def inner(k):
            g = lambda kp: kp * ang(k, kp) * fkp(kp)
            return (quad(g, 0, k, limit=400)[0] + quad(g, k, 2 * k + 5, limit=400)[0]
                    + quad(g, 2 * k + 5, np.inf, limit=400)[0])
        return quad(lambda k: k * fk(k) * inner(k), 0, np.inf, limit=400)[0] * (2 * PI) / (2 * PI) ** 4

    S_k = I(lambda k: phi(k) ** 2, lambda k: phi(k) ** 2)
    T_k = I(lambda k: phi(k) ** 3, lambda k: phi(k))

    print("=== Closed forms vs independent quadrature (units E_b lambda^2) ===")
    print(f"S : closed {S_closed:.8f} | Bessel quad {S_bessel:.8f} | k-space quad {S_k:.6f}")
    print(f"T : closed {T_closed:.8f} |                       | k-space quad {T_k:.6f}")
    print(f"like-carrier, both species  -4S = {-4*S_closed:+.4f} E_b lam^2 = {-2*S_closed:+.5f} E0 aB^2")
    print(f"e-h lines,    both species  +4T = {+4*T_closed:+.4f} E_b lam^2 = {+2*T_closed:+.5f} E0 aB^2")
    print(f"total vertex zeta_1s1s = 8pi - 315pi^3/512 = {zeta_closed:.6f} E0 aB^2 "
          f"= {2*zeta_closed:.4f} E_b lam^2 = 2 x {zeta_closed:.4f} (per carrier)")

    # ---------- 3. the paper's solver (run of exchange_mean_field.py coulomb) ----------

    solver = {"ccr": -19.0761597295266, "cca_herm": 25.132741112535737, "total": 6.056581383009135}
    if os.path.exists(json_path):
        solver = json.load(open(json_path))["vertices"]["1s,1s"]
        print(f"\n(read solver values from {json_path})")
    else:
        print("\n(solver JSON not found; using the values recorded from vertices_coulomb_200_8000_250_100.json)")
    print("=== Solver (Wannier code, Coulomb kernel) vs closed form, E0 aB^2 ===")
    for key, ref in (("ccr", -2 * S_closed), ("cca_herm", 2 * T_closed), ("total", zeta_closed)):
        v = solver[key]
        print(f"{key:9s} solver {v:+.6f}  closed {ref:+.6f}  rel. diff {abs(v-ref)/abs(ref):.1e}")

    # ---------- 4. Ciuti et al. 1998 ----------
    # Eq. (22): H_e^exch(0,0) = -(1/A)(e^2/eps) lambda_2D (2/pi)^2 I_exch = (E_b lambda^2 / A)(4/pi^2)(-I_exch)
    # -I_exch(q=0) is only given graphically (Fig. 2, Monte Carlo); value read off that figure below.
    I0, dI = 14.66, 0.06          # read off Fig. 2 of Ciuti et al. (300 dpi; curve centre row 100.5, frame 88.5, 35.13 px/unit)
    print(f"\nFig. 2 read-off: -I_exch(q=0) = {I0:.2f} +/- {dI:.2f}")
    ciuti = 4 / PI**2 * I0
    print("=== Per exchanged carrier, E_b lambda_2D^2 ===")
    print(f"this work (solver)   {solver['total']:.4f}")
    print(f"closed form          {zeta_closed:.4f}")
    print(f"Ciuti Fig.2 + Eq.22  (4/pi^2) x {I0:.2f} = {ciuti:.3f}  (range {4/PI**2*(I0-dI):.3f}-{4/PI**2*(I0+dI):.3f})")
    print(f"difference (solver - Ciuti)/Ciuti = {100*(solver['total']-ciuti)/ciuti:+.1f} %")
    print(f"Ciuti's q=0 value implied by the closed form: -I_exch = {zeta_closed*PI**2/4:.2f}")


def check_lattice():
    """(e) Brute-force lattice check of the general-Delta exchange vertex."""
    rng = np.random.default_rng(7)
    L = 12                                   # lattice sites (momenta are integers mod L, units 2pi/L)
    ae, ah = 1/3, 2/3                        # mass fractions
    K0, Q = 3, 0                             # CM momenta (units 2pi/L); Delta = 3 -> ae*D = 1, ah*D = 2
    D = K0 - Q
    ks = np.arange(L)
    x = np.arange(L)

    def env():
        v = rng.normal(size=L) + 1j*rng.normal(size=L)
        return v/np.linalg.norm(v)
    phi = {s: env() for s in ("n", "l", "np", "lp")}
    Vq = rng.normal(size=L); Vq = 0.5*(Vq + Vq[(-ks) % L])          # even kernel V_q = V_{-q}
    V = lambda q: Vq[int(round(q)) % L]
    P = lambda s, k: phi[s][int(round(k)) % L]

    def Psi(s, K):
        # Psi(e,h) = (1/L) sum_k phi(k) exp(i(k+ae K)e) exp(i(-k+ah K)h)
        E = np.exp(2j*np.pi/L*np.outer(ks + ae*K, x))      # [k, e]
        H = np.exp(2j*np.pi/L*np.outer(-ks + ah*K, x))     # [k, h]
        return np.einsum('k,ke,kh->eh', phi[s], E, H)/L

    v = np.array([sum(Vq[q]*np.exp(2j*np.pi*q*d/L) for q in range(L)) for d in range(L)])  # v(d)
    vx = lambda a, b: v[(a - b) % L]

    Pn, Pl, Pnp, Plp = Psi("n", K0), Psi("l", Q), Psi("np", K0), Psi("lp", Q)
    e1, h1, e2, h2 = np.meshgrid(x, x, x, x, indexing="ij")
    ket = Pn[e1, h1]*Pl[e2, h2]
    W_ee, W_hh = vx(e1, e2), vx(h1, h2)
    W_eh1, W_eh2 = -vx(e1, h2), -vx(e2, h1)

    def X(bra, W):  # fermionic exchange sign included
        return -np.sum(np.conj(bra)*W*ket)

    braE = Pnp[e2, h1]*Plp[e1, h2]          # electrons exchanged
    braH = Pnp[e1, h2]*Plp[e2, h1]          # holes exchanged
    bf = {"e: ee": X(braE, W_ee), "e: hh": X(braE, W_hh), "e: eh": X(braE, W_eh1 + W_eh2),
          "h: ee": X(braH, W_ee), "h: hh": X(braH, W_hh), "h: eh": X(braH, W_eh1 + W_eh2)}

    c = np.conj
    S = lambda f: sum(f(k, kp) for k in ks for kp in ks)
    fm = {
     "e: ee": -S(lambda k, kp: V(k-kp+ae*D)*c(P("np",k))*P("n",k)*c(P("lp",kp))*P("l",kp)),
     "e: hh": -S(lambda k, kp: V(k-kp+ae*D)*c(P("np",kp-ae*D))*P("l",kp)*c(P("lp",k+ae*D))*P("n",k)),
     "e: eh": +S(lambda k, kp: V(k-kp)*(c(P("np",k-ae*D))*P("n",k-ae*D)*P("l",k)*c(P("lp",kp))
                                      + c(P("np",kp))*P("n",k)*c(P("lp",k+ae*D))*P("l",k+ae*D))),
     "h: ee": -S(lambda k, kp: V(k-kp-ah*D)*c(P("np",kp+ah*D))*P("l",kp)*c(P("lp",k-ah*D))*P("n",k)),
     "h: hh": -S(lambda k, kp: V(k-kp-ah*D)*c(P("np",k))*P("n",k)*c(P("lp",kp))*P("l",kp)),
     "h: eh": +S(lambda k, kp: V(k-kp)*(c(P("np",k+ah*D))*P("n",k+ah*D)*P("l",k)*c(P("lp",kp))
                                      + c(P("np",kp))*P("n",k)*c(P("lp",k-ah*D))*P("l",k-ah*D))),
    }
    ok = True
    for key in bf:
        d = abs(bf[key]-fm[key])
        ok &= d < 1e-10
        print(f"{key}:  brute force {bf[key]: .6e}   formula {fm[key]: .6e}   |diff| {d:.1e}")
    print("ALL TERMS MATCH" if ok else "MISMATCH")


CHECKS = {"wannier": check_wannier, "sign": check_sign, "offdiag": check_offdiag,
          "hydro": check_hydro, "lattice": check_lattice}

if __name__ == "__main__":
    args = sys.argv[1:]
    json_path = "vertices_coulomb_200_8000_250_100.json"
    if "--json" in args:
        i = args.index("--json"); json_path = args[i + 1]; del args[i:i + 2]
    which = args[0] if args else "all"
    for name, fn in CHECKS.items():
        if which in (name, "all"):
            print(f"\n######## check ({'abcde'[list(CHECKS).index(name)]}) {name}")
            fn(json_path) if name == "hydro" else fn()
