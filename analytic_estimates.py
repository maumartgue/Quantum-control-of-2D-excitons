"""
analytic_estimates.py -- closed-form and semi-analytic estimates quoted in the paper "Dark-State
Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley Lindblad Framework for WSe2".

  ionization   bound-to-continuum (2p -> e-h continuum) ionization by the resonant THz/mid-IR tone
               of the Xi manifold (Sect. V item 7, Eqs. 56-57): S_d(E_f) and kappa_ion = 0.304.
  intra_rate   intra-excitonic radiative rates of the Xi manifold, fixed-K0 against free space
               (App. C 2, Eq. C22; Sect. V item 6): writes intra_rate_check.json.
  v_offdiag    first-order effect of the neglected off-diagonal exchange element Sigma_1s,2s on the
               V results of Sect. VI E (App. D 6, Sect. V A): writes v_offdiag_uncertainty.json.
  thermal      Delta -> 0 control parameter alpha_{e,h} Q_th a_B / <k>_1s a_B (App. D 6).
  window       cascade transparency window as a probe of the 0-2p dephasing (Sect. VI D).
  yield        conversion yield per incident photon, in the normalization of the paper (Sect. VI B).

ionization -- Golden rule with the RWA field amplitude E/2 (Eq. 40 convention), linear polarization
  e_d = x, retained real state |p_x> (Sect. IV C). Final states: m=0 and m=2 Keldysh continuum at
  E_f = hbar*w_L23 - E_b(2p) = E_b(1s) - 2 E_b(2p)  (Sect. V item 7).
     Gamma_ion = (pi/2) (hbar Omega_23)^2 S_x(E_f) / (hbar d^2),   d = |<1s|x|p_x>|,
     S_x(E) = sum_f |<f|x|p_x>|^2 delta(E_f - E) = 1/2 |R_0(E)|^2 + 1/4 |R_2(E)|^2,
     R_m(E) = int u_E^{(m)} u_2p r dr  with u_E energy-normalised: int u_E u_E' dr = delta(E-E').
  Natural units (a_B, E0) as App. D. Bound states from the independent shooting solver
  solver_indep.py. Continuum normalised by the WKB amplitude at large r and cross-checked by box
  discretisation (option --box; ~5-10 min).

intra_rate -- compares, for the retained 2p -> 1s channel and the omitted 2p <-> 2s channel:
  (1) the strict fixed-K0 rate, Eq. (C22a) of App. C,
        gamma_K0 = |d|^2 omega / (hbar eps0 A c) * [n(omega)+1]          (omega^1, A^-1)
  (2) the recoil-allowed free-space rate of the same dipole (vacuum, all photon
      directions and both polarizations, recoil energy neglected),
        gamma_fs = |d|^2 omega^3 / (3 pi eps0 hbar c^3) * [n(omega)+1]   (omega^3, A-independent)
  They coincide at A* = 3 lambda^2 / (4 pi).
  Inputs are the values of the paper: |d_1s,2p| = 0.854 a_B, |d_2s,2p| = 2.740 a_B (Table VIII),
  a_B = 1.19 nm, E_b(1s,2p,2s) = 161.59, 48.39, 37.43 meV (Table VII; transition energies 113.19
  and 10.97 meV as quoted in Sect. V item 6), A = 9.62 um^2 (Sect. VI A).

v_offdiag -- chain (App. D 6 / Sect. V A, first order in chi = |Sigma_1s,2s|/(eps_2s - eps_1s)):
    phi_2s(0) changes by  d2 = chi * r,   r = |phi_1s(0)/phi_2s(0)|    -> Omega_13 by d2
    phi_1s(0) changes by  d1 = chi / r                                  -> Omega_12 by d1
    interband rates ~ |phi(0)|^2: gamma_2s->0 by 2 d2, gamma_1s->0 by 2 d1
    Gamma^V_13 = gamma_2s->0 / 2                    -> 2 d2
    beat amplitude ~ Omega_12 Omega_13 (weak pulses) -> d1 + d2
    1/T2 = Gamma^V_23 = (gamma_1s + gamma_2s)/2      -> (g1 2 d1 + g2 2 d2)/(g1 + g2)
    density below which the largest of the three (Gamma^V_13) stays < 10 %: linear in n_X.
  Inputs: propagated.json of exchange_mean_field.py (chi at 1e11, Hermitian and in/out; r from the
  solver), rates 3.20559 and 0.360901 meV (params.yaml).

thermal -- Q_th = sqrt(2 M k_B T)/hbar, carrier masses of Jin et al. PRB 90, 045422 (2014),
  Table I, WSe2: m_e=0.39, m_h=0.51 (M=0.90); mu kept at 0.20 (Stier) for the Wannier problem.

window -- weak-probe steady state of the Xi ladder (Sect. VI D, normalized susceptibility
  chi = 2 Gamma12 rho12 / Omega12, two-level peak = 1). At one- and two-photon resonance:
      Im chi(0) = 1 / (1 + Omega23^2 / (4 Gamma12 Gamma13))
  which reduces to the paper's 4 Gamma12 Gamma13 / Omega23^2 for Gamma13 -> 0.
  Inverted:  Gamma13 = Omega23^2 * A / (4 Gamma12 (1 - A)),  A = residual absorption at the centre.
  Inputs (meV, all as hbar*rate):
    Gamma12 = 1.60   paper value (radiative 1s width 3.21 meV FWHM / 2)
    Gamma13 = 5.6e-7 paper value (radiative-only model)
    Gamma13 = 3.96   if the measured 2p width (Kusaba 2021: 2 gamma0 = 7.91 meV FWHM) were
                     homogeneous dephasing of the 0-2p coherence (HWHM)

yield -- calibration (weak probe, normal incidence from one side, radiative-only 1s vertex):
  With H_int = (hbar*Omega12/2)(|1><2| + h.c.) and coherence damping Gamma12 = gamma/2,
  a cw resonant probe gives rho22 = Omega12^2/gamma^2 and an emission rate gamma*rho22 = Omega12^2/gamma.
  A radiatively limited 2D sheet is a perfect reflector at resonance: it emits the incident photon
  flux forward (cancelling transmission) and the same flux backward, i.e. 2 x incident.
  Hence the incident photon rate per normalization area is Omega12(t)^2 / (2 gamma), and
      N_in = (1 / 2 gamma) * int dt Omega12(t)^2  = Omega0^2 sigma sqrt(pi) / (2 gamma)   (Gaussian, exp[-t^2/2sigma^2])
      Y    = rho33(t_f) / N_in        (fraction of incident photons per area A that end in 2p)
  Upper bound: a mirror-symmetric sheet driven from one side couples only to the symmetric mode,
  which carries half the incident energy, so Y <= 1/2.
  Inputs: paper Sect. VI B (case 2, baseline) and Sect. VI (pulse parameters).

Usage:  python analytic_estimates.py [ionization|intra_rate|v_offdiag|thermal|window|yield|all] [--box]
        (default all; --box adds the box-discretisation check of the ionization estimate)
"""
import json
import math
import sys

import numpy as np
from scipy.constants import hbar, epsilon_0 as eps0, c, e, k as kB, pi
from scipy.optimize import brentq

from solver_indep import E0, V, shoot, bound, bound_wf


def ionization(box=False):
    """2p ionization: S_d(E_f), hbar*Gamma_ion per (hbar*Omega23)^2, accumulated P_ion."""
    # bound states from the independent shooting solver (validated against Tables VII-VIII:
    # E_b(1s)=161.585, E_b(2p)=48.391 meV, |d_1s,2p|=0.8539 a_B)
    E1s=bound(0,-161.59/E0,24); E2p=bound(1,-48.39/E0,45,0.03)
    r1=np.linspace(1e-6,24,400001); u1=bound_wf(0,E1s,24,r1)
    r2=np.linspace(1e-6,45,400001); u2=bound_wf(1,E2p,45,r2)
    I=np.trapezoid(u1*np.interp(r1,r2,u2)*r1,r1); d=abs(I)/np.sqrt(2)
    print('E_b(1s)=%.3f  E_b(2p)=%.3f meV   d(1s,2p)=%.4f a_B'%(-E1s*E0,-E2p*E0,d))
    Ef=2*E2p-E1s   # = E_2p + hbar w23 ,  hbar w23 = E_2p - E_1s  (positive kinetic energy)
    print('E_f = %.4f E0 = %.2f meV above threshold'%(Ef,Ef*E0))

    def cont_energy_norm(m,E,R=500.0):
        sol=shoot(E,m,R,dense=True)
        rr=np.linspace(R-150,R,6001); y=sol.sol(rr)
        c=(m*m-0.25)/2.0; kl=np.sqrt(2*(E-V(rr)-c/rr**2))
        C2=kl*(y[0]**2+(y[1]/kl)**2)
        scale=np.sqrt(2/np.pi)/np.sqrt(C2.mean())
        return sol, scale, C2.std()/C2.mean()
    def Rm(m,E):
        sol,sc,spread=cont_energy_norm(m,E)
        uE=sol.sol(r2)[0]*sc
        return np.trapezoid(uE*u2*r2,r2), spread
    def S_x(E):
        R0,s0=Rm(0,E); R2,s2=Rm(2,E)
        return 0.5*R0**2+0.25*R2**2, R0, R2, max(s0,s2)
    S,R0,R2,spr=S_x(Ef)
    print('R0=%.5f R2=%.5f  (energy-normalised, a_B*E0^-1/2); WKB-amplitude spread %.1e'%(R0,R2,spr))
    print('S_x(E_f) = %.5f a_B^2/E0 ; d=%.4f a_B'%(S,d))

    # ---- box cross-check: states with u(L)=0 near Ef; S_box = |<f|x|p>|^2 / dE
    def box_states(m,L,Ec,n=3):
        f=lambda E: shoot(E,m,L).y[0,-1]
        Es=np.linspace(Ec-0.02,Ec+0.02,24); vals=[f(e) for e in Es]; roots=[]
        for i in range(len(Es)-1):
            if np.sign(vals[i])!=np.sign(vals[i+1]): roots.append(brentq(f,Es[i],Es[i+1],xtol=1e-14))
        return np.array(roots)
    for L in ([300.0,600.0] if box else []):
        tot=0.0
        for m,w in [(0,0.5),(2,0.25)]:
            Es=box_states(m,L,Ef); j=np.argmin(abs(Es-Ef)); dE=(Es[j+1]-Es[j-1])/2
            sol=shoot(Es[j],m,L,dense=True); rr=np.linspace(1e-6,L,400001); u=sol.sol(rr)[0]
            u/=np.sqrt(np.trapezoid(u*u,rr)); Rb=np.trapezoid(u*np.interp(rr,r2,u2,right=0)*rr,rr)
            tot+=w*Rb**2/dE
        print('box L=%4.0f : S_x ~ %.5f'%(L,tot))

    # ---- rate per unit Rabi energy
    K=(np.pi/2)*S/d**2/E0     # hbar*Gamma_ion [meV] = K * (hbar*Omega23 [meV])^2
    print('hbar Gamma_ion = %.4e meV^-1 x (hbar Omega_23)^2'%K)
    hb=0.6582119569
    for W in [0.5,1.0,1.6,3.0,5.0]:
        g=K*W**2; print('  hbarOmega23=%4.1f meV: hbarGamma_ion=%.4f meV, Gamma_ion=%.4f ps^-1, 1/Gamma=%.1f ps, Gamma/gamma_1s=%.4f'%(W,g,g/hb,hb/g,g/3.21))

    # ---- adiabatic-following estimate of the accumulated ionization in the counterintuitive sequence
    # rho_33 = sin^2 theta, tan theta = Omega12/Omega23 ; equal widths sigma, equal peaks Omega0, delay x*sigma
    from scipy.integrate import quad
    def P_ion(W0_meV, sigma_ps, x):
        g=lambda s: (np.exp(-s**2)*np.exp(-(s-x)**2))/(np.exp(-s**2)+np.exp(-(s-x)**2))
        J=quad(g,-12,12+x)[0]
        return K*W0_meV**2/hb*sigma_ps*J
    print('\nAccumulated ionization probability (adiabatic following), hbarOmega0 = 5 meV:')
    for x in [1.0,1.25,1.5]:
        for s in [1.1,1.5,1.9]:
            print('   x=%.2f sigma=%.1f ps : P_ion ~ %.3f'%(x,s,P_ion(5.0,s,x)))
    print('   at hbarOmega0 = 3 meV, x=1.25, sigma=1.5 ps : %.4f'%P_ion(3.0,1.5,1.25))


def intra_rate():
    """Intra-excitonic radiative rates, fixed K0 against free space; writes intra_rate_check.json."""
    aB = 1.19e-9
    meV = 1e-3 * e
    A = 9.62e-12                       # m^2
    d12 = 0.854 * aB * e               # |d_1s,2p|  (C m)
    d23 = 2.740 * aB * e               # |d_2s,2p|
    w_ret = 113.19 * meV / hbar        # 2p -> 1s
    w_om = 10.97 * meV / hbar          # 2s <-> 2p


    def nB(w, T):
        return 0.0 if T == 0 else 1.0 / np.expm1(hbar * w / (kB * T))


    def g_K0(d, w, A):                 # Eq. (C22a), zero-temperature part
        return d ** 2 * w / (hbar * eps0 * A * c)


    def g_fs(d, w):                    # free-space dipole rate
        return d ** 2 * w ** 3 / (3 * pi * eps0 * hbar * c ** 3)


    to_neV = lambda g: hbar * g / e * 1e9
    out = {}
    gK, gF = g_K0(d12, w_ret, A), g_fs(d12, w_ret)
    lam = 2 * pi * c / w_ret
    Astar = 3 * lam ** 2 / (4 * pi)
    print("Retained 2p -> 1s channel, T = 0:")
    print(f"  fixed-K0 (A = 9.62 um^2): hbar*gamma = {to_neV(gK):.3f} neV,  1/gamma = {1e6/gK:.3f} us,"
          f"  coherence hbar/(gamma/2) = {2e6/gK:.2f} us")
    print(f"  free space (omega^3):     hbar*gamma = {to_neV(gF):.3f} neV,  1/gamma = {1e6/gF:.3f} us,"
          f"  coherence = {2e6/gF:.2f} us")
    print(f"  ratio fixed-K0 / free space = {gK/gF:.2f};  lambda = {lam*1e6:.2f} um;"
          f"  rates coincide at A* = 3 lambda^2/4pi = {Astar*1e12:.1f} um^2")
    out.update(gK_neV=to_neV(gK), gF_neV=to_neV(gF), ratio=gK / gF, lam_um=lam * 1e6, Astar_um2=Astar * 1e12)

    r1 = (d23 ** 2 * w_om) / (d12 ** 2 * w_ret)
    r3 = (d23 ** 2 * w_om ** 3) / (d12 ** 2 * w_ret ** 3)
    print(f"\nOmitted/retained spectral-density ratio: dipoles {d23**2/d12**2:.2f}, "
          f"frequencies {w_om/w_ret:.4f}")
    print(f"  fixed-K0 (omega^1): {r1:.3f}      free space (omega^3): {r3:.2e}")
    out.update(r1=r1, r3=r3)
    rows = []
    print(f"\n{'T (K)':>6s} {'n(10.97)':>10s} {'ratio fixed-K0':>15s} {'ratio free space':>17s}")
    for T in (4, 77, 150, 300):
        th = nB(w_om, T) / (1 + nB(w_ret, T))
        rows.append(dict(T=T, n_om=nB(w_om, T), ratio_K0=r1 * th, ratio_fs=r3 * th))
        print(f"{T:6d} {nB(w_om,T):10.3e} {r1*th:15.3e} {r3*th:17.3e}")
    out["thermal"] = rows
    json.dump(out, open("intra_rate_check.json", "w"), indent=1)


def v_offdiag(propagated="propagated.json"):
    """Effect of the neglected Sigma_1s,2s on the V results; writes v_offdiag_uncertainty.json."""
    P = json.load(open(propagated))
    r = P["admix"]["ratio"]
    g1, g2 = 3.20559, 0.360901


    def effects(chi):
        d2, d1 = chi * r, chi / r
        e = dict(Omega13=d2, gamma2s=2 * d2, Omega12=d1, gamma1s=2 * d1, Gamma13=2 * d2,
                 beat=d1 + d2, T2=(g1 * 2 * d1 + g2 * 2 * d2) / (g1 + g2))
        e["n_10pc"] = 1e11 * 0.10 / max(e["Gamma13"], e["beat"], e["T2"])
        return e


    out = {"ratio_phi1s_phi2s": r}
    for tag, chi in (("new chi, Hermitian", P["V_herm"]["chi"]),
                     ("new chi, prior (in)", P["V_prior"]["chi"]), ("new chi, post (out)", P["V_post"]["chi"])):
        e = effects(chi); out[tag] = dict(chi=chi, **e)
        print(f"{tag:22s} chi={chi:.4f}: gamma_2s {100*e['gamma2s']:.1f}%  Omega_13 {100*e['Omega13']:.1f}%  "
              f"Gamma^V_13 {100*e['Gamma13']:.1f}%  beat {100*e['beat']:.1f}%  T2 {100*e['T2']:.1f}%  "
              f"all < 10% for n_X < {e['n_10pc']:.2e}")
    json.dump(out, open("v_offdiag_uncertainty.json", "w"), indent=1)


def thermal():
    """Delta -> 0 control parameter at 4, 77, 150, 300 K."""
    hbar=1.054571817e-34; m0=9.1093837015e-31; kB=1.380649e-23
    aB=1.19e-9; k1s=0.6458   # <k>_1s a_B from the Wannier envelope (independently re-evaluated)
    def row(me,mh,T):
        M=me+mh; Q=np.sqrt(2*M*m0*kB*T)/hbar*aB
        return Q, me/M*Q/k1s, mh/M*Q/k1s
    for lab,me,mh in [('Jin 2014 (0.39,0.51)',0.39,0.51)]:
        print(lab, ' M=%.2f alpha_e=%.4f alpha_h=%.4f'%(me+mh,me/(me+mh),mh/(me+mh)))
        for T in [4,77,150,300]:
            Q,re,rh=row(me,mh,T); print('   T=%3d K  Q_th aB=%.3f  alpha_e ratio=%.3f  alpha_h ratio=%.3f'%(T,Q,re,rh))


def window():
    """Residual absorption at the centre of the cascade transparency window."""
    G12 = 1.60
    cases = {"model (radiative only)": 5.6e-7, "measured 2p width, if homogeneous": 7.91 / 2}

    def A_centre(W23, G13, G12=G12):
        return 1.0 / (1.0 + W23**2 / (4 * G12 * G13))

    def G13_from_A(A, W23, G12=G12):
        return W23**2 * A / (4 * G12 * (1 - A))

    print("Residual absorption at the window centre, Im chi(0)  [two-level peak = 1]")
    print(f"{'hbar*Omega23 (meV)':>20s}" + "".join(f"{k:>38s}" for k in cases))
    for W in (0.5, 1.0, 2.0, 3.0, 5.0):
        print(f"{W:20.1f}" + "".join(f"{A_centre(W, g):38.3g}" for g in cases.values()))

    print("\nControl needed for a window of depth 1 - A (A = residual absorption), Gamma13 = 3.96 meV:")
    for A in (0.5, 0.2, 0.1):
        W = (4 * G12 * cases["measured 2p width, if homogeneous"] * (1 - A) / A) ** 0.5
        print(f"  A = {A:.1f}:  hbar*Omega23 = {W:.1f} meV")

    print("\nInversion check: A = 0.5 at hbar*Omega23 = 5 meV  ->  hbar*Gamma13 =",
          f"{G13_from_A(0.5, 5.0):.2f} meV")
    print("Gamma13 that halves the window at the baseline control (5 meV):",
          f"{5.0**2 / (4 * G12):.2f} meV")


def yield_per_photon():
    """Conversion yield per incident photon."""
    HBAR = 0.6582119569          # meV ps
    gamma_meV = 3.21             # radiative 1s linewidth (FWHM) = hbar*gamma_{1s->0}
    Omega23_meV = 5.0
    Omega12_meV = 0.05 * min(Omega23_meV, gamma_meV / 2)   # paper's weak-probe rule -> 0.080 meV
    sigma_ps = 1.5
    rho33 = 1.34e-3              # case 2, baseline (Sect. VI B)
    N_emit = 1.46e-3             # case 2, baseline (Sect. VI B)

    gamma = gamma_meV / HBAR     # 1/ps
    Om0 = Omega12_meV / HBAR     # rad/ps
    N_in = Om0**2 * sigma_ps * math.sqrt(math.pi) / (2 * gamma)
    Y = rho33 / N_in
    eta = rho33 / (rho33 + N_emit)
    print(f"hbar*Omega12 = {Omega12_meV:.3f} meV, sigma = {sigma_ps} ps, hbar*gamma = {gamma_meV} meV")
    print(f"incident photons per normalization area  N_in = {N_in:.4e}")
    print(f"eta (paper)                     = {eta:.3f}")
    print(f"yield per incident photon  Y    = {Y:.3f}   (bound 1/2; Y/Y_max = {Y/0.5:.2f})")
    print(f"re-emitted / incident           = {N_emit/N_in:.3f}")
    print("Finite-bandwidth check: with no THz pulse, emitted/incident = 2 x (spectral overlap) < 2.")


PARTS = {"ionization": ionization, "intra_rate": intra_rate, "v_offdiag": v_offdiag,
         "thermal": thermal, "window": window, "yield": yield_per_photon}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--box"]
    which = args[0] if args else "all"
    for name, fn in PARTS.items():
        if which in (name, "all"):
            print(f"\n######## {name}")
            fn(box="--box" in sys.argv) if name == "ionization" else fn()
