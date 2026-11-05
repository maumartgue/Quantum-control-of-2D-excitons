"""
===============================================================================
 lindblad_solver.py
 Rotating-frame three-level Lindblad solver for the Xi and V excitonic
 manifolds of hBN-encapsulated WSe2 (paper "Dark-State Coherent Control of
 Excitons in TMD Monolayers: A Reduced Intravalley Lindblad Framework for WSe2")
===============================================================================

MASTER EQUATION
---------------
The master equation of the paper, Eqs. (63)-(68) for Xi and (70)-(75) for V, in
the one-exciton Fock space of Sect. III:

    d rho/dt = -(i/hbar) [H(t), rho] + sum_mu gamma_mu D[L_mu] rho,
    D[L] rho = L rho L^+ - 1/2 {L^+ L, rho}

    H(t) = sum_a h_a |a><a| + sum_legs (hbar Omega_l(t)/2) (e^{i phi_l}|j><i| + h.c.)

    Xi: |1> = 0, |2> = 1s, |3> = 2p;  legs (1,2) optical, (2,3) THz;
        L = |1><2| (gamma_1s->0), |2><1| (gamma_0->1s), |2><3| (gamma_2p->1s), |3><2| (gamma_1s->2p)
    V : |1> = 0, |2> = 1s, |3> = 2s;  legs (1,2), (1,3) optical;
        L = |1><2|, |2><1|, |1><3|, |3><1| (interband rates)

  * Topology (states, driven legs, photon jump operators) is data (XI, V below).
  * Photon-only dissipator, Eq. (53). Rates are read from params.yaml at the
    chosen photon temperature T.
  * The exciton-gas exchange mean field enters ONLY as diagonal level shifts
    Sigma_a (Eqs. 58, 62), added to the frame energies h_a. With lasers given by
    their bare detunings this reproduces the renormalized detunings of
    Eqs. (65) and (72); with Sigma switched off the frame energies ARE the
    renormalized detunings.
  * Frame energies (Eqs. 64, 71): h_1 = 0,
        Xi : h_2 = Delta~12,  h_3 = delta~Xi = Delta~12 + Delta~23
        V  : h_2 = Delta~12,  h_3 = Delta~13,  delta~V = Delta~13 - Delta~12
    In the signed-frequency convention of the paper, hbar*Delta~ = (renormalized
    resonance energy) - (photon energy); params.yaml 'calibration' uses the same.

NUMERICS (App. E)
-----------------
  * Units: energies and hbar*Omega in meV, times in ps, rates as hbar*gamma in meV;
    HB = hbar in meV ps.
  * params.yaml stores every photon rate twice, each to 6 significant digits:
    'hbar_gamma_meV' (hbar*gamma in meV) and 'dimensionless_gamma_tau0' (gamma*hbar/E0).
    The two agree to ~1e-6 relative (rounding). The rate block is chosen per run
    (Params.rates(T, block), Generator(..., rate_block=...)):
        RATES_MEV  = 'hbar_gamma_meV'           : conversion runs (Sect. VI B-C) [default]
        RATES_TAU0 = 'dimensionless_gamma_tau0' : response, EIT and V runs (Sect. VI D-E,
                                                  App. E), converted as gamma*tau0 x E0
  * Generator L(t) = L0 + sum_l Omega_l(t) K_l, Eq. (E1); vectorization by column
    stacking, vec(A X B) = (B^T kron A) vec(X). Basis index 0,1,2 <-> |1>,|2>,|3>.
  * One time propagator, integrate(): adaptive explicit Runge-Kutta DOP853, piecewise
    between envelope breakpoints with a capped step min(sigma, tau)/20 (so no pulse
    can be stepped over from the zero-RHS initial state), dense output every 3 fs.
    Implicit methods (Radau, BDF, LSODA) are available for cross-checks.
  * Running integrals are carried in the state vector: N_emit = int gamma_1s->0 rho22 dt,
    the 2s photon flow int gamma_2s->0 rho33 dt (V), int rho22 dt, and the 2p ionization
    P_ion = int Gamma_ion(t) rho33 dt (Eq. 57, first order; the generator is not modified).
  * Steady state from the null space of the Liouvillian (least squares on the system
    augmented with the trace condition).
  * Observables: dressed populations of Xi (dark state, Eq. 69), Bloch vector of the
    outer pair, lab-frame V coherence (inverse of the rotation of Eq. 9).
  * HDF5 output with params.yaml stored verbatim as metadata.

TWO INTERFACES TO THE SAME ENGINE
---------------------------------
  Run dictionaries (conversion runs of Sect. VI B-C):
      P = Params("params.yaml")
      r = xi_conversion(P, W12, W23, sigma, x, D12=0.0, D23=0.0)
      s = summary(r)
  Generator / Protocol objects (response and V runs of Sect. VI D-E, App. E):
      gen = Generator(P, XI, T=4, frame_meV=xi_frame(0.0, 0.0), rate_block=RATES_TAU0)
      pro = plateau_protocol(P, [W12, W23], tau_on_ps=0.5, T_pulse_ps=50.0)
      tr  = propagate(gen, pro)
      ob  = observables(tr, gen)
===============================================================================
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.integrate import solve_ivp

__version__ = "1.0"

HB = 0.658211957          # meV ps
RTOL, ATOL = 1e-8, 1e-10
RATES_MEV = "hbar_gamma_meV"
RATES_TAU0 = "dimensionless_gamma_tau0"
OUTPUT_DT_PS = 0.003


# =============================================================================
# 1. PARAMETERS
# =============================================================================
class Params:
    """Read-only view of params.yaml (meV, ps)."""

    def __init__(self, path="params.yaml"):
        self.path = str(path)
        self.text = Path(path).read_text()
        self.raw = yaml.safe_load(self.text)
        self.E0 = float(self.raw["units"]["E0_meV"])
        self.tau0_ps = float(self.raw["units"]["tau0_ps"])
        self.hbar_meV_ps = float(self.raw["units"]["hbar_meV_ps"])
        self.kion = float(self.raw["ionization"]["kappa_ion"])
        self.Sig1e11 = {k: float(v) for k, v in self.raw["exchange"]["Sigma_at_1e11_meV"].items()}
        self.proto = self.raw["protocol"]
        self.sha256 = hashlib.sha256(self.text.encode()).hexdigest()

    def rates(self, T=4, block=RATES_MEV):
        """hbar*gamma (meV) of every photon channel at T = 4, 77 or 300 K, from the rate
        block `block` of params.yaml (RATES_MEV as stored; RATES_TAU0 as gamma*tau0 x E0)."""
        b = self.raw["rates"][f"T_{int(T)}K"][block]
        if block == RATES_MEV:
            return {k: float(v) for k, v in b.items()}
        if block == RATES_TAU0:
            return {k: float(v) * self.E0 for k, v in b.items()}
        raise KeyError(block)

    rates_meV = rates

    def sigma(self, n):
        """Diagonal exchange shifts Sigma_a(n_X) in meV (Eq. 62; linear in n_X)."""
        return {k: v * n / 1e11 for k, v in self.Sig1e11.items()}

    sigma_meV = sigma

    def weak(self, W23):
        """Weak optical leg hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2) (meV)."""
        g1s = self.rates(4)["gamma_1s_to_0"]
        return 0.05 * min(W23, g1s / 2)

    @property
    def kappa_ion(self) -> float:
        return self.kion

    def laser_energies_meV(self, n_cal_cm2: float = 0.0) -> Dict[str, float]:
        """Carrier photon energies that give Delta~ = 0 at n_cal (0 -> bare)."""
        if n_cal_cm2 == 0.0:
            return dict(self.raw["calibration"]["laser_photon_energy_bare_meV"])
        for c in self.raw["calibration"]["mismatch_scan"]:
            if np.isclose(c["n_X_nominal_cm2"], n_cal_cm2):
                return dict(c["laser_photon_energy_meV"])
        raise KeyError(f"n_cal = {n_cal_cm2:g} not tabulated in params.yaml")


# =============================================================================
# 2. TOPOLOGIES (data only)
# =============================================================================
@dataclass(frozen=True)
class Topology:
    name: str
    labels: Tuple[str, str, str]
    legs: Tuple[Tuple[int, int], ...]             # driven legs (i, j): coupling |j><i| + h.c.
    jumps: Tuple[Tuple[int, int, str], ...]       # (src, dst, rate key): L = |dst><src|
    sigma_labels: Tuple[Optional[str], ...]       # Sigma label per state (None: crystal ground)
    outer: Tuple[int, int]                        # outer pair of the two-photon coherence
    leg_names: Tuple[str, ...]


XI = Topology(
    name="Xi", labels=("0", "1s", "2p"),
    legs=((0, 1), (1, 2)),
    jumps=((1, 0, "gamma_1s_to_0"), (0, 1, "gamma_0_to_1s"),
           (2, 1, "gamma_2p_to_1s"), (1, 2, "gamma_1s_to_2p")),
    sigma_labels=(None, "1s", "2p"), outer=(0, 2), leg_names=("12", "23"))

V = Topology(
    name="V", labels=("0", "1s", "2s"),
    legs=((0, 1), (0, 2)),
    jumps=((1, 0, "gamma_1s_to_0"), (0, 1, "gamma_0_to_1s"),
           (2, 0, "gamma_2s_to_0"), (0, 2, "gamma_0_to_2s")),
    sigma_labels=(None, "1s", "2s"), outer=(1, 2), leg_names=("12", "13"))

TOPOLOGIES = {"Xi": XI, "V": V}


def xi_frame(D12_meV: float, D23_meV: float) -> Tuple[float, float]:
    """Frame energies (h2, h3) of Eq. (64) from one-photon detunings (meV)."""
    return (D12_meV, D12_meV + D23_meV)


def v_frame(D12_meV: float, deltaV_meV: float) -> Tuple[float, float]:
    """Frame energies (h2, h3) of Eq. (71) from Delta12 and delta_V = Delta13 - Delta12."""
    return (D12_meV, D12_meV + deltaV_meV)


# =============================================================================
# 3. GENERATOR
# =============================================================================
I3 = np.eye(3)


def _op(a, b):                      # |a><b|
    m = np.zeros((3, 3), complex); m[a, b] = 1
    return m


def _comm(H):                       # vec(-i[H, rho]), column stacking
    return -1j * (np.kron(I3, H) - np.kron(H.T, I3))


def _diss(L):                       # vec(D[L] rho)
    LdL = L.conj().T @ L
    return np.kron(L.conj(), L) - 0.5 * np.kron(I3, LdL) - 0.5 * np.kron(LdL.T, I3)


def vec(rho):
    return np.asarray(rho).reshape(9, order="F")


def unvec(v):
    return np.asarray(v).reshape(3, 3, order="F")


class Generator:
    """Rotating-frame Liouvillian L(t) = L0 + sum_l Omega_l(t) K_l (1/ps; Omega in meV).

    Parameters
    ----------
    P          : Params
    topo       : Topology (XI or V)
    T          : photon temperature (4, 77, 300 K) for the rates
    frame_meV  : (h2, h3) laser-frame energies in meV (xi_frame / v_frame)
    n_X_cm2    : exciton density for the diagonal mean field (used if meanfield)
    meanfield  : add Sigma_a(n_X) to the frame energies (case 3)
    dissipator : include the photon dissipator (case 1 switches it off)
    phases     : per-leg phases phi_l (default 0)
    rates      : optional override {rate_key: hbar*gamma in meV}
    rate_block : rate block of params.yaml used when rates is None (RATES_MEV or RATES_TAU0)
    """

    def __init__(self, P: Params, topo: Topology, T: float = 4.0,
                 frame_meV: Sequence[float] = (0.0, 0.0), n_X_cm2: float = 0.0,
                 meanfield: bool = False, dissipator: bool = True,
                 phases: Optional[Sequence[float]] = None,
                 rates: Optional[Dict[str, float]] = None, rate_block: str = RATES_MEV):
        self.P, self.topo, self.T = P, topo, T
        self.rate_block = rate_block if rates is None else "override"
        self.meanfield, self.dissipator, self.n_X = meanfield, dissipator, n_X_cm2
        self.frame_meV = (0.0, frame_meV[0], frame_meV[1])
        sig = P.sigma(n_X_cm2) if meanfield else {}
        self.sigma_meV = tuple(0.0 if lab is None else sig.get(lab, 0.0) for lab in topo.sigma_labels)
        h = ([0.0, frame_meV[0] + self.sigma_meV[1], frame_meV[1] + self.sigma_meV[2]] if meanfield
             else [0.0, frame_meV[0], frame_meV[1]])
        self.h_meV = np.array(h, dtype=float)
        self.H0 = np.diag(h).astype(complex)                      # meV
        self.phases = tuple(phases) if phases is not None else (0.0,) * len(topo.legs)

        self.rates = dict(P.rates(T, rate_block)) if rates is None else dict(rates)   # meV
        self.jumps = []                                            # (src, dst, gamma [1/ps], key)
        L0 = _comm(self.H0 / HB)
        for src, dst, key in topo.jumps:
            g = self.rates.get(key, 0.0) / HB if dissipator else 0.0
            self.jumps.append((src, dst, g, key))
            if g > 0:
                L0 = L0 + g * _diss(_op(dst, src))
        self.L0 = L0
        self.hleg = [0.5 * (_op(j, i) + _op(i, j)) if ph == 0 else
                     0.5 * (np.exp(1j * ph) * _op(j, i) + np.exp(-1j * ph) * _op(i, j))
                     for (i, j), ph in zip(topo.legs, self.phases)]      # meV per meV of hbar*Omega
        self.K = [_comm(h) / HB for h in self.hleg]

    # --- instantaneous operators (hbar*Omega in meV) ---------------------------
    def H(self, omegas: Sequence[float]) -> np.ndarray:
        """Hamiltonian in meV."""
        return self.H0 + sum(w * h for w, h in zip(omegas, self.hleg))

    def L(self, omegas: Sequence[float]) -> np.ndarray:
        """Liouvillian in 1/ps."""
        return self.L0 + sum(w * k for w, k in zip(omegas, self.K))

    def gamma(self, key: str) -> float:
        """Rate of a jump channel in 1/ps (0 if absent or dissipator off)."""
        for _, _, g, k in self.jumps:
            if k == key:
                return g
        return 0.0

    def describe(self) -> Dict:
        return {"topology": self.topo.name, "T_K": self.T, "meanfield": self.meanfield,
                "dissipator": self.dissipator, "n_X_cm2": self.n_X,
                "frame_meV": list(map(float, self.frame_meV)), "sigma_meV": list(self.sigma_meV),
                "h_meV": list(map(float, self.h_meV)), "phases": list(self.phases),
                "rate_block": self.rate_block, "rates_per_ps": {k: g for _, _, g, k in self.jumps}}


# =============================================================================
# 4. ENVELOPES AND PROTOCOLS
# =============================================================================
@dataclass
class Envelope:
    """Gaussian, Eq. (43), or plateau, Eq. (42); times in ps."""
    kind: str                         # 'gauss' | 'plateau' | 'off'
    tc: float = 0.0
    sigma: float = 1.0
    t_on: float = 0.0
    t_off: float = 0.0
    tau_on: float = 1.0
    tau_off: float = 1.0

    def __call__(self, t):
        if self.kind == "gauss":
            return np.exp(-0.5 * ((t - self.tc) / self.sigma) ** 2)
        if self.kind == "plateau":
            return 0.25 * (1 + np.tanh((t - self.t_on) / self.tau_on)) * (1 - np.tanh((t - self.t_off) / self.tau_off))
        return 0.0 * np.asarray(t, dtype=float)

    def breakpoints(self):
        if self.kind == "gauss":
            return [self.tc - 6 * self.sigma, self.tc, self.tc + 6 * self.sigma]
        if self.kind == "plateau":
            return [self.t_on - 6 * self.tau_on, self.t_on, self.t_on + 6 * self.tau_on,
                    self.t_off - 6 * self.tau_off, self.t_off, self.t_off + 6 * self.tau_off]
        return []

    def min_scale(self):
        if self.kind == "gauss":
            return self.sigma
        if self.kind == "plateau":
            return min(self.tau_on, self.tau_off)
        return np.inf


@dataclass
class Protocol:
    """Drive of every leg: amplitude hbar*Omega (meV) x envelope; times in ps."""
    label: str
    amps: List[float]
    envs: List[Envelope]
    t_end: float
    meta: Dict = field(default_factory=dict)

    def omegas(self, t) -> np.ndarray:
        return np.array([a * e(t) for a, e in zip(self.amps, self.envs)])

    def breakpoints(self) -> List[float]:
        b = {0.0, self.t_end}
        for e in self.envs:
            b.update(x for x in e.breakpoints() if 0.0 < x < self.t_end)
        return sorted(b)

    def max_step(self) -> float:
        """min(sigma, tau_on)/20 over the drives."""
        s = min((e.min_scale() for e in self.envs), default=np.inf)
        return s / 20.0 if np.isfinite(s) else self.t_end / 200.0


def xi_transfer_protocol(P: Params, sigma_ps: float, x: float, W0_meV: float,
                         order: str = "counterintuitive", W12_meV: Optional[float] = None,
                         W23_meV: Optional[float] = None) -> Protocol:
    """Xi Gaussian pair of equal widths, delay x sigma, first centre 6 sigma after t0,
    end 6 sigma after the last centre; amplitudes W12, W23 (default W0 on both legs)."""
    first, second = 6 * sigma_ps, 6 * sigma_ps + x * sigma_ps
    t23, t12 = (first, second) if order == "counterintuitive" else (second, first)
    a12 = W0_meV if W12_meV is None else W12_meV
    a23 = W0_meV if W23_meV is None else W23_meV
    return Protocol(label=f"Xi_transfer[{order}]", amps=[float(a12), float(a23)],
                    envs=[Envelope("gauss", tc=t12, sigma=sigma_ps), Envelope("gauss", tc=t23, sigma=sigma_ps)],
                    t_end=second + 6 * sigma_ps,
                    meta={"sigma_ps": sigma_ps, "delay_over_sigma": x, "W0_meV": W0_meV,
                          "order": order, "t12_ps": float(t12), "t23_ps": float(t23)})


def plateau_protocol(P: Params, W_meV: Sequence[float], tau_on_ps: float, T_pulse_ps: float,
                     tau_off_ps: Optional[float] = None, label: str = "plateau") -> Protocol:
    """Plateau envelope (Eq. 42) on every leg: t_on = 6 tau_on, t_off = t_on + T_pulse,
    end 6 tau_off after t_off."""
    ton_ = float(tau_on_ps); toff_ = float(tau_off_ps if tau_off_ps else tau_on_ps)
    t_on = 6 * ton_; t_off = t_on + float(T_pulse_ps)
    envs = [Envelope("plateau", t_on=t_on, t_off=t_off, tau_on=ton_, tau_off=toff_) for _ in W_meV]
    return Protocol(label=label, amps=[float(w) for w in W_meV], envs=envs,
                    t_end=t_off + 6 * toff_,
                    meta={"W_meV": list(W_meV), "tau_on_ps": tau_on_ps,
                          "T_pulse_ps": T_pulse_ps, "t_on_ps": float(t_on), "t_off_ps": float(t_off)})


def v_beat_protocol(P: Params, sigma_ps: float, W12_meV: float, W13_meV: float,
                    T_after_ps: Optional[float] = None) -> Protocol:
    """Synchronous Gaussian pair t12 = t13 = 6 sigma; end = pulses (+6 sigma) + ~5 T2^V."""
    s = float(sigma_ps); tc = 6 * s
    if T_after_ps is None:
        r = P.rates(4)
        T2 = HB / (0.5 * (r["gamma_1s_to_0"] + r["gamma_2s_to_0"]))
        T_after_ps = 5 * T2
    t_end = tc + 6 * s + float(T_after_ps)
    return Protocol(label="V_beat", amps=[float(W12_meV), float(W13_meV)],
                    envs=[Envelope("gauss", tc=tc, sigma=s), Envelope("gauss", tc=tc, sigma=s)],
                    t_end=t_end,
                    meta={"sigma_ps": sigma_ps, "W12_meV": W12_meV, "W13_meV": W13_meV,
                          "t_c_ps": float(tc), "free_decay_from_ps": float(tc + 6 * s)})


def no_drive_protocol(P: Params, T_ps: float, n_legs: int = 2) -> Protocol:
    return Protocol(label="no_drive", amps=[0.0] * n_legs, envs=[Envelope("off")] * n_legs,
                    t_end=float(T_ps), meta={"T_ps": T_ps})


# =============================================================================
# 5. TIME PROPAGATION (the single engine)
# =============================================================================
def integrate(gen: Generator, amps, envs, t_end, bps=(), rho0=None, rtol=1e-8, atol=1e-13,
              max_step=None, dt_out=OUTPUT_DT_PS, method="DOP853"):
    """Integrate the master equation of `gen` over [0, t_end] (ps).

    amps = peak hbar*Omega per leg (meV); envs = envelope functions per leg; bps =
    breakpoints (ps), integration is piecewise between them with max_step (ps).
    rho0 defaults to |1><1| (crystal ground). Output on a uniform grid of dt_out (ps).
    Returns a dict with t, rho, W (hbar*Omega per leg), the running integrals N_emit,
    N_emit_2s, int_rho22 (ps), P_ion, the frame energies h and solver statistics.
    """
    P = gen.P
    L0, K = gen.L0, gen.K
    g10 = gen.gamma("gamma_1s_to_0")
    g20 = gen.gamma("gamma_2s_to_0")
    is_xi = gen.topo.name == "Xi"
    kion = P.kion / P.E0 / HB          # Gamma_ion = kion * (hbar Omega23)^2 [1/ps]

    def rhs(t, y):
        W = [a * e(t) for a, e in zip(amps, envs)]
        Lt = L0 + W[0] * K[0] + W[1] * K[1]
        r = y[:9]
        dr = Lt @ r
        p11, p22, p33 = r[0].real, r[4].real, r[8].real
        dacc = [g10 * p22, g20 * p33, p22,
                (kion * W[1] ** 2 * p33) if is_xi else 0.0]
        return np.concatenate([dr, np.asarray(dacc, complex)])

    y = np.zeros(13, complex)
    if rho0 is None:
        y[0] = 1.0
    else:
        y[:9] = vec(rho0)
    tg = np.arange(0.0, t_end + 0.5 * dt_out, dt_out); tg[-1] = min(tg[-1], t_end)
    edges = sorted({0.0, t_end, *[b for b in bps if 0 < b < t_end]})
    ms = max_step if max_step is not None else np.inf
    out = np.empty((len(tg), 13), complex); filled = np.zeros(len(tg), bool)
    nfev, nsteps = 0, 0
    # implicit scipy methods do not accept complex states: integrate [Re y, Im y]
    real_mode = method in ("Radau", "BDF", "LSODA")
    n = len(y)

    def rhs_real(t, u):
        d = rhs(t, u[:n] + 1j * u[n:])
        return np.concatenate([d.real, d.imag])

    for a, b in zip(edges[:-1], edges[1:]):
        if real_mode:
            s = solve_ivp(rhs_real, (a, b), np.concatenate([y.real, y.imag]), method=method,
                          rtol=rtol, atol=atol, max_step=ms, dense_output=True)
        else:
            s = solve_ivp(rhs, (a, b), y, method=method, rtol=rtol, atol=atol, max_step=ms,
                          dense_output=True)
        if not s.success:
            raise RuntimeError(s.message)
        nfev += s.nfev; nsteps += len(s.t) - 1
        sel = (tg >= a) & (tg <= b) & ~filled
        v = s.sol(tg[sel]).T
        out[sel] = v[:, :n] + 1j * v[:, n:] if real_mode else v; filled |= sel
        y = s.y[:n, -1] + 1j * s.y[n:, -1] if real_mode else s.y[:, -1]
    rho = out[:, :9].reshape(-1, 3, 3, order="F")
    Wt = np.array([[a * e(t) for a, e in zip(amps, envs)] for t in tg])
    acc = out[:, 9:].real
    return dict(t=tg, rho=rho, W=Wt, N_emit=acc[:, 0], N_emit_2s=acc[:, 1], int_rho22=acc[:, 2],
                P_ion=acc[:, 3], h=gen.h_meV.copy(), nfev=nfev, topo=gen.topo.name,
                nsteps=nsteps, n_segments=len(edges) - 1, final_state=unvec(y[:9]),
                method=method, rtol=rtol, atol=atol, max_step=ms)


# --- run-dictionary interface (conversion runs) -----------------------------------
def run(P, topo="Xi", h=(0.0, 0.0), amps=(0.0, 0.0), envs=None, t_end=None, bps=(),
        T=4, dissipator=True, rtol=1e-8, atol=1e-13, dt_out=OUTPUT_DT_PS, max_step=None,
        rates=None):
    """Integrate one protocol. h = (h2, h3) frame energies (meV); amps = peak hbar*Omega per leg
    (meV); envs = envelope functions per leg; t_end, bps in ps. Returns a dict."""
    gen = Generator(P, TOPOLOGIES[topo], T=T, frame_meV=h, dissipator=dissipator, rates=rates)
    return integrate(gen, amps, envs, t_end, bps=bps, rtol=rtol, atol=atol, max_step=max_step,
                     dt_out=dt_out)


def xi_conversion(P, W12, W23, sigma, x, D12=0.0, D23=0.0, order="counterintuitive", **kw):
    """Xi Gaussian pair: THz first in the counterintuitive order (t23 < t12), delay x*sigma,
    first centre 6 sigma after t0, end 6 sigma after the last centre."""
    first, second = 6 * sigma, 6 * sigma + x * sigma
    t23, t12 = (first, second) if order == "counterintuitive" else (second, first)
    t_end = second + 6 * sigma
    bps = [t12 - 6 * sigma, t12, t12 + 6 * sigma, t23 - 6 * sigma, t23, t23 + 6 * sigma]
    r = run(P, "Xi", h=(D12, D12 + D23), amps=(W12, W23),
            envs=(Envelope("gauss", tc=t12, sigma=sigma), Envelope("gauss", tc=t23, sigma=sigma)),
            t_end=t_end, bps=bps, max_step=sigma / 20, **kw)
    r.update(t12=t12, t23=t23, sigma=sigma, x=x, order=order)
    return r


def xi_plateau(P, W12, W23, tau=0.5, T_pulse=50.0, D12=0.0, D23=0.0, **kw):
    """Xi plateau drive on both legs (CPT plateau): t_on = 6 tau, t_off = t_on + T_pulse."""
    t_on = 6 * tau; t_off = t_on + T_pulse; t_end = t_off + 6 * tau
    env = Envelope("plateau", t_on=t_on, t_off=t_off, tau_on=tau, tau_off=tau)
    bps = [t_on - 6 * tau, t_on, t_on + 6 * tau, t_off - 6 * tau, t_off, t_off + 6 * tau]
    r = run(P, "Xi", h=(D12, D12 + D23), amps=(W12, W23), envs=(env, env), t_end=t_end,
            bps=bps, max_step=tau / 20, **kw)
    r.update(t_on=t_on, t_off=t_off)
    return r


def v_beat(P, W12, W13, sigma, D12=0.0, D13=0.0, T_after=None, **kw):
    """V synchronous Gaussian pair at tc = 6 sigma, followed by T_after (default 5 T2^V)."""
    tc = 6 * sigma
    if T_after is None:
        R = P.rates(4)
        T2 = HB / (0.5 * (R["gamma_1s_to_0"] + R["gamma_2s_to_0"]))
        T_after = 5 * T2
    t_end = tc + 6 * sigma + T_after
    r = run(P, "V", h=(D12, D13), amps=(W12, W13),
            envs=(Envelope("gauss", tc=tc, sigma=sigma), Envelope("gauss", tc=tc, sigma=sigma)),
            t_end=t_end, bps=[tc, tc + 6 * sigma], max_step=sigma / 20, **kw)
    r.update(tc=tc, sigma=sigma)
    return r


# --- Generator / Protocol interface (response and V runs) -------------------------
@dataclass
class Trajectory:
    t: np.ndarray                     # ps
    rho: np.ndarray                   # (Nt, 3, 3) complex
    omegas: np.ndarray                # (Nt, n_legs), hbar*Omega in meV
    acc: Dict[str, np.ndarray]        # running integrals (N_emit, N_emit_2s, int_rho22 [ps], P_ion)
    stats: Dict
    protocol: Protocol
    gen_info: Dict


def default_tolerances(gen: Generator) -> Tuple[float, float]:
    """(rtol, atol) from params.yaml protocol.numerics: coherent-only runs keep rho rank-1
    and use the tighter pair so that positivity holds to 1e-9."""
    nm = gen.P.proto.get("numerics", {})
    if not gen.dissipator and "coherent_only_rtol" in nm:
        return float(nm["coherent_only_rtol"]), float(nm["coherent_only_atol"])
    return float(nm.get("rtol", RTOL)), float(nm.get("atol", ATOL))


def propagate(gen: Generator, pro: Protocol, rho0: Optional[np.ndarray] = None,
              dt_out_ps: float = OUTPUT_DT_PS, rtol: Optional[float] = None,
              atol: Optional[float] = None, method: str = "DOP853",
              max_step: Optional[float] = None) -> Trajectory:
    """Integrate a Protocol with integrate(); tolerances default to default_tolerances(gen),
    max_step to min(sigma, tau_on)/20."""
    rt0, at0 = default_tolerances(gen)
    r = integrate(gen, pro.amps, pro.envs, pro.t_end, bps=pro.breakpoints(), rho0=rho0,
                  rtol=rt0 if rtol is None else rtol, atol=at0 if atol is None else atol,
                  max_step=pro.max_step() if max_step is None else max_step,
                  dt_out=dt_out_ps, method=method)
    acc = {k: r[k] for k in ("N_emit", "N_emit_2s", "int_rho22", "P_ion")}
    stats = {k: r[k] for k in ("method", "rtol", "atol", "max_step", "nfev", "nsteps",
                               "n_segments", "final_state")}
    return Trajectory(t=r["t"], rho=r["rho"], omegas=r["W"], acc=acc, stats=stats,
                      protocol=pro, gen_info=gen.describe())


# =============================================================================
# 6. STEADY STATE (Liouvillian null space)
# =============================================================================
def steady_state(gen: Generator, omegas_meV: Sequence[float]) -> Tuple[np.ndarray, Dict]:
    """Null-space steady state of L(omegas) with Tr rho = 1.

    Returns rho_ss and diagnostics: the two smallest singular values of L (1/ps), the
    spectral gap (smallest |Re lambda| of the non-zero eigenvalues, 1/ps), the number of
    (near-)zero modes and the residual |L rho_ss|.
    """
    Lm = gen.L(omegas_meV)
    A = np.vstack([Lm, vec(np.eye(3, dtype=complex)).reshape(1, 9)])
    b = np.zeros(10, dtype=complex); b[-1] = 1.0
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    rho = unvec(x)
    rho = 0.5 * (rho + rho.conj().T)
    sv = np.linalg.svd(Lm, compute_uv=False)
    ev = np.linalg.eigvals(Lm)
    scale = max(np.abs(ev).max(), 1e-300)
    zero = np.abs(ev) < 1e-12 * max(scale, 1.0)
    nz = ev[~zero]
    info = {"sv_min": float(sv[-1]), "sv_second": float(sv[-2]),
            "n_zero_modes": int(zero.sum()),
            "spectral_gap_per_ps": float(np.min(np.abs(nz.real))) if nz.size else 0.0,
            "residual": float(np.abs(Lm @ vec(rho)).max()),
            "unique": bool(zero.sum() == 1)}
    return rho, info


# =============================================================================
# 7. OBSERVABLES
# =============================================================================
def diagnostics(rho: np.ndarray) -> Dict[str, np.ndarray]:
    """Trace error, Hermiticity error, smallest eigenvalue and purity, per time."""
    tr = np.einsum("tii->t", rho)
    herm = np.abs(rho - np.conj(np.transpose(rho, (0, 2, 1)))).max(axis=(1, 2))
    rh = 0.5 * (rho + np.conj(np.transpose(rho, (0, 2, 1))))
    lmin = np.linalg.eigvalsh(rh)[:, 0]
    purity = np.einsum("tij,tji->t", rho, rho).real
    return {"trace_err": np.abs(tr - 1.0), "herm_err": herm, "min_eig": lmin, "purity": purity}


def dressed_populations(rho: np.ndarray, omegas: np.ndarray, h: np.ndarray,
                        mask_frac: float = 1e-6) -> Dict[str, np.ndarray]:
    """Dressed populations of the Xi manifold (phases zero).

    |D> = cos(th)|1> - sin(th)|3>, tan th = Omega12/Omega23 (Eq. 69);
    |B> = sin(th)|1> + cos(th)|3>; B+- = eigenvectors of H in span{|B>, |2>}
    (exact eigenbasis when delta~Xi = 0). P_D + P_B+ + P_B- = 1.
    Entries where Omega0 < mask_frac * max(Omega0) are NaN.
    """
    W12, W23 = omegas[:, 0], omegas[:, 1]
    W0 = np.hypot(W12, W23)
    th = np.arctan2(W12, W23); c, s = np.cos(th), np.sin(th)
    PD = c ** 2 * rho[:, 0, 0].real + s ** 2 * rho[:, 2, 2].real - 2 * s * c * rho[:, 0, 2].real
    PBp = np.empty_like(PD); PBm = np.empty_like(PD)
    for k in range(len(PD)):
        Hk = np.diag(h).astype(complex); Hk[1, 0] = Hk[0, 1] = W12[k] / 2
        Hk[2, 1] = Hk[1, 2] = W23[k] / 2
        B = np.array([s[k], 0, c[k]]); E2 = np.array([0, 1, 0.0])
        bas = np.stack([B, E2], 1).astype(complex)
        e, u = np.linalg.eigh(bas.conj().T @ Hk @ bas)
        vm, vp = bas @ u[:, 0], bas @ u[:, 1]
        PBm[k] = (vm.conj() @ rho[k] @ vm).real; PBp[k] = (vp.conj() @ rho[k] @ vp).real
    bad = W0 < mask_frac * max(W0.max(), 1e-300)
    for a in (th, PD, PBp, PBm):
        a[bad] = np.nan
    return {"theta": th, "P_D": PD, "P_Bp": PBp, "P_Bm": PBm, "sincos": s * c}


def xi_dressed(r):
    """P_D, P_B+, P_B- of a run dictionary of the Xi manifold (phases zero)."""
    d = dressed_populations(r["rho"], r["W"], r["h"])
    return d["P_D"], d["P_Bp"], d["P_Bm"]


def bloch_outer(rho: np.ndarray, pair: Tuple[int, int], mask: float = 1e-9):
    """Bloch vector of the outer pair (a, b), normalized by n = rho_aa + rho_bb; NaN where n < mask."""
    a, b = pair
    n = rho[:, a, a].real + rho[:, b, b].real
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.stack([2 * rho[:, a, b].real, 2 * rho[:, a, b].imag,
                      rho[:, a, a].real - rho[:, b, b].real], axis=1) / n[:, None]
    r[n < mask] = np.nan
    return r, n


def lab_frame_coherence(tr: Trajectory, E_L12_meV: float, E_L13_meV: float) -> np.ndarray:
    """V: rho23 in the laboratory frame, inverse of the rotation of Eq. (9):
    rho23_lab(t) = exp[-i (omega_L13 - omega_L12) t] rho23_bar(t), with signed carrier
    frequencies omega_L = -E_photon/hbar, i.e. exp[+i (E_L13 - E_L12) t / hbar] rho23_bar(t)."""
    w = (E_L13_meV - E_L12_meV) / HB                 # rad/ps
    return np.exp(1j * w * tr.t) * tr.rho[:, 1, 2]


def observables(tr: Trajectory, gen: Generator) -> Dict[str, np.ndarray]:
    """Populations, coherences, diagnostics, drives, running integrals, Xi dressed
    populations and the outer-pair Bloch vector of a trajectory."""
    ob = {"t_ps": tr.t}
    for a in range(3):
        ob[f"rho{a+1}{a+1}"] = tr.rho[:, a, a].real
    for a, b in ((0, 1), (1, 2), (0, 2)):
        ob[f"rho{a+1}{b+1}"] = tr.rho[:, a, b]
    ob.update(diagnostics(tr.rho))
    for k, name in enumerate(gen.topo.leg_names):
        ob[f"Omega{name}_meV"] = tr.omegas[:, k]
    ob["N_emit"] = tr.acc["N_emit"]
    ob["int_rho22_ps"] = tr.acc["int_rho22"]
    if gen.topo.name == "Xi":
        ob["P_ion"] = tr.acc["P_ion"]
        ob.update(dressed_populations(tr.rho, tr.omegas, gen.h_meV))
    else:
        ob["N_emit_2s"] = tr.acc["N_emit_2s"]
    r, n = bloch_outer(tr.rho, gen.topo.outer)
    ob["bloch_outer"] = r
    ob["outer_norm"] = n
    return ob


def trajectory_summary(tr: Trajectory, gen: Generator) -> Dict[str, float]:
    """Scalar observables of one trajectory."""
    ob = observables(tr, gen)
    s = {"rho11_final": float(ob["rho11"][-1]), "rho22_final": float(ob["rho22"][-1]),
         "rho33_final": float(ob["rho33"][-1]), "max_rho22": float(np.nanmax(ob["rho22"])),
         "int_rho22_ps": float(ob["int_rho22_ps"][-1]),
         "purity_min": float(np.min(ob["purity"])),
         "max_trace_err": float(ob["trace_err"].max()),
         "max_herm_err": float(ob["herm_err"].max()),
         "min_eig": float(ob["min_eig"].min()),
         "N_emit": float(tr.acc["N_emit"][-1])}
    if gen.topo.name == "Xi":
        s["P_ion"] = float(tr.acc["P_ion"][-1])
        s["P_D_min"] = float(np.nanmin(ob["P_D"])) if np.any(np.isfinite(ob["P_D"])) else np.nan
    else:
        s["N_emit_2s"] = float(tr.acc["N_emit_2s"][-1])
    return s


def summary(r):
    """Scalar observables of a run dictionary (conversion runs): final populations,
    N_emit, P_ion, one-exciton validity max(1 - rho11), diagnostics; for Xi also the
    conversion efficiency eta (Eq. 76), vertex suppression, P_ion per 2p, dressed
    populations and the outer Bloch radius."""
    rho = r["rho"]
    p11, p22, p33 = (rho[:, k, k].real for k in range(3))
    tr = np.abs(np.einsum("tii->t", rho) - 1).max()
    lmin = np.linalg.eigvalsh(0.5 * (rho + np.conj(np.transpose(rho, (0, 2, 1)))))[:, 0].min()
    s = dict(rho11_f=float(p11[-1]), rho22_f=float(p22[-1]), rho33_f=float(p33[-1]),
             max_rho22=float(p22.max()), int_rho22_ps=float(r["int_rho22"][-1]),
             N_emit=float(r["N_emit"][-1]), P_ion=float(r["P_ion"][-1]),
             max_one_exciton=float((1 - p11).max()), trace_err=float(tr), min_eig=float(lmin),
             purity_min=float(np.einsum("tij,tji->t", rho, rho).real.min()), nfev=r["nfev"])
    if r["topo"] == "Xi":
        n33 = s["rho33_f"]
        s["eta"] = n33 / (n33 + s["N_emit"]) if (n33 + s["N_emit"]) > 0 else np.nan
        s["vertex_suppression"] = s["max_rho22"] / n33 if n33 > 0 else np.nan
        s["int_vertex_over_n2p_ps"] = s["int_rho22_ps"] / n33 if n33 > 0 else np.nan
        s["P_ion_per_2p"] = s["P_ion"] / n33 if n33 > 0 else np.nan
        PD, PBp, PBm = xi_dressed(r)
        s["P_D_min"] = float(np.nanmin(PD)); s["P_D_end"] = float(PD[np.isfinite(PD)][-1])
        s["P_Bp_end"] = float(PBp[np.isfinite(PBp)][-1]); s["P_Bm_end"] = float(PBm[np.isfinite(PBm)][-1])
        vec_, nn = bloch_outer(rho, (0, 2), mask=0.0)
        rr = np.linalg.norm(vec_, axis=1); s["r_final"] = float(rr[-1]); s["outer_norm_min"] = float(nn.min())
    else:
        s["N_emit_2s"] = float(r["N_emit_2s"][-1])
    return s


def chi(rho: np.ndarray, leg: Tuple[int, int], omega: float) -> complex:
    """Susceptibility proxy chi ∝ rho_ji / Omega_leg for the leg (i, j)."""
    i, j = leg
    return rho[j, i] / omega


# =============================================================================
# 8. HDF5 OUTPUT
# =============================================================================
def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_h5(path: str, P: Params, runs: Dict[str, Tuple[Trajectory, Generator]] = None,
             scans: Dict[str, Dict[str, np.ndarray]] = None, meta: Dict = None):
    """HDF5 file with params.yaml (verbatim, SHA-256), code SHA-256, runs (trajectory,
    generator, protocol, solver statistics, observables) and scans."""
    import h5py
    with h5py.File(path, "w") as f:
        m = f.create_group("meta")
        m.attrs["params_yaml"] = P.text
        m.attrs["params_sha256"] = P.sha256
        m.attrs["code_sha256"] = code_sha256()
        m.attrs["code_version"] = __version__
        m.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        m.attrs["units"] = "time ps, energy meV (hbar = %g meV ps)" % HB
        for k, v in (meta or {}).items():
            m.attrs[k] = json.dumps(v, default=str)
        g = f.create_group("runs")
        for name, (tr, gen) in (runs or {}).items():
            h = g.create_group(name)
            h.attrs["generator"] = json.dumps(gen.describe(), default=str)
            h.attrs["protocol"] = json.dumps({"label": tr.protocol.label, **tr.protocol.meta,
                                              "amps_meV": list(map(float, tr.protocol.amps))},
                                             default=str)
            h.attrs["solver_stats"] = json.dumps({k: v for k, v in tr.stats.items()
                                                  if k != "final_state"}, default=str)
            h.create_dataset("t_ps", data=tr.t, compression="gzip")
            h.create_dataset("rho", data=tr.rho, compression="gzip")
            for k, v in observables(tr, gen).items():
                if k != "t_ps":
                    h.create_dataset(k, data=np.asarray(v), compression="gzip")
        s = f.create_group("scans")
        for name, d in (scans or {}).items():
            h = s.create_group(name)
            for k, v in d.items():
                if isinstance(v, (str, dict)):
                    h.attrs[k] = json.dumps(v, default=str)
                else:
                    h.create_dataset(k, data=np.asarray(v))
