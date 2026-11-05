"""
table4_admissibility.py -- admissibility of the protocol families: Table IV of the paper
("Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley Lindblad
Framework for WSe2", Sect. VI A).

For every protocol family of Table III, the worst case over all its runs of the admissibility
ratios of Sect. IV C:
  RWA          hbar*Omega / (hbar*omega_B - |Delta|) on the interband (0->1s) and intra-excitonic
               (1s->2p) legs;
  selectivity  hbar*Omega / |Delta| against the nearest omitted transition (0->2s or 0->1s for the
               optical tones, 0->3s for the 0->2s tone, 1s->3p for the THz tone), with |Delta|
               reduced by the detuning of the run and corrected for the mean-field shift of the
               competing state, Sigma_a = n_X zeta_aa E0 a_B^2 (Eq. 62);
  bandwidth    sigma / sigma_min or tau_on / tau_min for the bounds of Sect. IV C (margin n = 3).

The Xi conversion family is every run of Figs. 5 and 6 and of the linearity check: the robustness
map and its optical-leg control, the adiabaticity grid, the density mismatch scan and the
bare-laser scan, with the weak optical leg hbar*Omega12 = 0.05 min(hbar*Omega23, hbar*gamma_1s/2).
The response, V-response and V-beat families are read from the protocol block of params.yaml.

Inputs (current directory): params.yaml; fig5_R3.json (adiabaticity grid), fig6_R2.json (map,
optical-leg map, mismatch and bare-laser scans) and sect6_numbers.json (linearity), written by
fig5_R3.py, fig6_R2.py and sect6_numbers.py. Output: printed table and table4_admissibility.json.
Usage:  python table4_admissibility.py
"""
import json

import numpy as np
import yaml

E0, aB = 269.0, 1.19                       # meV, nm

PY = yaml.safe_load(open("params.yaml"))
S = {}
for f in ("fig5_R3.json", "fig6_R2.json", "sect6_numbers.json"):
    S.update(json.load(open(f)))

g1s = float(PY["rates"]["T_4K"]["hbar_gamma_meV"]["gamma_1s_to_0"])       # meV
Sig = {k: float(v) for k, v in PY["exchange"]["Sigma_at_1e11_meV"].items()}
BO = PY["bohr_energies_meV"]
SEL = PY["selectivity_n3"]
s32, s12, s23 = SEL["THz 1s->2p vs 1s->3p"], SEL["opt 0->1s vs 0->2s"], SEL["opt 0->2s vs 0->3s"]
W12_weak = 0.05 * min(5.0, g1s / 2)        # weak optical leg at the baseline hbar*Omega23 = 5 meV

zeta = {k: v["tot"] for k, v in PY["exchange"]["zeta_E0aB2"].items()}
Sg = lambda a, n: n * 1e-14 * zeta[f"{a},{a}"] * E0 * aB ** 2       # meV
gap2s = lambda n: s12["detuning_meV"] + Sg("2s", n) - Sg("1s", n)
gap3p = lambda n: s32["detuning_meV"] + Sg("3p", n) - Sg("2p", n)
c3 = PY["protocol"]["Part_A"]["case3"]

# Xi conversion: every run as (n, D12, D23, W12, W23)
runs = [(0, 0, 0, W12_weak, 5.0)]
runs += [(0, 0, r["delta"], 0.05 * min(r["W23"], g1s / 2), r["W23"]) for r in S["map"]]
runs += [(0, r["delta"], 0, W12_weak, 5.0) for r in S["map_optical"]]
runs += [(0, 0, 0, 0.05 * min(r["W23"], g1s / 2), r["W23"]) for r in S["grid"]]
runs += [(r["n_act"], r["D12"], r["D23"], W12_weak, 5.0) for r in S["mismatch"]]
runs += [(r["n_X"], r["D12"], r["D23"], W12_weak, 5.0) for r in S["bare_scan"]]
runs += [(c3["n_X_cm2"], c3["Xi"]["Delta12_meV"], c3["Xi"]["Delta23_meV"], W12_weak, 5.0)]
lin = [(0, 0, 0, r["W12"], 5.0) for r in S["linearity"]]


def xi_rows(rs):
    return dict(rwa_inter=max(w12 / (BO["0-1s"] - abs(d12)) for n, d12, d23, w12, w23 in rs),
                rwa_intra=max(w23 / (BO["1s-2p"] - abs(d23)) for n, d12, d23, w12, w23 in rs),
                sel_inter=max(w12 / (gap2s(n) + d12) for n, d12, d23, w12, w23 in rs),
                sel_intra=max(w23 / (gap3p(n) + d23) for n, d12, d23, w12, w23 in rs))


xc = xi_rows(runs); xcl = xi_rows(runs + lin)
bw_xc = min(PY["protocol"]["Xi_conversion"]["grids"]["sigma_ps"]) / max(s32["sigma_min_ps"], s12["sigma_min_ps"])

# Xi response: EIT spectra (probe <= 0.0801, |Delta12| <= 15, Omega23 <= 5), CPT plateau (0.05, 1), window-centre runs
xr = PY["protocol"]["Xi_response"]
w12r = max(xr["hbar_Omega12_meV_derived"])
xr_rwa_inter = max(w12r / (BO["0-1s"] - 15.0), 0.05 / BO["0-1s"])
xr_sel_inter = max(w12r / (gap2s(0) - 15.0), 0.05 / gap2s(0))
xr_rwa_intra = max(xr["hbar_Omega23_meV"]) / BO["1s-2p"]
xr_sel_intra = max(xr["hbar_Omega23_meV"]) / gap3p(0)
xr_bw = min(xr["tau_on_off_ps"], PY["protocol"]["Part_A"]["A4_Xi_plateau_CPT"]["tau_on_off_ps"]) / s32["tanh_ramp_min_ps"]

# V response
vr = PY["protocol"]["V_response"]; dVmax = max(abs(np.array(vr["delta_V_grid"]["values_meV"])))
vr_rwa = max(vr["hbar_Omega12_meV"] / BO["0-1s"], vr["hbar_Omega13_meV"] / (BO["0-2s"] - dVmax))
vr_sel = max(vr["hbar_Omega12_meV"] / gap2s(0), vr["hbar_Omega13_meV"] / (s23["detuning_meV"] - dVmax))
vr_bw = vr["tau_on_off_ps"] / s23["tanh_ramp_min_ps"]

# V beat (1 meV both legs; case 3 at 1e11 with bare lasers: Delta12 = Sigma_1s, Delta13 = Sigma_2s)
vb = PY["protocol"]["V_beat"]
vb_rwa = max(vb["hbar_Omega12_meV"] / (BO["0-1s"] - Sig["1s"]), vb["hbar_Omega13_meV"] / (BO["0-2s"] - Sig["2s"]),
             vb["hbar_Omega12_meV"] / BO["0-1s"])
# competitor distance from the photon = gap + (Sigma_comp - Sigma_target)(n) + hbar*Delta~ ; 3s shift unknown -> resonant runs
vb_sel = max(vb["hbar_Omega12_meV"] / gap2s(0), vb["hbar_Omega13_meV"] / s23["detuning_meV"],
             vb["hbar_Omega12_meV"] / (gap2s(1e11) + Sig["1s"]))
vb_bw = min(vb["sigma_ps_grid"]) / s23["sigma_min_ps"]

table = {
    "Xi conversion": dict(rwa_inter=xc["rwa_inter"], rwa_intra=xc["rwa_intra"], sel_inter=xc["sel_inter"],
                          sel_intra=xc["sel_intra"], bandwidth_min=bw_xc,
                          with_linearity_control=dict(rwa_inter=xcl["rwa_inter"], sel_inter=xcl["sel_inter"])),
    "Xi response": dict(rwa_inter=xr_rwa_inter, rwa_intra=xr_rwa_intra, sel_inter=xr_sel_inter,
                        sel_intra=xr_sel_intra, bandwidth=xr_bw),
    "V response": dict(rwa_inter=vr_rwa, sel_inter=vr_sel, bandwidth=vr_bw),
    "V beat": dict(rwa_inter=vb_rwa, sel_inter=vb_sel, bandwidth=vb_bw),
    "max_rwa": max(xc["rwa_intra"], xr_rwa_intra),
    "max_selectivity": max(xc["sel_intra"], xr_sel_intra, vb_sel),
    "THz_1s3p_detuning_meV": s32["detuning_meV"],
    "n_runs_Xi_conversion": len(runs),
}
json.dump(table, open("table4_admissibility.json", "w"), indent=1)

f = lambda x: "--" if x is None else f"{x:.2g}"
print("Table IV: worst case over all runs of each protocol family")
print(f"{'protocol':15s} {'RWA inter':>10s} {'RWA intra':>10s} {'sel inter':>10s} {'sel intra':>10s} {'bandwidth':>10s}")
for name, r in [(k, table[k]) for k in ("Xi conversion", "Xi response", "V response", "V beat")]:
    bw = f">= {r['bandwidth_min']:#.3g}" if "bandwidth_min" in r else f"{r['bandwidth']:#.3g}"
    print(f"{name:15s} {f(r['rwa_inter']):>10s} {f(r.get('rwa_intra')):>10s} {f(r['sel_inter']):>10s} "
          f"{f(r.get('sel_intra')):>10s} {bw:>10s}")
print(f"\nXi conversion: {len(runs)} runs; with the 2x Omega12 linearity control included, "
      f"RWA inter {xcl['rwa_inter']:.1e}, selectivity inter {xcl['sel_inter']:.1e}")
print(f"largest RWA ratio {table['max_rwa']:.3f}, largest selectivity ratio {table['max_selectivity']:.3f} "
      f"(THz leg at 5 meV; 1s->3p at {s32['detuning_meV']:.1f} meV)")
