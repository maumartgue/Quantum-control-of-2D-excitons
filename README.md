# Dark-state coherent control of excitons in WSe₂: codes

Codes for the paper

> **Conditions for Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley Lindblad Framework for WSe₂**
>
> Mauricio Martinez and Martin A. Mosquera, [arXiv:2611.07315](https://arxiv.org/abs/2611.07315) (2026)

The first arXiv version (v1) was titled "Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley Lindblad Framework for WSe₂". See [Versions of the paper](#versions-of-the-paper) for the numbering of sections, appendices and equations.

With these scripts, every number, table and figure of the results can be regenerated from the model parameters alone. The system is hBN-encapsulated monolayer WSe₂ with two three-level configurations:
- the ladder **Ξ** (|0⟩, 1s, 2p), with an optical and a THz leg;
- the **V** (|0⟩, 1s, 2s), with two optical legs.

The pipeline:
1. the Wannier exciton problem, solved by Rayleigh–Ritz with the Rytova–Keldysh potential;
2. the solver inputs (`params.yaml`);
3. the Lindblad master equation in the one-exciton Fock space;
4. the figures and numbers;
5. the sensitivity analyses outside the model: pure dephasing and population decay of the 2p exciton, which give the conditions a real monolayer must meet (Sect. VI G and App. G of v2).

## Requirements

- Python ≥ 3.9
- numpy ≥ 2.0 (the scripts use `numpy.trapezoid`)
- scipy ≥ 1.6
- PyYAML
- matplotlib
- h5py
- optional: QuTiP. One cross-check of `appE_validation.py` (V7) uses it and is skipped if it is not installed.

```
python -m pip install numpy scipy pyyaml matplotlib h5py
```

All scripts are in one folder and import each other by name. Run them from that folder.

## Quick start

`params.yaml` is shipped, so every Lindblad script runs directly:

```
python fig3_R1.py
```

To regenerate everything from scratch, run the scripts in this order. The run times were measured on a desktop PC.

| Step | Command | Time |
|---|---|---|
| 1 | `python exchange_mean_field.py keldysh` | ~40 s |
| 2 | `python exchange_mean_field.py coulomb` | ~20 s |
| 3 | `python solver_inputs_from_wannier.py` | ~5 s |
| 4 | `python fig3_R1.py` | ~10 s |
| 5 | `python fig4_R1b.py` | ~10 s |
| 6 | `python fig5_R3.py` | ~1.5 min |
| 7 | `python fig6_R2.py` | ~1.5 min |
| 8 | `python fig7_R4.py` | ~2.5 min |
| 9 | `python fig8_R5.py` | ~20 s |
| 10 | `python sect6_numbers.py` | ~40 s |
| 11 | `python table4_admissibility.py` | < 1 s |
| 12 | `python appE_validation.py` | ~30 s |
| 13 | `python analytic_estimates.py` | ~5 s |
| 14 | `python appD6_exchange_checks.py` | ~20 s |
| 15 | `python exciton_spectrum.py` | ~10 s |
| 16 | `python rayleigh_ritz_2d.py` | ~5 s |
| 17 | `python appG_dephasing_relaxation.py` | ~4 min |

Where the order matters:
- Step 1 writes the vertex file that step 3 and `sect6_numbers.py` read.
- Step 2 writes the 2D-hydrogen vertex file that `appD6_exchange_checks.py` reads.
- `table4_admissibility.py` reads the outputs of `fig5_R3.py`, `fig6_R2.py` and `sect6_numbers.py`.
- `appG_dephasing_relaxation.py` needs only `params.yaml`; its tasks `halfpoint` and `figure` read the outputs of its own earlier tasks, so run it without arguments (all tasks, in order) or keep the order of its usage line.

Each script prints its numbers, and the figure scripts also write a PDF and a PNG.

## What each file does

**Shared modules**

| File | Content |
|---|---|
| `rayleigh_ritz_2d.py` | Rayleigh–Ritz Wannier solver (App. D). One angular-momentum sector per run, a multi-scale even-tempered 2D hydrogenic basis, and a single generalized eigenproblem. Run alone, it validates against the exact 2D Coulomb spectrum and prints the WSe₂/hBN spectrum. |
| `solver_indep.py` | Independent radial-shooting solver for the same Keldysh problem, used by the ionization estimate. |
| `lindblad_solver.py` | Rotating-frame three-level Lindblad solver for the Ξ and V three-level systems. It contains the master equation of the paper with the photon dissipator and the exchange mean field as level shifts. Other features: piecewise DOP853 integration, the Liouvillian null-space steady state, and HDF5 output. |

**Inputs**

| File | Content | Paper |
|---|---|---|
| `exchange_mean_field.py` | Exciton-gas exchange vertices ζ and the mean field Σ(n_X), χ. Writes `vertices_<model>_….json` and `propagated.json`. | Sect. V A, App. D 6, Tables IX–X |
| `solver_inputs_from_wannier.py` | Computes every solver input and writes `params.yaml` (see below). | Sects. IV–V, App. C |
| `params.yaml` | The input file of all Lindblad scripts (see below). | — |

**Tables and appendix checks**

| File | Content | Paper |
|---|---|---|
| `exciton_spectrum.py` | Exciton spectrum, optical primitives, configuration ranking, and the comparison with Stier et al. | Sect. IV, App. D, Tables I, II, V–VIII |
| `table4_admissibility.py` | Admissibility of the protocol families: rotating-wave, selectivity and bandwidth ratios. | Sect. VI A, Table IV |
| `appD6_exchange_checks.py` | Numerical checks of the exchange vertices: Wannier identity, sign, off-diagonal form, and the hydrogenic benchmark (check (v)). | App. D 6 |
| `analytic_estimates.py` | 2p ionization, intra-excitonic radiative rates, the V off-diagonal estimate, thermal momentum, the transparency-window depth, and the yield per incident photon. | Sects. V–VI, Apps. C–D |
| `appE_validation.py` | Validation of the Lindblad solver (V0–V7), the reference runs, and the amplitude-equation check of the conversion efficiency. | App. E |

**Figures of the results section, and the remaining numbers**

| File | Figure | Writes |
|---|---|---|
| `fig3_R1.py` | Fig. 3: time evolution of the Ξ conversion, cases 1–3 | `figR1.pdf/png`, `fig3_R1.json`, `fig3_R1_runs.npz` |
| `fig4_R1b.py` | Fig. 4: pulse-order control | `figR1b.pdf/png`, `fig4_R1b.json`, `fig4_R1b_runs.npz` |
| `fig5_R3.py` | Fig. 5: adiabaticity and loss channels | `figR3.pdf/png`, `fig5_R3.json` |
| `fig6_R2.py` | Fig. 6: two-photon detuning and density calibration | `figR2.pdf/png`, `fig6_R2.json` |
| `fig7_R4.py` | Fig. 7: CPT plateau and cascade response (EIT/Autler–Townes) | `figR4.pdf/png`, `fig7_R4.json`, `fig7_R4.h5`, `.npz` |
| `fig8_R5.py` | Fig. 8: the V system, response and quantum beat | `figR5.pdf/png`, `fig8_R5.json`, `fig8_R5.h5` |
| `sect6_numbers.py` | Numbers of Sect. VI that are not read off a figure: linearity, temperature, V-beat validity, window-centre calibration | `sect6_numbers.json` |

**Sensitivity analyses outside the model** (added with arXiv v2 and the NJP version; numbering of v2)

The fixed-K₀ model produces neither pure dephasing nor population decay of 2p beyond its radiative decay. This script adds both to the Ξ generator of `lindblad_solver.py` by hand and asks how the conversion and the transparency window respond. It writes its outputs into `appG_output/`.

| File | Content | Paper |
|---|---|---|
| `appG_dephasing_relaxation.py` | Pure dephasing of 2p, L_φ = √(2γ_φ)\|2p⟩⟨2p\| (Eq. G1), and population decay of 2p at γ_rel, into non-radiating states or back to 1s (Eq. G2), with the width split ħγ_rel/2 + ħγ_φ = 3.955 meV of the measured 2p line. Tasks: `sweep` (γ_φ from 0 to 4 meV, both pulse orders; window centre against the closed formula), `sigma` (σ = 1.1–1.9 ps; η halves at γ_φσ ≈ 1), `coherence_time` (evaluation time t_c = t₁₂ + 3σ of the coherence fraction), `halfpoint` (half of the 0–2p coherence lost at γ_φσ ≈ 0.12), `relax` (population decay: no 2p population survives when a quarter or more of the width is population decay; dark-sector fraction; half point γ_relσ ≈ 0.11), `independent` (re-integration with an own density-matrix right-hand side), `figure`. Every task runs its own checks (trace, positivity, convergence, γ = 0 equal to `xi_conversion`). | Sect. VI G, App. G, Table SVII (NJP) / XI (arXiv v2), Fig. S1 (NJP) / Fig. 9 (arXiv v2) |

It writes `gamma_phi_sweep.csv`, `gamma_phi_sigma.csv`, `coherence_time.csv`, `coherence_halfpoint.csv`, `relax_2p.csv`, `appG_summary.json` (checks, half points, the numbers of Table SVII / XI) and `fig_dephasing.pdf/png`. One task can be run alone, for example `python appG_dephasing_relaxation.py relax`.

Several scripts take an optional argument that selects one part, for example `python analytic_estimates.py ionization` or `python appE_validation.py lindblad`. The usage line at the top of each script lists them.

## `params.yaml`

`params.yaml` holds every input of the Lindblad solver. It is not a result of the simulations; it is generated by `solver_inputs_from_wannier.py` from:

- **the exciton spectrum** of the Rayleigh–Ritz Wannier solver `rayleigh_ritz_2d.py`: binding energies, envelopes at the origin and intraband dipoles, for hBN-encapsulated WSe₂ with the Rytova–Keldysh potential;
- **the material constants** in the script: reduced mass, carrier masses, screening length, dielectric environment, quasiparticle gap, the natural units E₀ and a_B, and the normalization area;
- **the exchange vertices** ζ of the exciton gas, read from `vertices_keldysh_200_8000_250_100.json`, which `exchange_mean_field.py` writes;
- **the 2p ionization coefficient** κ_ion = 0.304, computed by the ionization estimate (`analytic_estimates.py ionization`) and entered in the script as a constant;
- **the protocol choices** of the paper: pulse shapes, grids, temperatures and integrator settings.

From these it computes:
- the photon rates at 4, 77 and 300 K;
- the exchange shifts Σ(n_X);
- the laser calibration and mismatch detunings;
- the fields;
- the selectivity bounds.

Running `python solver_inputs_from_wannier.py` reproduces the shipped file, apart from the time stamp and round-off in the two orthonormality-error entries.

## Units

The paper writes the equations of the Lindblad solver (App. E) in the natural units of App. D:
- E₀ = 269 meV;
- ħ/E₀ = 2.45 fs;
- a_B = 1.19 nm.

`lindblad_solver.py` works in meV and ps; the results are the same. The photon rates are stored in `params.yaml` in both forms, each to six significant digits: `hbar_gamma_meV` and `dimensionless_gamma_tau0`. The Ξ conversion runs read the first; the V and EIT runs read the second, as stated in the scripts. This matches the runs of the paper.

## Reproducibility

- The outputs reproduce every number quoted in the paper at the precision at which it is quoted.
- Different Python, numpy and scipy versions can change the last digits of the results: typically by 10⁻¹¹ relative or less, and by up to about 10⁻⁴ in the adaptive-quadrature ionization estimate. Both are far below the quoted precision.
- Quantities that are themselves at the level of numerical noise can change by more in relative terms. Examples are the residuals of the EIT-model fits and Akaike weights of order 10⁻¹²; the paper quotes these only as bounds.
- Every Ξ run reports max_t (1 − ρ̄₁₁), the validity condition of the one-exciton truncation (Sect. VI A).

## Versions of the paper

The codes of steps 1–16 were released with arXiv v1, and the section, appendix, table and equation numbers in their comments and in the tables above refer to **v1**. arXiv v2 and the NJP version add an appendix with the Hamiltonians of the model at the start, so the appendices of v1 move by one letter:

| Content | v1 | v2 and NJP (Supplementary Material) |
|---|---|---|
| Hamiltonians of the model | Sect. III | App. A |
| Fixed-K₀ projection | App. A | App. B |
| Rotating-frame Lindblad equation | App. B | App. C |
| Bath correlators, rates and Lamb shifts | App. C | App. D |
| Rayleigh–Ritz Wannier solver | App. D | App. E |
| Numerical solver of the Lindblad dynamics | App. E | App. F |
| Dephasing and population decay of 2p | — | App. G |
| Spectral selectivity and the V-beat window | — | App. H |

Equation numbers also differ between the versions. The physics and every number are the same. `appG_dephasing_relaxation.py` uses the numbering of v2. In the NJP version, the tables and figures of the Supplementary Material carry an S (Table SVII, Fig. S1).

## Licence

MIT; see `LICENSE`.

## Citation

If you use these codes, please cite the paper:

M. Martinez and M. A. Mosquera, *Conditions for Dark-State Coherent Control of Excitons in TMD Monolayers: A Reduced Intravalley Lindblad Framework for WSe₂*, arXiv:2611.07315 (2026).

```bibtex
@misc{Martinez2026DarkState,
  author        = {Martinez, Mauricio and Mosquera, Martin A.},
  title         = {Conditions for Dark-State Coherent Control of Excitons in {TMD} Monolayers: A Reduced Intravalley {Lindblad} Framework for {WSe$_2$}},
  year          = {2026},
  eprint        = {2611.07315},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  url           = {https://arxiv.org/abs/2611.07315}
}
```
