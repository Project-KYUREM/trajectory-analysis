# Kyurem — Neptune multiple-gravity-assist trajectory explorer

Searches Earth→Neptune gravity-assist trajectories across flyby sequences and
propulsion models with pykep/pygmo, then publishes them as a static site where
every trajectory can be inspected in 3D and exported for GMAT/STK.

## Setup

```bash
conda env create -f environment.yml      # creates orbital_env
conda activate orbital_env
pip install -e . --no-deps                # editable install of the aerocapture package (environment.yml does this too)
scripts/fetch_kernels.sh                  # SPICE kernels and the Neptune-GRAM table into data/ (~225 MB, gitignored)
python -m pytest                          # aerocapture tests (core, neptune, triton)
```

The editable install makes `aerocapture` importable from any working directory,
so the `# %%` cell scripts run in an interactive window or as files.

## Pipeline

```bash
# 1. search: the cost / time-of-flight front per (model, sequence) -> results/<model>/<sequence>/
python orbit_planning/batch.py --models capture,ballistic \
    --sequences EVEEJN,EVEJN,EVJN,EVVEJN,EJN --sweep --polish 24
#    (anything batch.py does not recognise is forwarded to pareto.py)

# 2. publish: compact every run found under results/ into docs/data/
python orbit_planning/publish.py

# 3. view: any static server; GitHub Pages serves docs/ directly
python -m http.server -d docs 8899
```

Single runs: `python orbit_planning/optimize.py --model ballistic --sequence EJN`
finds the one cheapest trajectory (its `champion.json` feeds the notebook, not
the site); `pareto.py` finds the whole front, which is what gets published.
Both take `--sweep` (one search per synodic launch slice) and `--help`.

## Models (`--model`)

| model | pykep problem | cost axis |
|-------|---------------|-----------|
| `capture`   | mga_1dsm + insertion into an e=0.99 ellipse | deep-space maneuvers + capture burn (m/s) |
| `ballistic` | mga, gravity assists only, flyby arrival     | departure Δv beyond the launch allowance (m/s) |
| `nep`       | mga_lt_nep, 40 kWe nuclear-electric          | builds and runs but is not decoded or published; the NEP path is "export a `ballistic`/`capture` GMAT seed and convert the impulses to finite burns in GMAT" |

Sequences are strings over the body registry (`V E M J S N`), Earth first,
destination last: `EVEEJN`, `EVEJN`, `EVJN`, `EVVEJN`, `EJN`, …

## Layout

| file | role |
|------|------|
| `orbit_planning/paths.py`       | filesystem layout + default matrix (kernel-free) |
| `orbit_planning/mission.py`     | SPICE kernels, body registry, sequences, launch-window slicing |
| `orbit_planning/models.py`      | the models: build / decode / cost / objective, and `problem()` |
| `orbit_planning/search.py`      | archipelago, MBH polish, parallel polish pool, shared CLI |
| `orbit_planning/optimize.py`    | single-objective search (+ `--sweep`) |
| `orbit_planning/pareto.py`      | NSGA-II front (+ `--sweep`, `--polish`) |
| `orbit_planning/trajectory.py`  | decision vector → mission description with arc states |
| `orbit_planning/batch.py`       | run many pairs, resumable |
| `orbit_planning/publish.py`     | raw results → `docs/data/` |
| `orbit_planning/porkchop.py`    | static PNG of one run's cloud |
| `orbit_planning/slingshot_planning.py` | notebook for poking at one champion |
| `docs/`                         | the site: `common.js`, `run.js`, `gallery.js`, `kepler.js` |
| `pyproject.toml`                | the `aerocapture` package (`pip install -e .`) and pytest settings |
| `aerocapture/core/system.py`, `paths.py` | Neptune-system kernels, constants, NEQ frame, Triton circular fit; data and output directories |
| `aerocapture/core/atmosphere.py`, `heating.py`, `dynamics.py`, `arrival.py` | Neptune-GRAM atmosphere, heating fits, equations of motion and conics, arrival hyperbola |
| `aerocapture/core/ode.py` | shared OpenMDAO component helpers, `AltitudeComp` |
| `aerocapture/core/guess.py` | guess numerics: float dynamics, propagation and events, shooting, segment spacing |
| `aerocapture/core/solve.py` | `DymosProblem` base, IPOPT driver, staged-solve helpers, active-box-bound check |
| `aerocapture/core/config_base.py` | `ConfigBase` of both mission configs: validation, opt-flag rule, margins-and-assumptions registry, record, echo |
| `aerocapture/core/io.py` | shared reporting: `solution.json`, CSV and text tables, solver/constants records, active-constraint test, assumptions-and-margins block |
| `aerocapture/core/plotting.py` | shared matplotlib primitives (styles, limit lines, overlays, save) |
| `aerocapture/neptune/config.py`, `vehicles.py` | Neptune aerocapture: the `Config` of every input; entry-vehicle presets |
| `aerocapture/neptune/ode.py`, `guess.py`, `problem.py`, `case.py` | Dymos ODEs, initial guess, multi-phase problem, staged solve |
| `aerocapture/neptune/results.py`, `plots.py` | summary and `solution.json`, figures |
| `aerocapture/neptune/run_neptune.py`, `sweep_neptune.py`, `trade_neptune.py` | `# %%` cell scripts: one case, v∞ × Triton-mode sweep, vehicle trade |
| `aerocapture/neptune/PLAN.md`   | design, inputs, findings |
| `aerocapture/triton/config.py`, `ballute.py` | Triton ballute capture: the `Config` of every input; ballute/probe records, drag and heating |
| `aerocapture/triton/frame.py`, `atmosphere.py` | Triton-centred frame and gravity, hydrostatic N₂ atmosphere |
| `aerocapture/triton/ode.py`, `guess.py`, `problem.py`, `case.py` | Dymos ODEs and components, patched-conic and backward-shooting guess, multi-phase problem, staged solve (S0–S3) |
| `aerocapture/triton/results.py`, `plots.py` | summary, checks and `solution.json`, figures |
| `aerocapture/triton/run_triton.py`, `sweep_triton.py`, `trade_triton.py` | `# %%` cell scripts: one case, v∞ × rp band × encounter sweep, ballute trade |
| `aerocapture/triton/PLAN.md`    | design, inputs, findings |
| `aerocapture/tests/`            | pytest checks: `core/`, `neptune/`, `triton/` (`python -m pytest`) |

## Design notes

- **Front polish.** Raw NSGA-II fronts run 10–50 % high. `pareto.py` MBH-polishes
  a time-of-flight-spanning sub-sample of the front (epsilon-constraint: each
  point pinned to its own launch date and flight time), appends the results and
  recomputes dominance over the union.
- **Spawn guard.** pygmo drives these Python problems via mp_island, which forces
  multiprocessing *spawn*; spawn re-imports `__main__`, so every search lives in
  a script behind `if __name__ == '__main__'`, never in a notebook cell.
- **Kernels at import.** A pickled `planet.spice` resolves against the *child's*
  kernel pool, so `mission.py` loads kernels on import; every worker gets them.
- **Decision vectors.** `x[0]` is the launch epoch (MJD2000) for every model and
  is the only gene read outside `models.py`; the per-model layouts are
  documented at the top of that file and reached through `Model.decode`.
- **Result files** carry `schema: 2`; `publish.py` skips anything older.
- **In-browser propagation.** The site receives arc states (r, v, Δt, μ) and
  rebuilds trajectories with a universal-variable Kepler propagator
  (`docs/kepler.js`, validated against pykep to machine precision). No server.
- **Exports** (CSV of arc states in ecliptic and equatorial J2000, GMAT seed) are
  generated client-side from the published description.

## Aerocapture (Dymos)

Two missions share `aerocapture/core/` and the same workflow; each is solved as
one multi-phase Dymos problem with IPOPT.

| | Neptune aerocapture (`aerocapture/neptune/`) | Triton ballute capture (`aerocapture/triton/`) |
|---|---|---|
| trajectory | hyperbolic arrival → optional Triton flyby (`none`, `inbound`, `outbound`) → one bank-modulated pass through Neptune's atmosphere → minimal capture | arrival → close Neptune flyby (1.10–1.25 R_N, ring-plane crossings outside 38 000–85 000 km) → ~10 h cruise → ballute pass in Triton's N₂ atmosphere with a free-time release → periapsis-raise burn → stable retrograde Triton orbit |
| controls | B-plane aim, bank angle, v∞ direction or periapsis time | B-plane aim, v∞ direction or periapsis time, release time, burn vector |
| default objective | peak stagnation heat rate | peak stagnation heat rate on the ballute |
| design, inputs, findings | `aerocapture/neptune/PLAN.md` | `aerocapture/triton/PLAN.md` |

### Workflow (both missions)

Each mission has three `# %%` cell scripts with the same cells in the same
order. `run_<mission>.py`: imports → CONFIG → initial guess (+ guess figures)
→ solve (staged, then the simulate check) → summary → figures → save.
`sweep_<mission>.py` and `trade_<mission>.py`: imports → BASE (every input,
the per-case fields marked) → SWEEP / TRADE lists → CASES (the only overrides
of BASE) → run → table → figure.

Every input is a field of the mission's `config.Config`, spelled out in the
CONFIG cell with value, unit and meaning, tagged [physics] (changes the
problem), [numerics] (changes how it is solved), [guess] (only shapes the
starting point) or [margin] (see below). Field names are shared where the
concept is shared (`vinf_kms`, `direction`, `opt_direction`, `opt_t_peri`,
`objective`, `heat_weight`, `g_max`, `q_max_w_cm2`, `sim_tol`, `order`,
`segments`, `max_iter`, `tol`, `print_level`, `tag`, …). The opt flags
(None = automatic, resolved values echoed and saved) choose what of the
arrival is optimised besides the B-plane:

| flag | Neptune automatic value | Triton automatic value |
|---|---|---|
| `opt_direction` | True when `direction="free"` and `triton_mode != "none"` | True when `direction="free"` |
| `opt_t_peri` | True when `direction != "free"` and `triton_mode != "none"` | True when `direction != "free"` |

### Margins and assumptions

Every safety or design margin, assumed model value, objective weight and check
tolerance that shapes the answer sits in one CONFIG group, "margins and
assumptions" ([margin]), with its unit and a one-line reason and cost in the
comment; `config.ASSUMPTIONS` holds the same list. The echo marks these fields,
and every printed summary ends with an "Assumptions & margins in effect" block:
each entry with its value and reason, the fixed model assumptions (`NOTES` in
`results.py`), the inequality constraints active at the solution, and any
numerical box bound the solver hit (a warning: an unstated constraint). The
same lists are saved in `solution.json` (`assumptions`, `active_constraints`,
`active_box_bounds`). Neptune: `triton_alt_min_km`, `triton_clear_km`,
`triton_exit_km`, `ra_target_hill`, `min_inclination_deg`, `rp_raise_km`,
`rad_coupling`, `rad_kappa`, `rad_rn_exp`, `heat_weight`, `sim_tol`. Triton:
`h_min_km`, `rp_final_alt_km`, `final_rp_margin_km`, `ra_max_hill`,
`final_ra_margin`, `i_retro_min_deg`, `i_pro_max_deg`, `atm_density_scale`,
`fm_accommodation`, `rad_in_limits`, `heat_weight`, `dv_weight`,
`stability_days`, `sim_tol`, `sim_rp_tol_km`.

Running several cases at once: set `OMP_NUM_THREADS=1`, or IPOPT/BLAS threads
oversubscribe the cores.

### Neptune aerocapture

```bash
python aerocapture/neptune/run_neptune.py     # one case -> results/aerocapture/neptune/<tag>/
python aerocapture/neptune/sweep_neptune.py   # v∞ × Triton mode -> results/aerocapture/neptune/sweep/<name>/
python aerocapture/neptune/trade_neptune.py   # presets × β × L/D × R_n × v∞ × mode -> .../trade/<name>/
```

Physical records in the CONFIG cell: the vehicle is a preset (placeholder
numbers) with its tuned fields overridden in the open.

```python
VEHICLE = replace(PRESETS["heritage-70deg"], beta_kg_m2=200.0, lift_to_drag=0.4,
                  nose_radius_m=0.5, tps="ablative", emissivity=0.85,
                  t_wall_max_k=None, q_max_w_cm2=None, g_max=20.0)
CONFIG = config.Config(vinf_kms=5.0, direction="free", triton_mode="inbound",
                       vehicle=VEHICLE, objective="qpeak", ...,
                       triton_alt_min_km=300.0, min_inclination_deg=90.0, ...)
```

Staged solve (`case.solve`): free direction from the guess (warm restarts) →
for a fixed direction, a direct solve or a continuation of the free asymptote
in ≤ 10° steps → simulate check (final apoapsis within `sim_tol`).

Outputs, `results/aerocapture/neptune/<tag>/`: `solution.json` (inputs with
the opt flags resolved, constants, arrival, Triton encounter, entry/exit,
loads, coupled and uncoupled heating, final orbit, solver stages, simulate
check, assumptions and margins, active constraints, warnings); figures
`guess_system`, `guess_atmosphere`, `system`, `atmosphere`, `corridor` and
`triton` (Triton modes), with the guess dotted and the simulation dashed on
the solution figures; the Dymos `.db` files. Sweeps and trades add
`<name>.csv` (columns in `aerocapture.neptune.results.COLUMNS`) and
`<name>.png`.

Notes:

- The target is minimal capture: at 0.1 R_Hill the exit is ~25 m/s below
  escape; final ra, rp, period and the periapsis-raise Δv are reported, not
  constrained.
- The free-direction optimum sits on `min_inclination_deg` (90°): a polar orbit
  minimises the airspeed against the rotating atmosphere.
- `triton_mode="both"` is rejected: a same-pass inbound + outbound flyby is
  geometrically impossible.

### Triton ballute capture

```bash
python aerocapture/triton/run_triton.py      # one case -> results/aerocapture/triton/<tag>/
python aerocapture/triton/sweep_triton.py    # v∞ × rp band × encounter -> results/aerocapture/triton/sweep/<name>/
python aerocapture/triton/trade_triton.py    # ballute size / mass / C_D / release / density -> .../trade/<name>/
```

Physical records in the CONFIG cell: the ballute and the probe.

```python
BALLUTE = Ballute(name="ead-70", diameter_m=70.0, envelope_mass_kg=330.0,
                  cd_continuum=0.9, cd_free_molecular=2.0, cd_model="wilmoth",
                  emissivity=0.85, radiating_sides=2, t_film_max_k=773.0)
PROBE = Probe(mass_kg=200.0, cd_area_m2=2.0, nose_radius_m=0.5)
CONFIG = config.Config(vinf_kms=5.0, direction="free", scope="full",
                       ballute=BALLUTE, probe=PROBE, objective="qpeak", ...,
                       rp_final_alt_km=1000.0, final_rp_margin_km=50.0, ...)
```

Staged solve (`case.solve`): S0 Triton-only problem (`scope="triton_only"`
stops here; the engine of ballute trades) → S1 full problem with a ≤ 0.2 km/s
post-flyby TCM → S2 TCM removed (halving continuation on failure) → S3
fixed-direction continuation when `direction != "free"` → simulate check and
`stability_days` tidal propagation.

Outputs, `results/aerocapture/triton/<tag>/`: `solution.json` (inputs,
constants and atmosphere, arrival, Neptune periapsis with ring crossings and
the GRAM drag estimate, Triton encounter with the direct-intercept comparison,
ballute/probe pass with every heating term, film temperature, release, burn,
final orbit, stability, corridor, solver stages, simulate check, assumptions
and margins, active constraints, warnings); figures `guess_system`,
`guess_neptune_flyby`, `guess_triton`, `guess_pass`, `guess_corridor`,
`system`, `neptune_flyby` (full scope), `triton`, `pass`, `pass_ballute`,
`corridor` and `stability`; the Dymos `.db` files. Sweeps and trades add
`<name>.csv` (columns in `aerocapture.triton.results.COLUMNS`) and
`<name>.png`.

Notes:

- The close Neptune pass costs ~4.4 km/s of Triton entry speed against a
  direct tangent intercept (8.2 vs 3.75 km/s at v∞ = 5); every summary prints
  this comparison.
- The Tauber–Sutton air radiation (linear in R_n) is an upper bound far outside
  its fit range: reported every run, constrained only with `rad_in_limits=True`.

## Aerocapture references

Sources behind `aerocapture/`: the shared Neptune system, atmosphere, heating and
software first, then the Triton-capture sources. The AMAT Neptune-GRAM table is GPL-3.0, so
`scripts/fetch_kernels.sh` downloads it into `data/` instead of committing it
to this MIT repo.

### Neptune system and ephemerides

- NAIF generic kernels (JPL): `de440.bsp`, `nep097.bsp`, `pck00011.tpc`,
  `gm_de440.tpc`, `naif0012.tls`. https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
  — planetary ephemeris, Triton (801) ephemeris for the circular-orbit fit,
  Neptune pole/rotation/radii for the frame, GM of Neptune (899) and Triton (801),
  leap seconds. Fetched by `scripts/fetch_kernels.sh`.
- R. A. Jacobson, "The Orbits of the Neptunian Satellites and the Orientation
  of the Pole of Neptune", *Astronomical Journal* 137(5), 4322–4329, 2009.
  https://doi.org/10.1088/0004-6256/137/5/4322 — Neptune J2 = 3408.428e-6 at
  reference radius 25 225 km (Table 5), used by `system.py`.
  - Note: AMAT `planet.py` pairs J2 = 3411e-6 (the NSSDC Neptune fact-sheet
    value) with R = 24 622 km. Rescaled to 24 622 km, Jacobson's J2 is
    3577e-6, so the AMAT pairing is ~5 % low.

### Atmosphere model

- A. P. Girija et al., AMAT (below): `atmdata/Neptune/neptune-gram-avg.dat`,
  the Neptune-GRAM mean density profile read by `atmosphere.py`.
  (Which Neptune-GRAM version AMAT tabulated is unverified.)
- C. G. Justus, A. Duvall, D. L. Johnson, "Engineering-Level Model Atmospheres
  for Titan and Neptune", AIAA Joint Propulsion Conference, Huntsville AL, 2003.
  https://ntrs.nasa.gov/citations/20030066242 — original Neptune-GRAM.
- C. G. Justus, A. L. Duvall, V. W. Keller, "Atmospheric Models for
  Aerocapture", 40th AIAA/ASME/SAE/ASEE Joint Propulsion
  Conference, 2004. https://ntrs.nasa.gov/citations/20040085958 — Neptune-GRAM
  in the NASA aerocapture studies.
- H. L. Justh, J. Hoffman, "Neptune Global Reference Atmospheric Model
  (Neptune-GRAM): User Guide", NASA/TM-20205001193, 2020.
  https://ntrs.nasa.gov/citations/20205001193 — current Neptune-GRAM; source
  for future min/max or dispersed profiles.

### Aerocapture mission studies and vehicles

- A. P. Girija, S. J. Saikia, J. M. Longuski, S. Bhaskaran, M. S. Smith,
  J. A. Cutts, "Feasibility and Performance Analysis of Neptune Aerocapture
  Using Heritage Blunt-Body Aeroshells", *J. Spacecraft and Rockets* 57(6),
  1186–1203, 2020. https://doi.org/10.2514/1.A34719 — L/D 0.3–0.4
  heritage-vehicle feasibility and corridor width; `heritage-70deg` preset.
- M. K. Lockwood et al., "Aerocapture Systems Analysis for a Neptune Mission",
  NASA/TM-2006-214300, 2006. https://ntrs.nasa.gov/citations/20060012088 —
  mid-L/D ellipsled; `ellipsled` preset and entry-speed context.
- M. K. Lockwood, "Neptune Aerocapture Systems Analysis", AIAA 2004-4951, 2004.
  https://ntrs.nasa.gov/citations/20040111217 — conference summary of the above.
- B. R. Starr, C. H. Westhelle, J. P. Masciarelli, "Aerocapture Performance
  Analysis for a Neptune-Triton Exploration Mission", AIAA 2004-4955, 2004.
  https://ntrs.nasa.gov/citations/20040111218 — companion guidance/performance
  study for the same mission.
- R. E. Dyke, G. A. Hrinda, "Structural Design for a Neptune Aerocapture
  Mission", AIAA 2004-5179, 2004. https://ntrs.nasa.gov/citations/20040111219
  — companion structures study; vehicle mass context.
- C. R. Heidrich, S. Dutta, R. D. Braun, "Modern Aerocapture Guidance to Enable
  Reduced-Lift Vehicles at Neptune", AAS 19-221, 2019.
  https://ntrs.nasa.gov/citations/20200002455 — low-L/D vehicles at Neptune.
- J. M. DiNonno, F. M. Cheatwood, "Low-Earth Orbit Flight Test of an Inflatable
  Decelerator (LOFTID) Mission Overview and Science Return", AIAA SciTech 2024.
  https://ntrs.nasa.gov/citations/20230017347 — `loftid` inflatable preset
  (LOFTID flew Nov 2022).

### Aerothermal heating correlations

- B. Bienstock et al., NASA Vision Mission "Neptune Orbiter with Probes",
  Contract NNH04CC41C final report, 2005 — Neptune convective fit
  `2.24008e-7 ρ^0.45213 v^2.6918 √(0.291/Rn)` and radiative fit
  `8.125812e-3 ρ^0.49814 (v/10 km/s)^15.113 (Rn/0.291)` W/cm² in
  `heating.py`, taken from AMAT `vehicle.py` (`qStagConvective`,
  `qStagRadiative`). The report itself was not found publicly, so the fits are
  unverified against it. Related public paper: B. Bienstock et al., "Neptune
  Polar Orbiter with Probes", 2005, https://ntrs.nasa.gov/citations/20070014629.
- K. Sutton, R. A. Graves Jr., "A General Stagnation-Point Convective-Heating
  Equation for Arbitrary Gas Mixtures", NASA TR R-376, 1971.
  https://ntrs.nasa.gov/citations/19720003329 — H₂–He constant for the
  alternative convective set.
- J. A. Fay, F. R. Riddell, "Theory of Stagnation Point Heat Transfer in
  Dissociated Air", *J. Aeronautical Sciences* 25(2), 73–85, 1958.
  https://doi.org/10.2514/8.7517 — the theory Sutton–Graves fits; background,
  not implemented.
- NASA TFAWS 2012 Aerothermodynamics Course, Lecture 1 "Stagnation Point
  Heating". https://tfaws.nasa.gov/TFAWS12/Proceedings/Aerothermodynamics%20Course.pdf
  — Fay–Riddell / Sutton–Graves background, radiative-equilibrium wall
  temperature, Goulard/Tauber–Wakefield coupling (κ = 2 Titan, 3.45 Earth,
  ~3 Mars/Venus). Notes that tabulated engineering radiation fits usually
  already include radiative cooling (the double-counting question).
  (Lecturer not named in the PDF.)

### Radiative–convective coupling

- R. Goulard, "The Coupling of Radiation and Convection in Detached Shock
  Layers", *J. Quant. Spectrosc. Radiat. Transfer* 1(3–4), 249–257, 1961.
  https://doi.org/10.1016/0022-4073(61)90025-5 — Goulard number
  Γ = 2 q_rad,unc / (½ ρ∞ V∞³).
- R. Goulard, "Preliminary Estimates of Radiative Transfer Effects on Detached
  Shock Layers", *AIAA Journal* 2(3), 494–502, 1964.
  https://doi.org/10.2514/3.2359 — radiative cooling of the shock layer.
- M. E. Tauber, R. M. Wakefield, "Heating Environment and Protection during
  Jupiter Entry", *J. Spacecraft and Rockets* 8(6), 630–636, 1971
  (AIAA 70-1324). https://doi.org/10.2514/3.59703 — coupled/uncoupled ratio
  1/(1 + κΓ^0.7) applied with `rad_coupling="goulard"` (Neptune).
- B. R. Hollis, S. A. Striepe, M. J. Wright, D. Bose, K. Sutton,
  N. Takashima, "Prediction of the Aerothermodynamic Environment of the Huygens
  Probe", AIAA 2005-4816, 38th AIAA Thermophysics Conference, 2005.
  https://doi.org/10.2514/6.2005-4816 — states κ depends on gas composition
  (3.0 reported, 2.0 fits Titan); reason `rad_kappa` is a [margin] input.
- J. D. Anderson Jr., *Hypersonic and High-Temperature Gas Dynamics*, 2nd ed.,
  AIAA Education Series, 2006. https://doi.org/10.2514/4.861956 — textbook
  treatment of radiative gas dynamics and coupling.

### Software

- A. P. Girija, S. J. Saikia, J. M. Longuski, J. A. Cutts, "AMAT: A Python
  package for rapid conceptual design of aerocapture and atmospheric Entry,
  Descent, and Landing (EDL) missions in a Jupyter environment", *JOSS* 6(67),
  3710, 2021. https://doi.org/10.21105/joss.03710 ·
  https://github.com/athulpg007/AMAT · https://amat.readthedocs.io/ (GPL-3.0)
  — GRAM table and heating-fit implementations; cross-check case.
- R. Falck, J. S. Gray, K. Ponnapalli, T. Wright, "dymos: A Python package for
  optimal control of multidisciplinary systems", *JOSS* 6(59), 2809, 2021.
  https://doi.org/10.21105/joss.02809 — multi-phase collocation.
- J. S. Gray, J. T. Hwang, J. R. R. A. Martins, K. T. Moore, B. A. Naylor,
  "OpenMDAO: an open-source framework for multidisciplinary design, analysis,
  and optimization", *Struct. Multidisc. Optim.* 59(4), 1075–1104, 2019.
  https://doi.org/10.1007/s00158-019-02211-z — component framework and
  derivatives.
- E. Wu, G. Kenway, C. A. Mader, J. Jasa, J. R. R. A. Martins, "pyOptSparse: A
  Python framework for large-scale constrained nonlinear optimization of sparse
  systems", *JOSS* 5(54), 2564, 2020. https://doi.org/10.21105/joss.02564 —
  optimizer interface.
- A. Wächter, L. T. Biegler, "On the implementation of an interior-point filter
  line-search algorithm for large-scale nonlinear programming", *Math.
  Programming* 106(1), 25–57, 2006. https://doi.org/10.1007/s10107-004-0559-y
  — IPOPT, the NLP solver.
- cyipopt, Python wrapper for IPOPT. https://github.com/mechmotum/cyipopt —
  required by pyOptSparse to reach IPOPT.

### Triton capture

Sources behind `aerocapture/triton/` (the Neptune-system sources above apply too).

- N. Brisby, J. E. Lyne, "Triton Aerogravity Assist Using a Flight-Proven,
  Inflatable Aerobrake for Neptune Orbit Capture", *J. Spacecraft Rockets*
  61(4), 1124–1127, 2024. https://doi.org/10.2514/1.A35964 — direct-intercept
  Triton entry speeds (4.01–9.41 km/s), unit-tested.
- D. F. Strobel, X. Zhu, "Comparative planetary nitrogen atmospheres: Density
  and thermal structures of Pluto and Triton", *Icarus* 291, 55–64, 2017.
  https://doi.org/10.1016/j.icarus.2017.03.013 — surface pressure and
  temperature of the hydrostatic model.
- B. Benne et al., "A photochemical model of Triton's atmosphere paired with an
  uncertainty propagation study", *A&A* 667, A169, 2022.
  https://doi.org/10.1051/0004-6361/202244447 — Triton-3 profile, p(1026 km)
  anchor of the Bates scale L.
- B. Benne et al., "Impact of the transport of magnetospheric electrons on the
  composition of the Triton atmosphere", *A&A* 686, A22, 2024.
  https://doi.org/10.1051/0004-6361/202346699 —
  n_N₂ = 4.0 ± 0.4e8 cm⁻³ at 575 km (Voyager UVS), independent check.
- J. Marques Oliveira et al., "Constraints on the structure and seasonal
  variations of Triton's atmosphere from the 5 October 2017 stellar occultation
  and previous observations", *A&A* 659, A136, 2022. https://doi.org/10.1051/0004-6361/202141443 — 1.18 ± 0.03 µbar at 47 km.
- B. Sicardy et al., Triton occultations 1989–2022, arXiv:2402.02476, 2024 —
  surface pressure 14–16 µbar.
- Global Aerospace Corp., "Pluto Hop, Skip, and Jump", NIAC Phase I final
  report, 2018, NTRS 20180007811; Phase II report, NTRS 20250000440 — enveloping
  ballute concept, film temperature calibration (3.5 W/cm² → 520–550 °C),
  LAURA radiation finding, g range.
- K. Rohrschneider, R. D. Braun, "Survey of Ballute Technology for
  Aerocapture", *J. Spacecraft Rockets* 44(1), 10–23, 2007.
  https://doi.org/10.2514/1.19288
- Z. R. Putnam, R. D. Braun, "Drag-Modulation Flight-Control System Options for
  Planetary Aerocapture", *J. Spacecraft Rockets* 51(1), 139–150, 2014.
  https://doi.org/10.2514/1.A32589 — single-stage jettison drag modulation.
- R. G. Wilmoth, R. A. Mitcheltree, J. N. Moss, "Low-Density
  Aerodynamics of the Stardust Sample Return Capsule", *J. Spacecraft Rockets*
  36(3), 436–441, 1999. https://doi.org/10.2514/2.3464 — sine-squared
  Knudsen bridging of C_D.
- K. Sutton, R. A. Graves, "A General Stagnation-Point Convective-Heating
  Equation for Arbitrary Gas Mixtures", NASA TR R-376, 1971 — N₂ constant
  1.7407e-4 (SI) as in AMAT `vehicle.py`.
- M. E. Tauber, K. Sutton, "Stagnation-point radiative heating relations for
  Earth and Mars entries", *J. Spacecraft Rockets* 28(1), 40–42, 1991.
  https://doi.org/10.2514/3.26206 — radiative upper bound and f(V) table.
- M. Pedersen, N. Meyer-Vernet, M. G. Aubier, P. Zarka, "Dust distribution
  around Neptune: Grain impacts near the ring plane measured by the Voyager
  Planetary Radio Astronomy Experiment", *J. Geophys. Res.* 96, 19187–19196,
  1991. https://doi.org/10.1029/91JA01601 — Voyager 2 ring-plane crossings.
- D. P. Hamilton, J. A. Burns, "Orbital stability zones about asteroids",
  *Icarus* 92(1), 118–131, 1991.
  https://doi.org/10.1016/0019-1035(91)90039-V
- R. C. Domingos, O. C. Winter, T. Yokoyama, "Stable satellites around
  extrasolar giant planets", *MNRAS* 373(3), 1227–1234, 2006.
  https://doi.org/10.1111/j.1365-2966.2006.11104.x — prograde 0.49 /
  retrograde 0.93 R_H limits.
- D. J. Scheeres, M. D. Guman, B. F. Villac, "Stability Analysis of Planetary
  Satellite Orbiters: Application to the Europa Orbiter", *J. Guid. Control
  Dyn.* 24(4), 778–787, 2001. https://doi.org/10.2514/2.4778 — Kozai band of
  satellite orbiters.
- Y. Kozai, "Secular perturbations of asteroids with high inclination and
  eccentricity", *Astron. J.* 67, 591, 1962. https://doi.org/10.1086/108790
