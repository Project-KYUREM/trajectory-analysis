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
#    (also builds each model's combined front, docs/data/<model>/fronts.json)
python orbit_planning/publish.py

# 3. view locally: any static server
python -m http.server -d docs 8899

# 4. deploy: push docs/ (site + data) to the gh-pages branch as one snapshot
scripts/deploy_site.sh
```

**Site data.** `docs/data/` is generated and not committed to `main`. Each
run is `cloud.json` (metadata), `points.bin.gz` (every point of the cloud as
binary columns: launch, cost, time of flight, raw index, flags) and
`detail.json.gz` (trajectory descriptions); the format is documented in
`orbit_planning/sitedata.py`, and the page decompresses it in the browser.
A chemical run with 66 000 points is ~1.3 MB instead of ~11 MB as plain
JSON, with nothing dropped. Older single-file runs still load, and
`python orbit_planning/sitedata.py` (or any publish) converts them in place.
Publishing merges into the existing index, so runs published earlier (from
another machine) stay listed while their files remain in `docs/data/`.

**Deploying.** `scripts/deploy_site.sh` commits a copy of `docs/` into a
fresh, parent-less commit and force-pushes it to `gh-pages`, so that branch
always holds exactly one snapshot and the repository does not grow with
each publish. One-time setup: GitHub → Settings → Pages → Deploy from a
branch → `gh-pages`, `/ (root)`. `DRY_RUN=1` builds the snapshot without
pushing. Because the data lives only in `docs/data/` on the machine that
publishes, deploy from that machine (the cluster checkout, after
`publish.py`).

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
| `dfd`       | Sims-Flanagan MGA (`lowthrust.MgaLowThrust`), Direct Fusion Drive 10 N @ 10 000 s (2 MW), 20 t at Earth escape, continuous thrust + gravity assists, Q-law low-thrust capture into a 1.25 R_N × 443 449 km Neptune orbit | propellant, cruise + capture (kg); time axis includes the capture spiral |
| `dfd-1mw`   | as `dfd` with the 1 MW engine, 5 N @ 10 000 s | as `dfd` |

Sequences are strings over the body registry (`V E M J S U N`), Earth first,
destination last: `EVEEJN`, `EVEJN`, `EVJN`, `EVVEJN`, `EJN`, `EJUN`, `EN`, …

## Low-thrust: Direct Fusion Drive to Neptune orbit (`dfd`)

The same pipeline (archipelago → MBH polish → Pareto front → site) applied to a
continuous-thrust spacecraft. Everything model-specific lives in
`orbit_planning/lowthrust.py` (the problem, engine presets, capture model) and
the `DFD_SPACECRAFT` block of `orbit_planning/models.py` (the vehicle).

Mission model:

- **Launch**: 20 000 kg delivered to Earth escape (C3 = 0, "20 t to TLI").
  `launch_c3_km2s2 > 0` adds launch v∞ genes.
- **Cruise**: Sims-Flanagan transcription (pykep `sims_flanagan.leg`),
  `n_seg = 10` impulses per leg, each bounded by `thrust · dt / m`; constant
  thrust at any solar distance (fusion, not solar power). Each leg's forward
  and backward halves must meet: 7 match-point equality constraints per leg.
- **Gravity assists**: unpowered and exact by construction (`pykep.fb_prop` with
  periapsis-radius and plane-angle genes, periapsis ≥ the body's safe radius);
  thrust on the legs on either side.
- **Capture into a highly eccentric orbit**: periapsis 1.25 R_N (30 778 km),
  apoapsis 1.25 × Triton's orbital radius (443 449 km); e = 0.87, period
  ~3.2 days. Set in `mission.BODIES['N']['lowthrust_orbit']`. The final leg
  arrives with a free v∞ (a gene, ≤ 4 km/s, or less where the table caps it). The planetocentric phase is
  flown numerically (`lowthrust.capture_sim`): planar two-body motion from
  5 × the target apoapsis on the arrival hyperbola, thrust steered by a
  Q-law (Petropoulos 2004) that drives orbital energy and periapsis radius to
  their targets together, so the spiral ends on the target orbit and never
  dips into the planet. It thrusts continuously until the orbit is bound with
  apoapsis below 5 × the target's (`capture_kswitch`), then coasts wherever
  thrust is less than 20 % as effective as at the best point of the orbit
  (`capture_eta`; 0 = never coast: about twice the Δv in half the time).
  Δv and duration are tabulated once over arrival v∞ × arrival mass
  (`lowthrust.CaptureTable`, built in parallel on first use, a few minutes,
  then cached in `results/cache/`) and interpolated in every fitness call;
  the duration is added to the time objective. Lookups use a bicubic
  spline, so the polish sees a smooth problem. The table keeps the
  v∞ rows where every mass converged (efficient law first, continuous thrust
  as fallback); arrivals faster than that are refused by a constraint. Periapsis at 1.25 R_N lies inside the
  ring system (rings to ~2.5 R_N), so ring-plane crossings need checking in
  a 3-D follow-up; the planar capture does not model them, nor the
  orientation of the final orbit.
- **Cost**: propellant = 20 000 kg − mass in the final orbit; the final mass
  must stay ≥ `min_final_mass_kg` (10 000 kg).

Engine numbers (`lowthrust.ENGINES`) are published Princeton Satellite Systems
/ PPPL DFD design estimates (~10 000 s, 2.5–10 N per MW depending on the
source; see [Low-thrust (DFD) references](#low-thrust-dfd-references)), not
flight data; put
your engine's thrust, Isp and dry mass in `lowthrust.ENGINES` /
`models.DFD_SPACECRAFT` and every result records them (`assumptions` in the
result header).

Search: the archipelago (SADE in `optimize.py`, NSGA-II in `pareto.py`)
searches the problem with its constraints folded into a weighted penalty; the
MBH polish then solves the *true* constrained problem with IPOPT
(finite-difference gradients, unit-box scaling, `search.UnitBox`). Raw
Sims-Flanagan individuals are rarely feasible, so the polished points are the
result: give `--polish` generously, and use `--sweep`. On the penalised
problem a single NSGA-II population collapses to a few near-identical
trajectories, so the polish seeds (the least-penalised individual per
cruise-time group, one per distinct launch day / cruise day) are only as
diverse as the slices they come from. `models.PENALTY_WEIGHT` sets how hard
the global stage is pushed towards feasibility. Every point carries `feasible` and
`violation`; only feasible points can be on the front, and only feasible
points are published. Expect only part of the polished seeds to close.
Low-thrust seeds are polished in a wide band (±20 % cruise time, ±180 days of
launch; the chemical models keep ±3 % and ±45 days) because a penalised seed
is not yet a trajectory and the nearest feasible one can sit far from it,
and "feasible" means residuals under 1e-5 (`models.C_TOL`, ~1500 km and
~0.3 m/s at a match point). On 16 fixed seeds (analytic ephemerides, 4-island
searches) that closed 5 of 8 EJN seeds (cheapest 5 530 kg of propellant,
against 8 150 kg with the old narrow band) and 3 of 8 EVEJN seeds (against
1 of 8), in less CPU time. Seeds that do not close mostly stall at residuals
of 0.1-0.3 on longer sequences: the global stage only places the seeds and
the polish does the work, so more seeds (`--polish`), more slices
(`--sweep`) and more hops (`--polish-stop`) raise the yield. A polish of one point (MBH, 5 hops of up to 1000 IPOPT
iterations) takes ~5–10 min for a 2-leg sequence on one core, more for longer
ones, so `--polish 32` on 8 cores is roughly 30–50 min per sequence on top of
the search. The first run of a vehicle also builds its capture table (~40 s
on 4 cores for `dfd`). `--polish-maxeval` and `--polish-stop` trade time for feasibility.

```bash
# one sequence, cheapest trajectory
python orbit_planning/optimize.py --model dfd --sequence EJN

# one sequence, propellant / flight-time front over every synodic launch slice
python orbit_planning/pareto.py --model dfd --sequence EJN --sweep --polish 32
```

### Sweeping gravity-assist sequences

`orbit_planning/sequence_sweep.py` runs one search per sequence of an array
and ranks them. The array is `SEQUENCES` at the top of the file (a Python
list, defaulting to `paths.DFD_SEQUENCES`), or pass it on the command line:

```bash
# the SEQUENCES list in the file
python orbit_planning/sequence_sweep.py

# an explicit array (space separated), swept launch windows, 32 polished points each
python orbit_planning/sequence_sweep.py --model dfd \
    --sequences EN EJN ESN EUN EJSN EJUN EVEJN EVEEJN --sweep --polish 32

# cheapest-only per sequence (optimize.py) instead of the front
python orbit_planning/sequence_sweep.py --mode champion --sequences EJN EJSN

# re-rank existing results only, by the cheapest trajectory within 10 years
python orbit_planning/sequence_sweep.py --summary-only --max-years 10
```

On a many-core machine, `scripts/run_dfd_sweep.sh` runs the 77 Earth →
Neptune swing-by sequences (at most 5 assists; only Venus, up to 3×, and
Earth, up to 2×, back to back) as a bash array, `JOBS` sequences at once
with `WORKERS` processes each (default 16 × 8 = 128 cores), longest
sequences first, one log per sequence in `results/<model>/logs/`. It builds
the capture table once up front, skips finished sequences (so it resumes),
and ends with the ranking. Settings are environment variables:

```bash
scripts/run_dfd_sweep.sh                         # all 77, dfd, 16 x 8
JOBS=32 WORKERS=4 POLISH=16 scripts/run_dfd_sweep.sh
MODELS="dfd dfd-1mw" SEQUENCES="EJN EVEJN" scripts/run_dfd_sweep.sh
DRY_RUN=1 scripts/run_dfd_sweep.sh               # print the commands only
EXTRA="--polish-maxeval 1500" scripts/run_dfd_sweep.sh   # any other pareto.py option
```

Each sequence runs in its own process; ones that already have a result of
the same kind are skipped unless `--force`, so an interrupted sweep resumes.
Unrecognised options (`--sweep`, `--islands`, `--pop`, `--gen`, `--rounds`,
`--polish`, `--polish-maxeval`, `--seed`, …) go to the search script. The
ranking (feasible-point count, cheapest propellant and its flight time and
final mass, fastest feasible trajectory) is printed and written to
`results/<model>/sequence_sweep.{csv,json}`. `batch.py --models dfd
--sequences EJN,EJSN` does the same searches without the ranking, and
`publish.py` puts the runs on the site like any other model (thrust arcs are
drawn as the coasts between Sims-Flanagan impulses).

## Layout

| file | role |
|------|------|
| `orbit_planning/paths.py`       | filesystem layout + default matrix (kernel-free) |
| `orbit_planning/mission.py`     | SPICE kernels, body registry, sequences, launch-window slicing |
| `orbit_planning/models.py`      | the models: build / decode / cost / objective, `problem()`, `evaluate()`; the DFD vehicle |
| `orbit_planning/lowthrust.py`   | Sims-Flanagan gravity-assist problem, DFD engine presets, Q-law capture simulation and its cached table |
| `orbit_planning/search.py`      | archipelago, MBH polish, parallel polish pool, shared CLI |
| `orbit_planning/optimize.py`    | single-objective search (+ `--sweep`) |
| `orbit_planning/pareto.py`      | NSGA-II front (+ `--sweep`, `--polish`) |
| `orbit_planning/trajectory.py`  | decision vector → mission description with arc states |
| `orbit_planning/batch.py`       | run many pairs, resumable |
| `orbit_planning/sequence_sweep.py` | run an array of sequences for one model and rank them |
| `scripts/run_dfd_sweep.sh` | the 77-sequence DFD sweep, many sequences in parallel on one machine |
| `orbit_planning/publish.py`     | raw results → `docs/data/` |
| `orbit_planning/sitedata.py`    | the compact per-run site format (write, read, convert old runs) |
| `scripts/deploy_site.sh`        | push `docs/` (site + data) to `gh-pages` as one replaced snapshot |
| `orbit_planning/fronts.py`      | each model's combined Pareto front over all sequences → `docs/data/<model>/fronts.json` (the "Best by trip time" tab; kernel-free, rerunnable on its own) |
| `orbit_planning/porkchop.py`    | static PNG of one run's cloud |
| `orbit_planning/slingshot_planning.py` | notebook for poking at one champion |
| `docs/`                         | the site: `common.js`, `run.js`, `gallery.js`, `fronts.js`, `kepler.js` |
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

## Low-thrust (DFD) references

Sources behind `orbit_planning/lowthrust.py` and the `dfd` models. The
Neptune-system sources above (Jacobson 2009 for Triton's orbit; Domingos et
al. 2006 for Hill-sphere stability) apply too. No published source was found
for a Princeton engine named "Starfire"; the engine presets use the Princeton
Satellite Systems / PPPL Direct Fusion Drive numbers below.

### Direct Fusion Drive

- S. J. Thomas, M. A. Paluszek, S. A. Cohen, "Fusion-Enabled Pluto Orbiter
  and Lander", AIAA, 2017 (NASA NIAC Phase I results).
  https://w3.pppl.gov/ppst/docs/thomas2017aiaa.pdf — 5 N of constant thrust
  delivers ~1000 kg to Pluto orbit in ~4 years; up to 1 MW of electric power
  to the payload on arrival. Basis of the `dfd-1mw` preset (5 N).
- S. J. Thomas, M. A. Paluszek, S. A. Cohen, "Fusion-Enabled Pluto Orbiter
  and Lander", NIAC Phase II final report (award 80NSSC18K0040), 2019, NTRS
  20190031807. https://ntrs.nasa.gov/citations/20190031807 — the PFRC-based
  engine and mission design.
- NASA NIAC project summary, "Fusion Enabled Pluto Orbiter and Lander".
  https://www.nasa.gov/directorates/spacetech/niac/2017_Phase_I_Phase_II/Fusion_Enabled_Pluto_Orbiter_and_Lander
  — 2.5–5 N of thrust per MW of fusion power at an Isp of ~10 000 s.
- S. A. Cohen et al., "Direct Fusion Drive for Interstellar Exploration",
  *J. British Interplanetary Society* 72(2), 37–50, 2019.
  https://w3.pppl.gov/ppst/docs/cohen2019jbis.pdf — 1–10 MW engine range,
  thrust augmentation, subsystem mass budget.
- Gajeri, Aime, Kezerashvili, "A Titan mission using the Direct Fusion
  Drive", arXiv:2009.12621, 2020. https://arxiv.org/abs/2009.12621 — DFD
  trajectory study with constant thrust and Isp; quotes ~10 000 s Isp and
  5–10 N per MW of thrust power (PSS estimate), 0.75–1.25 kW/kg specific power.
  Basis of the `dfd-2mw` preset (10 N).
- "Direct Fusion Drive", Wikipedia. https://en.wikipedia.org/wiki/Direct_Fusion_Drive
  — overview and further references.

### Low-thrust trajectory methods

- J. A. Sims, S. N. Flanagan, "Preliminary Design of Low-Thrust
  Interplanetary Missions", AAS/AIAA Astrodynamics Specialist Conference,
  AAS 99-338, 1999 — the transcription (impulses at segment midpoints,
  forward/backward propagation to a match point) in `MgaLowThrust`.
- C. H. Yam, D. di Lorenzo, D. Izzo, "Low-Thrust Trajectory Design as a
  Constrained Global Optimization Problem", *Proc. IMechE Part G* 225(11),
  1243–1251, 2011 — Sims-Flanagan multiple-gravity-assist as a global
  optimisation problem (pykep's `mga_lt_nep`); the constraint
  non-dimensionalisation follows it.
- A. E. Petropoulos, "Low-Thrust Orbit Transfers Using Candidate Lyapunov
  Functions with a Mechanism for Coasting", AIAA/AAS Astrodynamics
  Specialist Conference, AIAA 2004-5089, 2004. https://doi.org/10.2514/6.2004-5089
  — the Q-law: Lyapunov steering and effectivity-based coasting, used in the
  Neptune capture spiral (`lowthrust.capture_sim`).
- D. Izzo, "PyGMO and PyKEP: open source tools for massively parallel
  optimization in astrodynamics (the case of interplanetary trajectory
  optimization)", 5th International Conference on Astrodynamics Tools and
  Techniques (ICATT), 2012. https://github.com/esa/pykep — Sims-Flanagan legs,
  flyby propagation, Kepler propagation.
- F. Biscani, D. Izzo, "A parallel global multiobjective framework for
  optimization: pagmo", *JOSS* 5(53), 2338, 2020.
  https://doi.org/10.21105/joss.02338 — archipelagos, SADE, NSGA-II, MBH, and the
  IPOPT / NLopt wrappers of the polish.

### Neptune target orbit

- NASA NSSDC Neptune fact sheet. https://nssdc.gsfc.nasa.gov/planetary/factsheet/neptunefact.html
  — equatorial radius 24 764 km at 1 bar; the code uses the 24 622 km
  volumetric mean radius, as `mission.BODIES` and AMAT do (the target
  periapsis, 1.25 R_N, is 30 778 km).
- JPL Solar System Dynamics, planetary satellite mean elements.
  https://ssd.jpl.nasa.gov/sats/elem/ — Triton's semi-major axis, 354 759 km
  (Jacobson 2009 ephemeris); the target apoapsis is 1.25 × that, 443 449 km.
