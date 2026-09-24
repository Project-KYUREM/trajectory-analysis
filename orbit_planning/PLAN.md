# Kyurem — Neptune MGA trajectory explorer

## STATUS: pipeline built and verified (impulsive models)

DONE: mission body-registry, 3 models (capture/ballistic working; nep = GMAT
seed exporter), optimize/pareto/batch/publish all --model/--sequence aware,
trajectory decoder + CSV/GMAT exports, static site (docs/) with in-browser
Kepler propagator (validated vs pykep to 1e-13), 6 runs published as demo data.
REMAINING: full-quality --sweep batch across all 5 sequences x 2 models;
front MBH-polish before publish; enable GitHub Pages (gh token invalid -> do in
repo Settings); legacy slingshot_planning.py notebook points at old paths.


A pipeline that searches Earth→Neptune multiple-gravity-assist trajectories for
many flyby sequences, then publishes them as a browsable GitHub Pages site where
each trajectory can be inspected, visualized in 3D, and exported for GMAT/STK.

## Current state (done)

- **`mission.py`** — SPICE kernels, GMs, and a **body registry** (`V E M J S N`).
  Sequences are strings, always Earth→Neptune: `make_sequence("EVMJN")`.
  - Saturn safe-radius ratio = 2.4 (clears F ring), not pykep's 1.1.
  - Mars/Jupiter/Saturn flybys use barycenters (de440 has no planet centre;
    offset negligible at flyby scale).
  - `build_udp(multi_objective=, orbit_insertion=, launch_window=, sequence=)`.
  - `launch_slices()` — synodic-period slicing of the launch window, derived
    from the first two bodies; resonant-first-leg guard falls back to 2 yr.
  - `orbit_insertion=True` default: final DV term is the Neptune capture burn
    into a loose ellipse (rp=1.1 R_N, e=0.99), not raw arrival vinf.
- **`optimize.py`** — single-objective SADE archipelago + MBH/COBYLA polish.
  `--sweep` runs one archipelago per launch slice. Overwrite guard re-scores
  the existing champion under the current objective before replacing it.
- **`pareto.py`** — NSGA-II (DV vs TOF). `--sweep` pools per-slice fronts and
  recomputes global dominance. Every decision vector retained (not just front).
- **`porkchop.py`** — two-panel plot: launch/arrival porkchop + DV/TOF trade.
  Sequential blue ramp (theme-selected, not flipped); Pareto front ringed;
  line breaks across real gaps; validated palette.

## Key facts / gotchas (learned)

- pygmo drives `mga_1dsm` (a Python UDP, thread_safety.none) via **mp_island**,
  which forces multiprocessing **spawn** → re-imports `__main__`. So every
  `evolve()` MUST live behind `if __name__ == '__main__'` in a script, never in
  a notebook cell. That is why optimize/pareto are separate files.
- A pickled `planet.spice` is ~250 bytes and resolves against whatever kernel
  pool the child has → every process must furnsh kernels (mission.py does it at
  import).
- `alpha` TOF encoding: leg times come from `x[-1]` + alpha genes, NOT from the
  TOF bounds, so a sliced-window `x` decodes/scores identically under the
  full-window UDP. (Would NOT hold under `eta` encoding.)
- `de440.bsp` has no Triton/Neptune-centre → **`nep097.bsp`** required.
- `data/` is gitignored (de440 is 115 MB > GitHub's 100 MB limit). Site must be
  fully precomputed; browser never touches kernels.
- Raw swept `pareto.json` ≈ 36 MB per sequence → cannot be the site payload;
  needs a compaction/publish step.
- `champion.json` currently stale/self-inconsistent (stored 7527, rescores
  30741) — regenerate, don't trust it.

## Propulsion / objective models (models.py) — coexist via --model

- **capture** (existing): mga_1dsm + orbit_insertion (e=0.99). DONE, works.
- **ballistic** (Trident): mga, no DSMs, flyby arrival. Cost = departure C3;
  arrival vinf reported not charged. DONE, works. dim 4-7, unconstrained.
- **nep** (40 kWe flagship: Tmax 1 N, Isp 3500 s, wet 4000 kg): mga_lt_nep,
  constrained Sims-Flanagan (nx 77-191, nec 15-39, nic 23-56). CONSTRUCTS and
  RUNS via pg.unconstrain(kuri) + models.safe() wrapper, but does NOT yet find
  feasible solutions with impulsive-scale budgets. NEEDS: feasibility-first
  optimisation (global penalty -> local SLSQP/IPOPT restoration with gradients),
  bigger budget, tuned penalty weights. Separate effort.

Sequences to run (5): EVEEJN, EVEJN, EVJN, EVVEJN, EJN.

models.safe(): wraps any UDP so a fitness exception (epoch past kernel coverage,
invalid conic) returns a large penalty instead of crashing -- required because
NLopt COBYLA and global samplers probe out-of-box points.

--model / --sequence now wired into optimize.py; results -> results/<model>/<sequence>/.
TODO: same wiring into pareto.py; then batch.py, publish.py, docs/ site.

## Plan — remaining phases

### A. Generalize to arbitrary sequences — DONE

### B. Sequence enumerator + batch runner
- Enumerate all **4-flyby** E→N sequences (6 bodies visited, like EVEEJN) over
  {V, E, M, J, S}. Up to 5^4 = 625; prune: no immediate repeats (`JJ`), no
  outer→inner backtracking (`J…V`), cap consecutive same body. Show filtered
  list + count + time estimate; user trims before running.
- `batch.py --sequences ...` runs the pareto sweep per sequence into
  `results/<seq>/`. Resumable (skip existing), background-friendly, scoreboard.
- **Quality: full sweep + MBH-polish the pooled front before publishing**
  (~45–70 min/seq; measure per-point polish cost first). Unpolished fronts ran
  2–3× high — matters because these seed GMAT.

### C. Publish step (`publish.py`) — raw results → site data
- `points.json` per sequence: compact cloud (dv, tof, launch, arrival, pareto,
  id) ≈ 1–2 MB.
- `front.json`: per selectable point, the **mission table** (legs, DSM
  epochs/magnitudes, flyby rp/β, C3, RLA/DLA, capture burn) **plus initial
  state vectors (r, v, Δt, μ) for every ballistic arc** → browser propagates
  Kepler arcs itself (~80-line universal-variable propagator), no kernels/server.
- Planet orbit polylines per sequence for 3D context.
- All states in **ECLIPJ2000 and J2000 equatorial** (GMAT/STK use EME2000).
- Stable point IDs: `s<slice>-i<index>` so a trajectory can be named across runs.

### D. Static site (`docs/`, Plotly.js vendored, no CDN)
- Index: card per sequence (porkchop thumbnail + best DV, knee TOF).
- Sequence page: interactive scatter, front/cloud toggle, axis toggle
  (DV–TOF trade ↔ launch/arrival porkchop). Click a dot → mission table +
  3D trajectory + downloads.
- Clickable detail = front points + cheapest-few-per-TOF-bin; dominated blob
  drawn for context only.

### E. Export + ship
- **JSON + CSV state exports** (chosen): epochs, r/v (both frames), DSM ΔV
  vectors, flyby params, C3/RLA/DLA. (GMAT script / STK .e = possible later.)
- `scripts/fetch_kernels.sh` so a fresh clone can reproduce.
- Gitignore: `results/` raw stays local; `docs/` committed. README.
- Enable Pages from `main:/docs`. NOTE: `gh` token currently invalid — re-auth
  or enable in repo Settings by hand. Confirm before pushing.

## Open decisions
- Enumeration pruning strictness (B).
- Whether to add Mars-heavy sequences (included per user).
- Objective for the site runs: `orbit_insertion` (capture burn) is current
  default. See also the **Trident-style all-initial-ΔV / ballistic** variant
  under investigation.
