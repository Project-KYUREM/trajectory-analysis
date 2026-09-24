# Kyurem — Neptune multiple-gravity-assist trajectory explorer

Searches Earth→Neptune gravity-assist trajectories across flyby sequences and
propulsion models with pykep/pygmo, then publishes them as a static site where
every trajectory can be inspected in 3D and exported for GMAT/STK.

## Setup

```bash
conda env create -f environment.yml      # creates orbital_env
conda activate orbital_env
scripts/fetch_kernels.sh                  # SPICE kernels into data/ (~225 MB, gitignored)
```

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
