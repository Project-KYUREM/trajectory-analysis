# Kyurem — Neptune multiple-gravity-assist trajectory explorer

Searches Earth→Neptune gravity-assist trajectories across many flyby sequences
and three propulsion/objective models, then publishes them as a browsable static
site where each trajectory can be inspected in 3D and exported for GMAT/STK.

## Setup

```bash
conda env create -f environment.yml      # creates orbital_env
conda activate orbital_env
scripts/fetch_kernels.sh                  # downloads SPICE kernels into data/ (~225 MB)
```

Kernels and raw results are gitignored; the site payload under `docs/data/` is
committed.

## Models (`--model`)

| model | what | cost axis |
|-------|------|-----------|
| `capture`   | mga_1dsm + orbit insertion (e=0.99 ellipse) | DSMs + capture burn (m/s) |
| `ballistic` | Trident-style: gravity assists only, flyby arrival | departure C3 Δv (m/s) |
| `nep`       | 40 kWe nuclear-electric — **seed exporter, not optimiser** | (export a `ballistic`/`capture` GMAT seed and convert to a finite/EP burn in GMAT) |

Sequences are strings, Earth→Neptune: `EVEEJN`, `EVEJN`, `EVJN`, `EVVEJN`, `EJN`, …

## Pipeline

```bash
# 1. search — one Pareto sweep per (model, sequence) into results/<model>/<sequence>/
python orbit_planning/batch.py --models capture,ballistic \
    --sequences EVEEJN,EVEJN,EVJN,EVVEJN,EJN --sweep

# 2. publish — compact raw results into docs/data/ (the site payload)
python orbit_planning/publish.py --models capture,ballistic \
    --sequences EVEEJN,EVEJN,EVJN,EVVEJN,EJN

# 3. view — any static server; GitHub Pages serves docs/ directly
python -m http.server -d docs 8899   # then open localhost:8899
```

Single runs: `python orbit_planning/optimize.py --model ballistic --sequence EJN`
(best-of), or `pareto.py` (full front). Both take `--sweep` to partition the
launch window by synodic period.

## Layout

- `orbit_planning/mission.py` — kernels, body registry, launch-window slicing
- `orbit_planning/models.py` — the three models + a crash-safe fitness wrapper
- `orbit_planning/optimize.py` / `pareto.py` — single-objective / Pareto search
- `orbit_planning/trajectory.py` — decode x → mission table, arc states, exports
- `orbit_planning/batch.py` / `publish.py` — run many, compact for the site
- `docs/` — the static site (Plotly vendored, in-browser Kepler propagator)

## Notes

- Every `evolve()` lives behind `if __name__ == '__main__'` in a script, never a
  notebook cell: pygmo drives these via mp_island → multiprocessing spawn, which
  re-imports `__main__`.
- The browser reproduces trajectories from arc states with a universal-variable
  Kepler propagator (`docs/kepler.js`), validated against pykep to machine
  precision — no server, no kernels needed to view.
- Arc states are exported in ECLIPJ2000 **and** J2000 (EME2000) for GMAT/STK.
