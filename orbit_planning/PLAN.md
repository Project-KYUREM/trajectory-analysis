# Kyurem — plan and decisions

Layout, commands and design notes live in the top-level README; this file
tracks status, the decisions that are not obvious from the code, and what is
left.

## Status

Built and verified end to end: body registry and string sequences; three
models on one pipeline (`capture`, `ballistic` searchable; `nep` builds and
runs but is a GMAT-seed model, not decoded or published); `optimize.py` /
`pareto.py` / `batch.py` / `publish.py` / site all model-aware; front polish
(parallel, epsilon-constraint, dominance recomputed) in `pareto.py`; sweep
champions polished in parallel in `optimize.py`; static site with in-browser
Kepler propagation (validated against pykep to ~1e-13 relative).

Remaining:
- Full-quality run: `batch.py --sweep --polish 24 --islands 20` over the five
  default sequences x two models (hours; resumable), then `publish.py`.
- Enable GitHub Pages: repo Settings -> Pages -> branch `main`, folder `/docs`.

## Decisions worth remembering

- **Why scripts, not notebook cells.** pygmo drives Python UDPs through
  mp_island, which forces multiprocessing *spawn*; spawn re-imports
  `__main__`, so a search in a notebook cell forks the notebook. Every search
  sits behind `if __name__ == '__main__'`.
- **Why kernels load at import.** A pickled `planet.spice` resolves against
  the child's kernel pool; every worker must furnsh them itself.
- **Why `orbit_insertion`.** Arrival vinf is not the cost of arriving; the
  capture burn at periapsis is ~10x smaller. Charging raw vinf biased the
  search towards slow trajectories.
- **Why the launch window is sliced.** Opportunities recur at the first leg's
  synodic period; one search over decades wastes its budget between them.
- **Why the front is polished.** NSGA-II spreads well but stalls short of
  each local optimum (10-50 % high). Polished points are appended and
  dominance recomputed, because refinement moves points more than the gaps
  between them.
- **Why NEP is a seed, not a search.** `mga_lt_nep` is a 77-191 dimensional
  constrained Sims-Flanagan problem; heuristic search did not reach
  feasibility at impulsive budgets. The practical path is to export an
  impulsive trajectory's GMAT seed and convert the impulses to finite burns
  in GMAT, where the burn arcs (~5 % of a leg at 1 N / 4000 kg) fit easily.
- **Saturn safe radius 2.4 R** (not pykep's 1.1): 1.1 flies through the B ring.
- **Ballistic "maneuvers".** `mga` reports a per-flyby velocity mismatch the
  assist cannot supply; it is shown as a burn at the flyby, not as a
  deep-space maneuver on a leg.
- **Exports are client-side.** One implementation (docs/run.js) rotates
  ECLIPJ2000 states with the matrix publish.py emits; there is no Python copy
  to drift.

## Data facts

- `de440.bsp` has no Neptune centre or Triton: `nep097.bsp` supplies both.
- Kernels (`data/`) and raw results (`results/`) are gitignored; `docs/data/`
  is the committed, compacted payload (~0.4 MB per run).
- Result files carry `schema: 2`.
