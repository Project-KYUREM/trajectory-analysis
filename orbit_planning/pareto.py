"""Multi-objective (DV vs time-of-flight) search over the EVEEJN sequence.

Run as a script, not from a cell -- same mp_island/spawn reason as optimize.py:

    python orbit_planning/pareto.py --islands 8 --rounds 10

Single-objective optimisation returns one arbitrary point on the DV/TOF trade.
NSGA-II returns the whole front, which is what the porkchop plot samples.

Writes every individual from every island's final population, not just the
champions: the porkchop wants the cloud, and the non-dominated subset is
flagged rather than filtered so the plot can show what was dominated too.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygmo as pg

import mission

PARETO_PATH = mission.HERE / "pareto.json"


def search(n_islands, pop_size, gen, rounds, seed=None, launch_window=None,
           quiet=False):
    if pop_size % 4:
        raise ValueError(f"NSGA-II needs a population size divisible by 4, got {pop_size}")

    prob = pg.problem(mission.build_udp(multi_objective=True,
                                        launch_window=launch_window))
    algo = pg.algorithm(pg.nsga2(gen=gen))

    kwargs = dict(n=n_islands, algo=algo, prob=prob, pop_size=pop_size, t=pg.ring())
    if seed is not None:
        kwargs['seed'] = seed
    archi = pg.archipelago(**kwargs)

    if not quiet:
        print(f"dimension {prob.get_nx()} | {prob.get_nobj()} objectives | "
              f"{n_islands} islands x pop {pop_size} | {rounds} rounds x {gen} gen")

    t0 = time.time()
    archi.evolve(rounds)
    archi.wait_check()
    wall = time.time() - t0
    if not quiet:
        print(f"archipelago done in {wall:.1f}s")

    # Pool every individual from every island, not just champions.
    xs, fs = [], []
    for isl in archi:
        pop = isl.get_population()
        xs.extend([list(v) for v in pop.get_x()])
        fs.extend([list(v) for v in pop.get_f()])

    front_idx = set(pg.non_dominated_front_2d(fs))
    if not quiet:
        print(f"pooled {len(fs)} individuals | {len(front_idx)} on the non-dominated front")

    points = []
    for i, (x, f) in enumerate(zip(xs, fs)):
        dv, tof = f[0], f[1]
        launch = x[0]
        points.append({
            'dv_ms': dv,
            'tof_days': tof,
            'launch_mjd2000': launch,
            'arrival_mjd2000': launch + tof,
            'pareto': i in front_idx,
            # Keep EVERY decision vector, not just the front's. Dominance here
            # is computed on unpolished fitness, and local refinement moves
            # points by far more than the margins that separate them -- so a
            # point dominated now may well win after polishing. Discarding its
            # x makes that impossible to ever test.
            'x': x,
        })

    return points, wall


def sweep(args):
    """One NSGA-II run per launch slice, pooled into a single point cloud.

    A single search over the whole window concentrates on whichever launch
    opportunities look cheapest and abandons the rest -- on the previous run,
    7 of 17 synodic slices were never sampled at all. The porkchop then shows
    empty bands that read as "no solutions here" when they only mean "never
    searched". Per-slice runs guarantee every launch opportunity is covered.

    Dominance is recomputed over the POOLED set at the end, so the flagged
    front is the true global front, not a union of per-slice fronts.
    """
    slices = mission.launch_slices(width_days=args.slice_width,
                                   overlap=args.overlap)
    print(f"sweeping {len(slices)} launch slices "
          f"({args.islands} islands x pop {args.pop}, "
          f"{args.rounds} rounds x {args.gen} gen each)")
    print(f"{'#':>3}  {'from':>10}  {'to':>10}  {'pts':>5}  {'best DV':>10}  {'secs':>6}")

    pooled, t_all = [], time.time()
    for i, window in enumerate(slices):
        pts, wall = search(args.islands, args.pop, args.gen, args.rounds,
                           args.seed, launch_window=window, quiet=True)
        for pt in pts:
            pt['slice'] = i
        best = min(p['dv_ms'] for p in pts)
        print(f"{i:3d}  {window[0].mjd:10.1f}  {window[1].mjd:10.1f}  "
              f"{len(pts):5d}  {best:10.0f}  {wall:6.1f}")
        pooled.extend(pts)

    # Recompute dominance across every slice at once.
    fs = [[p['dv_ms'], p['tof_days']] for p in pooled]
    front_idx = set(pg.non_dominated_front_2d(fs))
    for i, pt in enumerate(pooled):
        pt['pareto'] = i in front_idx

    total = time.time() - t_all
    print(f"\nsweep done in {total/60:.1f} min | {len(pooled)} points pooled | "
          f"{len(front_idx)} on the global front")
    return pooled, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--islands', type=int, default=8)
    ap.add_argument('--pop', type=int, default=100)
    ap.add_argument('--gen', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', type=Path, default=PARETO_PATH)
    ap.add_argument('--sweep', action='store_true',
                    help='partition the launch window into synodic slices and '
                         'run NSGA-II on each, so every launch opportunity is '
                         'covered')
    ap.add_argument('--slice-width', type=float, default=None, dest='slice_width',
                    help='slice width in days (default: Earth-Venus synodic, ~583 d)')
    ap.add_argument('--overlap', type=float, default=0.2,
                    help='fractional overlap between adjacent slices')
    args = ap.parse_args()

    if args.sweep:
        points, wall = sweep(args)
    else:
        points, wall = search(args.islands, args.pop, args.gen, args.rounds, args.seed)

    front = [p for p in points if p['pareto']]
    best_dv = min(front, key=lambda p: p['dv_ms'])
    fastest = min(front, key=lambda p: p['tof_days'])
    print(f"  cheapest on front : {best_dv['dv_ms']:8.0f} m/s @ {best_dv['tof_days'] / 365.25:5.2f} yr")
    print(f"  fastest on front  : {fastest['dv_ms']:8.0f} m/s @ {fastest['tof_days'] / 365.25:5.2f} yr")

    args.out.write_text(json.dumps({
        'sequence': mission.LABELS,
        'tof_bounds_days': mission.MISSION_TOF_DAYS,
        'settings': {'islands': args.islands, 'pop': args.pop, 'gen': args.gen,
                     'rounds': args.rounds, 'seed': args.seed, 'topology': 'ring',
                     'wall_seconds': round(wall, 1),
                     'swept': args.sweep,
                     'slice_width_days': args.slice_width if args.sweep else None,
                     'overlap': args.overlap if args.sweep else None,
                     **mission.objective_config(mission.build_udp(multi_objective=True))},
        'points': points,
    }, indent=2))
    print(f"wrote {args.out}  ({len(points)} points)")


if __name__ == '__main__':
    main()
