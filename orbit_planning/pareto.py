"""Multi-objective search: the cost / time-of-flight front for one model and sequence.

    python orbit_planning/pareto.py --model capture --sequence EVEEJN --sweep

Single-objective optimisation returns one arbitrary point on the trade;
NSGA-II returns the whole front, which is what the site's porkchop samples.
Every individual from every island is kept, not just the front: the plot wants
the cloud, and dominance is a flag, not a filter.

Raw NSGA-II fronts run high (it spreads well but stalls short of each local
optimum), so a TOF-spanning sub-sample of the front is MBH-polished afterwards
and dominance is recomputed over the union. Run as a script, never from a
notebook cell -- see ``optimize.py`` for why.
"""
import argparse
import json
import time

import numpy as np
import pygmo as pg

import mission
import models
import search


def nsga2(gen):
    """Non-dominated sorting genetic algorithm II."""
    return pg.algorithm(pg.nsga2(gen=gen))


def _point(x, f, **tags):
    """One cloud record from a decision vector and its ``(cost, tof)`` fitness."""
    record = dict(cost=float(f[0]), tof_days=float(f[1]),   # x[0]: launch epoch (models.py contract)
                  launch_mjd2000=float(x[0]), arrival_mjd2000=float(x[0] + f[1]),
                  pareto=False, polished=False, x=list(x))
    record.update(tags)
    return record


def flag_front(points):
    """Set ``pareto`` on the non-dominated points; returns how many there are."""
    front = set(pg.non_dominated_front_2d([[p['cost'], p['tof_days']] for p in points]))
    for i, p in enumerate(points):
        p['pareto'] = i in front
    return len(front)


def optimise_front(args, launch_window=None, log=print):
    """NSGA-II archipelago over one launch window; returns the pooled cloud.

    Args:
        args: Parsed CLI namespace (see ``search.add_search_args``).
        launch_window: Optional ``[epoch, epoch]`` narrowing (slice sweep).
        log: Print-like callable for progress lines.

    Returns:
        ``(points, wall_seconds)`` with every individual from every island.

    Raises:
        ValueError: If the population size is not a multiple of four
            (an NSGA-II requirement).
    """
    if args.pop % 4:
        raise ValueError(f"NSGA-II needs a population size divisible by 4, got {args.pop}")
    prob = models.problem(args.model, args.sequence, launch_window=launch_window,
                          multi_objective=True)
    archi, wall = search.run_archipelago(prob, nsga2(args.gen), args.islands, args.pop,
                                         args.rounds, args.seed, log)
    points = [_point(x, f) for island in archi
              for x, f in zip(island.get_population().get_x(),
                              island.get_population().get_f())]
    log(f"pooled {len(points)} individuals")
    return points, wall


def sweep(args):
    """One NSGA-II run per launch slice, pooled into a single cloud.

    A single search over the whole window concentrates on whichever launch
    opportunities look cheapest and abandons the rest; per-slice runs
    guarantee every opportunity is sampled, so the porkchop never shows an
    empty band that only means "never searched".

    Args:
        args: Parsed CLI namespace.

    Returns:
        ``(points, wall_seconds)``: the pooled cloud (each point tagged with
        its ``slice``) and the total elapsed time.
    """
    slices = mission.launch_slices(args.sequence, width_days=args.slice_width,
                                   overlap=args.overlap)
    print(f"sweeping {len(slices)} launch slices "
          f"({args.islands} islands x pop {args.pop}, {args.rounds} rounds x {args.gen} gen)")
    pooled, t_all = [], time.time()
    for i, window in enumerate(slices):
        points, wall = optimise_front(args, window, log=search.silent)
        for p in points:
            p['slice'] = i
        print(f"  slice {i:2d}  {window[0].mjd:9.1f}..{window[1].mjd:9.1f}  "
              f"{len(points):5d} pts  best {min(p['cost'] for p in points):9.0f}  {wall:5.1f}s")
        pooled.extend(points)
    print(f"sweep done in {(time.time() - t_all) / 60:.1f} min | {len(pooled)} points pooled")
    return pooled, time.time() - t_all


def polish_front(points, args):
    """MBH-polish a TOF-spanning sub-sample of the front and re-rank the cloud.

    Each chosen point is pinned to a narrow band around its own launch date
    (``x[0]``, the one gene every model shares) and time of flight (an
    epsilon-constraint applied through the problem's TOF bounds) and locally
    minimised. Polished
    points are appended as new individuals and dominance is recomputed over
    the union: refinement moves points by more than the margins between them,
    so the old front cannot be assumed to survive.

    Args:
        points: The cloud; mutated in place (new points appended, flags reset).
        args: Parsed CLI namespace with the ``--polish*`` knobs.

    Returns:
        A summary dict recorded into the result's settings.
    """
    front = sorted((i for i, p in enumerate(points) if p['pareto']),
                   key=lambda i: points[i]['tof_days'])
    if not front or args.polish <= 0:
        return {'n_polished': 0}
    picks = sorted({int(round(k)) for k in np.linspace(0, len(front) - 1,
                                                       min(args.polish, len(front)))})
    jobs = []
    for k in picks:
        src = points[front[k]]
        jobs.append(dict(model=args.model, sequence=args.sequence, x=src['x'],
                         window_mjd2000=[src['x'][0] - args.polish_pad_days,
                                         src['x'][0] + args.polish_pad_days],
                         tof_days=[src['tof_days'] * (1 - args.polish_band),
                                   src['tof_days'] * (1 + args.polish_band)],
                         maxeval=args.polish_maxeval, stop=args.polish_stop))

    print(f"\npolishing {len(jobs)} of {len(front)} front points in parallel "
          f"(MBH stop={args.polish_stop}, COBYLA maxeval={args.polish_maxeval})")
    print(f"{'#':>3}  {'TOF yr':>7}  {'before':>9}  {'after':>9}  {'gain':>6}  {'secs':>6}")
    t0 = time.time()
    results = search.parallel_polish(jobs, args.polish_workers)
    scorer = models.build(args.model, args.sequence, multi_objective=True)

    gains = []
    for k, r in zip(picks, results):
        src = points[front[k]]
        gain = 100 * (r['before'] - r['after']) / r['before'] if r['before'] else 0.0
        gains.append(gain)
        print(f"{k:3d}  {src['tof_days'] / 365.25:7.2f}  {r['before']:9.0f}  {r['after']:9.0f}  "
              f"{gain:5.1f}%  {r['seconds']:6.1f}")
        points.append(_point(r['x'], scorer.fitness(r['x']), polished=True,
                             polished_from=front[k], slice=src.get('slice')))

    n_front = flag_front(points)
    on_front = sum(1 for p in points if p['polished'] and p['pareto'])
    print(f"polish done in {time.time() - t0:.1f}s | mean gain {np.mean(gains):.1f}% | "
          f"{on_front}/{len(jobs)} polished points on the recomputed front ({n_front} total)")
    return dict(n_polished=len(jobs), band=args.polish_band, pad_days=args.polish_pad_days,
                maxeval=args.polish_maxeval, stop=args.polish_stop,
                mean_gain_pct=float(np.mean(gains)))


def main():
    """CLI: search (or sweep), flag the front, polish it, write the cloud."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    search.add_search_args(parser, islands=8, pop=100, gen=200, rounds=10)
    parser.add_argument('--polish', type=int, default=16, metavar='N',
                        help='MBH-polish N TOF-spanning front points (0 disables)')
    parser.add_argument('--polish-band', type=float, default=0.03, dest='polish_band',
                        help='fractional TOF band a polished point is pinned to')
    parser.add_argument('--polish-pad-days', type=float, default=45.0, dest='polish_pad_days',
                        help='launch-date half-window a polished point is pinned to')
    args = parser.parse_args()
    out = search.output_path(args, 'pareto.json')

    points, wall = sweep(args) if args.sweep else optimise_front(args)
    print(f"{flag_front(points)} on the non-dominated front")
    polish_info = polish_front(points, args)

    front = [p for p in points if p['pareto']]
    cheapest, fastest = min(front, key=lambda p: p['cost']), min(front, key=lambda p: p['tof_days'])
    print(f"  cheapest on front : {cheapest['cost']:8.0f} @ {cheapest['tof_days'] / 365.25:5.2f} yr")
    print(f"  fastest on front  : {fastest['cost']:8.0f} @ {fastest['tof_days'] / 365.25:5.2f} yr")

    out.write_text(json.dumps({**search.result_header(args),
                               'settings': search.run_settings(args, wall, polish=polish_info),
                               'points': points}, indent=2))
    print(f"wrote {out}  ({len(points)} points)")


if __name__ == '__main__':
    main()
