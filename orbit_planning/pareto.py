"""Find the cost / time-of-flight Pareto front for one model and sequence.

Example: ``python orbit_planning/pareto.py --model capture --sequence EVEEJN
--sweep``. NSGA-II individuals from every island are kept and the
non-dominated ones are flagged. A TOF-spanning sub-sample of the front is then
MBH-polished and dominance is recomputed over the whole cloud. Run it as a
script, never from a notebook cell (see ``optimize.py``).
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
    """Return pygmo's NSGA-II algorithm for ``gen`` generations."""
    return pg.algorithm(pg.nsga2(gen=gen))


def _point(x, f, **tags):
    """Build one cloud record from a decision vector and its fitness.

    Args:
        x (Sequence[float]): Decision vector; ``x[0]`` is the launch epoch in
            MJD2000.
        f (Sequence[float]): Fitness ``(cost, tof_days)``.
        **tags (Any): Extra fields set on the record.

    Returns:
        dict: The point record.
    """
    record = dict(cost=float(f[0]), tof_days=float(f[1]),   # x[0]: launch epoch (models.py contract)
                  launch_mjd2000=float(x[0]), arrival_mjd2000=float(x[0] + f[1]),
                  pareto=False, polished=False, x=list(x))
    record.update(tags)
    return record


def flag_front(points):
    """Flag the non-dominated points of a cloud in place.

    Args:
        points (list[dict]): Cloud records; each gets its ``pareto`` field set.

    Returns:
        int: Number of points on the front.
    """
    front = set(pg.non_dominated_front_2d([[p['cost'], p['tof_days']] for p in points]))
    for i, p in enumerate(points):
        p['pareto'] = i in front
    return len(front)


def optimise_front(args, launch_window=None, log=print):
    """Run an NSGA-II archipelago over one launch window and pool every island.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from
            ``search.add_search_args``.
        launch_window (list[pykep.epoch], optional): ``[start, end]`` launch
            window; ``None`` uses the default window. Defaults to None.
        log (Callable[[str], None], optional): Print-like callable for progress
            lines. Defaults to print.

    Returns:
        tuple[list[dict], float]: ``(points, wall_seconds)``, with one record per
        individual of every island.

    Raises:
        ValueError: If ``args.pop`` is not a multiple of four, as NSGA-II requires.
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
    """Run one NSGA-II search per launch slice and pool the results.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        tuple[list[dict], float]: The pooled cloud, each point tagged with its
        ``slice``, and the total wall time in seconds.
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
    """MBH-polish a TOF-spanning sub-sample of the front and re-flag the cloud.

    Each chosen point is pinned to a narrow band around its own launch date and
    time of flight, then locally minimised. Polished points are appended as new
    records and dominance is recomputed over the whole cloud.

    Args:
        points (list[dict]): The cloud; polished points are appended and the
            ``pareto`` flags recomputed in place.
        args (argparse.Namespace): Parsed CLI arguments, including the
            ``--polish*`` options.

    Returns:
        dict: Polish summary recorded into the result's settings.
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
    """Search or sweep, flag and polish the front, and write the cloud."""
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
