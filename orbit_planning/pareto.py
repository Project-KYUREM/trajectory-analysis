"""Find the cost / time-of-flight Pareto front for one model and sequence.

Example: ``python orbit_planning/pareto.py --model capture --sequence EVEEJN
--sweep``. NSGA-II individuals from every island are kept and the
non-dominated ones are flagged. A TOF-spanning sub-sample of the front is then
MBH-polished and dominance is recomputed over the whole cloud. Run it as a
script, never from a notebook cell (see ``optimize.py``).

Constrained models (``dfd``): NSGA-II searches the penalised problem, every
individual is re-scored on the true one (``feasible``, ``violation``), the
polish seeds are spread over the cloud's cruise times (``_penalised_seeds``),
and only feasible points can be on the published front. Raw Sims-Flanagan
individuals are rarely feasible; the polished points are the result, so give
``--polish`` generously, and use ``--sweep``: one population per launch slice
is what gives the seeds real diversity (a single NSGA-II population on the
penalised problem collapses to a few near-identical trajectories).
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


def _point(x, score, search_f=None, **tags):
    """Build one cloud record from a decision vector and its score.

    Args:
        x (Sequence[float]): Decision vector; ``x[0]`` is the launch epoch in
            MJD2000.
        score (dict): ``models.evaluator`` output for ``x``.
        search_f (Sequence[float], optional): Fitness ``(cost, tof)`` the
            global search saw (penalised for a constrained model); defaults to
            the true ``(cost, tof_days)``.
        **tags (Any): Extra fields set on the record.

    Returns:
        dict: The point record.
    """
    record = dict(cost=score['cost'], tof_days=score['tof_days'],   # x[0]: launch epoch (models.py contract)
                  launch_mjd2000=float(x[0]), arrival_mjd2000=float(x[0] + score['cruise_days']),
                  cruise_days=score['cruise_days'],
                  feasible=score['feasible'], violation=score['violation'],
                  search_f=[float(v) for v in (search_f if search_f is not None
                                               else (score['cost'], score['tof_days']))],
                  pareto=False, polished=False, x=list(x))
    record.update(tags)
    return record


def flag_front(points):
    """Flag the non-dominated feasible points of a cloud in place.

    Args:
        points (list[dict]): Cloud records; each gets its ``pareto`` field set.

    Returns:
        int: Number of points on the front.
    """
    feasible = [i for i, p in enumerate(points) if p.get('feasible', True)]
    front = set()
    if len(feasible) > 1:
        nd = pg.non_dominated_front_2d([[points[i]['cost'], points[i]['tof_days']] for i in feasible])
        front = {feasible[k] for k in nd}
    elif feasible:
        front = set(feasible)
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
    score = models.evaluator(args.model, args.sequence, launch_window)
    points = [_point(x, score(x), search_f=f) for island in archi
              for x, f in zip(island.get_population().get_x(),
                              island.get_population().get_f())]
    n_feasible = sum(p['feasible'] for p in points)
    log(f"pooled {len(points)} individuals ({n_feasible} feasible)")
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
        n_feasible = sum(p['feasible'] for p in points)
        best = min((p['cost'] for p in points if p['feasible']), default=float('nan'))
        print(f"  slice {i:2d}  {window[0].mjd:9.1f}..{window[1].mjd:9.1f}  "
              f"{len(points):5d} pts ({n_feasible} feasible)  best {best:9.0f}  {wall:5.1f}s")
        pooled.extend(points)
    print(f"sweep done in {(time.time() - t_all) / 60:.1f} min | {len(pooled)} points pooled")
    return pooled, time.time() - t_all


def _penalised_seeds(points, n):
    """Pick polish seeds for a constrained model, spread over cruise time.

    The penalised front is degenerate (the penalty dominates both objectives,
    so it collapses to a handful of clones), so instead the distinct
    individuals of the whole cloud are sorted by cruise time and split into
    ``n`` equal-count groups (NSGA-II packs a population into a narrow time
    range, so equal-width bins would leave most empty); the individual with the
    lowest penalised cost in each group is a seed.

    Args:
        points (list[dict]): The cloud, with ``search_f`` and ``cruise_days``.
        n (int): Number of seeds wanted.

    Returns:
        tuple[list[int], list[int]]: ``(front, picks)`` in the shape
        ``polish_front`` uses: ``front`` the seed indices sorted by time,
        ``picks`` positions into it (all of them).
    """
    if not points or n <= 0:
        return [], []
    # NSGA-II populations carry near-clones; polishing two from the same launch
    # day and cruise day wastes a worker. Keep the least penalised of each.
    distinct = {}
    for i, p in enumerate(points):
        key = (round(p['launch_mjd2000']), round(p['cruise_days']))
        if key not in distinct or p['search_f'][0] < points[distinct[key]]['search_f'][0]:
            distinct[key] = i
    order = sorted(distinct.values(), key=lambda i: points[i]['cruise_days'])
    seeds = {min(group, key=lambda i: points[i]['search_f'][0])
             for group in np.array_split(np.array(order), min(n, len(order))) if len(group)}
    seeds = {int(i) for i in seeds}
    front = sorted(seeds, key=lambda i: points[i]['tof_days'])
    return front, list(range(len(front)))


def polish_front(points, args):
    """MBH-polish a TOF-spanning sub-sample of the front and re-flag the cloud.

    The seeds come from the front (for a constrained model, from
    ``_penalised_seeds``: the least-penalised individual per cruise-time bin,
    so infeasible seeds are allowed).
    Each chosen point is pinned to a narrow band around its own launch date and
    cruise time, then locally minimised. Polished points are appended as new
    records and dominance is recomputed over the whole cloud.

    Args:
        points (list[dict]): The cloud; polished points are appended and the
            ``pareto`` flags recomputed in place.
        args (argparse.Namespace): Parsed CLI arguments, including the
            ``--polish*`` options.

    Returns:
        dict: Polish summary recorded into the result's settings.
    """
    if models.get(args.model).constrained:
        front, picks = _penalised_seeds(points, args.polish)
    else:
        front = sorted((i for i, p in enumerate(points) if p['pareto']),
                       key=lambda i: points[i]['tof_days'])
        picks = sorted({int(round(k)) for k in np.linspace(0, len(front) - 1,
                                                           min(args.polish, len(front)))})
    if not front or args.polish <= 0:
        return {'n_polished': 0}
    jobs = []
    for k in picks:
        src = points[front[k]]
        jobs.append(dict(model=args.model, sequence=args.sequence, x=src['x'],
                         window_mjd2000=[src['x'][0] - args.polish_pad_days,
                                         src['x'][0] + args.polish_pad_days],
                         tof_days=[src['cruise_days'] * (1 - args.polish_band),
                                   src['cruise_days'] * (1 + args.polish_band)],
                         maxeval=args.polish_maxeval, stop=args.polish_stop))

    local = models.get(args.model).local.upper()
    print(f"\npolishing {len(jobs)} of {len(front)} front points in parallel "
          f"(MBH stop={args.polish_stop}, {local} maxeval={args.polish_maxeval})")
    print(f"{'#':>3}  {'TOF yr':>7}  {'before':>9}  {'after':>9}  {'gain':>6}  {'secs':>6}  feasible")
    t0 = time.time()
    results = search.parallel_polish(jobs, args.polish_workers)
    score = models.evaluator(args.model, args.sequence)

    gains = []
    for k, r in zip(picks, results):
        src = points[front[k]]
        # a gain is only meaningful between two feasible points
        gain = (100 * (r['before'] - r['after']) / r['before']
                if r['before'] and r['before_feasible'] and r['feasible'] else None)
        if gain is not None:
            gains.append(gain)
        print(f"{k:3d}  {src['tof_days'] / 365.25:7.2f}  {r['before']:9.0f}  {r['after']:9.0f}  "
              f"{'   --' if gain is None else f'{gain:5.1f}'}%  {r['seconds']:6.1f}  "
              f"{'yes' if r['feasible'] else 'NO'}")
        points.append(_point(r['x'], score(r['x']), polished=True,
                             polished_from=front[k], slice=src.get('slice')))

    n_front = flag_front(points)
    on_front = sum(1 for p in points if p['polished'] and p['pareto'])
    n_feasible = sum(1 for r in results if r['feasible'])
    mean_gain = float(np.mean(gains)) if gains else float('nan')
    print(f"polish done in {time.time() - t0:.1f}s | mean gain {mean_gain:.1f}% | "
          f"{n_feasible}/{len(jobs)} feasible | "
          f"{on_front}/{len(jobs)} polished points on the recomputed front ({n_front} total)")
    return dict(n_polished=len(jobs), n_feasible=n_feasible, band=args.polish_band,
                pad_days=args.polish_pad_days, maxeval=args.polish_maxeval, stop=args.polish_stop,
                local=models.get(args.model).local,
                mean_gain_pct=mean_gain if gains else None)


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
    args = search.finish_args(parser.parse_args())
    out = search.output_path(args, 'pareto.json')

    points, wall = sweep(args) if args.sweep else optimise_front(args)
    print(f"{flag_front(points)} on the non-dominated front")
    polish_info = polish_front(points, args)

    front = [p for p in points if p['pareto']]
    if front:
        cheapest, fastest = min(front, key=lambda p: p['cost']), min(front, key=lambda p: p['tof_days'])
        print(f"  cheapest on front : {cheapest['cost']:8.0f} @ {cheapest['tof_days'] / 365.25:5.2f} yr")
        print(f"  fastest on front  : {fastest['cost']:8.0f} @ {fastest['tof_days'] / 365.25:5.2f} yr")
    else:
        print("  NO feasible point: raise --polish, --rounds / --gen, --polish-maxeval, or --sweep")

    out.write_text(json.dumps({**search.result_header(args),
                               'settings': search.run_settings(args, wall, polish=polish_info),
                               'points': points}, indent=2))
    print(f"wrote {out}  ({len(points)} points)")


if __name__ == '__main__':
    main()
