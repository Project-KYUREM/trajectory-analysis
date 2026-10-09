"""Find the cheapest trajectory for one model and sequence.

Example: ``python orbit_planning/optimize.py --model ballistic --sequence EJN``;
``--sweep`` runs one search per launch slice. Constrained models (``dfd``) are
searched with the constraints folded into a penalty and polished on the true
constrained problem; their champions are ranked feasible-first. Run it as a script, never from a
notebook cell: pygmo evaluates pykep problems in spawned workers that
re-import ``__main__``, and the ``if __name__ == '__main__'`` guard keeps them
from re-running the search.
"""
import argparse
import json
import time

import pygmo as pg

import mission
import models
import search


def sade(gen):
    """Return pygmo's self-adaptive differential evolution for ``gen`` generations."""
    return pg.algorithm(pg.sade(gen=gen, memory=True))


def optimise_champion(args, launch_window=None, log=print):
    """Run an archipelago search, then polish its champion with MBH.

    The local solver is the model's (``models.Model.local``): COBYLA for the
    chemical models, IPOPT on the constrained problem for ``dfd``.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from
            ``search.add_search_args``.
        launch_window (list[pykep.epoch], optional): ``[start, end]`` launch
            window; ``None`` uses the default window. Defaults to None.
        log (Callable[[str], None], optional): Print-like callable for progress
            lines. Defaults to print.

    Returns:
        tuple[dict, float]: The best trajectory found (``cost``, ``x``,
        ``feasible``, ``violation``) and the wall time in seconds.
    """
    t0 = time.time()
    model = models.get(args.model)
    prob = models.problem(args.model, args.sequence, launch_window=launch_window)
    archi, _ = search.run_archipelago(prob, sade(args.gen), args.islands, args.pop,
                                      args.rounds, args.seed, log)
    cost, x, island = search.best_champion(archi)
    log("island champions: " + ", ".join(f"{f[0]:.0f}" for f in archi.get_champions_f()))
    log(f"best of archipelago : {cost:9.0f}  (island {island}{', penalised' if model.constrained else ''})")

    window = launch_window or mission.LAUNCH_WINDOW
    raw = dict(models.evaluate(args.model, args.sequence, x, window), x=x)
    if not model.constrained:
        raw['cost'] = cost   # the archipelago's own objective, as before
    r = search.polish_job(dict(model=args.model, sequence=args.sequence, x=x,
                               window_mjd2000=[window[0].mjd2000, window[1].mjd2000],
                               tof_days=None, maxeval=args.polish_maxeval, stop=args.polish_stop))
    best = min([raw, dict(cost=r['after'], x=r['x'], feasible=r['feasible'],
                          violation=r['violation'])], key=search.rank)
    if best is raw:
        log(f"{model.local.upper()} polish did not improve on the archipelago champion")
    else:
        log(f"after {model.local.upper()} polish : {best['cost']:9.0f}")
    if not best['feasible']:
        log(f"WARNING: champion infeasible (max violation {best['violation']:.2e}); "
            f"run more rounds, a larger --polish-maxeval or --sweep")
    return best, time.time() - t0


def sweep(args, slices_out):
    """Search each launch slice separately, then polish every slice champion.

    The archipelagos run one after another because each uses every core; the
    serial polishes then run together in a process pool.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        slices_out (pathlib.Path): File that receives every slice champion.

    Returns:
        tuple[dict, float]: The best slice record and the total wall time in
        seconds.
    """
    slices = mission.launch_slices(args.sequence, width_days=args.slice_width,
                                   overlap=args.overlap)
    print(f"sweeping {len(slices)} launch slices "
          f"({args.islands} islands x pop {args.pop}, {args.rounds} rounds x {args.gen} gen)")
    t_all = time.time()

    champions = []
    for i, window in enumerate(slices):
        prob = models.problem(args.model, args.sequence, launch_window=window)
        archi, wall = search.run_archipelago(prob, sade(args.gen), args.islands, args.pop,
                                             args.rounds, args.seed, log=search.silent)
        cost, x, _ = search.best_champion(archi)
        score = models.evaluate(args.model, args.sequence, x, window)
        if models.get(args.model).constrained:
            cost = score['cost']   # the true objective, not the penalised one
        champions.append(dict(slice=i, window_mjd2000=[window[0].mjd2000, window[1].mjd2000],
                              raw_cost=cost, cost=cost, x=x, feasible=score['feasible'],
                              violation=score['violation'], wall_seconds=round(wall, 1)))
        print(f"  slice {i:2d}  {window[0].mjd:9.1f}..{window[1].mjd:9.1f}  raw {cost:9.0f}"
              f"{'' if score['feasible'] else ' (infeasible)'}  {wall:5.1f}s")

    print(f"\npolishing {len(champions)} slice champions in parallel")
    jobs = [dict(model=args.model, sequence=args.sequence, x=c['x'],
                 window_mjd2000=c['window_mjd2000'], tof_days=None,
                 maxeval=args.polish_maxeval, stop=args.polish_stop) for c in champions]
    print(f"{'#':>3}  {'launch':>10}  {'raw':>9}  {'polished':>9}  {'gain':>6}  {'secs':>6}  feasible")
    for c, r in zip(champions, search.parallel_polish(jobs, args.polish_workers)):
        polished = dict(cost=r['after'], feasible=r['feasible'])
        if search.rank(polished) < search.rank(c):
            c.update(cost=r['after'], x=r['x'], feasible=r['feasible'], violation=r['violation'])
        c['launch_date'] = mission.mjd2000_to_datetime(c['x'][0]).strftime('%Y-%m-%d')  # x[0]: launch epoch
        gain = 100 * (c['raw_cost'] - c['cost']) / c['raw_cost'] if c['raw_cost'] else 0.0
        print(f"{c['slice']:3d}  {c['launch_date']:>10}  {c['raw_cost']:9.0f}  "
              f"{c['cost']:9.0f}  {gain:5.1f}%  {r['seconds']:6.1f}  {'yes' if c['feasible'] else 'NO'}")

    wall = time.time() - t_all
    best = min(champions, key=search.rank)
    print(f"\nsweep done in {wall / 60:.1f} min | best slice #{best['slice']} "
          f"at {best['cost']:.0f} (launch {best['launch_date']}"
          f"{'' if best['feasible'] else ', INFEASIBLE'})")
    slices_out.write_text(json.dumps({**search.result_header(args),
                                      'settings': search.run_settings(args, wall),
                                      'slices': champions}, indent=2))
    print(f"wrote {slices_out}  ({len(champions)} slices, every x retained)")
    return best, wall


def main():
    """Search or sweep, then write the champion unless an existing one scores better."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    search.add_search_args(parser, islands=20, pop=60, gen=250, rounds=16)
    args = search.finish_args(parser.parse_args())
    out = search.output_path(args, 'champion.json')

    if args.sweep:
        best, wall = sweep(args, out.with_name('slices.json'))
        extra = {'best_slice': best['slice']}
    else:
        best, wall = optimise_champion(args)
        extra = {}

    # Never silently replace a better champion. Re-score the existing file
    # under the current model rather than trusting its stored number, which
    # may have been produced under a different objective.
    if out.is_file():
        try:
            previous = json.loads(out.read_text())
            prev = models.evaluate(args.model, args.sequence, previous['x'])
            if search.rank(prev) < search.rank(best):
                print(f"\nREFUSING to overwrite {out}: existing champion scores {prev['cost']:.0f} "
                      f"(feasible: {prev['feasible']}) under the current objective, better than "
                      f"this run's {best['cost']:.0f} (feasible: {best['feasible']}).\n"
                      f"   Pass --out <other path> to keep both, or delete the file.")
                return
        except (KeyError, ValueError) as exc:
            print(f"(could not score existing {out}: {exc} -- overwriting)")

    out.write_text(json.dumps({**search.result_header(args), 'cost': best['cost'], 'x': best['x'],
                               'feasible': best['feasible'], 'violation': best['violation'],
                               'settings': search.run_settings(args, wall, **extra)}, indent=2))
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
