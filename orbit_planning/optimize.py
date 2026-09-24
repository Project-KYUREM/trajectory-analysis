"""Global search for the EVEEJN trajectory. Run as a script, not from a cell:

    python orbit_planning/optimize.py --islands 12 --rounds 16

This file exists separately from the notebook for exactly one reason: the
``if __name__ == "__main__"`` guard below.

mga_1dsm is a Python UDP and reports thread_safety.none, so pygmo cannot use
thread_island for it and falls back to mp_island. mp_island in turn forces
multiprocessing's "spawn" context (pygmo/_py_islands.py, _get_spawn_context)
even on Linux where fork is the default. spawn rebuilds each child by
re-importing __main__ -- so if the evolve call sits at module level in a
notebook, every child re-runs the whole script, reaches the evolve again, and
tries to spawn grandchildren. Python aborts that with "An attempt has been made
to start a new process before the current process has finished its
bootstrapping phase." The guard is what stops the recursion.

Unlike a stock archipelago, this one passes a real topology. pygmo's default is
Unconnected -- zero edges, no migration, i.e. N independent runs. A ring lets
islands exchange migrants, which is the actual point of the island model on a
landscape this multi-modal.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygmo as pg

import mission


def optimise(n_islands, pop_size, gen, rounds, seed=None, launch_window=None,
             quiet=False):
    prob = pg.problem(mission.build_udp(launch_window=launch_window))
    algo = pg.algorithm(pg.sade(gen=gen, memory=True))

    # Migration happens between evolution rounds, so the rounds count is also
    # the number of migration events. One big evolve() would migrate never.
    # pygmo rejects seed=None outright, so only pass it when it is set.
    archi_kwargs = dict(n=n_islands, algo=algo, prob=prob,
                        pop_size=pop_size, t=pg.ring())
    if seed is not None:
        archi_kwargs['seed'] = seed
    archi = pg.archipelago(**archi_kwargs)

    topo_name = str(archi.get_topology()).splitlines()[0].strip()
    if not quiet:
        print(f"dimension {prob.get_nx()} | {n_islands} islands x pop {pop_size} "
              f"| {rounds} rounds x {gen} gen | {topo_name}")

    t0 = time.time()
    archi.evolve(rounds)
    archi.wait_check()
    wall = time.time() - t0

    champs_f = archi.get_champions_f()
    champs_x = archi.get_champions_x()
    best_i = min(range(n_islands), key=lambda i: champs_f[i][0])
    best_dv, best_x = champs_f[best_i][0], list(champs_x[best_i])

    if not quiet:
        print(f"archipelago done in {wall:.1f}s")
        print("island champions (m/s): " + ", ".join(f"{f[0]:.0f}" for f in champs_f))
        print(f"best of archipelago  : {best_dv:9.0f} m/s  (island {best_i})")

    # Polish. mga_1dsm exposes no analytic gradient (prob.has_gradient() is
    # False), so SLSQP is unusable -- NLopt raises on the missing objective
    # gradient. COBYLA is derivative-free.
    local = pg.nlopt('cobyla')
    local.maxeval = 5000
    pop = pg.population(prob)
    pop.push_back(best_x)
    pop = pg.algorithm(pg.mbh(pg.algorithm(local), stop=5, perturb=1e-3)).evolve(pop)
    if pop.champion_f[0] < best_dv:
        best_dv, best_x = pop.champion_f[0], list(pop.champion_x)
        if not quiet:
            print(f"after COBYLA polish  : {best_dv:9.0f} m/s")
    elif not quiet:
        print("COBYLA polish did not improve on the archipelago champion")

    # Return the FULL elapsed time. `wall` above covers only the archipelago;
    # the MBH polish is serial and routinely costs more than the parallel
    # global phase, so reporting `wall` made the sweep look ~10x faster than it
    # was.
    return best_dv, best_x, time.time() - t0


def sweep(args):
    """One archipelago per launch slice.

    Launch opportunities recur at the Earth-Venus synodic period, separated by
    stretches of very poor DV. A single search over the whole window has to
    resolve ~20 nearly disconnected basins at once and wastes most of its
    budget in the valleys between them; per-slice searches are each as easy as
    the original narrow-window problem was, and the sweep doubles as the
    launch-window scan you actually want to look at.
    """
    slices = mission.launch_slices(width_days=args.slice_width,
                                   overlap=args.overlap)
    print(f"sweeping {len(slices)} launch slices "
          f"({args.islands} islands x pop {args.pop}, "
          f"{args.rounds} rounds x {args.gen} gen each)")
    print(f"{'#':>3}  {'from':>10}  {'to':>10}  {'best DV':>10}  {'launch':>12}  {'secs':>6}")

    results, t_all = [], time.time()
    for i, window in enumerate(slices):
        dv, x, wall = optimise(args.islands, args.pop, args.gen, args.rounds,
                               args.seed, launch_window=window, quiet=True)
        launch = mission.mjd2000_to_datetime(x[0]).strftime('%Y-%m-%d')
        print(f"{i:3d}  {window[0].mjd:10.1f}  {window[1].mjd:10.1f}  "
              f"{dv:10.0f}  {launch:>12}  {wall:6.1f}")
        results.append({'slice': i,
                        'window_mjd': [window[0].mjd, window[1].mjd],
                        'best_dv_ms': dv, 'x': x,
                        'launch_date': launch,
                        'wall_seconds': round(wall, 1)})

    total = time.time() - t_all
    best = min(results, key=lambda r: r['best_dv_ms'])
    print(f"\nsweep done in {total/60:.1f} min")
    print(f"best slice: #{best['slice']} at {best['best_dv_ms']:.0f} m/s "
          f"(launch {best['launch_date']})")

    payload = {
        'sequence': mission.LABELS,
        'tof_bounds_days': mission.MISSION_TOF_DAYS,
        'vinf_bounds_kms': mission.LAUNCH_VINF_KMS,
        'settings': {'islands': args.islands, 'pop': args.pop, 'gen': args.gen,
                     'rounds': args.rounds, 'seed': args.seed, 'topology': 'ring',
                     'slice_width_days': args.slice_width, 'overlap': args.overlap,
                     'wall_seconds': round(total, 1),
                     **mission.objective_config()},
        'slices': results,
    }
    args.slices_out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.slices_out}  ({len(results)} slices, every x retained)")
    return best, payload['settings']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--islands', type=int, default=20)
    ap.add_argument('--pop', type=int, default=60)
    ap.add_argument('--gen', type=int, default=250)
    ap.add_argument('--rounds', type=int, default=16)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', type=Path, default=mission.CHAMPION_PATH)
    ap.add_argument('--sweep', action='store_true',
                    help='partition the launch window into synodic slices and '
                         'search each one separately')
    ap.add_argument('--slice-width', type=float, default=None, dest='slice_width',
                    help='slice width in days (default: the Earth-Venus synodic '
                         'period, ~583 d)')
    ap.add_argument('--overlap', type=float, default=0.2,
                    help='fractional overlap between adjacent slices')
    ap.add_argument('--slices-out', type=Path, dest='slices_out',
                    default=mission.HERE / 'slices.json')
    args = ap.parse_args()

    if args.sweep:
        best, settings = sweep(args)
        best_dv, best_x = best['best_dv_ms'], best['x']
        settings = dict(settings, best_slice=best['slice'])
    else:
        best_dv, best_x, wall = optimise(args.islands, args.pop, args.gen,
                                         args.rounds, args.seed)
        settings = {'islands': args.islands, 'pop': args.pop, 'gen': args.gen,
                    'rounds': args.rounds, 'seed': args.seed,
                    'topology': 'ring', 'wall_seconds': round(wall, 1),
                    **mission.objective_config()}

    # Do not silently overwrite a better champion. Compare against the current
    # objective, not the stored fitness -- a champion saved under a different
    # objective has a different score now, and the stored number is meaningless.
    if args.out.is_file():
        try:
            prev = json.loads(args.out.read_text())
            prev_dv = mission.build_udp().fitness(prev['x'])[0]
            if prev_dv < best_dv:
                print(f"\nREFUSING to overwrite {args.out}: the existing champion "
                      f"scores {prev_dv:.0f} m/s under the current objective, "
                      f"better than this run's {best_dv:.0f} m/s.")
                print("   Pass --out <other path> to keep both, or delete the file "
                      "to force a replacement.")
                return
        except (KeyError, ValueError) as exc:
            print(f"(could not score existing {args.out}: {exc} -- overwriting)")

    args.out.write_text(json.dumps({
        'sequence': mission.LABELS,
        'best_dv_ms': best_dv,
        'x': best_x,
        'tof_bounds_days': mission.MISSION_TOF_DAYS,
        'vinf_bounds_kms': mission.LAUNCH_VINF_KMS,
        'settings': settings,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
