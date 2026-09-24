"""Search plumbing shared by ``optimize.py`` and ``pareto.py``.

Archipelago setup, the MBH + COBYLA polish, the parallel polish pool, the
common CLI surface and the result-file provenance all live here so the two
runners differ only in their algorithm and in what they collect.
"""
import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pygmo as pg
import pykep as pk

import mission
import models
import paths


def silent(*_args, **_kwargs):
    """A ``log`` callable that prints nothing."""


def run_archipelago(prob, algo, islands, pop_size, rounds, seed=None, log=print):
    """Evolve a ring-topology archipelago and return it with its wall time.

    Migration happens between evolution rounds, so ``rounds`` is also the
    number of migration events; one big ``evolve()`` would never migrate.
    pygmo's default topology is *unconnected* (no migration at all), which is
    why the ring is passed explicitly.

    Args:
        prob: A ``pg.problem`` (see ``models.problem``).
        algo: A ``pg.algorithm``.
        islands: Number of islands (each is a worker process).
        pop_size: Individuals per island.
        rounds: Evolution rounds; migration occurs between them.
        seed: Optional archipelago seed (pygmo rejects ``None`` outright).
        log: Print-like callable for progress lines.

    Returns:
        ``(archipelago, wall_seconds)``.
    """
    kwargs = dict(n=islands, algo=algo, prob=prob, pop_size=pop_size, t=pg.ring())
    if seed is not None:
        kwargs['seed'] = seed
    archi = pg.archipelago(**kwargs)
    log(f"dimension {prob.get_nx()} | {prob.get_nobj()} objective(s) | "
        f"{islands} islands x pop {pop_size} | {rounds} rounds x {algo.get_name()}")
    t0 = time.time()
    archi.evolve(rounds)
    archi.wait_check()
    wall = time.time() - t0
    log(f"archipelago done in {wall:.1f}s")
    return archi, wall


def best_champion(archi):
    """``(cost, x, island_index)`` of the best island champion."""
    fs, xs = archi.get_champions_f(), archi.get_champions_x()
    best = min(range(len(fs)), key=lambda i: fs[i][0])
    return float(fs[best][0]), list(xs[best]), best


def polish(prob, x, maxeval=5000, stop=5, perturb=1e-3):
    """Monotonic basin hopping around COBYLA from a starting vector.

    The pykep UDPs expose no gradient, so SLSQP is unusable; COBYLA is
    derivative-free. MBH repeatedly perturbs and re-descends, which walks down
    the funnel of nested local minima typical of gravity-assist landscapes.

    Args:
        prob: Problem to minimise (single objective).
        x: Starting decision vector, must lie inside ``prob``'s bounds.
        maxeval: COBYLA evaluation cap per local descent.
        stop: Consecutive non-improving hops before MBH stops.
        perturb: Hop size as a fraction of each variable's range.

    Returns:
        ``(cost, x)`` of the best point found (never worse than the start).
    """
    local = pg.nlopt('cobyla')
    local.maxeval = maxeval
    pop = pg.population(prob)
    pop.push_back(x)
    pop = pg.algorithm(pg.mbh(pg.algorithm(local), stop=stop, perturb=perturb)).evolve(pop)
    return float(pop.champion_f[0]), list(pop.champion_x)


def polish_job(job):
    """Worker body for ``parallel_polish``; runs in a spawned process.

    Args:
        job: Dict with ``model``, ``sequence``, ``x``, ``window_mjd2000``
            (``[lo, hi]`` launch bounds), optional ``tof_days`` (``[lo, hi]``,
            the epsilon-constraint band), ``maxeval`` and ``stop``.

    Returns:
        Dict with ``before``, ``after``, ``x`` and ``seconds``.
    """
    lo, hi = job['window_mjd2000']
    prob = models.problem(job['model'], job['sequence'],
                          launch_window=[pk.epoch(lo, 'mjd2000'), pk.epoch(hi, 'mjd2000')],
                          tof_days=job.get('tof_days'))
    before = float(prob.fitness(job['x'])[0])
    t0 = time.time()
    after, x = polish(prob, job['x'], maxeval=job['maxeval'], stop=job['stop'])
    return dict(before=before, after=after, x=x, seconds=time.time() - t0)


def parallel_polish(jobs, workers=None):
    """Run ``polish_job`` over many jobs in a spawn process pool.

    Each polish is strictly serial, so independent points are refined one per
    process. spawn re-imports ``__main__``, which is why every caller keeps its
    entry point behind ``if __name__ == '__main__'``.

    Args:
        jobs: Job dicts as ``polish_job`` expects.
        workers: Pool size; defaults to the CPU count.

    Returns:
        One result dict per job, in order.
    """
    if not jobs:
        return []
    workers = min(workers or os.cpu_count() or 1, len(jobs))
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(polish_job, jobs))


# --------------------------------------------------------------------- CLI --
def add_search_args(parser, islands, pop, gen, rounds):
    """Attach the CLI surface both runners share.

    Per-script defaults are passed in so the difference between the runners is
    visible at their call sites rather than buried in two copies of this list.

    Args:
        parser: The ``argparse.ArgumentParser`` to extend.
        islands: Default island count (one worker process each).
        pop: Default individuals per island.
        gen: Default generations per island per round.
        rounds: Default number of rounds (migration happens between rounds).
    """
    parser.add_argument('--model', default='capture', choices=sorted(models.MODELS),
                        help='propulsion/objective model')
    parser.add_argument('--sequence', default=paths.DEFAULT_SEQUENCES[0],
                        help='flyby sequence string, e.g. EVEEJN (Earth -> destination)')
    parser.add_argument('--islands', type=int, default=islands,
                        help='islands in the archipelago; each is a worker process')
    parser.add_argument('--pop', type=int, default=pop, help='individuals per island')
    parser.add_argument('--gen', type=int, default=gen,
                        help='generations each island evolves per round')
    parser.add_argument('--rounds', type=int, default=rounds,
                        help='rounds of evolution; islands exchange migrants between rounds')
    parser.add_argument('--seed', type=int, default=None, help='archipelago seed (reproducible runs)')
    parser.add_argument('--out', type=str, default=None,
                        help='output file (default: results/<model>/<sequence>/...)')
    parser.add_argument('--sweep', action='store_true',
                        help='partition the launch window into synodic slices and '
                             'search each one separately')
    parser.add_argument('--slice-width', type=float, default=None, dest='slice_width',
                        help='slice width in days (default: first-leg synodic period)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='fractional overlap between adjacent slices')
    parser.add_argument('--polish-maxeval', type=int, default=5000, dest='polish_maxeval',
                        help='COBYLA evaluations per local descent during the polish')
    parser.add_argument('--polish-stop', type=int, default=5, dest='polish_stop',
                        help='consecutive non-improving basin hops before the polish stops')
    parser.add_argument('--polish-workers', type=int, default=None, dest='polish_workers',
                        help='parallel polish processes (default: CPU count)')


def output_path(args, filename):
    """Resolve ``--out`` or ``results/<model>/<sequence>/<filename>``, creating the directory."""
    out = paths.Path(args.out) if args.out else paths.results_path(args.model, args.sequence.upper(), filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def result_header(args):
    """Provenance every result file starts with.

    Args:
        args: Parsed CLI namespace with ``model`` and ``sequence``.

    Returns:
        Dict with the schema version, model, sequence, labels, cost axis
        metadata and the search bounds in force.
    """
    model = models.get(args.model)
    setup = mission.resolve_bounds(args.sequence)
    return {
        'schema': paths.RESULT_SCHEMA,
        'model': model.name,
        'sequence': setup.sequence,
        'labels': setup.labels,
        'cost_label': model.label,
        'cost_units': model.units,
        'tof_bounds_days': mission.MISSION_TOF_DAYS,
        'vinf_bounds_kms': mission.LAUNCH_VINF_KMS,
    }


def run_settings(args, wall_seconds, **extra):
    """The ``settings`` block of a result file.

    Args:
        args: Parsed CLI namespace.
        wall_seconds: Total elapsed time of the run.
        **extra: Anything run-specific to record alongside (e.g. polish stats).

    Returns:
        Dict of search knobs, timing, the model's objective and the
        destination's capture parameters (which belong to the body, not the
        model, so they are recorded per run).
    """
    return {
        'islands': args.islands, 'pop': args.pop, 'gen': args.gen,
        'rounds': args.rounds, 'seed': args.seed, 'topology': 'ring',
        'swept': args.sweep,
        'slice_width_days': args.slice_width if args.sweep else None,
        'overlap': args.overlap if args.sweep else None,
        'wall_seconds': round(wall_seconds, 1),
        'objective': models.get(args.model).objective,
        'destination_capture': mission.resolve_bounds(args.sequence).destination.get('capture'),
        **extra,
    }
