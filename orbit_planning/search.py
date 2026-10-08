"""Provide the search plumbing shared by ``optimize.py`` and ``pareto.py``.

Holds the archipelago setup, the MBH + COBYLA polish and its process pool, the
common CLI options and the result-file provenance.
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
    """Accept any arguments and print nothing, for use as a ``log`` callable."""


def run_archipelago(prob, algo, islands, pop_size, rounds, seed=None, log=print):
    """Evolve a ring-topology archipelago and return it with its wall time.

    Islands exchange migrants between evolution rounds, so ``rounds`` is also the
    number of migration events.

    Args:
        prob (pygmo.problem): Problem to evolve (see ``models.problem``).
        algo (pygmo.algorithm): Algorithm every island runs.
        islands (int): Number of islands, each a worker process.
        pop_size (int): Individuals per island.
        rounds (int): Evolution rounds.
        seed (int, optional): Archipelago seed. Defaults to None.
        log (Callable[[str], None], optional): Print-like callable for progress
            lines. Defaults to print.

    Returns:
        tuple[pygmo.archipelago, float]: The evolved archipelago and its wall
        time in seconds.
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
    """Return the best island champion of an archipelago.

    Args:
        archi (pygmo.archipelago): An evolved single-objective archipelago.

    Returns:
        tuple[float, list[float], int]: ``(cost, x, island_index)`` of the best
        champion.
    """
    fs, xs = archi.get_champions_f(), archi.get_champions_x()
    best = min(range(len(fs)), key=lambda i: fs[i][0])
    return float(fs[best][0]), list(xs[best]), best


def polish(prob, x, maxeval=5000, stop=5, perturb=1e-3):
    """Refine a decision vector by monotonic basin hopping around COBYLA.

    COBYLA is derivative-free, which suits the pykep problems: they expose no
    gradient.

    Args:
        prob (pygmo.problem): Single-objective problem to minimise.
        x (Sequence[float]): Starting decision vector, inside ``prob``'s bounds.
        maxeval (int, optional): COBYLA evaluation cap per local descent.
            Defaults to 5000.
        stop (int, optional): Consecutive non-improving hops before MBH stops.
            Defaults to 5.
        perturb (float, optional): Hop size as a fraction of each variable's
            range. Defaults to 1e-3.

    Returns:
        tuple[float, list[float]]: ``(cost, x)`` of the best point found, never
        worse than the start.
    """
    local = pg.nlopt('cobyla')
    local.maxeval = maxeval
    pop = pg.population(prob)
    pop.push_back(x)
    pop = pg.algorithm(pg.mbh(pg.algorithm(local), stop=stop, perturb=perturb)).evolve(pop)
    return float(pop.champion_f[0]), list(pop.champion_x)


def polish_job(job):
    """Polish one point inside a ``parallel_polish`` worker process.

    Args:
        job (dict): ``model``, ``sequence``, ``x``, ``window_mjd2000`` (``[lo,
            hi]`` launch bounds in MJD2000), optional ``tof_days`` (``[lo, hi]``
            time-of-flight band), ``maxeval`` and ``stop``.

    Returns:
        dict: ``before``, ``after``, ``x`` and ``seconds``.
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

    Spawned workers re-import ``__main__``, so every caller keeps its entry point
    behind ``if __name__ == '__main__'``.

    Args:
        jobs (list[dict]): Job dicts as ``polish_job`` expects.
        workers (int, optional): Pool size, capped at the number of jobs;
            ``None`` uses the CPU count. Defaults to None.

    Returns:
        list[dict]: One result per job, in order.
    """
    if not jobs:
        return []
    workers = min(workers or os.cpu_count() or 1, len(jobs))
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(polish_job, jobs))


# --------------------------------------------------------------------- CLI --
def add_search_args(parser, islands, pop, gen, rounds):
    """Add the CLI options both runners share to a parser.

    Args:
        parser (argparse.ArgumentParser): Parser to extend.
        islands (int): Default island count.
        pop (int): Default individuals per island.
        gen (int): Default generations per island per round.
        rounds (int): Default number of evolution rounds.
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
    """Resolve the output file of a run and create its directory.

    Args:
        args (argparse.Namespace): Parsed CLI arguments with ``out``, ``model``
            and ``sequence``.
        filename (str): File name under ``results/<model>/<sequence>/``, used
            when ``--out`` is not given.

    Returns:
        pathlib.Path: The output path.
    """
    out = paths.Path(args.out) if args.out else paths.results_path(args.model, args.sequence.upper(), filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def result_header(args):
    """Build the provenance header every result file starts with.

    Args:
        args (argparse.Namespace): Parsed CLI arguments with ``model`` and
            ``sequence``.

    Returns:
        dict: Schema version, model, sequence, labels, cost-axis metadata and the
        search bounds.
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
    """Build the ``settings`` block of a result file.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        wall_seconds (float): Total elapsed time of the run.
        **extra (Any): Run-specific entries to record, such as polish statistics.

    Returns:
        dict: Search options, timing, the model's objective and the destination's
        capture parameters.
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
