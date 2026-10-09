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


class UnitBox:
    """Present a problem on the unit box with its objective divided by a scale.

    Gradient-based local solvers (SLSQP, IPOPT) start from an identity Hessian
    and step in raw units, so genes spanning 1e4 days next to throttles in
    [-1, 1], and an objective in kilograms next to residuals of 1e-6, stall
    them. Mapping every gene to [0, 1] and the objective to O(1) fixes that;
    constraints are left as they are (the UDP already non-dimensionalises them).

    Args:
        prob (pygmo.problem): Problem to wrap.
        scale (float): Divisor of every objective.
    """

    def __init__(self, prob, scale):
        """Store the problem, its bounds and the objective scale."""
        self._prob = prob
        self._lb, self._ub = (list(b) for b in prob.get_bounds())
        self._scale = float(scale)
        self._nobj = prob.get_nobj()

    def to_x(self, u):
        """Map a unit-box vector to the problem's decision vector."""
        return [lo + ui * (hi - lo) for ui, lo, hi in zip(u, self._lb, self._ub)]

    def to_u(self, x):
        """Map a decision vector to the unit box (zero-width genes map to 0)."""
        return [(xi - lo) / (hi - lo) if hi > lo else 0.0
                for xi, lo, hi in zip(x, self._lb, self._ub)]

    def fitness(self, u):
        """Return the wrapped fitness with its objectives scaled."""
        f = list(self._prob.fitness(self.to_x(u)))
        return [v / self._scale for v in f[:self._nobj]] + f[self._nobj:]

    def gradient(self, u):
        """Return a central-difference gradient on the unit box."""
        return pg.estimate_gradient(self.fitness, u, dx=1e-7)

    def get_bounds(self):
        """Return the unit box."""
        return [0.0] * len(self._lb), [1.0] * len(self._lb)

    def get_nobj(self):
        """Return the number of objectives."""
        return self._nobj

    def get_nec(self):
        """Return the number of equality constraints."""
        return self._prob.get_nec()

    def get_nic(self):
        """Return the number of inequality constraints."""
        return self._prob.get_nic()


def local_algorithm(name, maxeval):
    """Return the local NLP solver MBH wraps.

    Args:
        name (str): ``cobyla`` (derivative-free, the chemical models' default),
            ``slsqp`` (nlopt, finite-difference gradients) or ``ipopt``
            (interior point, finite-difference gradients; the most reliable at
            closing the match-point equalities of a Sims-Flanagan problem).
        maxeval (int): Evaluation (``cobyla``) or iteration (``slsqp``,
            ``ipopt``) cap per local descent.

    Returns:
        pygmo.nlopt or pygmo.ipopt: The configured solver.
    """
    if name == 'ipopt':
        local = pg.ipopt()
        local.set_integer_option('max_iter', maxeval)
        local.set_integer_option('print_level', 0)
        local.set_numeric_option('tol', 1e-8)
        # well inside models.C_TOL, so a converged descent is feasible by pygmo's test
        local.set_numeric_option('constr_viol_tol', 1e-2 * models.C_TOL)
        local.set_string_option('hessian_approximation', 'limited-memory')
        local.set_string_option('sb', 'yes')   # no IPOPT banner per descent
        return local
    local = pg.nlopt(name)
    local.maxeval = maxeval
    if name == 'slsqp':
        local.ftol_rel = 1e-10
        local.xtol_rel = 1e-10
    return local


def polish(prob, x, maxeval=5000, stop=5, perturb=1e-3, local='cobyla', scale=None):
    """Refine a decision vector by monotonic basin hopping around a local solver.

    COBYLA is derivative-free, which suits the pykep problems: they expose no
    gradient. On a constrained problem (``prob.c_tol`` set) MBH ranks points
    feasibility first, so the result is the best *feasible* point found when
    there is one. On the scaled path, a champion that ends just outside the
    tolerance gets one more descent with three times the iterations.

    Args:
        prob (pygmo.problem): Single-objective problem to minimise.
        x (Sequence[float]): Starting decision vector, inside ``prob``'s bounds.
        maxeval (int, optional): COBYLA evaluation cap per local descent.
            Defaults to 5000.
        stop (int, optional): Consecutive non-improving hops before MBH stops.
            Defaults to 5.
        perturb (float, optional): Hop size as a fraction of each variable's
            range. Defaults to 1e-3.
        local (str, optional): Local solver, see ``local_algorithm``. Defaults
            to ``cobyla``.
        scale (float, optional): Solve on the unit box with the objective
            divided by ``scale`` (see ``UnitBox``); ``None`` solves the problem
            as given. Defaults to None.

    Returns:
        tuple[float, list[float]]: ``(cost, x)`` of the best point found, never
        worse than the start (feasibility first on a constrained problem).
    """
    algo = local_algorithm(local, maxeval)
    if scale is None:
        pop = pg.population(prob)
        pop.push_back(x)
        pop = pg.algorithm(pg.mbh(pg.algorithm(algo), stop=stop, perturb=perturb)).evolve(pop)
        return float(pop.champion_f[0]), list(pop.champion_x)
    box = UnitBox(prob, scale)
    unit = pg.problem(box)
    unit.c_tol = list(prob.c_tol)
    pop = pg.population(unit)
    pop.push_back(box.to_u(x))
    pop = pg.algorithm(pg.mbh(pg.algorithm(algo), stop=stop, perturb=perturb)).evolve(pop)
    if not unit.feasibility_x(pop.champion_x) and _violation(unit, pop.champion_f) < 1e-3:
        # Close but not inside c_tol: one longer descent from the champion
        # usually finishes the job the MBH hops ran out of iterations for.
        finish = pg.population(unit)
        finish.push_back(pop.champion_x, pop.champion_f)
        finish = pg.algorithm(local_algorithm(local, 3 * maxeval)).evolve(finish)
        if unit.feasibility_x(finish.champion_x):
            pop = finish
    x_best = box.to_x(pop.champion_x)
    return float(prob.fitness(x_best)[0]), x_best


def _violation(prob, f):
    """Return the largest constraint violation of a fitness vector."""
    n_obj, n_eq = prob.get_nobj(), prob.get_nec()
    return max([abs(c) for c in f[n_obj:n_obj + n_eq]]
               + [max(0.0, c) for c in f[n_obj + n_eq:]], default=0.0)


def polish_job(job):
    """Polish one point inside a ``parallel_polish`` worker process.

    Args:
        job (dict): ``model``, ``sequence``, ``x``, ``window_mjd2000`` (``[lo,
            hi]`` launch bounds in MJD2000), optional ``tof_days`` (``[lo, hi]``
            time-of-flight band), ``maxeval`` and ``stop``.

    Constrained models are polished on their true constrained problem with the
    model's local solver; ``feasible`` then reports whether the result meets
    every constraint (always True for unconstrained models).

    Returns:
        dict: ``before``, ``before_feasible``, ``after``, ``x``, ``feasible``,
        ``violation`` and ``seconds``.
    """
    lo, hi = job['window_mjd2000']
    window = [pk.epoch(lo, 'mjd2000'), pk.epoch(hi, 'mjd2000')]
    model = models.get(job['model'])
    prob = models.problem(job['model'], job['sequence'], launch_window=window,
                          tof_days=job.get('tof_days'), constrained=True)
    x0 = clip(prob, job['x'])
    before = float(prob.fitness(x0)[0])
    before_feasible = bool(prob.feasibility_x(x0))
    t0 = time.time()
    after, x = polish(prob, x0, maxeval=job['maxeval'], stop=job['stop'], local=model.local,
                      scale=model.polish_scale)
    score = models.evaluate(job['model'], job['sequence'], x, window, job.get('tof_days'))
    return dict(before=before, before_feasible=before_feasible, after=after, x=x,
                feasible=score['feasible'],
                violation=score['violation'], seconds=time.time() - t0)


def clip(prob, x):
    """Clip a decision vector into a problem's bounds.

    A point pinned to a narrow polish band (pareto.py) can sit a hair outside
    it, which pygmo refuses.
    """
    lb, ub = prob.get_bounds()
    return [min(max(v, lo), hi) for v, lo, hi in zip(x, lb, ub)]


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
    parser.add_argument('--polish-maxeval', type=int, default=None, dest='polish_maxeval',
                        help='evaluations (COBYLA) or iterations (SLSQP) per local descent '
                             'during the polish (default: 5000 COBYLA, 300 SLSQP)')
    parser.add_argument('--polish-stop', type=int, default=5, dest='polish_stop',
                        help='consecutive non-improving basin hops before the polish stops')
    parser.add_argument('--polish-workers', type=int, default=None, dest='polish_workers',
                        help='parallel polish processes (default: CPU count)')


# Default --polish-maxeval per local solver: COBYLA counts evaluations, the
# gradient solvers count iterations (each a finite-difference gradient).
POLISH_MAXEVAL = {'cobyla': 5000, 'slsqp': 500, 'ipopt': 1000}


def finish_args(args):
    """Fill the CLI defaults that depend on the chosen model, in place.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from ``add_search_args``.

    Returns:
        argparse.Namespace: ``args``, with ``polish_maxeval`` resolved and
        ``sequence`` upper-cased.
    """
    args.sequence = args.sequence.upper()
    if args.polish_maxeval is None:
        args.polish_maxeval = POLISH_MAXEVAL[models.get(args.model).local]
    return args


def rank(record):
    """Sort key of a scored record: feasible before infeasible, then cheaper first."""
    return (not record.get('feasible', True), record['cost'])


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
        'tof_bounds_days': models.tof_bounds(model.name),
        'vinf_bounds_kms': model.launch_vinf_kms or mission.LAUNCH_VINF_KMS,
        'constrained': model.constrained,
        **({'assumptions': model.assumptions} if model.assumptions else {}),
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
        'polish_local': models.get(args.model).local,
        'polish_maxeval': args.polish_maxeval, 'polish_stop': args.polish_stop,
        'objective': models.get(args.model).objective,
        'destination_capture': mission.resolve_bounds(args.sequence).destination.get('capture'),
        **extra,
    }
