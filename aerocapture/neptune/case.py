"""Run aerocapture cases: Config -> epoch and arrival -> guess -> staged solve -> summary.

``run_neptune.py`` calls the steps one cell at a time; ``sweep_neptune.py`` and
``trade_neptune.py`` hand lists of configs to ``solve_chain``.

Solve stages (``solve``):
    1. Free direction: one solve from the guess, with warm restarts from the
       last iterate if IPOPT stops short.
    2. Fixed direction (``radec`` / ``from_run``): a direct solve from a guess
       built around the requested asymptote; if that fails, a free-direction
       solve whose asymptote is then rotated to the requested one in steps of
       at most ``max_step_deg``, each warm-started from the previous step.
    3. The Dymos simulate check on the final iterate.

The stage helpers (warm restarts, simulate check, asymptote rotation) are
``core.solve``.
"""
from dataclasses import replace

from aerocapture.core import arrival
from aerocapture.core import paths
from aerocapture.core import solve as core_solve
from aerocapture.neptune import guess as guess_mod
from aerocapture.neptune import problem
from aerocapture.neptune import results

RESULTS_DIR = paths.mission_dir("neptune")


def default_tag(cfg):
    """Return the results sub-directory name of a config (``cfg.tag`` if set)."""
    if cfg.tag:
        return cfg.tag
    d = "" if cfg.direction == "free" else f"_{cfg.direction}"
    return f"{cfg.triton_mode}_vinf{cfg.vinf_kms:.3g}_{cfg.vehicle.name}{d}"


def output_dir(cfg):
    """Return ``results/aerocapture/neptune/<tag>/`` for a config."""
    return RESULTS_DIR / default_tag(cfg)


def resolve_arrival(cfg, echo=True):
    """Validate a config, apply its arrival source and echo the inputs.

    For ``direction="from_run"`` the epoch becomes the published arrival epoch
    and ``vinf_kms`` the published |v-inf|.

    Args:
        cfg (config.Config): Inputs.
        echo (bool, optional): Print every input and the input warnings. Defaults to True.

    Returns:
        tuple: (config.Config, system.Epoch, (ra, dec) in rad in NEQ or None for a
        free direction).
    """
    cfg, epoch, target = arrival.resolve_config(cfg)
    if echo:
        cfg.echo()
    return cfg, epoch, target


def make_guess(cfg, epoch, target_radec=None, verbose=True):
    """Return the initial guess for a config, built around ``target_radec`` if given."""
    settings = guess_mod.GuessSettings.from_config(cfg, epoch, verbose, target_radec)
    return guess_mod.make_guess(settings)


def solve(cfg, epoch, guess, target_radec=None, restarts=2, max_step_deg=10.0):
    """Run the solve stages and the simulate check.

    Args:
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch.
        guess (guess.Guess): Initial guess.
        target_radec (tuple, optional): Fixed v-inf (ra, dec), rad. Defaults to None (free).
        restarts (int, optional): Warm restarts allowed per stage. Defaults to 2.
        max_step_deg (float, optional): Largest asymptote rotation per continuation
            step, deg. Defaults to 10.

    Returns:
        tuple: (problem.AeroProblem, dict) with the final IPOPT info, ``stages``
        (one entry per stage), ``active_box_bounds`` of the final stage and,
        after a continuation, ``continuation_complete``.
    """
    name = default_tag(cfg).replace("/", "_")
    rec = output_dir(cfg)
    stages = []
    if target_radec is not None and guess.info.get("fixed_asymptote"):
        ap = problem.AeroProblem(cfg, epoch, guess, fixed_radec=target_radec, name=name + "_fixed")
        info = core_solve.solve_with_restarts(ap, restarts, rec, stages,
                                              "fixed direction, direct", seed=True)
        if info.get("success"):
            return ap, core_solve.simulate_check(ap, info, stages, rec)
        print("[solve] direct fixed-direction solve failed; continuing from a free solve")
        guess = make_guess(cfg, epoch, verbose=False)
    ap = problem.AeroProblem(cfg, epoch, guess, name=name)
    info = core_solve.solve_with_restarts(ap, restarts, rec, stages, "free direction", seed=True)
    if target_radec is None:
        return ap, core_solve.simulate_check(ap, info, stages, rec)
    return _continue_to_direction(ap, cfg, epoch, guess, target_radec, restarts, max_step_deg,
                                  name, rec, stages)


def _continue_to_direction(ap, cfg, epoch, guess, target_radec, restarts, max_step_deg, name,
                           rec, stages):
    """Rotate a free solution's asymptote to ``target_radec``; return (problem, info)."""
    start = (float(ap.p.get_val("arrival_dv.ra")[0]), float(ap.p.get_val("arrival_dv.dec")[0]))
    vec = ap.design_vector()
    fixed = problem.AeroProblem(cfg, epoch, guess, fixed_radec=target_radec, name=name + "_cont")
    fixed.set_guess()
    info, complete = core_solve.rotate_asymptote(fixed, vec, start, target_radec, restarts,
                                                 max_step_deg, rec, stages)
    info = core_solve.simulate_check(fixed, info, stages, rec)
    info["continuation_complete"] = complete
    return fixed, info


def warm_start(ap, guess, cfg, out_dir):
    """Re-solve a solved problem in place for a config that changes only v-inf or beta, L/D, R_n.

    Args:
        ap (problem.AeroProblem): A solved problem; restored to its previous
            design vector if the warm start is not accepted.
        guess (guess.Guess): The guess ``ap`` was built from (recorded in the summary).
        cfg (config.Config): The new inputs.
        out_dir (pathlib.Path): Directory for the recorder files.

    Returns:
        dict or None: The summary if the warm start converged and is acceptable
        (``results.acceptable``), else None; also None, without solving, when
        ``cfg`` differs from ``ap.cfg`` in anything else.
    """
    old, veh = ap.cfg, cfg.vehicle
    same_vehicle = replace(veh, name=old.vehicle.name, beta_kg_m2=old.vehicle.beta_kg_m2,
                           lift_to_drag=old.vehicle.lift_to_drag,
                           nose_radius_m=old.vehicle.nose_radius_m) == old.vehicle
    if not same_vehicle or replace(cfg, vinf_kms=old.vinf_kms, vehicle=old.vehicle,
                                   tag=old.tag) != old:
        return None
    vec = ap.design_vector()
    ap.cfg = cfg
    ap.p.set_val("arrival_dv.vinf", cfg.vinf_kms)
    ap.set_vehicle(veh)
    stages = []
    info = core_solve.solve_with_restarts(ap, 0, out_dir, stages, "warm start")
    if info.get("success"):
        info = core_solve.simulate_check(ap, info, stages, out_dir)
        summary = results.summarize(ap, guess, info)
        if results.acceptable(summary):
            return summary
    print("[solve] warm start not accepted; falling back to a fresh guess")
    ap.cfg = old
    ap.p.set_val("arrival_dv.vinf", old.vinf_kms)
    ap.set_vehicle(old.vehicle)
    ap.load_design_vector(vec)
    return None


def solve_chain(cfgs):
    """Solve configs in order, warm-starting each from the last accepted one.

    A case is warm-started (``warm_start``) when the previous accepted case
    differs from it only in v-inf or vehicle beta, L/D, R_n and the direction
    is free; otherwise, or if the warm start is not accepted, it is solved from
    a fresh guess. Each case prints its summary and writes ``solution.json``
    into its output directory; failures are recorded as table rows.

    Args:
        cfgs (list): ``config.Config`` per case, in solve order.

    Returns:
        list: One ``results.table_row`` dict per case, with a ``start`` column
        ("warm start" or "fresh guess").
    """
    rows, prev = [], None
    for cfg in cfgs:
        print(f"\n=== {default_tag(cfg)} ===")
        cfg, epoch, target = resolve_arrival(cfg)
        out_dir = output_dir(cfg)
        summary, start = None, "warm start"
        if prev is not None and target is None:
            summary = warm_start(*prev, cfg, out_dir)
        if summary is None:
            start = "fresh guess"
            try:
                g = make_guess(cfg, epoch, target, verbose=False)
                ap, info = solve(cfg, epoch, g, target)
                summary = results.summarize(ap, g, info)
            except Exception as exc:       # recorded as a failed row
                print(f"failed: {type(exc).__name__}: {exc}")
                rows.append(dict(results.failed_row(cfg, f"{type(exc).__name__}: {exc}"),
                                 start=start))
                continue
            if results.acceptable(summary):
                prev = (ap, g)
        results.print_summary(summary)
        results.save(summary, out_dir)
        rows.append(dict(results.table_row(summary), start=start))
    return rows
