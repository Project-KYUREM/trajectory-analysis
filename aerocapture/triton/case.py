"""Run Triton capture cases: Config -> epoch, atmosphere -> guess -> staged solve -> summary.

``run_triton.py`` calls the steps one cell at a time; the sweep and
trade scripts hand lists of configs to ``solve_chain``.

Solve stages (``solve``):
    S0  Triton-only problem (Triton-frame hyperbola with the guess's v_inf
        vector, free B-plane): the aerocapture, release, coast, burn and final
        orbit. With ``scope="triton_only"`` this is the answer.
    S1  Full problem seeded with S0's Triton phases and Neptune legs
        back-propagated from S0's frame-switch state, with a post-flyby TCM of
        at most ``tcm_homotopy_kms``.
    S2  Full problem without the TCM (or with ``tcm_max_kms`` if set), seeded
        from S1. If it fails, S1 is re-solved with the TCM cap halved each
        time (from its current |TCM|) and S2 is retried.
    S3  Fixed v_inf direction (``radec`` / ``from_run``): the free asymptote is
        rotated to the requested one in steps of at most ``max_step_deg``, the
        periapsis time free, each warm-started from the previous step.
    Then the Dymos simulate check on the final problem.

The stage helpers (warm restarts, simulate check, asymptote rotation) are
``core.solve``.
"""
from dataclasses import replace

import numpy as np

from aerocapture.core import arrival
from aerocapture.core import paths
from aerocapture.core import solve as core_solve
from aerocapture.triton import guess as guess_mod
from aerocapture.triton import problem
from aerocapture.triton import results

RESULTS_DIR = paths.mission_dir("triton")
TCM_FLOOR_KMS = 1e-3          # the halving continuation stops below this cap


def default_tag(cfg):
    """Return the results sub-directory name of a config (``cfg.tag`` if set)."""
    if cfg.tag:
        return cfg.tag
    d = "" if cfg.direction == "free" else f"_{cfg.direction}"
    return f"{cfg.scope}_vinf{cfg.vinf_kms:.3g}_{cfg.ballute.name}{d}"


def output_dir(cfg):
    """Return ``results/aerocapture/triton/<tag>/`` for a config."""
    return RESULTS_DIR / default_tag(cfg)


def resolve(cfg, echo=True):
    """Validate a config, apply its arrival source, build the atmosphere and echo the inputs.

    Args:
        cfg (config.Config): Inputs.
        echo (bool, optional): Print every input and the input warnings. Defaults to True.

    Returns:
        tuple: (Config, system.Epoch, (ra, dec) rad or None, atmosphere.Atmosphere,
        atmosphere record dict).
    """
    cfg, epoch, target = arrival.resolve_config(cfg)
    atm, rec = cfg.make_atmosphere()
    if echo:
        cfg.echo()
    return cfg, epoch, target, atm, rec


def make_guess(cfg, epoch, atm, verbose=True):
    """Return the initial guess (free asymptote; a fixed one is reached in stage S3)."""
    return guess_mod.make_guess(cfg, epoch, atm, verbose)


def full_guess(cfg, epoch, atm, seed):
    """Return a full-scope guess: ``seed``'s Triton phases plus Neptune legs back-propagated.

    Args:
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch.
        atm (atmosphere.Atmosphere): Triton atmosphere.
        seed (guess.TritonGuess): Guess or S0 solution with Triton-frame arcs.

    Returns:
        guess.TritonGuess: Guess with every phase of the full scope.
    """
    gen = guess_mod.Generator(cfg, epoch, atm, verbose=False)
    arc = seed.arcs["triton_approach"]
    legs = guess_mod.neptune_arcs(gen, float(arc.t[0]), arc.y[0])
    arcs = dict(seed.arcs)
    arcs.update(legs["arcs"])
    info = dict(seed.info, neptune=legs["info"])
    g = guess_mod.TritonGuess(arcs=arcs, arrival=legs["arrival"],
                                 triton_arrival=seed.triton_arrival, burn_dv=seed.burn_dv,
                                 ring_sign=legs["ring_sign"], info=info)
    g.impulses = {"burn": seed.burn_dv}
    return g


def solve(cfg, epoch, atm, guess, target_radec=None, restarts=1, max_step_deg=10.0):
    """Run the solve stages (module docstring) and the simulate check.

    Args:
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch.
        atm (atmosphere.Atmosphere): Triton atmosphere.
        guess (guess.TritonGuess): Initial guess.
        target_radec (tuple, optional): Fixed v-inf (ra, dec), rad. Defaults to None.
        restarts (int, optional): Warm restarts allowed per stage. Defaults to 1.
        max_step_deg (float, optional): Largest asymptote rotation per S3 step. Defaults to 10.

    Returns:
        tuple: (problem.TritonProblem, dict) with the final IPOPT info,
        ``stages`` (one entry per stage) and ``active_box_bounds`` of the final stage.
    """
    name = default_tag(cfg).replace("/", "_")
    rec = output_dir(cfg)
    stages = []
    s0 = problem.TritonProblem(cfg, epoch, atm, guess, scope="triton_only", name=name + "_s0")
    info = core_solve.solve_with_restarts(s0, restarts, rec, stages, "S0 triton_only", seed=True)
    if cfg.scope == "triton_only":
        return s0, core_solve.simulate_check(s0, info, stages, rec)
    seed = s0.solution_guess() if info.get("success") else guess
    fg = full_guess(cfg, epoch, atm, seed)
    s1 = problem.TritonProblem(cfg, epoch, atm, fg, scope="full",
                               tcm_cap=max(cfg.tcm_homotopy_kms, cfg.tcm_max_kms),
                               name=name + "_s1")
    info = core_solve.solve_with_restarts(s1, restarts, rec, stages,
                                          f"S1 full, TCM <= {s1.tcm_cap:g} km/s", seed=True)
    final = s1
    if cfg.tcm_max_kms < s1.tcm_cap:
        final, info = _remove_tcm(cfg, epoch, atm, s1, restarts, rec, stages, name)
    if target_radec is not None:
        final, info = _continue_to_direction(final, cfg, epoch, atm, target_radec, restarts,
                                             max_step_deg, name, rec, stages)
    return final, core_solve.simulate_check(final, info, stages, rec)


def _remove_tcm(cfg, epoch, atm, s1, restarts, rec, stages, name):
    """Drive the TCM to ``cfg.tcm_max_kms`` (stage S2); return (problem, info)."""
    seed = s1.solution_guess()
    s2 = problem.TritonProblem(cfg, epoch, atm, seed, scope="full",
                               tcm_cap=cfg.tcm_max_kms if cfg.tcm_max_kms > 0 else None,
                               name=name + "_s2")
    s2.set_guess(seed)
    info = core_solve.solve_with_restarts(s2, restarts, rec, stages, "S2 full, TCM removed")
    cap = float(np.linalg.norm(s1.impulse_dv("tcm")))
    while not info.get("success") and cap > TCM_FLOOR_KMS:
        cap *= 0.5
        print(f"[solve] S2 failed; re-solving S1 with the TCM capped at {cap * 1e3:.2f} m/s")
        s1.set_cap("tcm", cap)
        i1 = core_solve.solve_with_restarts(s1, restarts, rec, stages,
                                            f"S2 halving, TCM <= {cap:g} km/s")
        if not i1.get("success"):
            break
        cap = min(cap, float(np.linalg.norm(s1.impulse_dv("tcm"))))
        seed = s1.solution_guess()
        s2.set_guess(seed)
        info = core_solve.solve_with_restarts(s2, restarts, rec, stages, "S2 full, TCM removed")
    return s2, info


def _continue_to_direction(tp, cfg, epoch, atm, target_radec, restarts, max_step_deg, name, rec,
                           stages):
    """Rotate a free solution's asymptote to ``target_radec`` (stage S3); return (problem, info)."""
    start = (float(tp.p.get_val("arrival_dv.ra")[0]), float(tp.p.get_val("arrival_dv.dec")[0]))
    seed = tp.solution_guess()
    fixed = problem.TritonProblem(cfg, epoch, atm, seed, scope="full", tcm_cap=tp.tcm_cap,
                                  fixed_radec=target_radec, name=name + "_s3")
    fixed.set_guess(seed)
    info, complete = core_solve.rotate_asymptote(fixed, tp.design_vector(), start, target_radec,
                                                 restarts, max_step_deg, rec, stages,
                                                 label="S3 continuation")
    return fixed, dict(info, continuation_complete=complete)


def warm_start(tp, guess, cfg, out_dir, atm_rec):
    """Re-solve a solved triton_only problem in place for a config that changes only the vehicle.

    Args:
        tp (problem.TritonProblem): A solved ``triton_only`` problem.
        guess (guess.TritonGuess): The guess it was built from.
        cfg (config.Config): New inputs.
        out_dir (pathlib.Path): Recorder directory.
        atm_rec (dict): Atmosphere record for the summary.

    Returns:
        dict or None: The summary if the warm start converged, else None (also
        None, without solving, when ``cfg`` differs in anything but the
        ballute and probe masses, diameter and C_D values).
    """
    old = tp.cfg
    keep = replace(cfg, ballute=replace(cfg.ballute, name=old.ballute.name,
                                        diameter_m=old.ballute.diameter_m,
                                        envelope_mass_kg=old.ballute.envelope_mass_kg,
                                        cd_continuum=old.ballute.cd_continuum,
                                        cd_free_molecular=old.ballute.cd_free_molecular),
                   probe=old.probe, tag=old.tag)
    if tp.scope != "triton_only" or keep != old:
        return None
    vec = tp.design_vector()
    tp.cfg = cfg
    tp.set_vehicle(cfg)
    stages = []
    info = core_solve.solve_with_restarts(tp, 0, out_dir, stages, "warm start")
    if info.get("success"):
        info = core_solve.simulate_check(tp, info, stages, out_dir)
        return results.summarize(tp, guess, info, atm_rec)
    print("[solve] warm start not accepted; falling back to a fresh guess")
    tp.cfg = old
    tp.set_vehicle(old)
    tp.load_design_vector(vec)
    return None


def run_case(cfg, verbose=False):
    """Resolve, guess, solve, summarise and save one config; return (summary, problem, guess)."""
    cfg, epoch, target, atm, rec = resolve(cfg)
    g = make_guess(cfg, epoch, atm, verbose)
    tp, info = solve(cfg, epoch, atm, g, target)
    summary = results.summarize(tp, g, info, rec)
    return summary, tp, g


def solve_chain(cfgs):
    """Solve configs in order, warm-starting vehicle variants of ``triton_only`` cases.

    Each case prints its summary and writes ``solution.json``; failures are
    recorded as table rows.

    Args:
        cfgs (list): ``config.Config`` per case, in solve order.

    Returns:
        list: One ``results.table_row`` dict per case, with a ``start`` column.
    """
    rows, prev = [], None
    for cfg in cfgs:
        print(f"\n=== {default_tag(cfg)} ===")
        out_dir = output_dir(cfg)
        summary, start = None, "warm start"
        if prev is not None:
            summary = warm_start(*prev, cfg, out_dir, prev_rec)
        if summary is None:
            start = "fresh guess"
            try:
                summary, tp, g = run_case(cfg)
            except Exception as exc:       # recorded as a failed row
                print(f"failed: {type(exc).__name__}: {exc}")
                rows.append(dict(results.failed_row(cfg, f"{type(exc).__name__}: {exc}"),
                                 start=start))
                continue
            if summary["solver"].get("success") and tp.scope == "triton_only":
                prev, prev_rec = (tp, g), summary["atmosphere"]
        results.print_summary(summary)
        results.save(summary, out_dir)
        rows.append(dict(results.table_row(summary), start=start))
    return rows
