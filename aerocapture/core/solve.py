"""Solve plumbing shared by the mission problems: IPOPT driver, solve, staged-solve helpers.

``DymosProblem`` is the base of each mission's problem class. The mission
class builds the whole NLP (phases, constraints, objective) in its own
``_build``; the base adds what does not depend on the formulation: the
OpenMDAO work directory, ``solve`` (IPOPT through pyOptSparse, then the
optional Dymos simulate check), timeseries readback and the design-vector
copies that warm starts and continuations restore.

After every driver run ``solve`` lists the numerical box bounds that are
active at the final iterate (``active_box_bounds``). The box bounds
(``BOX`` in each ``problem.py``) only keep IPOPT near the guess: phase start
times and durations, every state, the min-max slacks and the arrival B-plane
and declination. An active one is a constraint nobody stated, so the solve
info reports it as ``info["active_box_bounds"]``, labels like
``"approach.x_upper"`` or ``"arrival.b_t_lower"``. Design-variable bounds
that are inputs of the problem (bank-angle limits, the periapsis-time window,
impulse caps) are not box bounds and are not listed.

The staged-solve helpers run one stage with warm restarts
(``solve_with_restarts``), the final simulate check (``simulate_check``) and
the rotation of a free asymptote onto a fixed one (``rotate_asymptote``).
"""
import fnmatch
import math
import os
import time as _time
from pathlib import Path

import dymos as dm
import numpy as np
import openmdao.api as om

from aerocapture.core import dynamics
from aerocapture.core import paths

IPOPT_OPTIONS = dict(mu_strategy="adaptive", nlp_scaling_method="gradient-based",
                     bound_mult_init_method="mu-based")
SIMULATE_OPTIONS = dict(method="DOP853", atol=1e-10, rtol=1e-10, times_per_seg=10)
BOX_DESIGN_VARS = (                 # design variables whose bounds come from a problem's BOX
    "traj.*.t_initial",             # phase start time window
    "traj.*.t_duration",            # phase duration factors
    "traj.*.states:*",              # state magnitude factor (and the heat-load factor)
    "traj.*.parameters:q_peak",     # peak-heat-rate slack range
    "traj.*.parameters:g_peak",     # peak-g slack floor
    "arrival_dv.b_t",               # B-plane aim point range
    "arrival_dv.b_r",
    "arrival_dv.dec",               # declination kept off the poles
)
BOUND_TOL = 1e-6                    # a bound is active within this scaled distance


def resolve_work_dir(work_dir, default):
    """Return and create the OpenMDAO output directory.

    Args:
        work_dir (str | Path | None): Requested directory.
        default (Path): Directory used when neither ``work_dir`` nor the
            ``OPENMDAO_WORKDIR`` environment variable is set.

    Returns:
        Path: The existing directory.
    """
    path = Path(work_dir or os.environ.get("OPENMDAO_WORKDIR") or default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_ipopt_driver(p, cfg):
    """Attach a colored pyOptSparse IPOPT driver with the solver fields of a mission config.

    Args:
        p (om.Problem): Problem before ``setup``.
        cfg (object): Config with ``max_iter``, ``tol``, ``constr_viol_tol``,
            ``print_level`` and ``linear_solver``.
    """
    p.driver = om.pyOptSparseDriver(optimizer="IPOPT", print_results=False)
    p.driver.declare_coloring()
    p.driver.opt_settings.update(max_iter=cfg.max_iter, tol=cfg.tol,
                                 constr_viol_tol=cfg.constr_viol_tol,
                                 print_level=cfg.print_level,
                                 linear_solver=cfg.linear_solver, **IPOPT_OPTIONS)


def _label(name):
    """Return the short label of a design variable: phase.var or arrival.var."""
    name = name.replace("arrival_dv.", "arrival.").removeprefix("traj.")
    for kind in ("states:", "parameters:", "controls:"):
        name = name.replace(kind, "")
    return name


def active_box_bounds(p, patterns=BOX_DESIGN_VARS, tol=BOUND_TOL):
    """Return the box bounds active at the current design point.

    Distances are measured in the driver's scaled units, where a design
    variable's ``ref`` is one unit.

    Args:
        p (om.Problem): A set-up problem.
        patterns (tuple, optional): ``fnmatch`` patterns of the design
            variables whose bounds are box bounds. Defaults to ``BOX_DESIGN_VARS``.
        tol (float, optional): Scaled distance (times max(1, |bound|)) within
            which a bound counts as active. Defaults to ``BOUND_TOL``.

    Returns:
        list[str]: Labels ``"<phase>.<var>_lower"`` / ``"_upper"``, or
        ``"arrival.<var>_..."``, in design-variable order.
    """
    values = p.driver.get_design_var_values(driver_scaling=False)
    out = []
    for name, meta in p.model.get_design_vars(get_sizes=False).items():
        if not any(fnmatch.fnmatchcase(name, pat) for pat in patterns):
            continue
        scaler = 1.0 if meta.get("scaler") is None else meta["scaler"]
        adder = 0.0 if meta.get("adder") is None else meta["adder"]
        x = (np.atleast_1d(values[name]) + adder) * scaler
        for side in ("lower", "upper"):
            bound = meta.get(side)
            if bound is None or np.all(np.abs(np.atleast_1d(bound)) >= 1e20):
                continue
            b = (np.broadcast_to(bound, x.shape) + adder) * scaler
            if np.any(np.abs(x - b) <= tol * np.maximum(1.0, np.abs(b))):
                out.append(f"{_label(name)}_{side}")
    return out


class DymosProblem:
    """Solve, readback and warm-start plumbing of a built mission problem.

    A subclass sets ``name`` and ``work_dir`` and builds ``p`` (the set-up
    ``om.Problem``) and ``traj`` (its Dymos trajectory).
    """

    def solve(self, simulate=True, run_driver=True, record_dir=None):
        """Run IPOPT and, optionally, the Dymos simulate check.

        Args:
            simulate (bool, optional): Re-integrate every phase after the solve. Defaults to True.
            run_driver (bool, optional): Run IPOPT; False only evaluates the model.
                Defaults to True.
            record_dir (str or Path, optional): Directory for the solution and
                simulation recordings. Defaults to ``work_dir``.

        Returns:
            dict: ``wall_s`` and, with the driver, ``ipopt_value``, ``ipopt_text``,
            ``success``, ``iterations`` and ``active_box_bounds``.
        """
        rec = Path(record_dir) if record_dir else self.work_dir
        rec.mkdir(parents=True, exist_ok=True)
        t0 = _time.time()
        with paths.workdir(self.work_dir):
            dm.run_problem(self.p, run_driver=run_driver, simulate=simulate,
                           solution_record_file=str(rec / f"{self.name}_solution.db"),
                           simulation_record_file=str(rec / f"{self.name}_simulation.db"),
                           simulate_kwargs=SIMULATE_OPTIONS)
        info = {"wall_s": _time.time() - t0}
        if run_driver:
            sol = self.p.driver.pyopt_solution
            inform = sol.optInform if sol is not None else {"value": -999, "text": "no solution"}
            info.update(ipopt_value=int(inform["value"]), ipopt_text=str(inform["text"]),
                        success=int(inform["value"]) == 0)
            if getattr(self.p.driver, "iter_count", None) is not None:
                info["iterations"] = int(self.p.driver.iter_count)
            info["active_box_bounds"] = active_box_bounds(self.p)
        return info

    def ts(self, phase, name, units=None):
        """Return a collocated timeseries output of a phase as a flat array."""
        return self.p.get_val(f"traj.{phase}.timeseries.{name}", units=units).ravel()

    def sim_ts(self, phase, name):
        """Return a simulated timeseries output of a phase, or None before a simulate run."""
        sp = getattr(self.traj, "sim_prob", None)
        if sp is None:
            return None
        return sp.get_val(f"traj.{phase}.timeseries.{name}").ravel()

    def design_vector(self):
        """Return a copy of every design variable, for warm starts."""
        return {n: self.p.get_val(meta["source"]).copy()
                for n, meta in self.p.model.get_design_vars(get_sizes=False).items()}

    def load_design_vector(self, vec):
        """Set the design variables present in ``vec`` (from ``design_vector``)."""
        for n, meta in self.p.model.get_design_vars(get_sizes=False).items():
            if n in vec:
                self.p.set_val(meta["source"], vec[n])


# ------------------------------------------------------------ staged solving --
def solve_with_restarts(prob, restarts, record_dir, stages, label, seed=False):
    """Run IPOPT, restarting from the last iterate up to ``restarts`` times; log the stage.

    Args:
        prob (DymosProblem): Problem to solve.
        restarts (int): Warm restarts allowed after a failed solve.
        record_dir (Path): Recorder directory.
        stages (list): Stage log; one ``dict(stage=label, **info)`` is appended.
        label (str): Stage name for the log.
        seed (bool, optional): Seed from the build guess first. Defaults to False.

    Returns:
        dict: The last ``DymosProblem.solve`` info.
    """
    if seed:
        prob.set_guess()
    print(f"[solve] {label} ...", flush=True)
    info = prob.solve(simulate=False, record_dir=record_dir)
    for _ in range(restarts):
        if info.get("success"):
            break
        print(f"[solve] IPOPT: {info.get('ipopt_text')}; warm restart from the last iterate")
        info = prob.solve(simulate=False, record_dir=record_dir)
    print(f"[solve] {label}: {info.get('ipopt_text')} ({info.get('iterations')} it, "
          f"{info['wall_s']:.0f} s)", flush=True)
    if info.get("active_box_bounds"):
        print(f"[solve] active box bounds: {', '.join(info['active_box_bounds'])}")
    stages.append(dict(stage=label, **info))
    return info


def simulate_check(prob, info, stages, record_dir):
    """Run the Dymos simulate check on a solved problem; return ``info`` with the stage log."""
    sim = prob.solve(simulate=True, run_driver=False, record_dir=record_dir)
    return dict(info, sim_wall_s=sim["wall_s"], stages=stages)


def rotate_asymptote(prob, vec, start_radec, target_radec, restarts, max_step_deg, record_dir,
                     stages, label="continuation"):
    """Rotate a solution's v-inf direction onto ``target_radec`` in warm-started steps.

    Each step moves the asymptote along the great circle by at most
    ``max_step_deg``, loads the last converged design vector with the new
    direction and solves; a failed step halves the step size.

    Args:
        prob (DymosProblem): Problem with the direction held fixed (``ra``,
            ``dec`` set but not optimised); seeded and set up.
        vec (dict): Design vector of the converged free-direction solution.
        start_radec (tuple): That solution's (ra, dec), rad.
        target_radec (tuple): Requested (ra, dec), rad.
        restarts (int): Warm restarts allowed per step.
        max_step_deg (float): Largest rotation per step, deg.
        record_dir (Path): Recorder directory.
        stages (list): Stage log.
        label (str, optional): Stage-name prefix. Defaults to "continuation".

    Returns:
        tuple: (dict info of the last step, bool True if the target was reached).
    """
    s0 = dynamics.unit_from_radec(*start_radec)
    s1 = dynamics.unit_from_radec(*target_radec)
    omega = math.acos(np.clip(s0 @ s1, -1.0, 1.0))
    frac, step = 0.0, min(1.0, max_step_deg / max(math.degrees(omega), 1e-9))
    print(f"[solve] {label} over {math.degrees(omega):.1f} deg to the fixed asymptote")
    info = {}
    while frac < 1.0:
        trial = min(1.0, frac + step)
        s = s1 if omega < 1e-9 else (math.sin((1 - trial) * omega) * s0
                                     + math.sin(trial * omega) * s1) / math.sin(omega)
        prob.load_design_vector(vec)
        prob.p.set_val("arrival_dv.ra", math.atan2(s[1], s[0]))
        prob.p.set_val("arrival_dv.dec", math.asin(s[2]))
        info = solve_with_restarts(prob, restarts, record_dir, stages, f"{label} {trial:.3f}")
        if info.get("success"):
            frac, vec = trial, prob.design_vector()
        else:
            step /= 2
            if step < 1e-3:
                print(f"[solve] {label} stalled")
                break
    return info, frac >= 1.0
