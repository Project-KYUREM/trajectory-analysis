"""Arrival hyperbola as an OpenMDAO component, and v-inf inputs from published kyurem runs.

The incoming conic is set by the v-inf vector, the B-plane aim point and the
periapsis time; ``ArrivalComp`` maps them to the state at the start radius r0,
and ``add_arrival_match`` adds it to a problem with the equality constraints
tying that state to the first node of a phase. ``free_arrival_parameters``
picks which of the parameters IPOPT optimises. ``resolve_direction`` and
``resolve_config`` turn the arrival-direction inputs into an epoch and an
optional fixed asymptote; they need no problem or case module.
"""
import json
import math
from dataclasses import replace

import numpy as np
import openmdao.api as om

from aerocapture.core import dynamics
from aerocapture.core import system


class ArrivalComp(om.ExplicitComponent):
    """Map ``(vinf, ra, dec, b_t, b_r, t_peri)`` to the state ``(r0, v0, t0)`` at radius r0.

    Partials are by complex step; ``dynamics.hyperbola_state`` is complex-safe.
    Option ``mu`` sets the central body (Neptune by default).
    """

    def initialize(self):
        """Declare the start-radius and central-GM options."""
        self.options.declare("r0", types=float, default=1.0e6,
                             desc="start radius on the incoming branch, km")
        self.options.declare("mu", types=float, default=system.MU_NEPTUNE,
                             desc="central GM, km^3/s^2")

    def setup(self):
        """Declare the hyperbola parameters as inputs and the r0 state as outputs."""
        self.add_input("vinf", val=10.0, units="km/s")
        self.add_input("ra", val=0.0, units="rad")
        self.add_input("dec", val=0.0, units="rad")
        self.add_input("b_t", val=6e4, units="km")
        self.add_input("b_r", val=0.0, units="km")
        self.add_input("t_peri", val=0.0, units="s")
        self.add_output("r0", shape=(3,), units="km")
        self.add_output("v0", shape=(3,), units="km/s")
        self.add_output("t0", units="s")
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        """Evaluate the state at r0 with ``dynamics.hyperbola_state``."""
        args = [inputs[n][0] for n in ("vinf", "ra", "dec", "b_t", "b_r", "t_peri")]
        outputs["r0"], outputs["v0"], outputs["t0"] = dynamics.hyperbola_state(
            *args, self.options["r0"], mu=self.options["mu"])


ARRIVAL_PARAMETERS = ("ra", "dec", "b_t", "b_r", "t_peri")
_UNITS = dict(ra="rad", dec="rad", b_t="km", b_r="km", t_peri="s")


def free_arrival_parameters(cfg, direction_fixed):
    """Return the arrival parameters IPOPT optimises.

    ``Config.resolved`` applies the automatic rule for ``opt_direction`` and
    ``opt_t_peri``. It is applied to the problem actually being built, whose
    direction is fixed exactly when a fixed asymptote is passed in; the free
    stage of a fixed-direction continuation is therefore a free problem. A
    fixed asymptote is never optimised.

    Args:
        cfg (object): Mission config with ``direction``, ``resolved()``.
        direction_fixed (bool): True when the problem holds the v-inf direction fixed.

    Returns:
        list[str]: Subset of ``ARRIVAL_PARAMETERS``; ``b_t`` and ``b_r`` always.
    """
    res = replace(cfg, direction="radec" if direction_fixed else "free").resolved()
    names = ["b_t", "b_r"]
    if res.opt_direction and not direction_fixed:
        names = ["ra", "dec"] + names
    if res.opt_t_peri:
        names.append("t_peri")
    return names


def add_arrival_match(p, params, vinf, r0, mu, phase, states, refs, bounds, free):
    """Add the arrival hyperbola, its match to the first node of ``phase`` and its design variables.

    Subsystems: ``arrival_dv`` (``vinf`` and ``ARRIVAL_PARAMETERS``),
    ``arrival`` (``ArrivalComp``) and ``arrival_match`` (state and time at r0
    minus the first node of ``phase``), with the seven match residuals
    constrained to zero.

    Args:
        p (om.Problem): Problem before ``setup``, with the trajectory as ``traj``.
        params (dict): Seed values of ``ARRIVAL_PARAMETERS``.
        vinf (float): Hyperbolic excess speed, km/s (never optimised).
        r0 (float): Radius of the matched state, km.
        mu (float): Central GM, km^3/s^2.
        phase (str): First phase of the trajectory.
        states (tuple): The six state names of ``phase``.
        refs (dict): ``pos`` (km), ``vel`` (km/s) and ``time`` (s) refs of the
            match residuals.
        bounds (dict): Parameter name -> ``add_design_var`` keywords
            (``lower``, ``upper``, ``ref``).
        free (list): Parameters added as design variables.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("vinf", val=vinf, units="km/s")
    for n in ARRIVAL_PARAMETERS:
        ivc.add_output(n, val=params[n], units=_UNITS[n])
    p.model.add_subsystem("arrival_dv", ivc)
    p.model.add_subsystem("arrival", ArrivalComp(r0=float(r0), mu=float(mu)))
    for n in ("vinf",) + ARRIVAL_PARAMETERS:
        p.model.connect(f"arrival_dv.{n}", f"arrival.{n}")

    pos, vel = ("xa", "ya", "za"), ("vxa", "vya", "vza")
    exprs = [f"d{c} = {c}a - r0[{i}]" for i, c in enumerate("xyz")]
    exprs += [f"dv{c} = v{c}a - v0[{i}]" for i, c in enumerate("xyz")] + ["dt = ta - t0"]
    units = {n: {"units": "km"} for n in pos + ("dx", "dy", "dz")}
    units |= {n: {"units": "km/s"} for n in vel + ("dvx", "dvy", "dvz")}
    units |= {n: {"units": "s"} for n in ("ta", "t0", "dt")}
    p.model.add_subsystem("arrival_match", om.ExecComp(
        exprs, r0={"shape": (3,), "units": "km"}, v0={"shape": (3,), "units": "km/s"},
        **units))
    for n in ("r0", "v0", "t0"):
        p.model.connect(f"arrival.{n}", f"arrival_match.{n}")
    for n, inp in zip(tuple(states) + ("time",), pos + vel + ("ta",)):
        p.model.connect(f"traj.{phase}.timeseries.{n}", f"arrival_match.{inp}", src_indices=[0])
    for n in ("dx", "dy", "dz"):
        p.model.add_constraint(f"arrival_match.{n}", equals=0.0, ref=refs["pos"])
    for n in ("dvx", "dvy", "dvz"):
        p.model.add_constraint(f"arrival_match.{n}", equals=0.0, ref=refs["vel"])
    p.model.add_constraint("arrival_match.dt", equals=0.0, ref=refs["time"])
    for n in free:
        p.model.add_design_var(f"arrival_dv.{n}", **bounds[n])


def load_published_arrival(run, point):
    """Return the arrival v-inf of one point of a published kyurem front.

    Args:
        run (str): ``"<model>/<sequence>"`` under ``docs/data/``, such as ``"capture/EJN"``.
        point (str): Point id in the ``detail`` table of that run's ``cloud.json``.

    Returns:
        dict: ``vinf_vec_eclip`` (km/s, ECLIPJ2000), ``epoch_mjd2000``, ``et``
        (s past J2000 TDB) and ``source``.

    Raises:
        FileNotFoundError: If the run has no ``cloud.json``.
        KeyError: If the point is not in its detail table.
    """
    from aerocapture.core import paths
    model, sequence = run.split("/")
    path = paths.DOCS_DATA_DIR / model / sequence / "cloud.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (published runs live in docs/data/)")
    detail = json.loads(path.read_text())["detail"]
    if isinstance(detail, dict):
        if str(point) not in detail:
            raise KeyError(f"point {point} not in {path} detail table")
        entry = detail[str(point)]
    else:
        entry = detail[int(point)]
    arr = entry["arrival"]
    return dict(vinf_vec_eclip=np.asarray(arr["vinf_vec_eclip_ms"], dtype=float) / 1000.0,
                epoch_mjd2000=float(arr["epoch_mjd2000"]),
                et=system.mjd2000_to_et(arr["epoch_mjd2000"]), source=f"{run}#{point}")


def radec_from_vector(vec):
    """Return the right ascension and declination (rad) of a vector."""
    u = np.asarray(vec, float) / np.linalg.norm(vec)
    return float(np.arctan2(u[1], u[0])), float(np.arcsin(u[2]))


def resolve_direction(direction, epoch, ra_deg=0.0, dec_deg=0.0, from_run=None, point=None):
    """Return the epoch, fixed asymptote and published |v-inf| of an arrival-direction choice.

    Args:
        direction (str): ``"free"``, ``"radec"`` or ``"from_run"``.
        epoch (str | float): Time zero for ``free`` and ``radec``, as a SPICE
            string or TDB seconds past J2000.
        ra_deg (float, optional): Fixed v-inf right ascension in NEQ, deg
            (``radec``). Defaults to 0.0.
        dec_deg (float, optional): Fixed v-inf declination in NEQ, deg
            (``radec``). Defaults to 0.0.
        from_run (str, optional): Published run ``"<model>/<sequence>"``
            (``from_run``). Defaults to None.
        point (str, optional): Point id in that run (``from_run``). Defaults to None.

    Returns:
        tuple: (system.Epoch, (ra, dec) in rad in NEQ or None for a free
        direction, |v-inf| in km/s from the published run or None).

    Raises:
        ValueError: If ``direction`` is unknown.
    """
    if direction == "from_run":
        pub = load_published_arrival(from_run, point)
        ep = system.make_epoch(pub["et"])
        vec = ep.neq_from_eclip @ pub["vinf_vec_eclip"]
        return ep, radec_from_vector(vec), float(np.linalg.norm(vec))
    if direction not in ("free", "radec"):
        raise ValueError(f"direction must be 'free', 'radec' or 'from_run', not {direction!r}")
    ep = system.make_epoch(epoch)
    target = (math.radians(ra_deg), math.radians(dec_deg)) if direction == "radec" else None
    return ep, target, None


def resolve_config(cfg):
    """Validate a mission config and apply its arrival source.

    For ``direction="from_run"`` the epoch becomes the published arrival epoch
    and ``vinf_kms`` the published |v-inf|.

    Args:
        cfg (object): Mission config with ``validate()`` and the arrival fields
            ``direction``, ``epoch``, ``ra_deg``, ``dec_deg``, ``from_run``,
            ``point`` and ``vinf_kms``.

    Returns:
        tuple: (config with the arrival applied, system.Epoch, (ra, dec) in rad
        in NEQ or None for a free direction).
    """
    cfg.validate()
    epoch, target, vinf = resolve_direction(cfg.direction, cfg.epoch, cfg.ra_deg, cfg.dec_deg,
                                            cfg.from_run, cfg.point)
    if vinf is not None:
        cfg = replace(cfg, vinf_kms=vinf, epoch=str(epoch.et_ref))
    return cfg, epoch, target
