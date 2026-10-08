"""Define the propulsion and objective models behind the shared search pipeline.

A ``Model`` holds everything model-specific: how to build its pykep UDP,
decode a decision vector and read the headline cost, plus the provenance
stamped into ``results/<model>/<sequence>/``. ``capture`` is ``mga_1dsm`` with
orbit insertion (cost: deep-space maneuvers plus the capture burn).
``ballistic`` is ``mga`` with gravity assists only and a flyby arrival (cost:
departure delta-v beyond the launch-vinf allowance). ``nep`` is a constrained
Sims-Flanagan ``mga_lt_nep`` that serves as a GMAT seed and has no decoder.

For every model ``x[0]`` is the launch epoch in MJD2000, the only gene code
outside this module may read; every other gene is reached through
``Model.decode``.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import pygmo as pg
import pykep as pk

import mission

# Decision-vector contract. For every model x[0] is the launch epoch in
# MJD2000 (both pykep UDPs put t0 first); that is the only gene any code
# outside this module may read. Everything else is model-private and reached
# through ``Model.decode``:
#   mga_1dsm  [t0, u, v, vinf, eta1, a1] + [beta, rp/rP, eta, a] per further
#             leg + [T]; two ballistic arcs per leg (coast to the DSM, then the
#             Lambert arc).
#   mga       [t0, T, a1 .. an]; one Lambert arc per leg.


class SafeUDP:
    """Wrap a pykep UDP so a failed fitness evaluation returns a penalty vector.

    Global samplers and COBYLA probe vectors beyond kernel coverage or valid
    conics; the penalty keeps one such query from ending the run.
    """

    # Far above any real cost (m/s or kg) yet small enough that pygmo's
    # penalty methods do not overflow when they combine it.
    PENALTY = 1e6

    def __init__(self, udp):
        """Wrap a pykep UDP and cache its dimensions and bounds."""
        self._udp = udp
        prob = pg.problem(udp)
        self._nobj, self._nec, self._nic = prob.get_nobj(), prob.get_nec(), prob.get_nic()
        self._bounds = prob.get_bounds()

    def fitness(self, x):
        """Return the wrapped fitness, or a uniform ``PENALTY`` vector on any failure."""
        try:
            f = self._udp.fitness(x)
            if any(v != v or abs(v) == float('inf') for v in f):
                raise ValueError('non-finite fitness')
            return f
        except Exception:
            return [self.PENALTY] * (self._nobj + self._nec + self._nic)

    def get_bounds(self):
        """Return the bounds of the wrapped problem."""
        return self._bounds

    def get_nobj(self):
        """Return the number of objectives of the wrapped problem."""
        return self._nobj

    def get_nec(self):
        """Return the number of equality constraints of the wrapped problem."""
        return self._nec

    def get_nic(self):
        """Return the number of inequality constraints of the wrapped problem."""
        return self._nic


@dataclass(frozen=True)
class Decoded:
    """Hold a decision vector decoded into model-neutral terms.

    Args:
        leg_tof_days (list[float]): Time of flight of each leg, in days.
        leg_dsm_ms (list[float]): Deep-space maneuver of each leg, in m/s; empty
            when the model has none.
        flyby_mismatch_ms (list[float]): Velocity change per flyby that a burn
            must supply, in m/s; empty when the model's flybys are consistent by
            construction.
        lamberts (list[pykep.lambert_problem]): Lambert solution of each leg.
        arc_states (list[tuple]): ``(r, v)`` at the start of each ballistic arc,
            heliocentric ECLIPJ2000, in m and m/s.
        arc_epochs (list[float]): MJD2000 epoch of each arc start.
        capture_dv_ms (Optional[float]): Insertion burn at the destination in
            m/s, or ``None`` for a flyby arrival.
        flyby_rp_radii (list[float], optional): Periapsis of each flyby in planet
            radii, when the model has it as a gene. Defaults to None.
        flyby_beta_rad (list[float], optional): Plane rotation angle of each
            flyby in radians, when the model has it as a gene. Defaults to None.
    """
    leg_tof_days: list
    leg_dsm_ms: list
    flyby_mismatch_ms: list
    lamberts: list
    arc_states: list
    arc_epochs: list
    capture_dv_ms: Optional[float]
    flyby_rp_radii: Optional[list] = None
    flyby_beta_rad: Optional[list] = None


@dataclass(frozen=True)
class Model:
    """Describe one propulsion and objective model.

    Args:
        name (str): CLI name for ``--model`` and the results directory.
        label (str): Plain-language name of the cost axis.
        units (str): Units of the cost axis.
        blurb (str): One-line description shown on the site.
        objective (dict): Provenance recorded into every result file.
        build (Callable[[mission.Setup, bool], object]): Builds the pykep UDP
            from a ``Setup`` and a multi-objective flag.
        decode (Callable[[object, Sequence[float]], Decoded], optional): Decodes
            ``(udp, x)``; ``None`` marks the model as not decodable, so
            ``publish.py`` skips it and ``trajectory.describe`` refuses it.
            Defaults to None.
        cost_of (Callable[[Sequence[float]], float], optional): Headline cost from
            a fitness vector. Defaults to the first fitness element.
        constrained (bool, optional): Whether ``problem()`` folds the UDP's
            constraints into a penalty. Defaults to False.
    """
    name: str
    label: str
    units: str
    blurb: str
    objective: dict
    build: Callable[[mission.Setup, bool], object]
    decode: Optional[Callable[[object, Sequence[float]], Decoded]] = None
    cost_of: Callable[[Sequence[float]], float] = field(default=lambda f: float(f[0]))
    constrained: bool = False


# ----------------------------------------------------------------- capture --
def _build_capture(setup, multi_objective):
    """Build an ``mga_1dsm`` that prices arrival as a burn into the capture ellipse.

    The burn happens at periapsis, where it sheds only the speed above escape
    speed, so it costs far less than the raw arrival vinf.

    Args:
        setup (mission.Setup): Resolved sequence and bounds.
        multi_objective (bool): Make the fitness ``(cost, tof)``.

    Returns:
        pykep.trajopt.mga_1dsm: The UDP.
    """
    capture = setup.destination['capture']
    return pk.trajopt.mga_1dsm(
        seq=setup.planets,
        t0=setup.window,
        tof=setup.tof_days,
        tof_encoding='alpha',
        vinf=mission.LAUNCH_VINF_KMS,
        add_vinf_dep=False,   # the launcher supplies departure vinf
        add_vinf_arr=True,    # required by orbit_insertion
        multi_objective=multi_objective,
        orbit_insertion=True,
        rp_target=capture['rp_ratio'] * setup.destination['radius'],
        e_target=capture['e'],
    )


def _decode_capture(udp, x):
    """Decode an ``mga_1dsm`` decision vector.

    Flyby ``i`` has its plane angle at gene ``6 + 4i`` and its periapsis ratio at
    gene ``7 + 4i``.

    Args:
        udp (pykep.trajopt.mga_1dsm): The UDP the vector belongs to.
        x (Sequence[float]): Decision vector.

    Returns:
        Decoded: The decoded trajectory.
    """
    dv, lamberts, tof, arcs, epochs = udp._compute_dvs(x)
    n_flybys = len(udp._seq) - 2
    return Decoded(
        leg_tof_days=[float(t) for t in tof],
        leg_dsm_ms=[float(d) for d in dv[:-1]],
        flyby_mismatch_ms=[],
        lamberts=lamberts,
        arc_states=arcs,
        arc_epochs=[float(e) for e in epochs],
        capture_dv_ms=float(dv[-1]),
        flyby_rp_radii=[float(x[7 + 4 * i]) for i in range(n_flybys)],
        flyby_beta_rad=[float(x[6 + 4 * i]) for i in range(n_flybys)],
    )


# --------------------------------------------------------------- ballistic --
def _build_ballistic(setup, multi_objective):
    """Build an ``mga`` with gravity assists only and a flyby arrival."""
    return pk.trajopt.mga(
        seq=setup.planets,
        t0=setup.window,
        tof=setup.tof_days,
        vinf=mission.LAUNCH_VINF_KMS[1],   # scalar launch-vinf allowance, km/s
        tof_encoding='alpha',
        multi_objective=multi_objective,
        orbit_insertion=False,
    )


def _decode_ballistic(udp, x):
    """Decode an ``mga`` decision vector.

    Args:
        udp (pykep.trajopt.mga): The UDP the vector belongs to.
        x (Sequence[float]): Decision vector.

    Returns:
        Decoded: The decoded trajectory; flyby geometry follows from the Lambert
        arcs.
    """
    _, dv_flyby, dv_arrival, lamberts, _, tof, arcs, epochs = udp._compute_dvs(x)
    return Decoded(
        leg_tof_days=[float(t) for t in tof],
        leg_dsm_ms=[],
        flyby_mismatch_ms=[float(d) for d in dv_flyby],
        lamberts=lamberts,
        arc_states=arcs,
        arc_epochs=[float(e) for e in epochs],
        capture_dv_ms=float(dv_arrival) if udp.orbit_insertion else None,
    )


# --------------------------------------------------------------------- nep --
NEP_PARAMS = dict(Tmax=1.0, Isp=3500.0, wet_mass=4000.0,
                  vinf_dep=4.0, vinf_arr=6.0, fb_rel_vel=8.0, n_seg=10)


def _build_nep(setup, multi_objective):
    """Build the 40 kWe nuclear-electric Sims-Flanagan ``mga_lt_nep`` problem.

    Each leg has its own time-of-flight bounds, from 30 days to the mission cap.
    The problem carries match-point constraints, so its ``Model`` sets
    ``constrained``.

    Args:
        setup (mission.Setup): Resolved sequence and bounds.
        multi_objective (bool): Make the fitness ``(-final_mass, tof)``.

    Returns:
        pykep.trajopt.mga_lt_nep: The UDP.
    """
    n_legs = len(setup.planets) - 1
    return pk.trajopt.mga_lt_nep(
        seq=setup.planets,
        n_seg=[NEP_PARAMS['n_seg']] * n_legs,
        t0=[setup.window[0].mjd2000, setup.window[1].mjd2000],
        tof=[[30.0, setup.tof_days[1]]] * n_legs,
        vinf_dep=NEP_PARAMS['vinf_dep'],
        vinf_arr=NEP_PARAMS['vinf_arr'],
        mass=[NEP_PARAMS['wet_mass'], NEP_PARAMS['wet_mass']],
        Tmax=NEP_PARAMS['Tmax'],
        Isp=NEP_PARAMS['Isp'],
        fb_rel_vel=NEP_PARAMS['fb_rel_vel'],
        multi_objective=multi_objective,
        high_fidelity=False,
    )


# ---------------------------------------------------------------- registry --
# ``objective`` is the model's own provenance; the destination's capture
# parameters are recorded per run by ``search.run_settings`` since they belong
# to the body, not the model.
MODELS = {m.name: m for m in (
    Model(name='capture',
          label='Delta-V — deep-space maneuvers + capture',
          units='m/s',
          blurb='Chemical: deep-space maneuvers plus a capture burn into a loose capture ellipse',
          objective=dict(kind='orbit_insertion'),
          build=_build_capture, decode=_decode_capture),
    Model(name='ballistic',
          label='Departure Delta-V (launch energy)',
          units='m/s',
          blurb='Trident-style: gravity assists only, no deep-space maneuvers, flyby arrival',
          objective=dict(kind='launch_dv_flyby'),
          build=_build_ballistic, decode=_decode_ballistic),
    Model(name='nep',
          label='Propellant mass',
          units='kg',
          blurb='40 kWe nuclear-electric propulsion (low thrust); GMAT seed only',
          objective=dict(kind='max_final_mass', **NEP_PARAMS),
          build=_build_nep, constrained=True,
          cost_of=lambda f: float(NEP_PARAMS['wet_mass'] + f[0])),
)}


def get(name):
    """Look up a model by name.

    Args:
        name (str): Model name.

    Returns:
        Model: The registered model.

    Raises:
        ValueError: If ``name`` is not in ``MODELS``.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]


def build(name, sequence, launch_window=None, tof_days=None, multi_objective=False):
    """Build the raw pykep UDP for a model and sequence.

    Args:
        name (str): Model name.
        sequence (str): Sequence string, such as ``"EVEEJN"``.
        launch_window (list[pykep.epoch], optional): ``[start, end]`` launch
            window override. Defaults to None.
        tof_days (list[float], optional): ``[min, max]`` total time-of-flight
            override in days. Defaults to None.
        multi_objective (bool, optional): Make the fitness ``(cost, tof)``.
            Defaults to False.

    Returns:
        object: The unwrapped pykep UDP (``mga_1dsm``, ``mga`` or
        ``mga_lt_nep``); ``problem()`` gives an optimiser-ready version.
    """
    setup = mission.resolve_bounds(sequence, launch_window, tof_days)
    return get(name).build(setup, multi_objective)


def problem(name, sequence, launch_window=None, tof_days=None, multi_objective=False):
    """Build an optimiser-ready ``pg.problem`` for a model and sequence.

    The UDP is wrapped in ``SafeUDP``, and a constrained model's constraints are
    folded into a penalty with ``pg.unconstrain``.

    Args:
        name (str): Model name.
        sequence (str): Sequence string.
        launch_window (list[pykep.epoch], optional): ``[start, end]`` launch
            window override. Defaults to None.
        tof_days (list[float], optional): ``[min, max]`` total time-of-flight
            override in days. Defaults to None.
        multi_objective (bool, optional): Make the fitness ``(cost, tof)``.
            Defaults to False.

    Returns:
        pygmo.problem: The problem, ready for any pygmo algorithm.
    """
    udp = SafeUDP(build(name, sequence, launch_window, tof_days, multi_objective))
    prob = pg.problem(udp)
    if get(name).constrained:
        prob = pg.problem(pg.unconstrain(prob, method='kuri'))
    return prob
