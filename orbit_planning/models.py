"""Define the propulsion and objective models behind the shared search pipeline.

A ``Model`` holds everything model-specific: how to build its pykep UDP,
decode a decision vector and read the headline cost, plus the provenance
stamped into ``results/<model>/<sequence>/``. ``capture`` is ``mga_1dsm`` with
orbit insertion (cost: deep-space maneuvers plus the capture burn).
``ballistic`` is ``mga`` with gravity assists only and a flyby arrival (cost:
departure delta-v beyond the launch-vinf allowance). ``dfd`` and ``dfd-1mw``
are continuous-thrust Direct Fusion Drive models: a Sims-Flanagan
gravity-assist problem (``lowthrust.MgaLowThrust``) from a 20 t Earth-escape
launch to a circular Neptune orbit (cost: propellant). ``nep`` is a
constrained Sims-Flanagan ``mga_lt_nep`` that serves as a GMAT seed and has no
decoder.

Constrained models (``nep``, ``dfd``) are searched in two forms: the global
archipelago sees the constraints folded into a penalty, and the local polish
(``search.polish``) sees the true constrained problem.

For every model ``x[0]`` is the launch epoch in MJD2000, the only gene code
outside this module may read; every other gene is reached through
``Model.decode``.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pygmo as pg
import pykep as pk

import lowthrust
import mission

# Decision-vector contract. For every model x[0] is the launch epoch in
# MJD2000 (both pykep UDPs put t0 first); that is the only gene any code
# outside this module may read. Everything else is model-private and reached
# through ``Model.decode``:
#   mga_1dsm  [t0, u, v, vinf, eta1, a1] + [beta, rp/rP, eta, a] per further
#             leg + [T]; two ballistic arcs per leg (coast to the DSM, then the
#             Lambert arc).
#   mga       [t0, T, a1 .. an]; one Lambert arc per leg.
#   dfd       [t0, T, a1 .. an, ...]; see lowthrust.py (Layout) for the rest.


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

    def gradient(self, x):
        """Return a central-difference gradient of the full fitness vector.

        Only gradient-based local solvers (``slsqp``) call this; the pykep
        problems expose no analytic gradient. The step is relative to each
        gene, so epochs (~1e4 days) and throttles (~1) are both resolved.
        """
        return pg.estimate_gradient(self.fitness, x, dx=1e-7)

    def get_name(self):
        """Return the wrapped problem's name, when it has one."""
        return getattr(self._udp, 'get_name', lambda: type(self._udp).__name__)()

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
    vinf_dep_vec: Optional[list] = None
    vinf_arr_vec: Optional[list] = None
    lowthrust: Optional[dict] = None


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
            constraints into a penalty for the global search. Defaults to False.
        penalty (str, optional): ``pg.unconstrain`` method used for that fold:
            ``kuri`` or ``weighted``. Defaults to ``kuri``.
        local (str, optional): Local solver of the MBH polish: ``cobyla``,
            ``slsqp`` or ``ipopt`` (see ``search.local_algorithm``). Defaults
            to ``cobyla``.
        polish_scale (float, optional): When set, the polish solves on the unit
            box with the objective divided by this (see ``search.UnitBox``);
            ``None`` polishes the raw problem. Defaults to None.
        tof_days (list[float], optional): Default ``[min, max]`` total time of
            flight; ``None`` uses ``mission.MISSION_TOF_DAYS``. Defaults to None.
        launch_vinf_kms (list[float], optional): Launch v-infinity bounds
            recorded in result headers; ``None`` uses
            ``mission.LAUNCH_VINF_KMS``. Defaults to None.
        assumptions (dict, optional): Spacecraft / engine record stored with
            every result. Defaults to None.
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
    penalty: str = 'kuri'
    local: str = 'cobyla'
    polish_scale: Optional[float] = None
    tof_days: Optional[list] = None
    launch_vinf_kms: Optional[list] = None
    assumptions: Optional[dict] = None


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


# --------------------------------------------------------------------- dfd --
# Direct Fusion Drive spacecraft. Edit these (or add a preset to
# lowthrust.ENGINES) to change the vehicle; every result file records them.
DFD_SPACECRAFT = {
    name: lowthrust.Spacecraft(
        engine=lowthrust.ENGINES[engine],
        wet_mass_kg=20000.0,        # launcher: 20 t to TLI, taken as Earth escape
        min_final_mass_kg=10000.0,  # everything that is not propellant
        launch_c3_km2s2=0.0,        # C3 = 0: no launch v-infinity genes
        n_seg=10,                   # Sims-Flanagan segments per leg
        flyby_vinf_max_kms=20.0,
        arrival_vinf_max_kms=10.0,
        orbit_radius_radii=mission.BODIES['N']['lowthrust_orbit']['radius_radii'],
        high_fidelity=False)
    for name, engine in (('dfd', 'dfd-2mw'), ('dfd-1mw', 'dfd-1mw'))}

# Total cruise bounds for the fusion models (1.9 .. 15 yr); the capture spiral
# comes on top. Far shorter than the chemical models' 8-33 yr.
DFD_TOF_DAYS = [700.0, 5500.0]


def _build_dfd(name):
    """Return a builder of the Sims-Flanagan DFD problem for one spacecraft preset."""
    def build(setup, multi_objective):
        """Build ``lowthrust.MgaLowThrust`` for a ``mission.Setup``."""
        return lowthrust.MgaLowThrust(
            setup.planets, DFD_SPACECRAFT[name],
            t0=[setup.window[0].mjd2000, setup.window[1].mjd2000],
            tof_days=setup.tof_days, multi_objective=multi_objective)
    return build


def _decode_dfd(udp, x):
    """Decode a ``MgaLowThrust`` decision vector.

    Every Sims-Flanagan sub-arc (impulse to impulse) becomes one ballistic arc,
    so the site's Kepler propagator draws the thrusting trajectory unchanged.

    Args:
        udp (lowthrust.MgaLowThrust): The UDP the vector belongs to.
        x (Sequence[float]): Decision vector.

    Returns:
        Decoded: The decoded trajectory; ``leg_dsm_ms`` holds each leg's total
        thrust delta-v and ``lowthrust`` the capture, masses, residuals and
        per-impulse history.
    """
    b = udp.boundary_states(x)
    segs = udp.segments(x)
    cap = udp.capture(x, b)
    f = udp.fitness(x)
    n_eq = udp.get_nec()
    eq, ineq = np.array(f[udp.get_nobj():udp.get_nobj() + n_eq]), np.array(f[udp.get_nobj() + n_eq:])
    arcs = [a for s in segs for a in s['arcs']]
    sc = udp.sc
    return Decoded(
        leg_tof_days=[float(d) for d in b['durations']],
        leg_dsm_ms=[s['dv_ms'] for s in segs],
        flyby_mismatch_ms=[],
        lamberts=[],
        arc_states=[(a['r'].tolist(), a['v'].tolist()) for a in arcs],
        arc_epochs=[float(a['epoch_mjd2000']) for a in arcs],
        capture_dv_ms=cap['dv_ms'],
        flyby_rp_radii=[fb['rp_radii'] for fb in b['flybys']],
        flyby_beta_rad=[fb['beta_rad'] for fb in b['flybys']],
        vinf_dep_vec=b['vinf_dep'].tolist(),
        vinf_arr_vec=b['vinf_arr'].tolist(),
        lowthrust=dict(
            spacecraft=sc.record(),
            wet_mass_kg=sc.wet_mass_kg,
            arrival_mass_kg=float(b['end'][-1][2]),
            final_mass_kg=cap['final_mass_kg'],
            propellant_kg=sc.wet_mass_kg - cap['final_mass_kg'],
            cruise_propellant_kg=sc.wet_mass_kg - float(b['end'][-1][2]),
            leg_end_mass_kg=[float(e[2]) for e in b['end']],
            flyby_vinf_in_ms=[float(np.linalg.norm(fb['vinf_in'])) for fb in b['flybys']],
            capture=dict(cap, orbit_radius_km=sc.orbit_radius_radii * udp.planets[-1].radius / 1e3,
                         orbit_radius_radii=sc.orbit_radius_radii),
            max_eq_residual=float(np.max(np.abs(eq))) if len(eq) else 0.0,
            max_ineq_violation=float(max(0.0, np.max(ineq))) if len(ineq) else 0.0,
            leg_end_miss_km=[s['end_miss_m'] / 1e3 for s in segs],
            impulses=[[dict(epoch_mjd2000=p['epoch_mjd2000'], dv_ms=p['dv'].tolist(),
                            mass_kg=p['mass_kg']) for p in s['impulses']] for s in segs]),
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
    *(Model(name=name,
            label='Propellant — cruise + low-thrust capture',
            units='kg',
            blurb=(f"Direct Fusion Drive {sc.engine.thrust_n:g} N @ {sc.engine.isp_s:g} s, "
                   f"{sc.wet_mass_kg / 1000:g} t at Earth escape; continuous thrust (Sims-Flanagan) "
                   f"with gravity assists, spiral into a {sc.orbit_radius_radii:g} R circular orbit"),
            objective=dict(kind='lowthrust_capture_propellant', **sc.record()),
            build=_build_dfd(name), decode=_decode_dfd,
            constrained=True, penalty='weighted', local='ipopt', polish_scale=sc.wet_mass_kg,
            tof_days=DFD_TOF_DAYS,
            launch_vinf_kms=[0.0, sc.launch_c3_km2s2 ** 0.5],
            assumptions=sc.record())
      for name, sc in DFD_SPACECRAFT.items()),
)}

# Constraint tolerance for constrained models: match-point residuals are
# non-dimensional (1e-6 AU ~ 150 km, 1e-6 Earth orbital speed ~ 3 cm/s).
C_TOL = 1e-6

# ``weighted`` penalty weight per constraint: a residual of 1e-3 AU costs
# ~100 kg of the propellant objective, steep enough to drive the global search
# towards feasibility, shallow enough not to flatten the objective.
PENALTY_WEIGHT = 1e5


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
    model = get(name)
    setup = mission.resolve_bounds(sequence, launch_window,
                                   tof_days if tof_days is not None else model.tof_days)
    return model.build(setup, multi_objective)


def tof_bounds(name):
    """Return the default ``[min, max]`` total time of flight of a model, in days."""
    return get(name).tof_days or mission.MISSION_TOF_DAYS


def problem(name, sequence, launch_window=None, tof_days=None, multi_objective=False,
            constrained=False):
    """Build an optimiser-ready ``pg.problem`` for a model and sequence.

    The UDP is wrapped in ``SafeUDP``. A constrained model's constraints are
    folded into a penalty with ``pg.unconstrain`` (the form the global
    archipelago searches), unless ``constrained`` asks for the true problem
    (the form the local polish solves).

    Args:
        name (str): Model name.
        sequence (str): Sequence string.
        launch_window (list[pykep.epoch], optional): ``[start, end]`` launch
            window override. Defaults to None.
        tof_days (list[float], optional): ``[min, max]`` total time-of-flight
            override in days. Defaults to None.
        multi_objective (bool, optional): Make the fitness ``(cost, tof)``.
            Defaults to False.
        constrained (bool, optional): Keep a constrained model's constraints
            (with ``C_TOL``) instead of folding them into a penalty; no effect on
            unconstrained models. Defaults to False.

    Returns:
        pygmo.problem: The problem, ready for any pygmo algorithm.
    """
    model = get(name)
    udp = SafeUDP(build(name, sequence, launch_window, tof_days, multi_objective))
    prob = pg.problem(udp)
    if not model.constrained:
        return prob
    if constrained:
        prob.c_tol = [C_TOL] * (prob.get_nec() + prob.get_nic())
        return prob
    if model.penalty == 'weighted':
        n_c = prob.get_nec() + prob.get_nic()
        return pg.problem(pg.unconstrain(prob, method='weighted', weights=[PENALTY_WEIGHT] * n_c))
    return pg.problem(pg.unconstrain(prob, method=model.penalty))


def evaluator(name, sequence, launch_window=None, tof_days=None):
    """Return a scorer of decision vectors on the true problem.

    Args:
        name (str): Model name.
        sequence (str): Sequence string.
        launch_window (list[pykep.epoch], optional): Launch window override.
            Defaults to None.
        tof_days (list[float], optional): Time-of-flight override. Defaults to
            None.

    Returns:
        Callable[[Sequence[float]], dict]: Maps ``x`` to ``cost`` (the model's
        headline cost), ``tof_days`` (the second objective of the
        multi-objective form), ``cruise_days`` (launch to arrival, without a
        low-thrust capture spiral), ``feasible`` and ``violation`` (largest
        constraint violation, 0 when unconstrained).
    """
    model = get(name)
    prob = problem(name, sequence, launch_window, tof_days, multi_objective=True,
                   constrained=True)
    udp = build(name, sequence, launch_window, tof_days, multi_objective=True)
    n_eq, n_obj = prob.get_nec(), prob.get_nobj()

    def score(x):
        """Score one decision vector."""
        f = prob.fitness(x)
        eq = [abs(c) for c in f[n_obj:n_obj + n_eq]]
        ineq = [max(0.0, c) for c in f[n_obj + n_eq:]]
        cruise = (float(sum(udp.leg_times(x)[1])) if hasattr(udp, 'leg_times')
                  else float(f[1]))
        return dict(cost=model.cost_of(f), tof_days=float(f[1]), cruise_days=cruise,
                    feasible=bool(prob.feasibility_f(f)) if (eq or ineq) else True,
                    violation=float(max(eq + ineq, default=0.0)))
    return score


def evaluate(name, sequence, x, launch_window=None, tof_days=None):
    """Score one decision vector on the true problem (see ``evaluator``)."""
    return evaluator(name, sequence, launch_window, tof_days)(x)
