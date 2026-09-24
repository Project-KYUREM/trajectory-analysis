"""Propulsion / objective models behind one shared pipeline.

Pick a model with ``--model``; ``optimize.py``, ``pareto.py``, ``publish.py``
and the site are otherwise identical, and results land in
``results/<model>/<sequence>/``.

    capture    mga_1dsm with orbit insertion. Cost = deep-space maneuvers plus
               the burn into a loose capture ellipse at the destination.
    ballistic  Trident-style mga: gravity assists only, flyby arrival. Cost =
               departure delta-v beyond the launch-vinf allowance. Arrival vinf
               is reported, never charged.
    nep        40 kWe nuclear-electric low thrust (mga_lt_nep). A constrained
               Sims-Flanagan problem that builds and runs but is not decoded or
               published: its role here is a GMAT seed exported from an
               impulsive solution (see README).

Everything model-specific lives on a ``Model``: how to build the pykep UDP,
how to decode a decision vector, how to read the headline cost, and the
provenance to stamp into results. Runners never branch on a model name.
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
    """Wrap a pykep UDP so a fitness exception becomes a penalty, not a crash.

    Global samplers and NLopt's COBYLA (which steps outside the box) routinely
    probe decision vectors that push an epoch past kernel coverage or a
    hyperbola past a valid conic. Without this one bad query kills the run.
    """

    # Far above any real cost (m/s or kg) yet small enough that pygmo's
    # penalty methods do not overflow when they combine it.
    PENALTY = 1e6

    def __init__(self, udp):
        """Cache the shape of ``udp`` so the penalty vector needs no probing."""
        self._udp = udp
        prob = pg.problem(udp)
        self._nobj, self._nec, self._nic = prob.get_nobj(), prob.get_nec(), prob.get_nic()
        self._bounds = prob.get_bounds()

    def fitness(self, x):
        """Delegate, returning a uniform penalty vector on any failure or NaN."""
        try:
            f = self._udp.fitness(x)
            if any(v != v or abs(v) == float('inf') for v in f):
                raise ValueError('non-finite fitness')
            return f
        except Exception:
            return [self.PENALTY] * (self._nobj + self._nec + self._nic)

    def get_bounds(self):
        """Bounds of the wrapped problem."""
        return self._bounds

    def get_nobj(self):
        """Number of objectives of the wrapped problem."""
        return self._nobj

    def get_nec(self):
        """Number of equality constraints of the wrapped problem."""
        return self._nec

    def get_nic(self):
        """Number of inequality constraints of the wrapped problem."""
        return self._nic


@dataclass(frozen=True)
class Decoded:
    """A decision vector, read back in model-neutral terms.

    Attributes:
        leg_tof_days: Time of flight of each leg.
        leg_dsm_ms: Deep-space maneuver per leg; empty when the model has none.
        flyby_mismatch_ms: Per flyby, the velocity change the gravity assist
            cannot supply and a burn must (``ballistic`` only; ``mga_1dsm``
            makes its flybys consistent by construction, so empty there).
        lamberts: pykep Lambert solutions, one per leg.
        arc_states: ``(r, v)`` at the start of each ballistic arc, heliocentric
            ECLIPJ2000, metres and m/s.
        arc_epochs: MJD2000 epoch of each arc start.
        capture_dv_ms: Insertion burn at the destination, or ``None`` for a flyby.
        flyby_rp_radii: Periapsis of each flyby in planet radii, if the model
            has it as a free parameter.
        flyby_beta_rad: Flyby plane rotation angle per flyby, likewise.
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
    """One propulsion / objective model.

    Attributes:
        name: CLI name (``--model``) and results directory.
        label: Plain-language name of the cost axis, for plots.
        units: Units of the cost axis.
        blurb: One-line description shown on the site.
        objective: Provenance recorded into every result file.
        build: ``(setup, multi_objective) -> pykep UDP``.
        decode: ``(udp, x) -> Decoded``, or ``None`` if the model is not
            decodable; ``publish.py`` skips such models and
            ``trajectory.describe`` refuses them.
        cost_of: Headline scalar from a fitness vector.
        constrained: Whether the UDP carries constraints that heuristic
            optimisers need folded into a penalty (``pg.unconstrain``).
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
    """mga_1dsm pricing the final term as a periapsis burn into a capture ellipse.

    Arrival vinf is the speed relative to the destination while still far
    away, not the cost of arriving: the burn happens at periapsis, where
    gravity has already accelerated the craft to near escape speed, and only
    the sliver above escape speed is shed. Charging raw vinf would overprice
    arrival ~10x and bias the search towards needlessly slow trajectories.

    Args:
        setup: Resolved sequence and bounds.
        multi_objective: Fitness becomes ``(cost, tof)`` when true.

    Returns:
        A ``pk.trajopt.mga_1dsm`` instance.
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
    """Read an mga_1dsm decision vector.

    Uses pykep's ``_compute_dvs`` (a 5-tuple: per-leg dv with the capture burn
    last, Lambert legs, leg times, arc start states, arc epochs). Flyby
    periapsis and plane angle are the two genes at offsets 7+4i and 6+4i (see
    the layout at the top of this module).

    Args:
        udp: The ``mga_1dsm`` the vector belongs to.
        x: Decision vector.

    Returns:
        A ``Decoded``.
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
    """Trident-style mga: no deep-space maneuvers, flyby arrival (no capture)."""
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
    """Read an mga decision vector.

    Uses pykep's ``_compute_dvs`` (an 8-tuple: launch dv, per-flyby mismatch,
    arrival dv, Lambert legs, total launch dv, leg times, arc states, arc
    epochs). Flyby geometry is implied by the Lambert arcs, not a gene.

    Args:
        udp: The ``mga`` the vector belongs to.
        x: Decision vector.

    Returns:
        A ``Decoded``.
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
    """40 kWe flagship NEP as a Sims-Flanagan ``mga_lt_nep``.

    Unlike the impulsive models this problem has no single total-TOF gene, so
    each leg gets its own bounds: at least a month, at most the mission cap.
    It also carries match-point constraints, which is why ``constrained`` is
    set and ``problem()`` folds them into a penalty.

    Args:
        setup: Resolved sequence and bounds.
        multi_objective: Fitness becomes ``(-final_mass, tof)`` when true.

    Returns:
        A ``pk.trajopt.mga_lt_nep`` instance.
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
    """Look up a model by name with a helpful error."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]


def build(name, sequence, launch_window=None, tof_days=None, multi_objective=False):
    """Construct the raw pykep UDP for a model and sequence.

    Args:
        name: Model name.
        sequence: Sequence string, e.g. ``"EVEEJN"``.
        launch_window: Optional ``[epoch, epoch]`` override.
        tof_days: Optional ``[min, max]`` total-TOF override.
        multi_objective: Fitness becomes ``(cost, tof)`` when true.

    Returns:
        The unwrapped pykep UDP (use ``problem()`` for something optimisable).
    """
    setup = mission.resolve_bounds(sequence, launch_window, tof_days)
    return get(name).build(setup, multi_objective)


def problem(name, sequence, launch_window=None, tof_days=None, multi_objective=False):
    """Build an optimiser-ready ``pg.problem``: safe-wrapped, constraints folded.

    This is the one place the wrapping recipe lives; every runner, scorer and
    polish worker calls it, so a new constrained model needs no caller edits.

    Args:
        name: Model name.
        sequence: Sequence string.
        launch_window: Optional ``[epoch, epoch]`` override.
        tof_days: Optional ``[min, max]`` total-TOF override.
        multi_objective: Fitness becomes ``(cost, tof)`` when true.

    Returns:
        A ``pg.problem`` ready for any pygmo algorithm.
    """
    udp = SafeUDP(build(name, sequence, launch_window, tof_days, multi_objective))
    prob = pg.problem(udp)
    if get(name).constrained:
        prob = pg.problem(pg.unconstrain(prob, method='kuri'))
    return prob
