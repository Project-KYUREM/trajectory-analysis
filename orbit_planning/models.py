"""Propulsion / objective models, all sharing one optimise -> pareto -> porkchop
-> publish pipeline. Pick one with --model; everything downstream is otherwise
identical, and results land in results/<model>/<sequence>/.

    capture   (existing) mga_1dsm + orbit_insertion. Cost = DSMs + Neptune
              capture burn into an e=0.99 ellipse. Impulsive, alpha-encoded.
    ballistic (Trident)  mga, no DSMs, flyby arrival (no capture). Cost =
              departure C3 delta-v; arrival vinf is reported, not charged.
    nep       (flagship) mga_lt_nep, nuclear-electric low thrust. "Cost" is
              propellant mass (start mass - delivered mass). Constrained
              Sims-Flanagan problem -- a different, much harder optimisation.

Every model exposes the same small interface so the runners stay model-blind:
    build(sequence, launch_window=None, multi_objective=False) -> pygmo-ready UDP
    cost_of(fitness) -> float        the headline scalar (m/s, or kg for nep)
    label, units, impulsive          metadata the plots and publish step read
"""
import pygmo as pg
import pykep as pk

import mission


class _Safe:
    """Wrap a pykep UDP so a fitness exception becomes a large penalty rather
    than a crash. Global/heuristic optimisers (and NLopt's COBYLA, which steps
    outside the box) routinely probe decision vectors that drive an epoch past
    kernel coverage or a hyperbola past a valid conic; without this the whole
    run dies on one bad query."""

    _BIG = 1e6

    def __init__(self, udp):
        self._udp = udp
        prob = pg.problem(udp)
        self._nobj = prob.get_nobj()
        self._nec = prob.get_nec()
        self._nic = prob.get_nic()
        self._nf = self._nobj + self._nec + self._nic
        self._bounds = prob.get_bounds()

    def fitness(self, x):
        try:
            f = self._udp.fitness(x)
            # guard NaN/inf that some solvers emit without raising
            if any(v != v or abs(v) == float('inf') for v in f):
                raise ValueError('non-finite fitness')
            return f
        except Exception:
            return [self._BIG] * self._nf

    def get_bounds(self):
        return self._bounds

    def get_nobj(self):
        return self._nobj

    def get_nec(self):
        return self._nec

    def get_nic(self):
        return self._nic


def safe(udp):
    return _Safe(udp)

# --------------------------------------------------------------- ballistic --
# Trident-style: gravity assists only, no deep-space maneuvers. The chemical
# upper stage delivers departure C3; arrival is a flyby. pk.trajopt.mga shares
# mga_1dsm's seq/t0/tof/vinf/orbit_insertion shape but has no eta (DSM) genes,
# so the decision vector is just [t0] + per-leg alphas -- far lower dimension.
def _build_ballistic(sequence, launch_window=None, multi_objective=False):
    planets, _ = mission.make_sequence(sequence)
    window = launch_window if launch_window is not None else mission.LAUNCH_WINDOW
    return pk.trajopt.mga(
        seq=planets,
        t0=list(window),
        tof=mission.MISSION_TOF_DAYS,
        vinf=mission.LAUNCH_VINF_KMS[1],   # scalar C3 cap, km/s
        tof_encoding='alpha',              # single total-TOF budget, like our alpha
        multi_objective=multi_objective,
        orbit_insertion=False,             # flyby: no capture burn
    )


# ----------------------------------------------------------------- capture --
def _build_capture(sequence, launch_window=None, multi_objective=False):
    return mission.build_udp(multi_objective=multi_objective,
                             orbit_insertion=True,
                             launch_window=launch_window,
                             sequence=sequence)


# --------------------------------------------------------------------- nep --
# 40 kWe flagship NEP. mga_lt_nep is a constrained Sims-Flanagan problem
# (per-segment throttles + match-point equality constraints), so it is NOT
# drop-in: it uses per-leg direct tof bounds (no alpha), and pg.unconstrain
# wraps the constraints into a penalty before any heuristic can touch it.
NEP_PARAMS = dict(Tmax=1.0, Isp=3500.0, wet_mass=4000.0,
                  vinf_dep=4.0, vinf_arr=6.0, fb_rel_vel=8.0, n_seg=10)


def _build_nep(sequence, launch_window=None, multi_objective=False):
    planets, _ = mission.make_sequence(sequence)
    n_legs = len(planets) - 1
    window = launch_window if launch_window is not None else mission.LAUNCH_WINDOW
    lo, hi = mission.MISSION_TOF_DAYS
    # Per-leg bounds: let any leg use most of the budget; the total is bounded
    # loosely by the sum. (mga_lt_nep has no single-window alpha option.)
    per_leg = [[30.0, hi]] * n_legs
    raw = pk.trajopt.mga_lt_nep(
        seq=planets,
        n_seg=[NEP_PARAMS['n_seg']] * n_legs,
        t0=[window[0].mjd2000, window[1].mjd2000],
        tof=per_leg,
        vinf_dep=NEP_PARAMS['vinf_dep'],
        vinf_arr=NEP_PARAMS['vinf_arr'],
        mass=[NEP_PARAMS['wet_mass'], NEP_PARAMS['wet_mass']],
        Tmax=NEP_PARAMS['Tmax'],
        Isp=NEP_PARAMS['Isp'],
        fb_rel_vel=NEP_PARAMS['fb_rel_vel'],
        multi_objective=multi_objective,
        high_fidelity=False,
    )
    return raw   # runners apply pg.unconstrain; see models.needs_unconstrain


MODELS = {
    'capture': dict(
        build=_build_capture, impulsive=True, needs_unconstrain=False,
        label='Delta-V — deep-space maneuvers + Neptune capture', units='m/s',
        cost_of=lambda f: float(f[0]),
        blurb='Chemical: deep-space maneuvers plus a capture burn into an e=0.99 ellipse'),
    'ballistic': dict(
        build=_build_ballistic, impulsive=True, needs_unconstrain=False,
        label='Departure Delta-V (launch energy)', units='m/s',
        cost_of=lambda f: float(f[0]),
        blurb='Trident-style: gravity assists only, no deep-space maneuvers, flyby arrival'),
    'nep': dict(
        build=_build_nep, impulsive=False, needs_unconstrain=True,
        label='Propellant mass', units='kg',
        # objective is -final_mass; propellant = wet - delivered
        cost_of=lambda f: float(NEP_PARAMS['wet_mass'] + f[0]),
        blurb='40 kWe nuclear-electric propulsion (low thrust)'),
}


def get(model_name):
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}; choose from {sorted(MODELS)}")
    return MODELS[model_name]


def build(model_name, sequence, launch_window=None, multi_objective=False):
    return get(model_name)['build'](sequence, launch_window, multi_objective)
