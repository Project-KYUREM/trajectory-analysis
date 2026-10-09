"""Define the Sims-Flanagan low-thrust gravity-assist problem and its engines.

``MgaLowThrust`` is a pygmo UDP for a continuous-thrust (Direct Fusion Drive)
spacecraft flying a gravity-assist sequence to a planet and spiralling down
into a stable orbit there. It depends only on pykep, not on SPICE, so it can be
built on any pykep planets (``mission`` hands it SPICE planets; tests can use
``pykep.planet.jpl_lp``).

Physics, in order of the mission:

* **Launch.** The launcher puts ``wet_mass_kg`` on an escape trajectory with
  ``C3 = launch_c3_km2s2`` (default 0: "20 t to TLI" is taken as 20 t to Earth
  escape). With ``C3 = 0`` the departure v-infinity genes are dropped and the
  spacecraft starts on Earth's heliocentric state.
* **Heliocentric legs.** Each leg is a Sims-Flanagan transcription
  (pykep ``sims_flanagan.leg``): ``n_seg`` segments, each a Kepler arc with one
  impulse at its midpoint whose size is bounded by ``thrust_n * dt / m``. The
  arcs propagate forward from the leg start and backward from the leg end and
  must meet at a match point: 7 equality constraints (position, velocity,
  mass) per leg. A fusion drive's thrust does not depend on solar distance, so
  ``thrust_n`` is constant across the whole cruise.
* **Gravity assists.** Unpowered and exact by construction: the outgoing
  velocity is ``pykep.fb_prop`` of the incoming v-infinity with a periapsis
  radius and plane angle that are genes, so no flyby constraint is needed.
  Thrusting happens on the legs on both sides of the flyby.
* **Capture.** The final leg ends at the destination with a v-infinity that is
  a gene. A low-thrust spiral from that hyperbola into a circular orbit of
  radius ``r`` costs, in the Edelbaum / thrust-much-less-than-gravity limit,
  ``dv = v_inf + sqrt(mu / r)``: the v-infinity is shed far out where gravity
  is weak, then the spiral from a parabolic orbit to a circular one costs the
  circular speed. The spiral takes ``dm * c / thrust`` seconds, which is added
  to the time objective. This is an upper bound on the capture cost (it ignores
  any Oberth gain from braking deep in the well).

Decision vector (all MJD2000 / days / km/s / fractions; see ``Layout``)::

    [t0, T, a_1 .. a_n]                       launch epoch, total cruise, leg weights
    [vinf, u, v]                              launch v-infinity (only if C3 > 0)
    [mf_1 .. mf_n]                            mass at the end of each leg / wet mass
    [vinf, u, v, beta, rp/R] * (n - 1)        incoming v-infinity and geometry per flyby
    [vinf, u, v]                              arrival v-infinity at the destination
    [throttles leg 1] .. [throttles leg n]    3 * n_seg per leg, each in [-1, 1]

Leg ``i`` lasts ``T * a_i / sum(a)`` days (pykep's "alpha" encoding), so ``T``
alone carries the cruise time and the optimiser splits it across legs.

Fitness::

    [propellant_kg, (total_days)] + equality constraints + inequality constraints

where ``propellant_kg = wet_mass - mass_in_orbit`` and ``total_days`` (the
second objective, multi-objective only) is the cruise plus the capture spiral.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pykep as pk
from pykep.sims_flanagan import leg as _sf_leg
from pykep.sims_flanagan import sc_state, spacecraft


# ------------------------------------------------------------------ engines --
@dataclass(frozen=True)
class Engine:
    """Describe a constant-thrust, constant-Isp engine.

    Args:
        name (str): Preset name.
        thrust_n (float): Thrust in newtons, available at any solar distance.
        isp_s (float): Specific impulse in seconds.
        power_mw (float): Fusion power in MW, recorded for provenance only.
        note (str): Where the numbers come from.
    """
    name: str
    thrust_n: float
    isp_s: float
    power_mw: float
    note: str

    @property
    def exhaust_ms(self):
        """Effective exhaust speed in m/s."""
        return self.isp_s * pk.G0


# Direct Fusion Drive (Princeton Satellite Systems / PPPL, PFRC-based). These
# are published design estimates, not flight data: ~10 000 s Isp and 5-10 N
# per MW of fusion power (Thomas et al. 2017; Cohen et al. 2019; Gajeri et al.
# 2020, arXiv:2009.12621). The 1 MW point matches the PSS 10 t Pluto orbiter
# study (5 N); the 2 MW point scales it to the 20 t spacecraft here. Replace
# with your own engine's numbers: nothing else in the pipeline is tuned to them.
ENGINES = {e.name: e for e in (
    Engine('dfd-1mw', thrust_n=5.0, isp_s=10000.0, power_mw=1.0,
           note='PSS DFD, 1 MW, 5 N @ 10 000 s (Pluto orbiter study scale)'),
    Engine('dfd-2mw', thrust_n=10.0, isp_s=10000.0, power_mw=2.0,
           note='PSS DFD, 2 MW (two 1 MW units or one 2 MW unit), 10 N @ 10 000 s'),
)}


# --------------------------------------------------------------- spacecraft --
@dataclass(frozen=True)
class Spacecraft:
    """Hold the spacecraft, launch and capture assumptions of a low-thrust model.

    Args:
        engine (Engine): The propulsion system.
        wet_mass_kg (float, optional): Mass delivered by the launcher to Earth
            escape. Defaults to 20 000 kg.
        min_final_mass_kg (float, optional): Floor on the mass in the final
            orbit (structure, engine, radiators, shielding, payload). An
            inequality constraint and the lower bound of every mass gene.
            Defaults to 10 000 kg.
        launch_c3_km2s2 (float, optional): Launch energy the launcher supplies at
            ``wet_mass_kg``. 0 is Earth escape. Defaults to 0.
        n_seg (int, optional): Sims-Flanagan segments per leg. More segments
            resolve the thrust profile better and cost more dimensions (3 each).
            Defaults to 10.
        flyby_vinf_max_kms (float, optional): Upper bound of every flyby
            v-infinity gene. Defaults to 20 km/s.
        arrival_vinf_max_kms (float, optional): Upper bound of the arrival
            v-infinity gene. Defaults to 10 km/s.
        orbit_radius_radii (float, optional): Radius of the circular target orbit
            at the destination, in planet radii. The default of 10 Neptune radii
            (~246 000 km) sits between Proteus (4.8 R_N) and Triton (14.3 R_N),
            outside the rings (< 2.6 R_N) and deep inside the Hill sphere
            (~4 700 R_N). Defaults to 10.
        high_fidelity (bool, optional): Propagate each segment with continuous
            thrust (pykep's Taylor integrator) instead of one impulse at its
            midpoint. Much slower. Defaults to False.
    """
    engine: Engine
    wet_mass_kg: float = 20000.0
    min_final_mass_kg: float = 10000.0
    launch_c3_km2s2: float = 0.0
    n_seg: int = 10
    flyby_vinf_max_kms: float = 20.0
    arrival_vinf_max_kms: float = 10.0
    orbit_radius_radii: float = 10.0
    high_fidelity: bool = False

    def record(self):
        """Return a JSON-serialisable record of every assumption."""
        return dict(engine=self.engine.name, thrust_n=self.engine.thrust_n,
                    isp_s=self.engine.isp_s, power_mw=self.engine.power_mw,
                    engine_note=self.engine.note, wet_mass_kg=self.wet_mass_kg,
                    min_final_mass_kg=self.min_final_mass_kg,
                    launch_c3_km2s2=self.launch_c3_km2s2, n_seg=self.n_seg,
                    flyby_vinf_max_kms=self.flyby_vinf_max_kms,
                    arrival_vinf_max_kms=self.arrival_vinf_max_kms,
                    orbit_radius_radii=self.orbit_radius_radii,
                    high_fidelity=self.high_fidelity)


def capture_cost(vinf_ms, mass_kg, mu, radius_m, engine):
    """Price a low-thrust capture from a hyperbola into a circular orbit.

    Args:
        vinf_ms (float): Arrival hyperbolic excess speed, m/s.
        mass_kg (float): Mass on arrival, kg.
        mu (float): Gravitational parameter of the destination, m^3/s^2.
        radius_m (float): Radius of the circular target orbit, m.
        engine (Engine): The engine doing the spiral.

    Returns:
        dict: ``dv_ms``, ``final_mass_kg``, ``propellant_kg``, ``days``
        (spiral duration at full thrust) and ``v_circ_ms``.
    """
    v_circ = math.sqrt(mu / radius_m)
    dv = vinf_ms + v_circ
    final = mass_kg * math.exp(-dv / engine.exhaust_ms)
    prop = mass_kg - final
    return dict(dv_ms=dv, final_mass_kg=final, propellant_kg=prop,
                days=prop * engine.exhaust_ms / engine.thrust_n / pk.DAY2SEC,
                v_circ_ms=v_circ)


def _vinf_vector(mag_kms, u, v):
    """Return a v-infinity vector in m/s from its magnitude and two unit-box angles.

    The same encoding as pykep's ``mga_1dsm`` launch v-infinity: ``u`` sets the
    azimuth, ``v`` the declination, both uniformly on the sphere.
    """
    theta = 2.0 * math.pi * u
    phi = math.acos(2.0 * v - 1.0) - math.pi / 2.0
    s = mag_kms * 1000.0
    return np.array([s * math.cos(phi) * math.cos(theta),
                     s * math.cos(phi) * math.sin(theta),
                     s * math.sin(phi)])


# -------------------------------------------------------------- the layout --
@dataclass(frozen=True)
class Layout:
    """Index map of the decision vector (see the module docstring).

    Args:
        n_legs (int): Number of legs (bodies in the sequence minus one).
        n_seg (int): Segments per leg.
        has_launch_vinf (bool): Whether launch v-infinity genes are present.
    """
    n_legs: int
    n_seg: int
    has_launch_vinf: bool
    t0: int = field(init=False)
    T: int = field(init=False)
    alpha: slice = field(init=False)
    launch: slice = field(init=False)
    mass: slice = field(init=False)
    flyby: list = field(init=False)
    arrival: slice = field(init=False)
    throttles: list = field(init=False)
    size: int = field(init=False)

    def __post_init__(self):
        """Lay the genes out in order."""
        n = self.n_legs
        i = 0
        set_ = lambda k, v: object.__setattr__(self, k, v)
        set_('t0', i); i += 1
        set_('T', i); i += 1
        set_('alpha', slice(i, i + n)); i += n
        k = 3 if self.has_launch_vinf else 0
        set_('launch', slice(i, i + k)); i += k
        set_('mass', slice(i, i + n)); i += n
        flyby = []
        for _ in range(n - 1):
            flyby.append(slice(i, i + 5)); i += 5
        set_('flyby', flyby)
        set_('arrival', slice(i, i + 3)); i += 3
        thr = []
        for _ in range(n):
            thr.append(slice(i, i + 3 * self.n_seg)); i += 3 * self.n_seg
        set_('throttles', thr)
        set_('size', i)


# ------------------------------------------------------------------ the UDP --
class MgaLowThrust:
    """Sims-Flanagan multiple-gravity-assist problem ending in a low-thrust capture.

    Args:
        planets (list): pykep planets in visiting order, Earth first. Each needs
            ``eph``, ``mu_self``, ``radius`` and ``safe_radius``.
        sc (Spacecraft): Spacecraft, engine and capture assumptions.
        t0 (list[float]): ``[start, end]`` launch window in MJD2000.
        tof_days (list[float]): ``[min, max]`` total cruise time in days (the
            capture spiral comes on top).
        multi_objective (bool, optional): Return ``(propellant, total_days)``
            instead of propellant alone. Defaults to False.
    """

    def __init__(self, planets, sc, t0, tof_days, multi_objective=False):
        """Store the problem data and build the reusable Sims-Flanagan leg."""
        self.planets = list(planets)
        self.sc = sc
        self.t0 = [float(t0[0]), float(t0[1])]
        self.tof_days = [float(tof_days[0]), float(tof_days[1])]
        self.multi_objective = multi_objective
        self.n_legs = len(self.planets) - 1
        self.layout = Layout(self.n_legs, sc.n_seg, sc.launch_c3_km2s2 > 0.0)
        self._leg = None

    # pykep's leg is not picklable; pygmo pickles UDPs into spawned workers, so
    # it is rebuilt lazily in whichever process evaluates.
    def __getstate__(self):
        """Drop the unpicklable leg before pickling."""
        state = self.__dict__.copy()
        state['_leg'] = None
        return state

    def _get_leg(self):
        """Return the cached pykep Sims-Flanagan leg, building it on first use."""
        if self._leg is None:
            self._leg = _sf_leg()
            self._leg.set_mu(pk.MU_SUN)
            self._leg.set_spacecraft(spacecraft(self.sc.wet_mass_kg, self.sc.engine.thrust_n,
                                                self.sc.engine.isp_s))
            self._leg.high_fidelity = self.sc.high_fidelity
        return self._leg

    # ------------------------------------------------------------ pygmo API --
    def get_bounds(self):
        """Return the box bounds of the decision vector."""
        L, sc = self.layout, self.sc
        lb, ub = [0.0] * L.size, [0.0] * L.size
        lb[L.t0], ub[L.t0] = self.t0
        lb[L.T], ub[L.T] = self.tof_days
        lb[L.alpha], ub[L.alpha] = [1e-3] * self.n_legs, [1.0] * self.n_legs
        if L.has_launch_vinf:
            lb[L.launch] = [0.0, 0.0, 0.0]
            ub[L.launch] = [math.sqrt(sc.launch_c3_km2s2), 1.0, 1.0]
        lb[L.mass] = [sc.min_final_mass_kg / sc.wet_mass_kg] * self.n_legs
        ub[L.mass] = [1.0] * self.n_legs
        for k, s in enumerate(L.flyby):
            body = self.planets[k + 1]
            lb[s] = [0.0, 0.0, 0.0, -2.0 * math.pi, body.safe_radius / body.radius]
            ub[s] = [sc.flyby_vinf_max_kms, 1.0, 1.0, 2.0 * math.pi, 30.0]
        lb[L.arrival] = [0.0, 0.0, 0.0]
        ub[L.arrival] = [sc.arrival_vinf_max_kms, 1.0, 1.0]
        for s in L.throttles:
            lb[s] = [-1.0] * (s.stop - s.start)
            ub[s] = [1.0] * (s.stop - s.start)
        return lb, ub

    def get_nobj(self):
        """Return 2 for (propellant, time), else 1."""
        return 2 if self.multi_objective else 1

    def get_nec(self):
        """Return the number of equality constraints: 7 match-point residuals per leg."""
        return 7 * self.n_legs

    def get_nic(self):
        """Return the number of inequality constraints: throttle norms plus the mass floor."""
        return self.n_legs * self.sc.n_seg + 1

    def get_name(self):
        """Return a readable problem name."""
        return f"Sims-Flanagan MGA ({self.sc.engine.name}, {self.n_legs} legs)"

    def fitness(self, x):
        """Evaluate objectives, match-point equalities and inequalities."""
        ev = self._evaluate(x)
        objectives = [ev['propellant_kg']]
        if self.multi_objective:
            objectives.append(ev['total_days'])
        return objectives + ev['eq'] + ev['ineq']

    # --------------------------------------------------------- the physics --
    def leg_times(self, x):
        """Return the encounter epochs in MJD2000 and each leg's duration in days."""
        L = self.layout
        alpha = np.asarray(x[L.alpha], dtype=float)
        durations = x[L.T] * alpha / alpha.sum()
        epochs = x[L.t0] + np.concatenate([[0.0], np.cumsum(durations)])
        return epochs, durations

    def boundary_states(self, x):
        """Return the start and end heliocentric state and mass of every leg.

        Returns:
            dict: ``epochs`` and ``durations`` (from ``leg_times``), ``body_r`` and
            ``body_v`` of each encounter, per-leg ``start`` and ``end`` as
            ``(r, v, m)``, ``vinf_dep`` and ``vinf_arr`` vectors (m/s), and per
            flyby ``vinf_in``, ``vinf_out`` vectors and ``rp_m``, ``beta_rad``.
        """
        L, sc = self.layout, self.sc
        epochs, durations = self.leg_times(x)
        body = [p.eph(pk.epoch(float(t), 'mjd2000')) for p, t in zip(self.planets, epochs)]
        body_r = [np.array(b[0]) for b in body]
        body_v = [np.array(b[1]) for b in body]
        masses = [sc.wet_mass_kg] + [float(m) * sc.wet_mass_kg for m in x[L.mass]]

        vinf_dep = _vinf_vector(*x[L.launch]) if L.has_launch_vinf else np.zeros(3)
        vinf_arr = _vinf_vector(*x[L.arrival])
        starts = [(body_r[0], body_v[0] + vinf_dep, masses[0])]
        ends = []
        flybys = []
        for k, s in enumerate(L.flyby):
            mag, u, v, beta, rp_ratio = x[s]
            planet = self.planets[k + 1]
            vinf_in = _vinf_vector(mag, u, v)
            v_in = body_v[k + 1] + vinf_in
            rp = rp_ratio * planet.radius
            v_out = np.array(pk.fb_prop(v_in.tolist(), body_v[k + 1].tolist(), rp, beta,
                                        planet.mu_self))
            ends.append((body_r[k + 1], v_in, masses[k + 1]))
            starts.append((body_r[k + 1], v_out, masses[k + 1]))
            flybys.append(dict(vinf_in=vinf_in, vinf_out=v_out - body_v[k + 1],
                               rp_m=rp, rp_radii=float(rp_ratio), beta_rad=float(beta)))
        ends.append((body_r[-1], body_v[-1] + vinf_arr, masses[-1]))
        return dict(epochs=epochs, durations=durations, body_r=body_r, body_v=body_v,
                    start=starts, end=ends, vinf_dep=vinf_dep, vinf_arr=vinf_arr,
                    flybys=flybys)

    def capture(self, x, boundary=None):
        """Return the capture cost (see ``capture_cost``) of a decision vector."""
        b = boundary if boundary is not None else self.boundary_states(x)
        dest = self.planets[-1]
        return capture_cost(float(np.linalg.norm(b['vinf_arr'])), b['end'][-1][2],
                            dest.mu_self, self.sc.orbit_radius_radii * dest.radius,
                            self.sc.engine)

    def _evaluate(self, x):
        """Run every leg and the capture; return objectives and constraints."""
        L, sc = self.layout, self.sc
        b = self.boundary_states(x)
        leg = self._get_leg()
        eq, ineq = [], []
        for i in range(self.n_legs):
            (r0, v0, m0), (r1, v1, m1) = b['start'][i], b['end'][i]
            leg.set(pk.epoch(float(b['epochs'][i]), 'mjd2000'),
                    sc_state(r0.tolist(), v0.tolist(), m0), list(x[L.throttles[i]]),
                    pk.epoch(float(b['epochs'][i + 1]), 'mjd2000'),
                    sc_state(r1.tolist(), v1.tolist(), m1))
            mm = leg.mismatch_constraints()
            # non-dimensional, as pykep's own low-thrust problems do
            eq += [mm[0] / pk.AU, mm[1] / pk.AU, mm[2] / pk.AU,
                   mm[3] / pk.EARTH_VELOCITY, mm[4] / pk.EARTH_VELOCITY,
                   mm[5] / pk.EARTH_VELOCITY, mm[6] / sc.wet_mass_kg]
            ineq += list(leg.throttles_constraints())
        cap = self.capture(x, b)
        ineq.append((sc.min_final_mass_kg - cap['final_mass_kg']) / sc.wet_mass_kg)
        return dict(propellant_kg=sc.wet_mass_kg - cap['final_mass_kg'],
                    total_days=float(x[L.T]) + cap['days'], eq=eq, ineq=ineq)

    # ------------------------------------------------------------ decoding --
    def segments(self, x):
        """Rebuild every impulse and Kepler arc of the low-fidelity transcription.

        Mirrors pykep's ``sims_flanagan.leg`` (low fidelity): segment ``j`` of a
        leg lasts ``dt = tof / n_seg`` and carries one impulse
        ``throttle * thrust * dt / m`` at its midpoint. The first
        ``(n_seg + 1) // 2`` impulses are applied forward from the leg start and
        the rest backward from the leg end; the halves are compared (the
        match-point residual) at the earliest backward impulse, exactly as pykep
        does. The arcs are then rebuilt chronologically by propagating forward
        from the leg start through every impulse, so a feasible leg ends on the
        target body (``end_miss_m`` measures how close).

        Returns:
            list[dict]: One entry per leg with ``arcs`` (``epoch_mjd2000``,
            ``dt_days``, ``r``, ``v`` at the arc start; ``n_seg + 1`` arcs),
            ``impulses`` (``epoch_mjd2000``, ``dv`` vector in m/s, ``mass_kg``
            before the impulse), ``dv_ms`` (sum of the impulse magnitudes),
            ``mismatch`` (pykep's 7 residuals, SI units) and ``end_miss_m``.
        """
        L, sc = self.layout, self.sc
        c, thrust = sc.engine.exhaust_ms, sc.engine.thrust_n
        b = self.boundary_states(x)
        prop = lambda r, v, t: tuple(map(np.array, pk.propagate_lagrangian(
            r.tolist(), v.tolist(), t, pk.MU_SUN)))
        out = []
        for i in range(self.n_legs):
            t_start, t_end = float(b['epochs'][i]), float(b['epochs'][i + 1])
            n = sc.n_seg
            dt_days = (t_end - t_start) / n
            dt = dt_days * pk.DAY2SEC
            mid = [t_start + (j + 0.5) * dt_days for j in range(n)]
            thr = np.asarray(x[L.throttles[i]], dtype=float).reshape(n, 3)
            n_fwd = (n + 1) // 2
            dvs, masses = [None] * n, [None] * n

            r, v = (np.asarray(s, dtype=float) for s in b['start'][i][:2])
            m, t_now = b['start'][i][2], t_start
            for j in range(n_fwd):
                r, v = prop(r, v, (mid[j] - t_now) * pk.DAY2SEC)
                t_now = mid[j]
                dvs[j], masses[j] = thr[j] * thrust / m * dt, m
                v = v + dvs[j]
                m *= math.exp(-np.linalg.norm(dvs[j]) / c)
            fwd = (r, v, m, t_now)

            r, v = (np.asarray(s, dtype=float) for s in b['end'][i][:2])
            m, t_now = b['end'][i][2], t_end
            for j in range(n - 1, n_fwd - 1, -1):
                r, v = prop(r, v, (mid[j] - t_now) * pk.DAY2SEC)
                t_now = mid[j]
                dvs[j] = thr[j] * thrust / m * dt
                v = v - dvs[j]
                m *= math.exp(np.linalg.norm(dvs[j]) / c)
                masses[j] = m
            rf, vf = prop(fwd[0], fwd[1], (t_now - fwd[3]) * pk.DAY2SEC)
            mismatch = np.concatenate([rf - r, vf - v, [fwd[2] - m]])

            # chronological rebuild: start -> impulse 1 -> ... -> impulse n -> end
            arcs = []
            r, v = (np.asarray(s, dtype=float) for s in b['start'][i][:2])
            t_now = t_start
            for j in range(n):
                arcs.append(dict(epoch_mjd2000=t_now, dt_days=mid[j] - t_now, r=r, v=v))
                r, v = prop(r, v, (mid[j] - t_now) * pk.DAY2SEC)
                v = v + dvs[j]
                t_now = mid[j]
            arcs.append(dict(epoch_mjd2000=t_now, dt_days=t_end - t_now, r=r, v=v))
            r, _ = prop(r, v, (t_end - t_now) * pk.DAY2SEC)
            out.append(dict(
                arcs=arcs,
                impulses=[dict(epoch_mjd2000=mid[j], dv=dvs[j], mass_kg=masses[j]) for j in range(n)],
                dv_ms=float(sum(np.linalg.norm(d) for d in dvs)),
                mismatch=mismatch,
                end_miss_m=float(np.linalg.norm(r - b['end'][i][0]))))
        return out
