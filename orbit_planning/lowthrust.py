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
  a gene. The planetocentric phase, from that hyperbola into a highly
  eccentric target orbit (default periapsis 1.25 R_N, apoapsis 1.25 x
  Triton's orbit), is flown numerically by ``capture_sim``: planar two-body
  dynamics with the engine steered by a Q-law (Petropoulos 2004), a Lyapunov
  feedback that drives orbital energy and periapsis radius to their targets
  together. It thrusts continuously until the orbit is bound and its apoapsis
  is below ``capture_kswitch`` times the target's, then coasts wherever
  thrusting is less than ``capture_eta`` as effective as at the best point of
  the current orbit. Its delta-v and duration are tabulated once over
  arrival v-infinity and arrival mass (``CaptureTable``, cached on disk) and
  interpolated in every fitness call; the duration is added to the time
  objective.

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
# are published design estimates, not flight data: Isp ~10 000 s; 2.5-5 N per
# MW of fusion power in the NIAC Phase I modelling (Thomas, Paluszek & Cohen
# 2017), 5-10 N per MW as quoted by Gajeri, Aime & Kezerashvili (2020,
# arXiv:2009.12621). The 1 MW / 5 N point is the PSS Pluto orbiter study's
# engine (~1000 kg to Pluto orbit in ~4 yr); the 2 MW / 10 N point is sized
# for the 20 t spacecraft here. Sources are listed in the README. Replace with
# your own engine's numbers: nothing else in the pipeline is tuned to them.
ENGINES = {e.name: e for e in (
    Engine('dfd-1mw', thrust_n=5.0, isp_s=10000.0, power_mw=1.0,
           note='PSS DFD, 1 MW, 5 N @ 10 000 s (PSS Pluto orbiter study engine)'),
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
            v-infinity gene and of the capture table (which may cap lower, at
            its last converged row; an inequality constraint enforces that).
            Low-thrust capture cost grows faster than v-infinity (shedding it
            in the heliocentric phase is cheaper), so trajectories arrive
            slow. Defaults to 4 km/s.
        periapsis_radii (float, optional): Periapsis of the target orbit, in
            radii of the destination. Defaults to 1.25.
        apoapsis_km (float, optional): Apoapsis of the target orbit, km.
            Defaults to 443 449 km, 1.25 x Triton's semi-major axis
            (354 759 km): the orbit crosses Triton's and stays deep inside
            Neptune's Hill sphere (~116 million km).
        capture_eta (float, optional): Q-law effectivity cut: once the orbit is
            small, coast wherever thrusting is less than this fraction as
            effective as at the best point of the orbit. 0 thrusts all the way
            (fastest, ~2x the delta-v); larger saves propellant and takes
            longer. Defaults to 0.2.
        capture_kswitch (float, optional): Coasting is allowed only once the
            orbit is bound with apoapsis below this multiple of the target's;
            before that the engine thrusts continuously so the first periapsis
            pass captures. Defaults to 5.
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
    arrival_vinf_max_kms: float = 4.0
    periapsis_radii: float = 1.25
    apoapsis_km: float = 1.25 * 354759.0
    capture_eta: float = 0.2
    capture_kswitch: float = 5.0
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
                    periapsis_radii=self.periapsis_radii, apoapsis_km=self.apoapsis_km,
                    capture_eta=self.capture_eta, capture_kswitch=self.capture_kswitch,
                    capture_model='planar Q-law spiral (lowthrust.capture_sim)',
                    high_fidelity=self.high_fidelity)


# ------------------------------------------------------------------ capture --
def _q_terms(x, y, vx, vy, mu, e_target, rp_target):
    """Return the capture Q-law's Lyapunov value and its velocity gradient.

    ``Q = dE^2 + dp^2`` with ``dE`` the energy error relative to the target
    energy and ``dp`` the periapsis-radius error relative to the target
    periapsis. Energy and periapsis (rather than apoapsis) stay defined on a
    hyperbola, so one law runs from arrival to the final orbit.

    Returns:
        tuple: ``(Q, (dQ/dvx, dQ/dvy), rp, E, h, e)``.
    """
    r = math.hypot(x, y)
    E = 0.5 * (vx * vx + vy * vy) - mu / r
    h = x * vy - y * vx
    e = math.sqrt(max(1.0 + 2.0 * E * h * h / (mu * mu), 1e-16))
    rp = h * h / (mu * (1.0 + e))
    ge = ((h * h * vx - 2.0 * E * h * y) / (mu * mu * e),
          (h * h * vy + 2.0 * E * h * x) / (mu * mu * e))
    grp = ((-2.0 * h * y * (1.0 + e) - h * h * ge[0]) / (mu * (1.0 + e) ** 2),
           (2.0 * h * x * (1.0 + e) - h * h * ge[1]) / (mu * (1.0 + e) ** 2))
    dE = (E - e_target) / abs(e_target)
    dp = (rp - rp_target) / rp_target
    grad = (2.0 * dE * vx / abs(e_target) + 2.0 * dp * grp[0] / rp_target,
            2.0 * dE * vy / abs(e_target) + 2.0 * dp * grp[1] / rp_target)
    return dE * dE + dp * dp, grad, rp, E, h, e


_NU = np.linspace(-math.pi, math.pi, 49)[:-1]


def _best_rate(h, e, E, rp, mu, e_target, rp_target):
    """Return the largest ``|grad_v Q|`` on the osculating orbit (its bound arc for a hyperbola)."""
    p = h * h / mu
    nu = _NU if e < 1.0 else _NU[np.abs(_NU) < math.acos(-1.0 / e) - 0.05]
    cn, sn = np.cos(nu), np.sin(nu)
    r = p / (1.0 + e * cn)
    x, y = r * cn, r * sn
    vx, vy = -mu / h * sn, mu / h * (e + cn)
    ge = ((h * h * vx - 2.0 * E * h * y) / (mu * mu * e),
          (h * h * vy + 2.0 * E * h * x) / (mu * mu * e))
    grp = ((-2.0 * h * y * (1.0 + e) - h * h * ge[0]) / (mu * (1.0 + e) ** 2),
           (2.0 * h * x * (1.0 + e) - h * h * ge[1]) / (mu * (1.0 + e) ** 2))
    dE = (E - e_target) / abs(e_target)
    dp = (rp - rp_target) / rp_target
    gx = 2.0 * dE * vx / abs(e_target) + 2.0 * dp * grp[0] / rp_target
    gy = 2.0 * dE * vy / abs(e_target) + 2.0 * dp * grp[1] / rp_target
    return float(np.max(np.hypot(gx, gy)))


class _Budget(Exception):
    """Raised inside the integrator when a capture exceeds its evaluation budget."""


def capture_sim(vinf_ms, mass_kg, thrust_n, exhaust_ms, mu, body_radius_m, rp_m, ra_m,
                eta=0.2, kswitch=5.0, start_radius_m=None, q_tol=1e-5, ramp=0.05,
                max_days=1500.0, max_evals=600000):
    """Fly a low-thrust capture from a hyperbola into an elliptical orbit.

    Planar two-body motion about the destination, starting inbound on the
    arrival hyperbola (aimed at periapsis ``rp_m``) at ``start_radius_m``.
    The thrust points along ``-grad_v Q`` (``_q_terms``) at full magnitude.
    Once the orbit is bound with apoapsis below ``kswitch * ra_m`` it is
    throttled smoothly to zero where its effectivity (``|grad_v Q|`` here over
    its maximum on the osculating orbit) is below ``eta``; before that, and
    always when ``eta`` is 0, it only fades out where ``|grad_v Q|`` is under
    ``ramp`` of its periapsis value, the points near apoapsis where the energy
    and periapsis terms cancel and the steering direction flips. The run ends when ``Q`` falls
    below ``q_tol`` (energy and periapsis within ~0.3 %).

    Args:
        vinf_ms (float): Arrival v-infinity, m/s.
        mass_kg (float): Mass at arrival, kg.
        thrust_n (float): Thrust, N.
        exhaust_ms (float): Exhaust speed, m/s.
        mu (float): Destination GM, m^3/s^2.
        body_radius_m (float): Destination radius; reaching 1.02 of it ends the
            run as a crash.
        rp_m (float): Target periapsis radius, m.
        ra_m (float): Target apoapsis radius, m.
        eta (float, optional): Effectivity cut; 0 never coasts. Defaults to 0.2.
        kswitch (float, optional): Apoapsis multiple below which coasting is
            allowed. Defaults to 5.
        start_radius_m (float, optional): Radius where the capture starts;
            ``None`` uses ``5 * ra_m``. Defaults to None.
        q_tol (float, optional): Convergence threshold on ``Q``. Defaults to 1e-5.
        ramp (float, optional): Width of the throttle ramp in effectivity.
            Defaults to 0.05.
        max_days (float, optional): Time limit. Defaults to 1500 days.
        max_evals (int, optional): Right-hand-side evaluation limit. Defaults to
            600 000.

    Returns:
        dict: ``status`` (``ok``, ``crash``, ``timeout`` or ``budget``),
        ``dv_ms``, ``days``, ``final_mass_kg``, ``rp_m``, ``ra_m``.
    """
    from scipy.integrate import solve_ivp   # only the table build needs scipy

    e_target = -mu / (rp_m + ra_m)
    r0 = start_radius_m or 5.0 * ra_m
    h0 = rp_m * math.sqrt(vinf_ms ** 2 + 2.0 * mu / rp_m)
    v0 = math.sqrt(vinf_ms ** 2 + 2.0 * mu / r0)
    vt = h0 / r0
    state0 = [r0, 0.0, -math.sqrt(max(v0 * v0 - vt * vt, 0.0)), vt, mass_kg]
    e_switch = -mu / (rp_m + kswitch * ra_m)
    evals = [0]

    def rhs(_t, s):
        evals[0] += 1
        if evals[0] > max_evals:
            raise _Budget
        _q, g, rp, E, h, e = _q_terms(s[0], s[1], s[2], s[3], mu, e_target, rp_m)
        n = math.hypot(*g)
        if eta > 0.0 and E < e_switch:
            ref, cut = _best_rate(h, e, E, rp, mu, e_target, rp_m), eta
        else:
            # Thrust everywhere, but fade out where the energy and periapsis
            # terms of the gradient cancel (near apoapsis): there the
            # steering direction flips and the integrator would stall.
            ref, cut = math.hypot(*_q_terms(rp, 0.0, 0.0, h / rp, mu, e_target, rp_m)[1]), 0.0
        ref = max(ref, n, 1e-300)
        throttle = min(max((n / ref - cut) / ramp, 0.0), 1.0)
        r = math.hypot(s[0], s[1])
        grav = -mu / r ** 3
        acc = throttle * thrust_n / s[4] / n if n > 0.0 else 0.0
        return [s[2], s[3], grav * s[0] - acc * g[0], grav * s[1] - acc * g[1],
                -throttle * thrust_n / exhaust_ms]

    def done(_t, s):
        return _q_terms(s[0], s[1], s[2], s[3], mu, e_target, rp_m)[0] - q_tol
    done.terminal, done.direction = True, -1

    def crash(_t, s):
        return math.hypot(s[0], s[1]) - 1.02 * body_radius_m
    crash.terminal = True

    try:
        sol = solve_ivp(rhs, [0.0, max_days * 86400.0], state0, method='DOP853', rtol=1e-9,
                        atol=[1.0, 1.0, 1e-6, 1e-6, 1e-6], events=[done, crash])
    except _Budget:
        return dict(status='budget', dv_ms=float('nan'), days=float('nan'),
                    final_mass_kg=float('nan'), rp_m=float('nan'), ra_m=float('nan'))
    status = ('ok' if sol.t_events[0].size else 'crash' if sol.t_events[1].size else 'timeout')
    _q, _g, rp, E, h, e = _q_terms(*sol.y[:4, -1], mu, e_target, rp_m)
    m_end = float(sol.y[4, -1])
    return dict(status=status, dv_ms=exhaust_ms * math.log(mass_kg / m_end),
                days=float(sol.t[-1]) / 86400.0, final_mass_kg=m_end, rp_m=rp,
                ra_m=(-mu / E - rp) if E < 0 else float('inf'))


def _capture_cell(job):
    """Fill one ``CaptureTable`` cell: the efficient law first, continuous thrust as fallback."""
    for eta in dict.fromkeys((job['eta'], 0.0)):
        result = capture_sim(job['vinf_ms'], job['mass_kg'], eta=eta, **job['common'])
        if result['status'] == 'ok':
            return dict(result, eta=eta)
    return dict(result, eta=None)


class CaptureTable:
    """Capture delta-v and duration over arrival v-infinity and arrival mass.

    Only fully converged v-infinity rows are kept (``capture_table`` trims the
    rest), and ``vinf_kms[-1]`` is the largest arrival v-infinity the problem
    accepts (an inequality constraint of ``MgaLowThrust``). Lookups use a
    bicubic spline, so the objective and the final-mass constraint stay smooth
    for the gradient-based polish.

    Args:
        vinf_kms (numpy.ndarray): v-infinity grid, km/s.
        mass_kg (numpy.ndarray): Arrival-mass grid, kg.
        dv_ms (numpy.ndarray): Delta-v, m/s, shape ``(len(vinf), len(mass))``.
        days (numpy.ndarray): Duration, days, same shape.
        eta_used (numpy.ndarray): Effectivity cut that converged per cell.
        key (str): Hash of every input; the cache file name.
    """

    def __init__(self, vinf_kms, mass_kg, dv_ms, days, eta_used, key):
        """Store the grids; the splines are built on first lookup."""
        self.vinf_kms, self.mass_kg = np.asarray(vinf_kms), np.asarray(mass_kg)
        self.dv_ms, self.days = np.asarray(dv_ms), np.asarray(days)
        self.eta_used, self.key = np.asarray(eta_used), key
        self._splines = None

    def __getstate__(self):
        """Drop the splines before pickling; they are rebuilt where used."""
        state = self.__dict__.copy()
        state['_splines'] = None
        return state

    @property
    def vinf_max_ms(self):
        """Largest arrival v-infinity the table covers, m/s."""
        return float(self.vinf_kms[-1]) * 1000.0

    def lookup(self, vinf_ms, mass_kg):
        """Return ``(dv_ms, days)`` at one point, clamped to the grid."""
        if self._splines is None:
            from scipy.interpolate import RectBivariateSpline
            kx, ky = min(3, len(self.vinf_kms) - 1), min(3, len(self.mass_kg) - 1)
            self._splines = [RectBivariateSpline(self.vinf_kms, self.mass_kg, grid, kx=kx, ky=ky)
                             for grid in (self.dv_ms, self.days)]
        v = min(max(vinf_ms / 1000.0, self.vinf_kms[0]), self.vinf_kms[-1])
        m = min(max(mass_kg, self.mass_kg[0]), self.mass_kg[-1])
        return tuple(float(sp(v, m)[0, 0]) for sp in self._splines)


# Capture-table grid: v-infinity in km/s (topped up to the arrival bound) and
# the number of arrival-mass points between the mass floor and the wet mass.
CAPTURE_VINF_KMS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
CAPTURE_MASS_POINTS = 4
_TABLES = {}


def capture_table(sc, mu, body_radius_m, cache_dir=None, workers=None, log=print):
    """Return the capture table of a spacecraft at a destination, building it once.

    The table is memoised per process and cached as ``capture_<key>.npz`` in
    ``cache_dir``, where ``key`` hashes every input, so a change to the engine,
    target orbit, Q-law settings or grid rebuilds it. Building runs one
    ``capture_sim`` per cell in a spawn process pool (seconds to a minute each).

    Args:
        sc (Spacecraft): Spacecraft, engine and target orbit.
        mu (float): Destination GM, m^3/s^2.
        body_radius_m (float): Destination radius, m.
        cache_dir (pathlib.Path, optional): Cache directory; ``None`` keeps the
            table in memory only. Defaults to None.
        workers (int, optional): Pool size; ``None`` uses the CPU count.
            Defaults to None.
        log (Callable[[str], None], optional): Progress printer. Defaults to print.

    Returns:
        CaptureTable: The table.
    """
    import hashlib
    import json
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    vinf = sorted({v for v in CAPTURE_VINF_KMS if v < sc.arrival_vinf_max_kms}
                  | {sc.arrival_vinf_max_kms})
    mass = list(np.linspace(sc.min_final_mass_kg, sc.wet_mass_kg, CAPTURE_MASS_POINTS))
    common = dict(thrust_n=sc.engine.thrust_n, exhaust_ms=sc.engine.exhaust_ms, mu=mu,
                  body_radius_m=body_radius_m, rp_m=sc.periapsis_radii * body_radius_m,
                  ra_m=sc.apoapsis_km * 1e3, kswitch=sc.capture_kswitch)
    key = hashlib.sha1(json.dumps(dict(common, eta=sc.capture_eta, vinf=vinf, mass=mass,
                                       version=3), sort_keys=True).encode()).hexdigest()[:16]
    if key in _TABLES:
        return _TABLES[key]
    path = cache_dir / f'capture_{key}.npz' if cache_dir is not None else None
    if path is not None and path.is_file():
        z = np.load(path)
        table = CaptureTable(z['vinf_kms'], z['mass_kg'], z['dv_ms'], z['days'], z['eta_used'], key)
        _TABLES[key] = table
        return table

    jobs = [dict(vinf_ms=v * 1000.0, mass_kg=m, eta=sc.capture_eta, common=common)
            for v in vinf for m in mass]
    log(f"building the {sc.engine.name} capture table ({len(vinf)} v-inf x {len(mass)} masses, "
        f"rp {sc.periapsis_radii:g} R, ra {sc.apoapsis_km:.0f} km); cached once built")
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(workers or os.cpu_count() or 1, len(jobs)),
                             mp_context=ctx) as pool:
        cells = list(pool.map(_capture_cell, jobs))
    shape = (len(vinf), len(mass))
    ok = np.array([c['eta'] is not None for c in cells]).reshape(shape)
    dv = np.array([c['dv_ms'] for c in cells]).reshape(shape)
    days = np.array([c['days'] for c in cells]).reshape(shape)
    eta_used = np.array([-1.0 if c['eta'] is None else c['eta'] for c in cells]).reshape(shape)
    for i, v in enumerate(vinf):
        log(f"  v-inf {v:4.1f} km/s: " + "  ".join(
            f"{dv[i, j]:6.0f} m/s {days[i, j]:5.0f} d" if ok[i, j] else "  FAILED      "
            for j in range(len(mass))))
    # keep the converged prefix of rows; arrivals faster than that are refused
    rows = next((i for i in range(len(vinf)) if not ok[i].all()), len(vinf))
    if rows < 2:
        raise RuntimeError("capture table: no two v-infinity rows converged; "
                           "check the engine and target orbit")
    if rows < len(vinf):
        log(f"  arrival v-inf capped at {vinf[rows - 1]:g} km/s (first unconverged row)")
    table = CaptureTable(np.array(vinf[:rows]), np.array(mass), dv[:rows], days[:rows],
                         eta_used[:rows], key)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vinf_kms=table.vinf_kms, mass_kg=table.mass_kg, dv_ms=table.dv_ms,
                 days=table.days, eta_used=table.eta_used)
    _TABLES[key] = table
    return table


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
        table (CaptureTable, optional): Capture costs; ``None`` builds (or loads)
            it with ``capture_table`` and no disk cache. Defaults to None.
    """

    def __init__(self, planets, sc, t0, tof_days, multi_objective=False, table=None):
        """Store the problem data and build the reusable Sims-Flanagan leg."""
        self.planets = list(planets)
        self.sc = sc
        self.t0 = [float(t0[0]), float(t0[1])]
        self.tof_days = [float(tof_days[0]), float(tof_days[1])]
        self.multi_objective = multi_objective
        self.n_legs = len(self.planets) - 1
        self.layout = Layout(self.n_legs, sc.n_seg, sc.launch_c3_km2s2 > 0.0)
        dest = self.planets[-1]
        self.table = table if table is not None else capture_table(sc, dest.mu_self, dest.radius)
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
        """Return the number of inequality constraints: throttle norms, mass floor, arrival speed."""
        return self.n_legs * self.sc.n_seg + 2

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
        """Return the low-thrust capture of a decision vector, from the capture table.

        Returns:
            dict: ``dv_ms``, ``days``, ``final_mass_kg``, ``propellant_kg``,
            ``vinf_ms`` and ``arrival_mass_kg``.
        """
        b = boundary if boundary is not None else self.boundary_states(x)
        vinf = float(np.linalg.norm(b['vinf_arr']))
        mass = float(b['end'][-1][2])
        dv, days = self.table.lookup(vinf, mass)
        final = mass * math.exp(-dv / self.sc.engine.exhaust_ms)
        return dict(dv_ms=dv, days=days, final_mass_kg=final, propellant_kg=mass - final,
                    vinf_ms=vinf, arrival_mass_kg=mass)

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
        # arrivals faster than the capture table covers (km/s)
        ineq.append((cap['vinf_ms'] - self.table.vinf_max_ms) / 1000.0)
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
