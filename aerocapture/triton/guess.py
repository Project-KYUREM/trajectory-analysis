"""Initial guess: patched conics, a scipy Triton pass with adaptive release, backward shooting.

Times are seconds from the epoch; NEQ arcs hold ``x y z vx vy vz``, Triton-frame
(TC) arcs hold ``xt yt zt vxt vyt vzt`` (``frame``). ``Generator.run``:

1. Patched conic. The Neptune hyperbola (|v_inf|, periapsis at the middle of
   the rp band) lies in Triton's orbit plane, moving with Triton (``co``) or
   against it (``counter``), with periapsis on the node line of that plane on
   Neptune's equator (both ring-plane crossings then clear the rings). Its
   outbound crossing of r = a_T gives the Triton-relative velocity v_rel and,
   placing Triton there, the encounter time.
2. Triton pass (TC). The entry state at the interface lies in Triton's plane
   with the incoming asymptote along v_rel, on the side that makes the capture
   orbit retrograde about Triton (``final_sense``). The ballute flies until the
   osculating Triton apoapsis falls to ``guess_ra_fraction`` x ra_max
   (adaptive release), then the probe flies to the exit interface. The
   corridor is [shallowest entry angle that releases, steepest with minimum
   altitude >= h_min]; the guess flies at ``guess_corridor_fraction`` of it.
3. Capture coast to apoapsis, a tangential burn raising the osculating
   periapsis to the design periapsis + ``guess_rp_margin_km``, and ``final_coast_h`` of
   final orbit.
4. Backward shooting (scaled Levenberg-Marquardt): entry speed, in-plane
   rotation of the pass and entry time are solved so that the full-dynamics
   back-propagation (TC to ``frame_switch_km``, then NEQ with Neptune J2 and
   Triton to r0) has the requested |v_inf| at r0, Neptune periapsis at the
   middle of the band and periapsis on the node line. Steps 2 and 4 alternate
   until the entry angle settles.
5. Phase seeding: arcs are cut at r_split (in and out), Neptune periapsis and
   the frame switch; the r0 state gives the Neptune B-plane parameters and the
   frame-switch state the Triton-frame hyperbola of the ``triton_only`` scope.

The constants below are numerical settings of the root-finds and integrations;
the ``guess_*`` fields of ``Config`` are the tuning knobs.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from aerocapture.core import dynamics
from aerocapture.core import guess as cg
from aerocapture.core import system
from aerocapture.triton import ballute
from aerocapture.triton import frame as tf
from aerocapture.triton import ode

MU_N, MU_T = system.MU_NEPTUNE, system.MU_TRITON
ATM_RTOL, ATM_ATOL = 1e-9, 1e-9          # atmosphere-pass tolerances
BALLUTE_MAX_S = 4000.0                   # longest ballute leg integrated, s
PROBE_MAX_S = 3.0e4                      # longest probe leg (release to exit), s
COAST_MAX_S = 2.0 * 86400.0              # longest coast from exit to apoapsis, s
BACK_SPAN_S = 3.0e6                      # longest back-propagation to r0, s
GAMMA_START_DEG, GAMMA_STEP_DEG = -10.0, 2.0   # shallow-boundary scan start and step
GAMMA_STEEP_STEP_DEG = 0.25              # steep-boundary scan step
GAMMA_LIMIT_DEG = -89.0                  # steepest entry angle scanned
CORRIDOR_XTOL_DEG = 1e-4                 # corridor boundary tolerance
SHOOT_TOL = 1.0                          # largest accepted scaled shooting residual
OUTER_ITERATIONS = 3                     # corridor / shooting alternations


@dataclass
class TritonGuess:
    """Seed of every phase plus the boundary parameters.

    Attributes:
        arcs (dict): Phase name -> ``core.guess.Arc`` (NEQ or TC states).
        arrival (dict): Neptune hyperbola parameters at r0 (``hyperbola_state`` keys).
        triton_arrival (dict): Triton-frame hyperbola parameters at the frame switch.
        burn_dv (np.ndarray): Periapsis-raise burn, km/s (3,).
        ring_sign (float): Sign of e_vec . N_asc at Neptune periapsis.
        info (dict): Patched conic, corridor, pass and shooting notes.
    """
    arcs: dict
    arrival: dict
    triton_arrival: dict
    burn_dv: np.ndarray
    ring_sign: float
    info: dict = field(default_factory=dict)


class GuessError(RuntimeError):
    """Raised when no consistent guess trajectory exists."""


# ------------------------------------------------------------- patched conics --
def neptune_crossing(vinf, rp, r_cross, mu=MU_N):
    """Return the outbound crossing of radius ``r_cross`` by a Neptune hyperbola.

    Args:
        vinf (float): Hyperbolic excess speed, km/s.
        rp (float): Periapsis radius, km.
        r_cross (float): Crossing radius, km.
        mu (float, optional): Central GM. Defaults to Neptune's.

    Returns:
        dict: ``e``, ``turn_deg`` (asymptote turn), ``f_deg`` (true anomaly),
        ``v_r``, ``v_t`` (km/s), ``fpa_deg`` and ``t_from_peri_s``.
    """
    e = 1.0 + rp * vinf ** 2 / mu
    p = rp * (1.0 + e)
    f = math.acos((p / r_cross - 1.0) / e)
    h = math.sqrt(mu * p)
    v_t = h / r_cross
    v_r = math.sqrt(vinf ** 2 + 2.0 * mu / r_cross - v_t ** 2)
    return dict(e=e, turn_deg=math.degrees(2.0 * math.asin(1.0 / e)), f_deg=math.degrees(f),
                v_r=v_r, v_t=v_t, fpa_deg=math.degrees(math.atan2(v_r, v_t)),
                t_from_peri_s=float(dynamics.time_from_periapsis(e, f, vinf, mu)))


def entry_speed(v_rel, r_interface):
    """Return the Triton-interface speed of a hyperbola with excess speed ``v_rel``, km/s."""
    return math.sqrt(v_rel ** 2 + 2.0 * MU_T / r_interface)


def encounter_speeds(vinf, rp, triton, r_interface):
    """Return the Triton-relative and entry speeds after a close Neptune pass, per sense.

    The trajectory crosses Triton's orbit outbound; ``co`` moves in Triton's
    plane and sense, ``counter`` in its plane against it, ``polar`` in a plane
    perpendicular to Triton's velocity.

    Args:
        vinf (float): Neptune v-inf, km/s.
        rp (float): Neptune periapsis radius, km.
        triton (system.TritonOrbit): Triton's orbit (radius a, speed a n).
        r_interface (float): Triton interface radius, km.

    Returns:
        dict: ``crossing`` (``neptune_crossing``) and, per sense, ``v_rel`` and
        ``v_entry`` (km/s).
    """
    c = neptune_crossing(vinf, rp, triton.a)
    v_tri = triton.a * triton.n
    rel = dict(co=math.hypot(c["v_r"], c["v_t"] - v_tri),
               counter=math.hypot(c["v_r"], c["v_t"] + v_tri),
               polar=math.sqrt(c["v_r"] ** 2 + c["v_t"] ** 2 + v_tri ** 2))
    out = dict(crossing=c)
    for k, v in rel.items():
        out[k] = dict(v_rel=v, v_entry=entry_speed(v, r_interface))
    return out


def direct_intercept(vinf, triton, r_interface, theta_deg=0.0):
    """Return the Triton speeds of a direct intercept tangent to Triton's orbit (no close pass).

    The hyperbola has its periapsis on Triton's orbit; ``theta_deg`` is the
    angle between its periapsis velocity and Triton's velocity (0: co-moving).

    Returns:
        dict: ``v_peri`` (Neptune-frame speed at a_T), ``v_rel`` and ``v_entry`` (km/s).
    """
    v_p = math.sqrt(vinf ** 2 + 2.0 * MU_N / triton.a)
    v_tri = triton.a * triton.n
    v_rel = math.sqrt(v_p ** 2 + v_tri ** 2 - 2.0 * v_p * v_tri * math.cos(math.radians(theta_deg)))
    return dict(v_peri=v_p, v_rel=v_rel, v_entry=entry_speed(v_rel, r_interface))


# ------------------------------------------------------------ scalar dynamics --
def f_tc(t, y, tr, j2k=cg.J2K):
    """Return the TC vacuum derivative (``frame.accel_translated`` on floats)."""
    (tx, ty, tz), _ = cg.triton_state_scalar(tr, t)
    ax, ay, az = cg.accel_neptune_scalar(y[0] + tx, y[1] + ty, y[2] + tz, j2k)
    d2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2]
    d3 = d2 * math.sqrt(d2)
    t3 = (tx * tx + ty * ty + tz * tz) ** 1.5
    k = tr.n ** 2 - MU_T / t3
    return np.array([y[3], y[4], y[5], ax - MU_T * y[0] / d3 + k * tx,
                     ay - MU_T * y[1] / d3 + k * ty, az - MU_T * y[2] / d3 + k * tz])


def f_tc_tidal(t, y, tr, j2k=cg.J2K):
    """Return the TC derivative with the differential Neptune tide (``accel_tidal``)."""
    (tx, ty, tz), _ = cg.triton_state_scalar(tr, t)
    ax, ay, az = cg.accel_neptune_scalar(y[0] + tx, y[1] + ty, y[2] + tz, j2k)
    bx, by, bz = cg.accel_neptune_scalar(tx, ty, tz, j2k)
    d2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2]
    d3 = d2 * math.sqrt(d2)
    return np.array([y[3], y[4], y[5], ax - bx - MU_T * y[0] / d3, ay - by - MU_T * y[1] / d3,
                     az - bz - MU_T * y[2] / d3])


class PassModel:
    """Scalar right-hand side of the TC atmosphere pass for one vehicle configuration.

    Args:
        cfg (config.Config): Inputs (heating options are not needed here).
        triton (system.TritonOrbit): Ephemeris.
        atm (atmosphere.Atmosphere): Triton atmosphere.
        phase (str): ``"ballute"`` or ``"probe"``.
    """

    def __init__(self, cfg, triton, atm, phase):
        self.tr, self.atm, self.phase = triton, atm, phase
        self.p = cfg.pass_params(phase)
        self.omega = tf.omega_vec(triton, cfg.atm_rotation)

    def drag(self, y):
        """Return (drag in km/s^2, airspeed unit vector, airspeed km/s, rho) of one state."""
        r = y[:3]
        rho = float(self.atm.density(math.sqrt(r @ r) - system.R_TRITON))
        w = self.omega
        vrel = y[3:6] - np.array([w[1] * r[2] - w[2] * r[1], w[2] * r[0] - w[0] * r[2],
                                  w[0] * r[1] - w[1] * r[0]])
        vm = math.sqrt(vrel @ vrel)
        p = self.p
        kn = ballute.knudsen(rho, p["length"])
        x = min(max(0.375 + math.log10(kn) / 8.0, 0.0), 0.5)
        cd = p["cd_c"] + (p["cd_fm"] - p["cd_c"]) * math.sin(math.pi * x) ** 2
        drag = 0.5 * rho * (vm * 1e3) ** 2 * cd * p["area"] / p["mass"] * 1e-3
        return drag, vrel / vm, vm, rho

    def rhs(self, t, y):
        """Return the TC state derivative with drag."""
        f = f_tc(t, y, self.tr)
        drag, u, _, _ = self.drag(y)
        f[3:6] -= drag * u
        return f


def _orbit_metric(y, ra_target):
    """Return the capture metric E + mu_T/(ra_target + rp) of a TC state (<= 0: ra <= target)."""
    r, v = y[:3], y[3:6]
    rm = math.sqrt(r @ r)
    energy = 0.5 * (v @ v) - MU_T / rm
    h = np.cross(r, v)
    h2 = h @ h
    e = math.sqrt(max(1.0 + 2.0 * energy * h2 / MU_T ** 2, 0.0))
    rp = h2 / (MU_T * (1.0 + e))
    return energy + MU_T / (ra_target + rp)


propagate = cg.propagate          # also reached as ``guess.propagate`` by ``results.stability``


# --------------------------------------------------------------- Triton pass --
@dataclass
class PassResult:
    """One scipy atmosphere pass: ballute leg, optional probe leg and the outcome flags."""
    ballute: object
    probe: object
    released: bool
    exited: bool
    crashed: bool
    min_alt_km: float
    metric_end: float


def fly_pass(t0, y0, ballute_model, probe_model, r_interface, ra_release, release=True):
    """Fly the pass from the entry interface: ballute until release, then probe to exit.

    Args:
        t0 (float): Entry time, s.
        y0 (np.ndarray): TC entry state (6,).
        ballute_model (PassModel): Ballute-phase model.
        probe_model (PassModel): Probe-phase model.
        r_interface (float): Interface radius, km.
        ra_release (float): Release when the osculating apoapsis falls to this radius, km.
        release (bool, optional): Release at all; False keeps the ballute to exit.
            Defaults to True.

    Returns:
        PassResult: Dense solutions of both legs (probe None without release) and flags.
    """
    evs = [cg.radius_event(r_interface, 1), cg.radius_event(system.R_TRITON, -1),
           cg.apsis_event(1)]
    if release:
        evs.append(cg.event(lambda t, y: _orbit_metric(y, ra_release), True, -1))
    sol_b = solve_ivp(lambda t, y: ballute_model.rhs(t, y), (t0, t0 + BALLUTE_MAX_S), y0,
                      method="DOP853", rtol=ATM_RTOL, atol=ATM_ATOL, events=evs,
                      dense_output=True, first_step=0.05)
    alts = [np.linalg.norm(sol_b.y[:3], axis=0).min()]
    alts += [np.linalg.norm(y[:3]) for y in sol_b.y_events[2]]
    released = release and len(sol_b.t_events[3]) > 0
    exited = len(sol_b.t_events[0]) > 0
    crashed = len(sol_b.t_events[1]) > 0
    sol_p = None
    if released:
        sol_p = solve_ivp(lambda t, y: probe_model.rhs(t, y),
                          (sol_b.t[-1], sol_b.t[-1] + PROBE_MAX_S), sol_b.y[:, -1],
                          method="DOP853", rtol=ATM_RTOL, atol=ATM_ATOL, events=evs[:3],
                          dense_output=True, first_step=0.05)
        alts.append(np.linalg.norm(sol_p.y[:3], axis=0).min())
        alts += [np.linalg.norm(y[:3]) for y in sol_p.y_events[2]]
        exited = len(sol_p.t_events[0]) > 0
        crashed = len(sol_p.t_events[1]) > 0
    last = (sol_p or sol_b).y[:, -1]
    return PassResult(sol_b, sol_p, released, exited, crashed,
                      float(min(alts)) - system.R_TRITON, _orbit_metric(last, ra_release))


def pass_diagnostics(arc, cfg, triton, atm, phase):
    """Return the ``ode.aero`` diagnostics (drag, heating, loads) along a TC arc."""
    r, v = arc.y[:, :3], arc.y[:, 3:6]
    rho = atm.density(np.linalg.norm(r, axis=1) - system.R_TRITON)
    opts = cfg.eom_options(phase)
    return ode.aero(r, v, rho, cfg.pass_params(phase), triton, opts["rotating"],
                           opts["conv_model"], opts["bridging"], opts["fm_accommodation"],
                           opts["rad_in_limits"], opts["emissivity"], opts["sides"])


# ------------------------------------------------------------------ generator --
class Generator:
    """Build a ``TritonGuess`` for one config (stages in the module docstring).

    Args:
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch.
        atm (atmosphere.Atmosphere): Triton atmosphere.
        verbose (bool, optional): Print progress. Defaults to True.
    """

    def __init__(self, cfg, epoch, atm, verbose=True):
        self.cfg, self.epoch, self.atm, self.verbose = cfg, epoch, atm, verbose
        self.tr = epoch.triton
        self.h_t = np.asarray(self.tr.h_hat)
        self.ra_max = cfg.ra_max_km(self.tr)
        self.ra_release = cfg.guess_ra_fraction * self.ra_max
        self.r_i = cfg.r_interface
        self.models = {ph: PassModel(cfg, self.tr, atm, ph) for ph in ("ballute", "probe")}
        sense = -1.0 if cfg.final_sense in ("retrograde", "any") else 1.0
        self.h_sc = sense * self.h_t                 # capture-orbit normal about Triton
        self.h_n = self.h_t if cfg.guess_encounter == "co" else -self.h_t
        self.info = {}

    def log(self, *a):
        """Print a progress line when verbose."""
        if self.verbose:
            print("[guess]", *a, flush=True)

    # ------------------------------------------------------ geometry helpers
    def entry_state(self, v_e, gamma, theta):
        """Return the TC interface state at in-plane angle ``theta`` (from Triton's p axis)."""
        p, q = np.asarray(self.tr.p), np.asarray(self.tr.q)
        rhat = math.cos(theta) * p + math.sin(theta) * q
        that = np.cross(self.h_sc, rhat)
        return np.concatenate([self.r_i * rhat,
                               v_e * (math.sin(gamma) * rhat + math.cos(gamma) * that)])

    def align_theta(self, v_e, gamma, s_target):
        """Return the in-plane angle whose two-body incoming asymptote is ``s_target``."""
        y = self.entry_state(v_e, gamma, 0.0)
        bp = dynamics.state_to_bplane(y[:3], y[3:], 0.0, mu=MU_T)
        s = dynamics.unit_from_radec(bp["ra"], bp["dec"])
        return math.atan2(np.cross(s, s_target) @ self.h_t, s @ s_target)

    def pass_at(self, t_e, v_e, gamma, theta, release=True):
        """Return the ``PassResult`` of an entry at time ``t_e``."""
        return fly_pass(t_e, self.entry_state(v_e, gamma, theta), self.models["ballute"],
                        self.models["probe"], self.r_i, self.ra_release,
                        release and self.cfg.release_mode == "free")

    # ------------------------------------------------------ stage 2: corridor
    def corridor(self, t_e, v_e, theta):
        """Return the entry-angle corridor (rad) and the guess angle at ``v_e``.

        Raises:
            GuessError: If no entry angle captures with the minimum altitude respected.
        """
        h_min = self.cfg.h_min_km
        free = self.cfg.release_mode == "free"

        def shallow(g):                      # <= 0: the pass reaches the release apoapsis
            res = self.pass_at(t_e, v_e, g, theta, release=False)
            return -1.0 if res.crashed else res.metric_end

        def steep(g):                        # >= 0: minimum altitude respected (and exits)
            res = self.pass_at(t_e, v_e, g, theta, release=True)
            return res.min_alt_km - h_min if res.exited else -h_min - 1.0

        g_prev, g = None, math.radians(GAMMA_START_DEG)
        while shallow(g) > 0.0:
            g_prev, g = g, g - math.radians(GAMMA_STEP_DEG)
            if g < math.radians(GAMMA_LIMIT_DEG):
                raise GuessError(f"no entry angle down to {GAMMA_LIMIT_DEG} deg brings the "
                                 f"apoapsis to {self.ra_release:.0f} km (ballute too small)")
        g_sh = g if g_prev is None else brentq(shallow, g, g_prev,
                                               xtol=math.radians(CORRIDOR_XTOL_DEG))
        if steep(g_sh) < 0.0:
            raise GuessError(
                f"no corridor at v_E {v_e:.3f} km/s: the shallowest capturing entry "
                f"({math.degrees(g_sh):.3f} deg) already dips below h_min {h_min} km; lower "
                "the ballistic coefficient (larger or lighter ballute)")
        g_hi, g_lo = g_sh, g_sh - math.radians(GAMMA_STEEP_STEP_DEG)
        while steep(g_lo) >= 0.0:
            g_hi, g_lo = g_lo, g_lo - math.radians(GAMMA_STEEP_STEP_DEG)
            if g_lo < math.radians(GAMMA_LIMIT_DEG):
                break
        g_st = brentq(steep, g_lo, g_hi, xtol=math.radians(CORRIDOR_XTOL_DEG))
        frac = self.cfg.guess_corridor_fraction
        g_guess = g_sh + frac * (g_st - g_sh)
        cor = dict(v_entry_kms=v_e, gamma_shallow_deg=math.degrees(g_sh),
                   gamma_steep_deg=math.degrees(g_st), width_deg=math.degrees(g_sh - g_st),
                   gamma_guess_deg=math.degrees(g_guess), release_mode=self.cfg.release_mode,
                   release_ra_km=self.ra_release, free_release=free)
        self.log(f"corridor at v_E {v_e:.4f} km/s: {cor['gamma_shallow_deg']:.3f} .. "
                 f"{cor['gamma_steep_deg']:.3f} deg (width {cor['width_deg']:.3f}); "
                 f"guess {cor['gamma_guess_deg']:.3f}")
        return cor, g_guess

    # ------------------------------------------------------ stage 4: Neptune leg
    def back_to_switch(self, t_e, y_e):
        """Return the TC back-propagation from the entry to the frame-switch distance."""
        sol = propagate(f_tc, t_e, y_e, t_e - 5.0 * 86400.0, (self.tr,),
                        events=[cg.radius_event(self.cfg.frame_switch_km)])
        if len(sol.t_events[0]) == 0:
            raise GuessError("back-propagation from Triton never reached the frame switch")
        return sol

    def back_to_r0(self, t_s, y_neq):
        """Return the NEQ back-propagation to r0 with the Neptune periapsis event."""
        sol = propagate(cg.f_neq, t_s, y_neq, t_s - BACK_SPAN_S, (self.tr,),
                        events=[cg.radius_event(self.cfg.r0_km), cg.apsis_event()])
        if len(sol.t_events[0]) == 0:
            raise GuessError("back-propagation did not reach r0 (orbit bound?)")
        return sol

    def neptune_leg(self, t_s, y_tc):
        """Return the NEQ back-propagation and its periapsis from a TC frame-switch state."""
        r, v = tf.to_neq(t_s, y_tc[:3], y_tc[3:6], self.tr)
        sol = self.back_to_r0(t_s, np.concatenate([r, v]))
        if len(sol.t_events[1]) == 0:
            raise GuessError("no Neptune periapsis on the back-propagated leg")
        i = int(np.argmin([np.linalg.norm(y[:3]) for y in sol.y_events[1]]))
        return sol, float(sol.t_events[1][i]), sol.y_events[1][i]

    def ring_angle(self, r_peri):
        """Return the in-plane angle from the Neptune periapsis to the nearer node, rad."""
        n = tf.node_line(self.tr)
        h = self.h_n
        ang = math.atan2(np.cross(r_peri, n) @ h, r_peri @ n)
        return cg.wrap_angle(2.0 * ang) / 2.0

    def _residual(self, x, gamma, theta0):
        """Return the scaled shooting residuals (v_inf, rp, ring angle) of (v_E, phi, t_E)."""
        v_e, phi, t_e = x
        try:
            sw = self.back_to_switch(t_e, self.entry_state(v_e, gamma, theta0 + phi))
            sol, _, y_p = self.neptune_leg(float(sw.t_events[0][0]), sw.y_events[0][0])
        except GuessError:
            return np.array([1e3, 1e3, 1e3])
        y0 = sol.y_events[0][0]
        vinf = math.sqrt(max(2.0 * (0.5 * y0[3:] @ y0[3:] - MU_N / np.linalg.norm(y0[:3])),
                             0.0))
        rp_mid = 0.5 * sum(self.cfg.rp_band_km)
        return np.array([(vinf - self.cfg.vinf_kms) / 1e-3,
                         (np.linalg.norm(y_p[:3]) - rp_mid) / 10.0,
                         self.ring_angle(y_p[:3]) / 1e-3])

    # ------------------------------------------------------ stage 1: patched conic
    def patched_conic(self):
        """Return the patched-conic encounter: v_rel direction, entry speed and time."""
        cfg, tr = self.cfg, self.tr
        rp_mid = 0.5 * sum(cfg.rp_band_km)
        speeds = encounter_speeds(cfg.vinf_kms, rp_mid, tr, self.r_i)
        c = speeds["crossing"]
        f_c = math.radians(c["f_deg"])
        best = None
        for sgn in (1.0, -1.0):
            e_p = sgn * tf.node_line(tr)
            r_c = cg.rotation(self.h_n, f_c) @ e_p
            theta_c = math.atan2(r_c @ np.asarray(tr.q), r_c @ np.asarray(tr.p))
            t_c = (theta_c - tr.theta0) / tr.n
            t_peri = t_c - c["t_from_peri_s"]
            t_c -= round(t_peri / tr.period) * tr.period
            t_peri = t_c - c["t_from_peri_s"]
            if best is None or abs(t_peri) < abs(best[0]):
                best = (t_peri, t_c, r_c)
        t_peri, t_c, r_c = best
        v_sc = c["v_r"] * r_c + c["v_t"] * np.cross(self.h_n, r_c)
        v_rel = v_sc - tr.velocity(t_c)
        vr = float(np.linalg.norm(v_rel))
        self.info["patched_conic"] = dict(
            rp_km=rp_mid, neptune_turn_deg=c["turn_deg"], crossing_true_anomaly_deg=c["f_deg"],
            crossing_fpa_deg=c["fpa_deg"], t_peri_to_triton_h=c["t_from_peri_s"] / 3600.0,
            encounter=cfg.guess_encounter, v_rel_kms=vr, v_entry_kms=entry_speed(vr, self.r_i),
            speeds={k: v for k, v in speeds.items() if k != "crossing"})
        self.log(f"patched conic: Neptune turn {c['turn_deg']:.1f} deg, crossing f "
                 f"{c['f_deg']:.1f} deg, v_rel {vr:.3f} km/s ({cfg.guess_encounter}), "
                 f"t_peri {t_peri / 3600:.2f} h")
        return v_rel / vr, entry_speed(vr, self.r_i), t_c

    # ------------------------------------------------------ run
    def run(self):
        """Return the ``TritonGuess``."""
        s_target, v_e, t_e = self.patched_conic()
        gamma = math.radians(-50.0)
        theta = self.align_theta(v_e, gamma, s_target)
        for it in range(OUTER_ITERATIONS):
            cor, g_new = self.corridor(t_e, v_e, theta)
            theta = self.align_theta(v_e, g_new, s_target)
            settled = it > 0 and abs(g_new - gamma) < math.radians(CORRIDOR_XTOL_DEG)
            gamma = g_new
            if settled:
                break
            res = cg.solve_scaled(lambda z: self._residual(z, gamma, theta),
                                          np.array([v_e, 0.0, t_e]), [1e-3, 1e-4, 10.0])
            self.log(f"shooting {it}: residual {np.round(res.fun, 4)}, v_E {res.x[0]:.5f} km/s, "
                     f"t_E {res.x[2]:.1f} s")
            if np.max(np.abs(res.fun)) > SHOOT_TOL:
                raise GuessError(f"backward shooting did not converge: {res.fun}")
            v_e, phi, t_e = res.x
            theta += phi
            s_target = self._asymptote(v_e, gamma, theta)
        self.info["corridor"] = cor
        return self._assemble(v_e, gamma, theta, t_e)

    def _asymptote(self, v_e, gamma, theta):
        """Return the two-body incoming asymptote of the entry state at ``theta``."""
        y = self.entry_state(v_e, gamma, theta)
        bp = dynamics.state_to_bplane(y[:3], y[3:], 0.0, mu=MU_T)
        return dynamics.unit_from_radec(bp["ra"], bp["dec"])

    # ------------------------------------------------------ stage 3 and 5
    def capture_and_final(self, t_x, y_x):
        """Return (capture_coast arc, burn dv, final arc) from the exit state."""
        cfg, tr = self.cfg, self.tr

        sol = propagate(f_tc, t_x, y_x, t_x + COAST_MAX_S, (tr,),
                        events=[cg.apsis_event(-1, terminal=True)])
        if len(sol.t_events[0]) == 0:
            raise GuessError("no apoapsis within the capture-coast window (orbit unbound?)")
        t_a, y_a = float(sol.t_events[0][0]), sol.y_events[0][0]
        coast = cg.sample(sol, t_x, t_a)
        rp_goal = cfg.rp_design + cfg.guess_rp_margin_km
        t_end = t_a + cfg.final_coast_h * 3600.0
        for _ in range(4):
            orb = dynamics.final_orbit(y_a[:3], y_a[3:6], self.ra_max, rp_goal, mu=MU_T)
            dv = float(orb["dv_raise"]) * y_a[3:6] / np.linalg.norm(y_a[3:6])
            y_b = np.concatenate([y_a[:3], y_a[3:6] + dv])
            fin = propagate(f_tc, t_a, y_b, t_end, (tr,))
            ts = np.linspace(t_a, t_end, 20000)
            alt_min = float(np.min(np.linalg.norm(fin.sol(ts)[:3], axis=0))) - system.R_TRITON
            short = cfg.rp_design - system.R_TRITON + 0.5 * cfg.guess_rp_margin_km - alt_min
            if short <= 0.0:
                break
            rp_goal += short
        self.info["final_guess"] = dict(burn_dv_kms=float(np.linalg.norm(dv)),
                                        min_alt_km=alt_min, t_burn_s=t_a)
        return coast, dv, cg.sample(fin, t_a, t_end, 4000)

    def _assemble(self, v_e, gamma, theta, t_e):
        """Fly the final pass, coasts and legs; cut the phase arcs and the boundary parameters."""
        cfg, tr = self.cfg, self.tr
        y_e = self.entry_state(v_e, gamma, theta)
        ps = self.pass_at(t_e, v_e, gamma, theta)
        if not ps.exited or ps.crashed:
            raise GuessError("the guess pass did not exit the atmosphere")
        arcs = {}
        sb = ps.ballute
        if ps.probe is not None:
            arcs["ballute"] = cg.sample(sb, sb.t[0], sb.t[-1])
            arcs["probe"] = cg.sample(ps.probe, ps.probe.t[0], ps.probe.t[-1])
            t_x, y_x = ps.probe.t[-1], ps.probe.y[:6, -1]
        else:
            arcs["ballute"] = cg.sample(sb, sb.t[0], sb.t[-1])
            t_x, y_x = sb.t[-1], sb.y[:6, -1]
        diag = pass_diagnostics(arcs["ballute"], cfg, tr, self.atm, "ballute")
        q = diag["q_heat"]
        arcs["ballute"].Q = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1])
                                                             * np.diff(arcs["ballute"].t))])
        arcs["capture_coast"], burn, arcs["final"] = self.capture_and_final(t_x, y_x)
        sw = self.back_to_switch(t_e, y_e)
        t_s, y_s = float(sw.t_events[0][0]), sw.y_events[0][0]
        arcs["triton_approach"] = cg.sample(sw, t_s, t_e)
        tri_arr = dynamics.state_to_bplane(y_s[:3], y_s[3:6], t_s, mu=MU_T)
        legs = neptune_arcs(self, t_s, y_s)
        arcs.update(legs["arcs"])
        peaks = [float(np.max(diag[k])) for k in ("q_heat", "g_load", "t_film")]
        self.info["pass"] = dict(
            entry_t_s=t_e, entry_speed_kms=v_e, entry_gamma_deg=math.degrees(gamma),
            released=ps.released, min_alt_km=ps.min_alt_km, peak_q_heat_w_cm2=peaks[0],
            peak_g=peaks[1], peak_t_film_k=peaks[2],
            release_t_s=float(sb.t[-1]) if ps.released else None)
        self.info["neptune"] = legs["info"]
        self.log(f"pass: gamma_E {math.degrees(gamma):.3f} deg, min alt {ps.min_alt_km:.1f} km, "
                 f"peak q {peaks[0]:.3f} W/cm2, g {peaks[1]:.1f}, film {peaks[2]:.0f} K; "
                 f"Neptune rp {legs['info']['rp_km']:.0f} km, v_inf "
                 f"{legs['arrival']['vinf']:.5f} km/s")
        return TritonGuess(arcs=arcs, arrival=legs["arrival"], triton_arrival=tri_arr,
                           burn_dv=np.asarray(burn), ring_sign=legs["ring_sign"], info=self.info)


def neptune_arcs(gen, t_s, y_tc):
    """Return the Neptune-phase arcs, arrival parameters and ring sign from a frame-switch state.

    Args:
        gen (Generator): Generator (config, ephemeris, propagators).
        t_s (float): Frame-switch time, s.
        y_tc (np.ndarray): TC state at the frame switch (6,).

    Returns:
        dict: ``arcs`` (approach, flyby_in, flyby_out, cruise), ``arrival``
        (``state_to_bplane`` at r0), ``ring_sign`` and ``info``.
    """
    sol, t_p, y_p = gen.neptune_leg(t_s, y_tc)
    t_r0 = float(sol.t_events[0][0])
    r_split = gen.cfg.r_split_km
    splits = cg.find_roots(sol, lambda t, r: np.linalg.norm(r, axis=1) - r_split,
                                   t_r0, t_s, 20000)
    if len(splits) < 2:
        raise GuessError("the Neptune leg does not cross r_split twice")
    t_in, t_out = min(splits), max(splits)
    arcs = dict(approach=cg.sample(sol, t_r0, t_in), flyby_in=cg.sample(sol, t_in, t_p),
                flyby_out=cg.sample(sol, t_p, t_out), cruise=cg.sample(sol, t_out, t_s))
    y0 = sol.sol(t_r0)
    arrival = dynamics.state_to_bplane(y0[:3], y0[3:6], t_r0)
    u_asc, u_desc = ode.node_inverse_radii(y_p[:3], y_p[3:6])
    sign = 1.0 if u_asc >= u_desc else -1.0
    rm = float(np.linalg.norm(y_p[:3]))
    info = dict(rp_km=rm, rp_rn=rm / system.R_NEPTUNE, t_peri_s=t_p,
                ring_angle_deg=math.degrees(gen.ring_angle(y_p[:3])),
                node_radii_km=[1.0 / u if u > 0 else None for u in (u_asc, u_desc)])
    return dict(arcs=arcs, arrival=arrival, ring_sign=sign, info=info)


def make_guess(cfg, epoch, atm, verbose=True):
    """Return the initial guess for a config."""
    return Generator(cfg, epoch, atm, verbose).run()


# ------------------------------------------------------------- segment weights --
def orbit_weight(arc):
    """Return |r|^-1.5 along a TC orbit arc (dense segments at periapsis)."""
    return np.linalg.norm(arc.y[:, :3], axis=1) ** -1.5
