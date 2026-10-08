"""Initial guess: one dynamically consistent trajectory that seeds every phase.

Dymos will not find a Triton encounter or a working atmosphere pass from a
straight-line guess, so ``Generator.run`` builds one scipy trajectory with the
same physics as the ODEs, in five stages:

1. Plane. A free direction uses a plane of inclination ``guess_inclination_deg``
   (None or ``match_triton_plane``: Triton's plane) sharing Triton's ascending
   node; a fixed direction uses the plane through the asymptote whose Triton
   crossing lies on Triton's orbit plane, retrograde and closest to polar.
2. Triton flyby, patched conic, at the crossing of Triton's circle: the turn is
   ``guess_turn_fraction`` of the maximum at ``triton_alt_min_km``, the B-vector
   lies in Triton's orbit plane, and the side lowers the Neptune energy.
3. Entry targeting. A constant bank from ``guess_banks_deg`` (lowest peak heat
   rate within ``g_max``) is flown; the entry flight-path angle is root-found so
   the exit energy is ``guess_capture_fraction`` times the capture bound
   (outbound: the pre-flyby energy the patched conic needs). The entry point is
   rotated in-plane until the Triton crossing lies on the node line (free
   direction) or the asymptote matches (fixed direction).
4. Full-dynamics shooting (scaled least squares) makes the arc hit |v_inf| at
   r0 and pass Triton at the patched-conic B-vector.
5. Phase seeding. The arc is shifted by whole Triton periods so the periapsis
   time is within half a period of the epoch, cut at the phase boundaries, and
   the r0 state gives the B-plane parameters.

Times are seconds from the epoch. The constants below are numerical settings
of the root-finds and integrations; the ``guess_*`` fields of ``Config`` are
the tuning knobs.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core import guess as cg
from aerocapture.core import heating
from aerocapture.core import system

MU, MU_T = system.MU_NEPTUNE, system.MU_TRITON
GAMMA_BRACKET_DEG = (-25.0, -2.0)           # entry flight-path angle search interval
ATM_MAX_S = 3000.0                          # longest atmosphere pass integrated
DEEP_CAPTURE_KM2S2 = -20.0                  # stop a pass once this bound (no exit expected)
CRASH_DEPTH_KM = 20.0                       # stop a pass this far below the reference sphere
BACK_SPAN_S = 3.0e6                         # longest back-propagation from entry to r0
OUT_SPAN_S = 1.0e6                          # longest outbound coast
OUT_STOP_TRITON_RADII = 2.5                 # outbound coast stops at this many Triton orbit radii
SHOOT_TOL = 1.0                             # largest accepted scaled shooting residual
INC_TOL_DEG = 0.5                           # outbound: accepted shortfall of the final inclination


@dataclass
class Guess:
    """Seed trajectory: arrival B-plane parameters, per-phase arcs and design notes."""
    mode: str
    arrival: dict
    arcs: dict
    info: dict = field(default_factory=dict)
    vehicle_params: dict = None


@dataclass
class GuessSettings:
    """The ``Config`` inputs the guess uses (see ``from_config`` for the mapping)."""
    mode: str
    vinf: float
    epoch: object
    vehicle_params: dict
    ra_max: float
    g_max: float
    h_min: float
    window: float
    window_exit_out: float
    r0: float
    r_split: float
    turn_fraction: float
    capture_fraction: float
    banks_deg: tuple
    inclination_deg: float
    min_inclination_deg: float
    r_interface: float
    asymptote_radec: tuple = None
    verbose: bool = True

    @classmethod
    def from_config(cls, cfg, epoch, verbose=True, asymptote_radec=None):
        """Return the settings for a ``config.Config``.

        Args:
            cfg (config.Config): Inputs.
            epoch (system.Epoch): Reference epoch.
            verbose (bool, optional): Print progress. Defaults to True.
            asymptote_radec (tuple, optional): Fixed v-inf (ra, dec) in rad. Defaults to None.

        Returns:
            GuessSettings: Settings with the plane in Triton's plane when
            ``guess_inclination_deg`` is None or ``match_triton_plane`` is set.
        """
        return cls(mode=cfg.triton_mode, vinf=cfg.vinf_kms, epoch=epoch,
                   vehicle_params=cfg.vehicle_params(), ra_max=cfg.ra_target,
                   g_max=cfg.vehicle.g_max, h_min=cfg.triton_alt_min_km,
                   window=cfg.triton_window_km, window_exit_out=cfg.triton_exit_km,
                   r0=cfg.r0_km, r_split=cfg.r_split_km, turn_fraction=cfg.guess_turn_fraction,
                   capture_fraction=cfg.guess_capture_fraction,
                   banks_deg=tuple(cfg.guess_banks_deg),
                   inclination_deg=None if cfg.match_triton_plane else cfg.guess_inclination_deg,
                   min_inclination_deg=cfg.min_inclination_deg, r_interface=cfg.r_interface,
                   asymptote_radec=asymptote_radec, verbose=verbose)


class GuessError(RuntimeError):
    """Raised when no consistent guess trajectory exists."""


# ------------------------------------------------------------------ dynamics --
_OM = system.OMEGA_NEPTUNE


def _bank_fn(sigma0):
    """Return a constant bank profile ``bank(r, v) = sigma0``."""
    return lambda r, v: sigma0


def _f_atm(t, y, bank, p, atm):
    """Return d(x y z vx vy vz Q)/dt in the atmosphere (``dynamics.atmosphere_rhs``)."""
    x, yy, z, vx, vy, vz = y[:6]
    r = math.sqrt(x * x + yy * yy + z * z)
    rho = float(atm.density(r - system.R_NEPTUNE))
    sig = bank(y[:3], y[3:6])
    wx, wy, wz = vx + _OM * yy, vy - _OM * x, vz          # v - omega x r
    vm = math.sqrt(wx * wx + wy * wy + wz * wz)
    ux, uy, uz = wx / vm, wy / vm, wz / vm
    rx, ry, rz = x / r, yy / r, z / r
    ru = rx * ux + ry * uy + rz * uz
    upx, upy, upz = rx - ru * ux, ry - ru * uy, rz - ru * uz
    un = math.sqrt(upx * upx + upy * upy + upz * upz)
    upx, upy, upz = upx / un, upy / un, upz / un
    sx, sy, sz = uy * upz - uz * upy, uz * upx - ux * upz, ux * upy - uy * upx
    drag = 0.5 * rho * (vm * 1e3) ** 2 / p["beta"] * 1e-3
    cs, sn = math.cos(sig), math.sin(sig)
    ld = p["ld"]
    gx, gy, gz = cg.accel_neptune_scalar(x, yy, z)
    ax = gx + drag * (-ux + ld * (cs * upx + sn * sx))
    ay = gy + drag * (-uy + ld * (cs * upy + sn * sy))
    az = gz + drag * (-uz + ld * (cs * upz + sn * sz))
    v_ms = vm * 1e3
    q_conv = heating.q_convective(rho, v_ms, p["rn"], p["conv_model"])
    q_rad = heating.q_radiative_uncoupled(rho, v_ms, p["rn"], p["rn_exp"])
    if p["coupled"]:
        q_rad = heating.q_radiative_coupled(q_rad, rho, v_ms, p["kappa"])
    return np.array([vx, vy, vz, ax, ay, az, q_conv + q_rad])


def atmosphere_diagnostics(t, y, sigma, p):
    """Return the heating and g-load diagnostics of ``dynamics.atmosphere_rhs`` along an arc."""
    rho = atmosphere.default().density(np.linalg.norm(y[:, :3], axis=1) - system.R_NEPTUNE)
    return dynamics.atmosphere_rhs(None, y[:, :3], y[:, 3:6], sigma, rho, p)[2]


def fly_atmosphere(t0, y0, bank, p, r_interface, rtol=1e-9):
    """Fly from the entry interface to the exit interface.

    The pass stops early, as not exited, on a crash ``CRASH_DEPTH_KM`` below
    the reference sphere or once the energy falls below ``DEEP_CAPTURE_KM2S2``.

    Args:
        t0 (float): Entry time, s.
        y0 (np.ndarray): Entry state (6,), km and km/s.
        bank (callable): ``bank(r, v)`` in rad.
        p (dict): Vehicle parameters (``Config.vehicle_params``).
        r_interface (float): Interface radius, km.
        rtol (float, optional): Integration tolerance. Defaults to 1e-9.

    Returns:
        tuple: (Arc with bank and heat load, bool exited).
    """
    atm = atmosphere.default()
    events = (cg.radius_event(r_interface, 1),
              cg.radius_event(system.R_NEPTUNE - CRASH_DEPTH_KM, -1),
              cg.event(lambda t, y, *_: cg.energy(y) - DEEP_CAPTURE_KM2S2, True, -1))
    sol = solve_ivp(_f_atm, (t0, t0 + ATM_MAX_S), np.concatenate([y0, [0.0]]),
                    args=(bank, p, atm), method="DOP853", rtol=rtol, atol=1e-8,
                    events=events, dense_output=True, first_step=0.1)
    tt = np.linspace(sol.t[0], sol.t[-1], 1201)
    yy = sol.sol(tt).T
    sig = np.array([bank(s[:3], s[3:6]) for s in yy])
    return cg.Arc(tt, yy[:, :6], sig, yy[:, 6]), len(sol.t_events[0]) > 0


def _triton_distance_event(triton, dist, direction=0, terminal=False):
    """Return a ``solve_ivp`` event at Triton distance ``dist`` (km)."""
    def ev(t, y, *_):
        d = y[:3] - triton.position(t)
        return math.sqrt(d @ d) - dist
    return cg.event(ev, terminal, direction)


def _closest_approach_event(triton):
    """Return a ``solve_ivp`` event at each Triton closest approach (d . d' = 0)."""
    def ev(t, y, *_):
        return (y[:3] - triton.position(t)) @ (y[3:6] - triton.velocity(t))
    return ev


# ----------------------------------------------------------- patched conics --
def turn_max(v_rel, r_min):
    """Return the largest flyby turn angle (rad) with periapsis radius ``r_min`` (km)."""
    return 2.0 * math.asin(1.0 / (1.0 + r_min * v_rel ** 2 / MU_T))


def flyby_radius(v_rel, delta):
    """Return the Triton periapsis radius (km) that turns ``v_rel`` (km/s) by ``delta`` (rad)."""
    return (1.0 / math.sin(delta / 2.0) - 1.0) * MU_T / v_rel ** 2


def _conic_velocity(energy, h_mag, pos, normal, outbound):
    """Return the Neptune-conic velocity at ``pos`` for a given energy, |h| and orbit normal."""
    r = float(np.linalg.norm(pos))
    vt = h_mag / r
    vr = math.sqrt(max(2.0 * (energy + MU / r) - vt ** 2, 0.0))
    rhat = pos / r
    return (vr if outbound else -vr) * rhat + vt * np.cross(normal, rhat)


def flyby_3d(v_known, pos, known_incoming, fraction, h_min, triton):
    """Return a patched-conic Triton flyby at ``pos`` on Triton's circle.

    The turn is ``fraction`` of the maximum at altitude ``h_min`` and lies in
    the plane of v_rel and v_rel x h_T, so the B-vector is in Triton's orbit
    plane. Of the two sides the one lowering the Neptune energy is taken.

    Args:
        v_known (np.ndarray): Spacecraft velocity on the known leg, km/s.
        pos (np.ndarray): Encounter position, km.
        known_incoming (bool): True if the known leg is the pre-flyby one.
        fraction (float): Turn as a fraction of the maximum.
        h_min (float): Minimum flyby altitude, km.
        triton (system.TritonOrbit): Triton's orbit.

    Returns:
        dict: ``energy_other`` (Neptune energy of the other leg, km^2/s^2),
        ``v_rel`` (km/s), ``delta`` (rad), ``r_fly`` (km), ``dv`` (km/s) and
        ``b_vec`` (km, Triton to the asymptote of the known leg).
    """
    h_t = np.asarray(triton.h_hat)
    v_t = math.sqrt(MU / triton.a) * np.cross(h_t, cg.unit(pos))
    vrel = v_known - v_t
    vr = float(np.linalg.norm(vrel))
    delta = fraction * turn_max(vr, system.R_TRITON + h_min)
    e_perp = cg.unit(np.cross(vrel, h_t))
    r_pos = float(np.linalg.norm(pos))
    sides = []
    for sgn in (1.0, -1.0):
        other_rel = math.cos(delta) * vrel + math.sin(delta) * vr * sgn * e_perp
        other = other_rel + v_t
        sides.append((0.5 * (other @ other) - MU / r_pos, sgn, other_rel))
    # pre-flyby known: lowest post-flyby energy; post-flyby known: highest pre-flyby energy
    e_other, _, other_rel = min(sides) if known_incoming else max(sides)
    v_in, v_out = (vrel, other_rel) if known_incoming else (other_rel, vrel)
    turn = v_out - v_in
    side = cg.unit(turn - (turn @ vrel) * vrel / vr ** 2)
    r_fly = flyby_radius(vr, delta)
    b = r_fly * math.sqrt(1 + 2 * MU_T / (r_fly * vr ** 2))
    return dict(energy_other=e_other, v_rel=vr, delta=delta, r_fly=r_fly,
                dv=2 * vr * math.sin(delta / 2), b_vec=-b * side)


# ---------------------------------------------------------------- generator --
class Generator:
    """Build a ``Guess`` for one ``GuessSettings`` (stages in the module docstring)."""

    def __init__(self, s: GuessSettings):
        self.s = s
        self.tr = s.epoch.triton
        self.h_t = np.asarray(self.tr.h_hat)
        self.fixed_dir = s.asymptote_radec is not None
        self.e_max = -MU / (s.ra_max + system.R_NEPTUNE)     # capture bound with rp ~ R_N
        self.e_goal = s.capture_fraction * self.e_max
        self.info = {"fixed_asymptote": self.fixed_dir}
        self._set_inclination(s.inclination_deg)

    def log(self, *a):
        """Print a progress line when verbose."""
        if self.s.verbose:
            print("[guess]", *a, flush=True)

    def run(self):
        """Return the ``Guess`` for the configured Triton mode."""
        mode = self.s.mode
        self.log(f"mode {mode}, v_inf {self.s.vinf} km/s, plane inclination "
                 f"{self.info['inclination_deg']:.1f} deg, exit energy goal "
                 f"{self.e_goal:.4f} km2/s2")
        if self.fixed_dir:
            self._plane_through_asymptote(mode)
        if mode == "none":
            arcs, y_e, t_e = self._none()
        elif mode == "inbound":
            arcs, y_e, t_e = self._inbound()
        else:
            arcs, y_e, t_e = self._outbound_retrograde()
        t_r0, y_r0 = arcs.pop("_r0")
        arrival = dynamics.state_to_bplane(y_r0[:3], y_r0[3:], t_r0)
        v = float(np.linalg.norm(y_e[3:]))
        self.info["entry"] = dict(t=float(t_e), v=v, gamma_deg=math.degrees(math.asin(
            (y_e[:3] @ y_e[3:]) / (self.s.r_interface * v))))
        self.info["exit_energy"] = cg.energy(arcs["atmosphere"].y[-1])
        self.log("arrival", {k: round(val, 6) for k, val in arrival.items()})
        return Guess(mode=mode, arrival=arrival, arcs=arcs, info=self.info,
                     vehicle_params=dict(self.s.vehicle_params))

    # ------------------------------------------------------- stage 1: plane
    def _set_normal(self, h):
        """Set the trajectory plane from its normal; the in-plane origin is its ascending node."""
        self.h = cg.unit(h)
        self.p = cg.unit(np.cross([0.0, 0.0, 1.0], self.h))
        self.q = np.cross(self.h, self.p)
        self.u_entry = 0.0
        self.info["inclination_deg"] = math.degrees(math.acos(self.h[2]))

    def _set_inclination(self, inclination_deg):
        """Set a plane sharing Triton's ascending node (None: Triton's plane)."""
        if inclination_deg is None:
            self._set_normal(self.h_t)
            return
        z = np.array([0.0, 0.0, 1.0])
        node = cg.unit(np.cross(z, self.h_t))
        inc = math.radians(inclination_deg)
        self._set_normal(math.sin(inc) * np.cross(node, z) + math.cos(inc) * z)

    def _plane_through_asymptote(self, mode):
        """Choose the plane through the fixed asymptote.

        ``none`` (or a Triton-plane request): the plane at the guess inclination.
        ``inbound`` / ``outbound``: the plane whose Triton crossing lies on
        Triton's orbit plane, retrograde and closest to polar.

        Raises:
            GuessError: If no plane through the asymptote reaches Triton.
        """
        s = self.s
        s_t = dynamics.unit_from_radec(*s.asymptote_radec)
        e1 = cg.unit(np.cross([0.0, 0.0, 1.0], s_t))
        e2 = np.cross(s_t, e1)
        normal = lambda phi: math.cos(phi) * e1 + math.sin(phi) * e2   # z-part: sin(phi) e2_z
        v_e = self._entry_speed_two_body()
        gamma, _ = self.design_atmosphere(v_e, self.e_goal)
        if mode == "none" or s.inclination_deg is None:
            inc = s.inclination_deg
            target = math.cos(math.radians(inc)) if inc is not None else float(self.h_t[2])
            sphi = float(np.clip(target / e2[2], -1.0, 1.0)) if abs(e2[2]) > 1e-9 else 0.0
            self._set_normal(normal(math.asin(sphi)))
            self._align_asymptote(v_e, gamma)
            return

        def crossing_offset(phi):
            self._set_normal(normal(phi))
            self._align_asymptote(v_e, gamma)
            sol = self._triton_free_leg(v_e, gamma)
            c = sol.sol(self.crossing(sol))[:3]
            return float(c @ self.h_t / np.linalg.norm(c))
        phis = np.linspace(0.0, 2 * math.pi, 37)
        vals = [crossing_offset(ph) for ph in phis]
        roots = []
        for a, b, fa, fb in zip(phis[:-1], phis[1:], vals[:-1], vals[1:]):
            if fa * fb < 0:
                r = brentq(crossing_offset, a, b, xtol=1e-10)
                if normal(r)[2] <= 0.0:
                    roots.append((abs(normal(r)[2]), r))
        if not roots:
            raise GuessError(
                f"no trajectory plane through the fixed v-inf direction puts the {mode} "
                f"crossing of Triton's orbit on Triton's circle with a retrograde orbit "
                f"(min |c.h_T| = {min(abs(v) for v in vals):.3f}); an {mode} flyby needs the "
                "asymptote within ~10-20 deg of Triton's orbit plane, so this arrival direction "
                f"cannot meet Triton {mode}")
        crossing_offset(min(roots)[1])
        self.log(f"fixed asymptote: plane inclination {self.info['inclination_deg']:.2f} deg")

    # ---------------------------------------------- stage 3: entry targeting
    def _entry_speed_two_body(self):
        """Return the entry speed of a two-body hyperbola with the requested v-inf."""
        return math.sqrt(self.s.vinf ** 2 + 2 * MU / self.s.r_interface)

    def entry_state(self, v_e, gamma, psi=0.0):
        """Return the interface state for entry speed, flight-path angle and heading offset.

        ``psi`` tilts the heading out of the plane; with a fixed asymptote it
        instead rotates the whole state about the asymptote, which keeps the
        incoming direction (to within J2) while moving the plane.
        """
        u = self.u_entry
        rhat = math.cos(u) * self.p + math.sin(u) * self.q
        that = np.cross(self.h, rhat)
        if self.fixed_dir:
            vdir = math.sin(gamma) * rhat + math.cos(gamma) * that
            rot = cg.rotation(dynamics.unit_from_radec(*self.s.asymptote_radec), psi)
            return np.concatenate([rot @ (self.s.r_interface * rhat), rot @ (v_e * vdir)])
        vdir = math.sin(gamma) * rhat + math.cos(gamma) * (math.cos(psi) * that
                                                             + math.sin(psi) * self.h)
        return np.concatenate([self.s.r_interface * rhat, v_e * vdir])

    def exit_energy(self, v_e, gamma, sigma0, psi=0.0):
        """Return (exit energy, arc) of a constant-bank pass; no exit counts as deeply captured."""
        arc, ok = fly_atmosphere(0.0, self.entry_state(v_e, gamma, psi), _bank_fn(sigma0),
                                 self.s.vehicle_params, self.s.r_interface, rtol=1e-8)
        e = cg.energy(arc.y[-1])
        return (e if ok else min(e, DEEP_CAPTURE_KM2S2)), arc

    def solve_gamma(self, v_e, sigma0, e_goal, psi=0.0):
        """Return the entry flight-path angle (rad) whose pass exits at ``e_goal``, or None."""
        f = lambda g: self.exit_energy(v_e, g, sigma0, psi)[0] - e_goal
        lo, hi = (math.radians(g) for g in GAMMA_BRACKET_DEG)
        if f(lo) > 0 or f(hi) < 0:
            return None
        g = brentq(f, lo, hi, xtol=1e-8)
        if abs(f(g)) > 0.5:          # converged onto the crash discontinuity
            return None
        return g

    def design_atmosphere(self, v_e, e_goal, psi=0.0, scan=True):
        """Return (entry angle, arc) for the bank with the lowest peak heat rate within ``g_max``.

        With ``scan=False`` the bank chosen by the last scan is reused.
        """
        banks = self.s.banks_deg if scan else (self.info["bank_deg"],)
        best = None
        for b in banks:
            g = self.solve_gamma(v_e, math.radians(b), e_goal, psi)
            if g is None:
                continue
            _, arc = self.exit_energy(v_e, g, math.radians(b), psi)
            diag = atmosphere_diagnostics(arc.t, arc.y, arc.sigma, self.s.vehicle_params)
            qmax, gmax = float(diag["q_total"].max()), float(diag["g_load"].max())
            if scan:
                self.log(f"bank {b:5.1f} deg: gamma_E {math.degrees(g):7.3f} deg, peak q "
                         f"{qmax:8.1f} W/cm2, peak g {gmax:5.1f}")
            key = (gmax > self.s.g_max, qmax)
            if best is None or key < best[0]:
                best = (key, b, g, arc)
        if best is None:
            raise GuessError("no entry angle captures the vehicle for any bank in "
                             f"{banks} (v_entry {v_e:.3f} km/s)")
        self.info["bank_deg"] = best[1]
        return best[2], best[3]

    def corridor(self, v_e, e_goal):
        """Return the lift-up and lift-down entry angles reaching ``e_goal``, or None."""
        up = self.solve_gamma(v_e, 0.0, e_goal)
        down = self.solve_gamma(v_e, math.pi, e_goal)
        if up is None or down is None:
            return None
        return dict(gamma_lift_up_deg=math.degrees(up), gamma_lift_down_deg=math.degrees(down),
                    width_deg=math.degrees(down - up))

    def atm_arc(self, v_e, gamma, psi, t_e):
        """Return the pass with the chosen bank from the entry state at time ``t_e``."""
        arc, ok = fly_atmosphere(t_e, self.entry_state(v_e, gamma, psi),
                                 _bank_fn(math.radians(self.info["bank_deg"])),
                                 self.s.vehicle_params, self.s.r_interface)
        if not ok:
            raise GuessError("atmosphere arc did not exit")
        return arc

    def _align_asymptote(self, v_e, gamma):
        """Rotate the entry point in-plane until the incoming asymptote is the fixed one."""
        s_t = dynamics.unit_from_radec(*self.s.asymptote_radec)
        for _ in range(4):
            sol = self.back_propagate(self.entry_state(v_e, gamma), 0.0, triton=False)
            y0 = sol.y_events[0][0]
            bp = dynamics.state_to_bplane(y0[:3], y0[3:], sol.t_events[0][0])
            s_g = dynamics.unit_from_radec(bp["ra"], bp["dec"])
            d = math.atan2(np.cross(s_g, s_t) @ self.h, s_g @ s_t)
            self.u_entry += d
            if abs(d) < 1e-9:
                break

    def _align(self, v_e, gamma):
        """Rotate the entry point onto the fixed asymptote or the crossing onto the node line."""
        if self.fixed_dir:
            self._align_asymptote(v_e, gamma)
            return
        for _ in range(3):
            sol = self._triton_free_leg(v_e, gamma)
            c = sol.sol(self.crossing(sol))[:3]
            self.u_entry -= math.atan2(c @ self.q, c @ self.p)

    # ------------------------------------------------------ vacuum helpers
    def back_propagate(self, y_e, t_e, triton=True):
        """Return the backward propagation from the entry state to r0 (events: r0, Triton CA)."""
        tr = self.tr if triton else None
        ev = [cg.radius_event(self.s.r0)] + ([_closest_approach_event(tr)] if tr else [])
        sol = cg.propagate(cg.f_neq, t_e, y_e, t_e - BACK_SPAN_S, (tr,), events=ev)
        if len(sol.t_events[0]) == 0:
            raise GuessError("back-propagation did not reach r0 (orbit bound?)")
        return sol

    def exit_coast(self, arc, triton):
        """Return the coast from atmosphere exit through the outbound Triton window.

        Events: 0 entering ``window``, 1 leaving at ``window_exit_out`` (terminal),
        2 Triton closest approach, 3 beyond ``OUT_STOP_TRITON_RADII`` Triton radii (terminal).
        """
        ev = [_triton_distance_event(self.tr, self.s.window, -1),
              _triton_distance_event(self.tr, self.s.window_exit_out, 1, terminal=True),
              _closest_approach_event(self.tr),
              cg.radius_event(OUT_STOP_TRITON_RADII * self.tr.a, 1)]
        return cg.propagate(cg.f_neq, arc.t[-1], arc.y[-1], arc.t[-1] + OUT_SPAN_S,
                            (self.tr if triton else None,), events=ev)

    def _triton_free_leg(self, v_e, gamma):
        """Return the Triton-free leg crossing Triton's orbit: approach (inbound) or exit coast."""
        if self.s.mode == "outbound":
            return self.exit_coast(self.atm_arc(v_e, gamma, 0.0, 0.0), triton=False)
        return self.back_propagate(self.entry_state(v_e, gamma), 0.0, triton=False)

    def closest_approach(self, sol, idx=1):
        """Return (t, relative position, distance) of the closest CA event ``idx``, or None."""
        if len(sol.t_events) <= idx or len(sol.t_events[idx]) == 0:
            return None
        best = None
        for t in sol.t_events[idx]:
            d = sol.sol(t)[:3] - self.tr.position(t)
            dist = float(np.linalg.norm(d))
            if best is None or dist < best[2]:
                best = (float(t), d, dist)
        return best

    def vinf_at_r0(self, sol):
        """Return the osculating v-inf at the r0 event."""
        return math.sqrt(max(2.0 * cg.energy(sol.y_events[0][0]), 0.0))

    def crossing(self, sol):
        """Return the first time a solution crosses Triton's orbital radius."""
        roots = cg.find_roots(sol, lambda t, r: np.linalg.norm(r, axis=1) - self.tr.a,
                       sol.t[0], sol.t[-1], 20000)
        if not roots:
            raise GuessError("arc never crosses Triton's orbit")
        return roots[0]

    def time_shift_for_triton(self, sol, t_guess, b_vec):
        """Return the time shift (s) putting Triton at r - B at the crossing nearest ``t_guess``."""
        roots = cg.find_roots(sol, lambda t, r: np.linalg.norm(r - b_vec, axis=1) - self.tr.a,
                       sol.t[0], sol.t[-1], 4000)
        if not roots:
            raise GuessError("arc never crosses Triton's orbit")
        t_star = min(roots, key=lambda t: abs(t - t_guess))
        target = sol.sol(t_star)[:3] - b_vec
        theta_needed = math.atan2(target @ np.asarray(self.tr.q), target @ np.asarray(self.tr.p))
        return cg.wrap_angle(theta_needed - (self.tr.theta0 + self.tr.n * t_star)) / self.tr.n

    def entry_speed_for_vinf(self, gamma, psi, t_e, v_e):
        """Return (v_e, solution) with the entry speed iterated so the v-inf at r0 matches."""
        for _ in range(10):
            sol = self.back_propagate(self.entry_state(v_e, gamma, psi), t_e)
            vinf = self.vinf_at_r0(sol)
            err = vinf - self.s.vinf
            if abs(err) < 1e-8:
                break
            v_e -= err * vinf / v_e          # d vinf / d v_e = v_e / vinf on a conic
        return v_e, sol

    # ------------------------------------------------- stage 5: phase seeding
    def snap(self, sol, t_e):
        """Return ``t_e`` shifted by whole Triton periods so t_peri is within half a period of 0."""
        y0 = sol.y_events[0][0]
        bp = dynamics.state_to_bplane(y0[:3], y0[3:], sol.t_events[0][0])
        return t_e - round(bp["t_peri"] / self.tr.period) * self.tr.period

    def split_approach(self, sol, t_e):
        """Return the r0 state and the approach and coast_in arcs, split at ``r_split``."""
        t_r0 = float(sol.t_events[0][0])
        t_split = cg.find_roots(sol, lambda t, r: np.linalg.norm(r, axis=1) - self.s.r_split,
                         t_r0, t_e, 20000)[0]
        return {"_r0": (t_r0, sol.sol(t_r0)), "approach": cg.sample(sol, t_r0, t_split),
                "coast_in": cg.sample(sol, t_split, t_e)}

    def split_triton_in(self, sol, t_e):
        """Return the r0 state and approach, triton_in, coast_in arcs, split at ``window``."""
        t_r0 = float(sol.t_events[0][0])
        tr = self.tr
        roots = cg.find_roots(sol, lambda t, r: np.linalg.norm(r - tr.position(t), axis=1)
                       - self.s.window, t_r0, t_e, 40000)
        if len(roots) < 2:
            raise GuessError("Triton window not crossed twice")
        t_in, t_out = roots[:2]
        return {"_r0": (t_r0, sol.sol(t_r0)), "approach": cg.sample(sol, t_r0, t_in),
                "triton_in": cg.sample(sol, t_in, t_out), "coast_in": cg.sample(sol, t_out, t_e)}

    # -------------------------------------------------------------- modes
    def _none(self):
        """Return (arcs, entry state, entry time) without Triton, timed to keep Triton far away."""
        v_e = self._entry_speed_two_body()
        gamma, _ = self.design_atmosphere(v_e, self.e_goal)
        t_e = 0.0
        for it in range(3):
            v_e, sol = self.entry_speed_for_vinf(gamma, 0.0, t_e, v_e)
            gamma, _ = self.design_atmosphere(v_e, self.e_goal, scan=False)
            if it == 0:
                ts = np.linspace(sol.t_events[0][0], t_e, 3000)
                rs = sol.sol(ts)[:3].T
                shifts = np.linspace(0, self.tr.period, 361)[:-1]
                mind = [np.min(np.linalg.norm(rs - self.tr.position(ts + dt), axis=1))
                        for dt in shifts]
                t_e += shifts[int(np.argmax(mind))]
                self.info["triton_min_dist_km"] = float(np.max(mind))
        if self.fixed_dir:
            self._align_asymptote(v_e, gamma)
        t_e = self.snap(self.back_propagate(self.entry_state(v_e, gamma), t_e), t_e)
        y_e = self.entry_state(v_e, gamma)
        arcs = self.split_approach(self.back_propagate(y_e, t_e), t_e)
        arcs["atmosphere"] = self.atm_arc(v_e, gamma, 0.0, t_e)
        self.info["corridor"] = self.corridor(v_e, self.e_goal)
        return arcs, y_e, t_e

    def _inbound_flyby(self, v_e, gamma):
        """Return the patched-conic flyby whose post-flyby leg enters at (v_e, gamma)."""
        e_post = 0.5 * v_e ** 2 - MU / self.s.r_interface
        h_post = self.s.r_interface * v_e * math.cos(gamma)
        pos = self.tr.a * self.p
        v_post = _conic_velocity(e_post, h_post, pos, self.h, outbound=False)
        return flyby_3d(v_post, pos, False, self.s.turn_fraction, self.s.h_min, self.tr)

    def _inbound(self):
        """Return (arcs, entry state, entry time) with the Triton flyby before entry."""
        s = self.s
        gamma = math.radians(-10.0)
        v_e = self._entry_speed_two_body()
        for i in range(3):          # entry speed whose pre-flyby leg has the requested v-inf
            vinf_err = lambda ve: math.sqrt(max(
                2 * self._inbound_flyby(ve, gamma)["energy_other"], 0.0)) - s.vinf
            v_e = brentq(vinf_err, v_e - 1.5, v_e + 0.5, xtol=1e-10)
            gamma, _ = self.design_atmosphere(v_e, self.e_goal, scan=i == 0)
        for _ in range(4):
            self._align(v_e, gamma)
            g_new, _ = self.design_atmosphere(v_e, self.e_goal, scan=False)
            done, gamma = abs(g_new - gamma) < 1e-8, g_new
            if done:
                break
        fb = self._inbound_flyby(v_e, gamma)
        self._record_flyby(fb)
        sol0 = self.back_propagate(self.entry_state(v_e, gamma), 0.0, triton=False)
        t_e = self.time_shift_for_triton(sol0, self.crossing(sol0), fb["b_vec"])
        x = np.array([v_e, t_e, 0.0])
        for _ in range(2):
            x = self._shoot_inbound(x, gamma, fb["r_fly"])
            g_new, _ = self.design_atmosphere(x[0], self.e_goal, x[2], scan=False)
            if abs(g_new - gamma) < 1e-7:
                break
            gamma = g_new
        v_e, t_e, psi = x
        sol = self.back_propagate(self.entry_state(v_e, gamma, psi), t_e)
        t_snap = self.snap(sol, t_e)
        if t_snap != t_e:
            v_e, t_e, psi = self._shoot_inbound(np.array([v_e, t_snap, psi]), gamma, fb["r_fly"])
            sol = self.back_propagate(self.entry_state(v_e, gamma, psi), t_e)
        y_e = self.entry_state(v_e, gamma, psi)
        ca_alt = self.closest_approach(sol)[2] - system.R_TRITON
        self.info["triton_ca_alt_km"] = ca_alt
        self.log(f"inbound: v_E {v_e:.5f} km/s, gamma_E {math.degrees(gamma):.4f} deg, "
                 f"Triton CA alt {ca_alt:.1f} km, v_inf {self.vinf_at_r0(sol):.6f}")
        arcs = self.split_triton_in(sol, t_e)
        arcs["atmosphere"] = self.atm_arc(v_e, gamma, psi, t_e)
        self.info["corridor"] = self.corridor(v_e, self.e_goal)
        return arcs, y_e, t_e

    def _shoot_inbound(self, x0, gamma, r_fly):
        """Return (v_e, t_e, psi) giving v-inf at r0 and a CA at ``r_fly`` in Triton's plane."""
        fail = np.array([1e3, 1e3, 1e3])

        def resid(x):
            v_e, t_e, psi = x
            try:
                sol = self.back_propagate(self.entry_state(v_e, gamma, psi), t_e)
            except GuessError:
                return fail
            ca = self.closest_approach(sol)
            if ca is None:
                return fail
            return np.array([(self.vinf_at_r0(sol) - self.s.vinf) / 1e-3, ca[2] - r_fly,
                             ca[1] @ self.h_t])
        res = cg.solve_scaled(resid, x0, [1e-3, 1.0, 1e-4])
        self.log(f"inbound shooting residual {np.round(res.fun, 4)}")
        if np.max(np.abs(res.fun)) > SHOOT_TOL:
            raise GuessError(f"inbound Triton shooting did not converge: {res.fun}")
        return res.x

    def _outbound_retrograde(self):
        """Return the outbound guess with the plane raised until the final orbit clears the bound.

        The flyby tilts the near-radial outbound leg by several degrees, so the
        plane is raised until the final inclination is within ``INC_TOL_DEG`` of
        ``min_inclination_deg`` + 2 ``INC_TOL_DEG``.
        """
        goal = self.s.min_inclination_deg + 2 * INC_TOL_DEG
        inc = self.info["inclination_deg"]
        for _ in range(4):
            arcs, y_e, t_e = self._outbound()
            y_f = arcs["triton_out"].y[-1]
            h = np.cross(y_f[:3], y_f[3:6])
            inc_f = math.degrees(math.acos(h[2] / np.linalg.norm(h)))
            self.info["final_inclination_deg"] = inc_f
            if inc_f >= goal - INC_TOL_DEG or self.fixed_dir:
                return arcs, y_e, t_e
            inc += goal - inc_f
            self.log(f"post-flyby inclination {inc_f:.2f} deg; guess plane -> {inc:.2f} deg")
            self._set_inclination(inc)
        raise GuessError("could not raise the post-flyby inclination to "
                         f"{goal:.1f} deg (last {inc_f:.2f} deg)")

    def _outbound(self):
        """Return (arcs, entry state, entry time) with the Triton flyby after exit."""
        s = self.s
        v_e = self._entry_speed_two_body()
        v_e, _ = self.entry_speed_for_vinf(math.radians(-10.0), 0.0, 0.0, v_e)
        gamma, _ = self.design_atmosphere(v_e, self.e_goal)
        pos = self.tr.a * self.p
        for _ in range(6):          # exit energy the patched conic needs to end at e_goal
            self._align(v_e, gamma)
            y_x = self.atm_arc(v_e, gamma, 0.0, 0.0).y[-1]
            h_vec = np.cross(y_x[:3], y_x[3:6])

            def flyby(e_pre):
                v_pre = _conic_velocity(e_pre, float(np.linalg.norm(h_vec)), pos, cg.unit(h_vec),
                                        outbound=True)
                return flyby_3d(v_pre, pos, True, s.turn_fraction, s.h_min, self.tr)
            e_pre = brentq(lambda e: flyby(e)["energy_other"] - self.e_goal,
                           self.e_goal - 0.5, self.e_goal + 8.0, xtol=1e-10)
            fb = flyby(e_pre)
            g_new, _ = self.design_atmosphere(v_e, e_pre, scan=False)
            done, gamma = abs(g_new - gamma) < 1e-8, g_new
            if done:
                break
        self._record_flyby(fb, exit_energy_pre=e_pre)
        sol0 = self.exit_coast(self.atm_arc(v_e, gamma, 0.0, 0.0), triton=False)
        t_e = self.time_shift_for_triton(sol0, self.crossing(sol0), fb["b_vec"])
        lsq = lambda x0: cg.solve_scaled(lambda x: self._outbound_resid(x, v_e, fb["b_vec"])[0],
                                       x0, [1e-6, 1.0, 1e-4])
        x = np.array([gamma, t_e, 0.0])
        for _ in range(3):
            res = lsq(x)
            x = res.x
            self.log(f"outbound shooting residual {np.round(res.fun, 4)}")
            v_new, sol = self.entry_speed_for_vinf(x[0], x[2], x[1], v_e)
            if abs(v_new - v_e) < 1e-8:
                break
            v_e = v_new
        gamma, t_e, psi = x
        t_snap = self.snap(sol, t_e)
        if t_snap != t_e:
            res = lsq(np.array([gamma, t_snap, psi]))
            gamma, t_e, psi = res.x
            sol = self.back_propagate(self.entry_state(v_e, gamma, psi), t_e)
        if np.max(np.abs(res.fun)) > SHOOT_TOL:
            raise GuessError(f"outbound Triton shooting did not converge: {res.fun}")
        _, (arc, fsol) = self._outbound_resid(np.array([gamma, t_e, psi]), v_e, fb["b_vec"])
        y_e = self.entry_state(v_e, gamma, psi)
        arcs = self.split_approach(sol, t_e)
        arcs["atmosphere"] = arc
        t_in, t_out = float(fsol.t_events[0][0]), float(fsol.t_events[1][0])
        arcs["coast_mid"] = cg.sample(fsol, arc.t[-1], t_in)
        arcs["triton_out"] = cg.sample(fsol, t_in, t_out)
        self.info["triton_ca_alt_km"] = self.closest_approach(fsol, idx=2)[2] - system.R_TRITON
        self.info["final_energy"] = cg.energy(fsol.y_events[1][0])
        self.log(f"outbound: exit energy {cg.energy(arc.y[-1]):.4f}, after Triton "
                 f"{self.info['final_energy']:.4f}, CA alt {self.info['triton_ca_alt_km']:.1f} km")
        self.info["corridor"] = self.corridor(v_e, e_pre)
        return arcs, y_e, t_e

    def _outbound_resid(self, x, v_e, b_target):
        """Return the scaled residuals (post-flyby energy, Triton B_T, B_R) and (arc, coast)."""
        gamma, t_e, psi = x
        arc, ok = fly_atmosphere(t_e, self.entry_state(v_e, gamma, psi),
                                 _bank_fn(math.radians(self.info["bank_deg"])),
                                 self.s.vehicle_params, self.s.r_interface)
        if not ok:
            return np.array([1e3, 1e3, 1e3]), None
        sol = self.exit_coast(arc, triton=True)
        if len(sol.t_events[1]) == 0 or len(sol.t_events[2]) == 0:
            ts = np.linspace(sol.t[0], sol.t[-1], 4000)
            d = np.linalg.norm(sol.sol(ts)[:3].T - self.tr.position(ts), axis=1)
            return np.array([1e2, d.min() / 10.0, 1e2]), None
        t_ca, d, _ = self.closest_approach(sol, idx=2)
        v_rel = sol.sol(t_ca)[3:] - self.tr.velocity(t_ca)
        bp = dynamics.state_to_bplane(d, v_rel, t_ca, mu=MU_T)
        t_ax, r_ax = dynamics.bplane_axes(dynamics.unit_from_radec(bp["ra"], bp["dec"]))
        e_post = cg.energy(sol.y_events[1][0])
        return np.array([(e_post - self.e_goal) / 1e-3, (bp["b_t"] - b_target @ t_ax) / 10.0,
                         (bp["b_r"] - b_target @ r_ax) / 10.0]), (arc, sol)

    def _record_flyby(self, fb, **extra):
        """Store and print the patched-conic flyby."""
        self.info["patched_conic"] = dict(v_rel=fb["v_rel"], delta_deg=math.degrees(fb["delta"]),
                                          r_fly=fb["r_fly"], dv=fb["dv"], **extra)
        self.log("patched conic flyby", self.info["patched_conic"])


def make_guess(settings: GuessSettings) -> Guess:
    """Return the initial guess for ``settings``."""
    return Generator(settings).run()


# ------------------------------------------------------- segment clustering --
def closest_approach_time(arc, triton):
    """Return the sample time of an arc closest to Triton, s."""
    return float(arc.t[np.argmin(np.linalg.norm(arc.y[:, :3] - triton.position(arc.t), axis=1))])


def triton_weight(triton):
    """Return the segment weight function 1/d_Triton along an NEQ arc (dense near Triton)."""
    return lambda arc: 1.0 / np.linalg.norm(arc.y[:, :3] - triton.position(arc.t), axis=1)


def atmosphere_weight(arc):
    """Return (rho / rho_max)^0.25 of the Neptune-GRAM density along an NEQ arc."""
    return cg.density_weight(atmosphere.default(), system.R_NEPTUNE)(arc)
