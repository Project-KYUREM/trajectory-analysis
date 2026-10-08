"""Equations of motion and conic geometry in the NEQ frame (km, s, km/s).

Gravity, Neptune point mass plus the J2 zonal term with the pole on z:

    a = -mu r / |r|^3 * [ (1 + k (1 - 5 z^2/r^2)) for x, y ;
                          (1 + k (3 - 5 z^2/r^2)) for z ],   k = 3/2 J2 (R_J2 / r)^2

Triton third body, direct plus indirect term (Neptune's own acceleration
towards Triton, since the frame is Neptune-centred):

    a_T = -mu_T (r - r_T) / |r - r_T|^3 - mu_T r_T / |r_T|^3

Aerodynamics, with the atmosphere co-rotating rigidly about +z:

    v_rel = v - omega x r,      omega = (0, 0, OMEGA_NEPTUNE)
    D     = rho |v_rel|^2 / (2 beta)                      (m/s^2, then km/s^2)
    a_aero = D (-u + L/D (cos sigma e_up + sin sigma e_side))

where u = v_rel / |v_rel|, e_up is the radial unit vector projected normal to
u, and e_side = u x e_up. sigma = 0 is lift up, sigma = pi lift down, and
sigma > 0 rotates the lift towards e_side (the vehicle's right). The sensed
g-load is D sqrt(1 + (L/D)^2) / G0. Heating uses ``heating.py`` with the
atmosphere-relative speed in m/s. Constants come from ``system.py``.

Every function the ODEs call is complex-step safe: no ``abs``,
``np.linalg.norm``, ``np.maximum`` or branches on values. Vector arrays have
shape ``(..., 3)``.
"""
import numpy as np

from aerocapture.core import heating
from aerocapture.core import system

_OMEGA_VEC = np.array([0.0, 0.0, system.OMEGA_NEPTUNE])
_POLE = np.array([0.0, 0.0, 1.0])


def norm(v):
    """Return the complex-step safe Euclidean norm along the last axis."""
    return np.sqrt(np.sum(v * v, axis=-1))


def dot(a, b):
    """Return the dot product along the last axis."""
    return np.sum(a * b, axis=-1)


# ------------------------------------------------------------------ gravity --
def accel_neptune(r, mu=system.MU_NEPTUNE, j2=system.J2_NEPTUNE, r_j2=system.R_J2):
    """Return the point-mass plus J2 acceleration in km/s^2 at positions ``r`` (km)."""
    rm = norm(r)
    k = 1.5 * j2 * (r_j2 / rm) ** 2
    zr2 = (r[..., 2] / rm) ** 2
    scale = np.stack([1.0 + k * (1.0 - 5.0 * zr2), 1.0 + k * (1.0 - 5.0 * zr2),
                      1.0 + k * (3.0 - 5.0 * zr2)], axis=-1)
    return -mu / rm[..., None] ** 3 * r * scale


def accel_triton(r, r_triton, mu=system.MU_TRITON):
    """Return Triton's direct plus indirect acceleration in km/s^2."""
    d = r - r_triton
    return -mu * (d / norm(d)[..., None] ** 3 + r_triton / norm(r_triton)[..., None] ** 3)


def vacuum_rhs(t, r, v, triton=None, j2=system.J2_NEPTUNE):
    """Return ``(r_dot, v_dot)`` for vacuum flight.

    Args:
        t (np.ndarray): Time(s) past the reference epoch, s.
        r (np.ndarray): Positions, km.
        v (np.ndarray): Velocities, km/s.
        triton (system.TritonOrbit, optional): Third body; None leaves it out.
            Defaults to None.
        j2 (float, optional): Neptune's J2. Defaults to ``system.J2_NEPTUNE``.

    Returns:
        tuple: ``(v, a)`` in km/s and km/s^2.
    """
    a = accel_neptune(r, j2=j2)
    if triton is not None:
        a = a + accel_triton(r, triton.position(t))
    return v, a


# ------------------------------------------------------------- aerodynamics --
def relative_velocity(r, v, omega_vec=None):
    """Return the velocity relative to an atmosphere rotating rigidly at ``omega_vec``, km/s.

    Args:
        r (np.ndarray): Positions from the body centre, km.
        v (np.ndarray): Inertial velocities, km/s.
        omega_vec (np.ndarray, optional): Atmosphere angular velocity, rad/s.
            Defaults to None (Neptune's rotation about +z).

    Returns:
        np.ndarray: ``v - omega x r``, km/s.
    """
    return v - np.cross(_OMEGA_VEC if omega_vec is None else omega_vec, r)


def aero_accel(r, v, sigma, rho, beta, lift_to_drag, omega_vec=None):
    """Return the lift plus drag acceleration and the airspeed.

    Args:
        r (np.ndarray): Positions, km.
        v (np.ndarray): Inertial velocities, km/s.
        sigma (np.ndarray): Bank angle, rad.
        rho (np.ndarray): Density, kg/m^3.
        beta (float): Ballistic coefficient m / (C_D A), kg/m^2.
        lift_to_drag (float): Lift-to-drag ratio.
        omega_vec (np.ndarray, optional): Atmosphere angular velocity, rad/s.
            Defaults to None (Neptune's rotation).

    Returns:
        tuple: ``(accel, v_rel, drag)``: acceleration in km/s^2, airspeed
        ``|v - omega x r|`` in km/s, and drag magnitude in km/s^2.
    """
    vrel = relative_velocity(r, v, omega_vec)
    vm = norm(vrel)
    u = vrel / vm[..., None]
    rhat = r / norm(r)[..., None]
    up = rhat - dot(rhat, u)[..., None] * u
    up = up / norm(up)[..., None]
    side = np.cross(u, up)
    drag = 0.5 * rho * (vm * 1e3) ** 2 / beta * 1e-3
    lift_dir = np.cos(sigma)[..., None] * up + np.sin(sigma)[..., None] * side
    return drag[..., None] * (-u + lift_to_drag * lift_dir), vm, drag


def atmosphere_rhs(t, r, v, sigma, rho, vehicle_params):
    """Return ``(r_dot, v_dot, diagnostics)`` for atmospheric flight.

    Gravity is Neptune point mass plus J2; Triton is left out of the pass.

    Args:
        t (np.ndarray | None): Time, unused (the atmosphere is time-invariant).
        r (np.ndarray): Positions, km.
        v (np.ndarray): Inertial velocities, km/s.
        sigma (np.ndarray): Bank angle, rad.
        rho (np.ndarray): Density, kg/m^3.
        vehicle_params (dict): ``beta`` (kg/m^2), ``ld``, ``rn`` (m),
            ``conv_model`` (key of ``heating.CONVECTIVE_SETS``), ``rn_exp``,
            ``kappa``, ``coupled`` (1.0 applies the Goulard correction to
            ``q_rad``, 0.0 does not) and ``eps`` (emissivity).

    Returns:
        tuple: ``(v, a, diag)``; ``diag`` holds ``v_rel`` (km/s), ``g_load``,
        ``q_dyn`` (kPa), ``q_conv``, ``q_rad``, ``q_rad_unc``, ``q_total``,
        ``q_total_unc`` (W/cm^2) and ``t_radeq`` (K).
    """
    p = vehicle_params
    a_aero, vm, drag = aero_accel(r, v, sigma, rho, p["beta"], p["ld"])
    v_ms = vm * 1e3
    q_conv = heating.q_convective(rho, v_ms, p["rn"], p["conv_model"])
    q_rad_unc = heating.q_radiative_uncoupled(rho, v_ms, p["rn"], p["rn_exp"])
    q_rad_cpl = heating.q_radiative_coupled(q_rad_unc, rho, v_ms, p["kappa"])
    q_rad = p["coupled"] * q_rad_cpl + (1.0 - p["coupled"]) * q_rad_unc
    q_total = q_conv + q_rad
    diag = dict(v_rel=vm,
                g_load=drag * np.sqrt(1.0 + p["ld"] ** 2) / system.G0,
                q_dyn=0.5 * rho * v_ms ** 2 * 1e-3,
                q_conv=q_conv, q_rad=q_rad, q_rad_unc=q_rad_unc, q_total=q_total,
                q_total_unc=q_conv + q_rad_unc,
                t_radeq=heating.t_radiative_equilibrium(q_total, p["eps"]))
    return v, accel_neptune(r) + a_aero, diag


# ------------------------------------------------------------ conic geometry --
def bplane_axes(s_hat, pole=_POLE):
    """Return the B-plane axes ``(T, R)``: T = S x pole normalised, R = S x T."""
    t = np.cross(s_hat, pole)
    t = t / norm(t)[..., None]
    return t, np.cross(s_hat, t)


def unit_from_radec(ra, dec):
    """Return the NEQ unit vector at right ascension ``ra`` and declination ``dec`` (rad)."""
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def time_from_periapsis(e, f, vinf, mu):
    """Return the hyperbolic time from periapsis to true anomaly ``f`` (rad), s."""
    big_f = 2.0 * np.arctanh(np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(f / 2.0))
    return (e * np.sinh(big_f) - big_f) * mu / vinf ** 3


def hyperbola_state(vinf, ra, dec, b_t, b_r, t_peri, r0, mu=system.MU_NEPTUNE):
    """Return the state on the incoming branch of a hyperbola at radius ``r0``.

    Complex-step safe in every argument.

    Args:
        vinf (float): Hyperbolic excess speed, km/s.
        ra (float): Right ascension of the approach velocity S in NEQ, rad.
        dec (float): Declination of the approach velocity S in NEQ, rad.
        b_t (float): B-vector component along T = S x z (normalised), km.
        b_r (float): B-vector component along R = S x T, km.
        t_peri (float): Periapsis time, s.
        r0 (float): Radius at which to place the state, km.
        mu (float, optional): Central GM, km^3/s^2. Defaults to Neptune's.

    Returns:
        tuple: ``(r, v, t0)``: position in km, velocity in km/s, and the time
        at ``r0`` in s (before ``t_peri``).
    """
    s = unit_from_radec(ra, dec)
    t_ax, r_ax = bplane_axes(s)
    b_vec = b_t * t_ax + b_r * r_ax
    b = norm(b_vec)
    h_hat = np.cross(b_vec, s) / b
    e = np.sqrt(1.0 + (b * vinf ** 2 / mu) ** 2)
    p = (b * vinf) ** 2 / mu
    # S = (1/e) P + (sqrt(e^2 - 1)/e) (h x P), solved for the periapsis direction P.
    p_hat = (s - np.sqrt(e ** 2 - 1.0) * np.cross(h_hat, s)) / e
    q_hat = np.cross(h_hat, p_hat)
    f0 = -np.arccos((p / r0 - 1.0) / e)
    r = p / (1.0 + e * np.cos(f0)) * (np.cos(f0) * p_hat + np.sin(f0) * q_hat)
    v = np.sqrt(mu / p) * (-np.sin(f0) * p_hat + (e + np.cos(f0)) * q_hat)
    return r, v, t_peri + time_from_periapsis(e, f0, vinf, mu)


def state_to_bplane(r, v, t, mu=system.MU_NEPTUNE):
    """Return the ``hyperbola_state`` parameters of a hyperbolic two-body state.

    Real inputs only.

    Args:
        r (np.ndarray): Position, km.
        v (np.ndarray): Velocity, km/s.
        t (float): Time of the state, s.
        mu (float, optional): Central GM, km^3/s^2. Defaults to Neptune's.

    Returns:
        dict: ``vinf``, ``ra``, ``dec``, ``b_t``, ``b_r``, ``t_peri`` as floats.
    """
    r, v = np.asarray(r, float), np.asarray(v, float)
    rm = np.linalg.norm(r)
    h = np.cross(r, v)
    hm = np.linalg.norm(h)
    e_vec = np.cross(v, h) / mu - r / rm
    e = np.linalg.norm(e_vec)
    vinf = np.sqrt(v @ v - 2.0 * mu / rm)
    h_hat, p_hat = h / hm, e_vec / e
    s = (p_hat + np.sqrt(e ** 2 - 1.0) * np.cross(h_hat, p_hat)) / e
    b_vec = hm / vinf * np.cross(s, h_hat)
    t_ax, r_ax = bplane_axes(s)
    f = np.arccos(np.clip((e_vec @ r) / (e * rm), -1.0, 1.0)) * np.sign(r @ v)
    return dict(vinf=float(vinf), ra=float(np.arctan2(s[1], s[0])),
                dec=float(np.arcsin(s[2])), b_t=float(b_vec @ t_ax),
                b_r=float(b_vec @ r_ax), t_peri=float(t - time_from_periapsis(e, f, vinf, mu)))


def final_orbit(r, v, ra_max, rp_target, mu=system.MU_NEPTUNE, pole=_POLE):
    """Return capture metrics of the osculating two-body orbit; complex-step safe.

    ``capture = energy + mu / (ra_max + rp)`` is <= 0 exactly when the orbit
    is bound with apoapsis <= ``ra_max``, and is smooth through parabolic.

    Args:
        r (np.ndarray): Positions, km.
        v (np.ndarray): Velocities, km/s.
        ra_max (float): Largest acceptable apoapsis radius, km.
        rp_target (float): Periapsis radius after the raise burn, km.
        mu (float, optional): Central GM, km^3/s^2. Defaults to Neptune's.
        pole (np.ndarray, optional): Unit reference normal of the inclination.
            Defaults to +z (Neptune's pole).

    Returns:
        dict: ``energy`` and ``capture`` (km^2/s^2), ``rp`` and ``ra`` (km),
        ``cos_inc`` (h . pole / |h|), and ``dv_raise`` (km/s, the apoapsis burn
        lifting periapsis to ``rp_target``; meaningful for bound orbits only).
    """
    energy = 0.5 * dot(v, v) - mu / norm(r)
    h = np.cross(r, v)
    h2 = dot(h, h)
    e = np.sqrt(1.0 + 2.0 * energy * h2 / mu ** 2)
    rp = h2 / (mu * (1.0 + e))
    ra = -mu / (2.0 * energy) * (1.0 + e)
    ra_abs = np.sqrt(ra * ra)          # equals ra when bound; keeps unbound iterates finite
    dv = np.sqrt(2.0 * mu * rp_target / (ra_abs * (ra_abs + rp_target))) - np.sqrt(h2) / ra_abs
    return dict(energy=energy, rp=rp, ra=ra, capture=energy + mu / (ra_max + rp),
                cos_inc=dot(h, pole) / np.sqrt(h2), dv_raise=dv)
