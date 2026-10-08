"""Triton-centred frame: transforms, gravity, rotation and orbit metrics (km, s, km/s, rad).

Frame TC: centred on Triton's analytic circular ephemeris ``r_T(t)``
(``system.TritonOrbit``), axes parallel to NEQ, non-rotating. A state maps as

    rho = r - r_T(t),        w = v - v_T(t).

Gravity in TC is the NEQ vacuum model of ``dynamics.vacuum_rhs`` translated
exactly (the analytic ephemeris accelerates as -n^2 r_T):

    rho'' = -mu_T rho/|rho|^3 + a_N(rho + r_T) - mu_T r_T/|r_T|^3 + n^2 r_T

with a_N Neptune's point mass + J2. A state integrated in NEQ and in TC
therefore agrees to round-off. Because the circular fit's n differs slightly
from the Keplerian rate of the model, this form carries a near-uniform residual
acceleration of order 1e-9 km/s^2 (a few km per day); the long stability
propagation uses the physical differential tide instead,

    rho'' = -mu_T rho/|rho|^3 + a_N(rho + r_T) - a_N(r_T).

Triton rotates synchronously (prograde about its own orbit normal), so the
atmosphere co-rotates at omega_T = n h_T (16.7 m/s at the surface). Triton's
orbit-plane inclination i_T of a Triton-centred orbit is measured from h_T;
i_T > 90 deg is retrograde relative to Triton's motion about Neptune. The
Triton Hill radius is a_T (mu_T / (3 mu_N))^(1/3).

Every function an ODE calls is complex-step safe.
"""
import math

import numpy as np

from aerocapture.core import dynamics
from aerocapture.core import system

MU_T = system.MU_TRITON
R_T = system.R_TRITON


def hill_radius(triton):
    """Return Triton's Hill radius in km for the fitted orbit radius."""
    return triton.a * (MU_T / (3.0 * system.MU_NEPTUNE)) ** (1.0 / 3.0)


def omega_vec(triton, rotating=True):
    """Return Triton's spin vector n h_T in rad/s (zero when ``rotating`` is False)."""
    return np.asarray(triton.h_hat) * (triton.n if rotating else 0.0)


def to_triton(t, r, v, triton):
    """Return the TC position and velocity of NEQ state(s) at time(s) ``t``."""
    return r - triton.position(t), v - triton.velocity(t)


def to_neq(t, rho, w, triton):
    """Return the NEQ position and velocity of TC state(s) at time(s) ``t``."""
    return rho + triton.position(t), w + triton.velocity(t)


def accel_translated(t, rho, triton, j2=system.J2_NEPTUNE):
    """Return the TC acceleration (km/s^2) equivalent to the NEQ vacuum model.

    Args:
        t (np.ndarray): Time(s) past the epoch, s.
        rho (np.ndarray): TC positions, km, shape ``(..., 3)``.
        triton (system.TritonOrbit): Triton's ephemeris.
        j2 (float, optional): Neptune's J2. Defaults to ``system.J2_NEPTUNE``.

    Returns:
        np.ndarray: Acceleration, km/s^2.
    """
    r_t = triton.position(t)
    a_n = dynamics.accel_neptune(rho + r_t, j2=j2)
    rt3 = dynamics.norm(r_t)[..., None] ** 3
    return (-MU_T * rho / dynamics.norm(rho)[..., None] ** 3 + a_n - MU_T * r_t / rt3
            + triton.n ** 2 * r_t)


def accel_tidal(t, rho, triton, j2=system.J2_NEPTUNE):
    """Return the TC acceleration with the differential Neptune tide, km/s^2."""
    r_t = triton.position(t)
    return (-MU_T * rho / dynamics.norm(rho)[..., None] ** 3
            + dynamics.accel_neptune(rho + r_t, j2=j2) - dynamics.accel_neptune(r_t, j2=j2))


def orbit(rho, w, triton, ra_max, rp_target):
    """Return ``dynamics.final_orbit`` about Triton with i_T measured from h_T."""
    return dynamics.final_orbit(rho, w, ra_max, rp_target, mu=MU_T,
                                pole=np.asarray(triton.h_hat))


def elements(rho, w, triton):
    """Return the osculating Triton-centred a, e, rp, ra (km) and i_T (deg) of real state(s)."""
    rho, w = np.atleast_2d(rho), np.atleast_2d(w)
    rm = np.linalg.norm(rho, axis=1)
    energy = 0.5 * np.sum(w * w, axis=1) - MU_T / rm
    h = np.cross(rho, w)
    h2 = np.sum(h * h, axis=1)
    e = np.sqrt(np.maximum(1.0 + 2.0 * energy * h2 / MU_T ** 2, 0.0))
    a = -MU_T / (2.0 * energy)
    rp = h2 / (MU_T * (1.0 + e))
    ra = np.where(energy < 0, a * (1.0 + e), np.inf)
    inc = np.degrees(np.arccos(np.clip(h @ np.asarray(triton.h_hat) / np.sqrt(h2), -1, 1)))
    return dict(a=a, e=e, rp=rp, ra=ra, i_t_deg=inc, energy=energy)


def node_line(triton):
    """Return the unit ascending-node vector of Triton's orbit plane on Neptune's equator."""
    n = np.cross([0.0, 0.0, 1.0], np.asarray(triton.h_hat))
    return n / np.linalg.norm(n)


def triton_inclination_deg(triton):
    """Return the inclination of Triton's orbit to Neptune's equator, deg."""
    return math.degrees(math.acos(triton.h_hat[2]))
