"""Neptune-Triton system: SPICE kernels, physical constants, frame, Triton orbit.

Frame NEQ: Neptune-centred and inertial. z is Neptune's IAU north pole at the
reference epoch (pck00011), x is the ascending node of Neptune's equator on the
J2000 equator (z_J2000 x pole), y completes the right-handed set.

Units: km, s, km/s, rad. Time is seconds past ``Epoch.et_ref`` (TDB).

Constants and sources:

    MU_NEPTUNE, MU_TRITON, GM_SUN   gm_de440.tpc (NAIF ids 899, 801, 10)
    R_TRITON, R_NEPTUNE_EQ          pck00011.tpc radii
    R_NEPTUNE = 24 622 km           altitude reference: the Neptune-GRAM table
                                    heights and orbit_planning's body registry
    J2_NEPTUNE = 3408.428e-6        Jacobson (2009), AJ 137:4322, referred to
    R_J2 = 25 225 km                its own reference radius R_J2
    OMEGA_NEPTUNE                   pck00011 prime-meridian rate (15.966 h)
    R_HILL = a (m_N / 3 M_sun)^1/3  a = 30.07 AU (JPL mean elements), m_N the
                                    Neptune-system GM (gm_de440, id 8)
    ENTRY_ALTITUDE = 1000 km        atmospheric interface (repo convention)
    G0 = 9.80665e-3 km/s^2          standard gravity, for g-loads

Triton follows a circular orbit fitted to nep097.bsp over +-6 d about the
reference epoch (true eccentricity ~2e-4): r(t) = a (cos th p + sin th q),
th = theta0 + n t. The analytic form keeps the ODE smooth and complex-step
differentiable in time; ``triton_residual_km`` measures the fit error.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spiceypy as spice

from aerocapture.core import paths

KERNELS = ("de440.bsp", "naif0012.tls", "pck00011.tpc", "gm_de440.tpc", "nep097.bsp")
ATMOSPHERE_TABLE = paths.DATA_DIR / "neptune-gram-avg.dat"


def _load_kernels():
    """Furnsh every kernel in ``KERNELS`` that is not already loaded.

    Raises:
        FileNotFoundError: If a kernel file is missing from the data directory.
    """
    loaded = {Path(spice.kdata(i, "ALL")[0]).name for i in range(spice.ktotal("ALL"))}
    for kernel in KERNELS:
        path = paths.DATA_DIR / kernel
        if not path.is_file():
            raise FileNotFoundError(f"SPICE kernel missing: {path}\n"
                                    f"Run scripts/fetch_kernels.sh to download the kernel set.")
        if kernel not in loaded:
            spice.furnsh(str(path))


_load_kernels()


def _gm(naif_id):
    """Return the gm_de440 GM of a NAIF body in km^3/s^2."""
    return float(spice.bodvcd(naif_id, "GM", 1)[1][0])


# ---------------------------------------------------------------- constants --
MU_NEPTUNE = _gm(899)                                       # km^3/s^2
MU_TRITON = _gm(801)                                        # km^3/s^2
GM_SUN = _gm(10)                                            # km^3/s^2
R_TRITON = float(spice.bodvcd(801, "RADII", 3)[1][0])       # km
R_NEPTUNE_EQ = float(spice.bodvcd(899, "RADII", 3)[1][0])   # km
R_NEPTUNE = 24622.0                                         # km, altitude reference
J2_NEPTUNE = 3408.428e-6
R_J2 = 25225.0                                              # km
OMEGA_NEPTUNE = math.radians(float(spice.bodvcd(899, "PM", 3)[1][1])) / 86400.0   # rad/s
A_NEPTUNE_HELIO = 4.4983e9                                  # km
R_HILL = A_NEPTUNE_HELIO * (_gm(8) / (3.0 * GM_SUN)) ** (1.0 / 3.0)   # km
G0 = 9.80665e-3                                             # km/s^2
ENTRY_ALTITUDE = 1000.0                                     # km
R_INTERFACE = R_NEPTUNE + ENTRY_ALTITUDE                    # km

DEFAULT_EPOCH = "2060-01-01 00:00:00 TDB"


# -------------------------------------------------------------------- frame --
def neq_from_j2000(et):
    """Return the rotation matrix taking J2000 vectors into NEQ at ``et``."""
    pole = np.array(spice.pxform("IAU_NEPTUNE", "J2000", et))[:, 2]
    x = np.cross([0.0, 0.0, 1.0], pole)
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(pole, x), pole])


def neq_from_eclipj2000(et):
    """Return the rotation matrix taking ECLIPJ2000 (pykep) vectors into NEQ at ``et``."""
    return neq_from_j2000(et) @ np.array(spice.pxform("ECLIPJ2000", "J2000", 0.0))


# ------------------------------------------------------------------- triton --
@dataclass(frozen=True)
class TritonOrbit:
    """Circular orbit of Triton in NEQ, complex-step safe in time.

    Attributes:
        a (float): Orbit radius, km.
        n (float): Mean motion, rad/s (positive; the sense is carried by ``q``).
        theta0 (float): Phase at t = 0, rad.
        p (tuple): In-plane unit vector, Triton's direction at the fit's mid-sample.
        q (tuple): In-plane unit vector ``h_hat x p``, 90 deg ahead along the motion.
        h_hat (tuple): Orbit normal; points near -z because Triton is retrograde.
    """
    a: float
    n: float
    theta0: float
    p: tuple
    q: tuple
    h_hat: tuple

    def _phase(self, t):
        """Return ``(cos th, sin th)`` as column arrays for time(s) ``t`` in s."""
        th = self.theta0 + self.n * np.asarray(t)
        return np.cos(th)[..., None], np.sin(th)[..., None]

    def position(self, t):
        """Return the position in km at time(s) ``t`` in s, shape ``(..., 3)``."""
        c, s = self._phase(t)
        return self.a * (c * np.asarray(self.p) + s * np.asarray(self.q))

    def velocity(self, t):
        """Return the velocity in km/s at time(s) ``t`` in s, shape ``(..., 3)``."""
        c, s = self._phase(t)
        return self.a * self.n * (-s * np.asarray(self.p) + c * np.asarray(self.q))

    @property
    def period(self):
        """Return the orbital period in s."""
        return 2.0 * math.pi / self.n


def _spice_triton(et_ref, t):
    """Return Triton's nep097 NEQ states, shape ``(len(t), 6)`` in km and km/s.

    Args:
        et_ref (float): Reference epoch, TDB seconds past J2000.
        t (np.ndarray): Times past ``et_ref``, s.

    Returns:
        np.ndarray: Rows ``[x, y, z, vx, vy, vz]`` relative to Neptune.
    """
    rot = neq_from_j2000(et_ref)
    states = np.array([spice.spkezr("TRITON", et_ref + ti, "J2000", "NONE", "NEPTUNE")[0]
                       for ti in t])
    return np.hstack([states[:, :3] @ rot.T, states[:, 3:] @ rot.T])


def fit_triton(et_ref, span_days=6.0, samples=241):
    """Fit the circular Triton model to nep097 over ``+-span_days`` about ``et_ref``.

    The normal is the mean unit angular momentum, ``a`` the mean radius, and
    ``theta0``, ``n`` a least-squares line through the unwrapped in-plane phase.

    Args:
        et_ref (float): Reference epoch, TDB seconds past J2000.
        span_days (float, optional): Half-width of the fit window. Defaults to 6.0.
        samples (int, optional): Number of SPICE samples. Defaults to 241.

    Returns:
        TritonOrbit: The fitted orbit.
    """
    t = np.linspace(-span_days, span_days, samples) * 86400.0
    states = _spice_triton(et_ref, t)
    r, v = states[:, :3], states[:, 3:]
    h = np.cross(r, v)
    h_hat = np.mean(h / np.linalg.norm(h, axis=1)[:, None], axis=0)
    h_hat /= np.linalg.norm(h_hat)
    r_mid = r[samples // 2] - h_hat * (r[samples // 2] @ h_hat)
    p = r_mid / np.linalg.norm(r_mid)
    q = np.cross(h_hat, p)
    n, theta0 = np.polyfit(t, np.unwrap(np.arctan2(r @ q, r @ p)), 1)
    return TritonOrbit(a=float(np.mean(np.linalg.norm(r, axis=1))), n=float(n),
                       theta0=float(theta0), p=tuple(p), q=tuple(q), h_hat=tuple(h_hat))


def triton_residual_km(orbit, et_ref, span_days=6.0, samples=121):
    """Return the largest distance in km between ``orbit`` and nep097 over ``+-span_days``."""
    t = np.linspace(-span_days, span_days, samples) * 86400.0
    err = orbit.position(t) - _spice_triton(et_ref, t)[:, :3]
    return float(np.max(np.linalg.norm(err, axis=1)))


@dataclass(frozen=True)
class Epoch:
    """Reference epoch and the quantities frozen at it.

    Attributes:
        et_ref (float): TDB seconds past J2000 at simulation time zero.
        label (str): The time string or ET value it was built from.
        triton (TritonOrbit): Triton orbit fitted about ``et_ref``.
        neq_from_eclip (np.ndarray): ECLIPJ2000 -> NEQ rotation matrix.
    """
    et_ref: float
    label: str
    triton: TritonOrbit
    neq_from_eclip: np.ndarray

    def iso(self, t_s):
        """Return the UTC ISO calendar string of simulation time ``t_s`` (s)."""
        return spice.et2utc(self.et_ref + float(t_s), "ISOC", 0)


def make_epoch(label=DEFAULT_EPOCH):
    """Build an ``Epoch`` from a SPICE time string or an ET value in s.

    Args:
        label (str | float, optional): SPICE-parsable time string, or TDB
            seconds past J2000. Defaults to ``DEFAULT_EPOCH``.

    Returns:
        Epoch: The epoch with its Triton fit and frame rotation.
    """
    et = float(label) if isinstance(label, (int, float)) else spice.str2et(label)
    return Epoch(et_ref=et, label=str(label), triton=fit_triton(et),
                 neq_from_eclip=neq_from_eclipj2000(et))


def mjd2000_to_et(mjd2000):
    """Return the ET of a pykep MJD2000 date, read as days past 2000-01-01 00:00 TDB."""
    return spice.str2et("2000-01-01 00:00:00 TDB") + float(mjd2000) * 86400.0
