"""Ballute and probe records, rarefied drag and stagnation heating in N2.

Units: density rho kg/m^3, airspeed v m/s, lengths m, heat rates W/cm^2.

Knudsen number on the ballute diameter D, hard-sphere N2 mean free path:

    lambda = m_N2 / (sqrt(2) pi d^2 rho),   d = 3.7e-10 m,   Kn = lambda / D

Drag coefficient, Wilmoth sine-squared bridging (Wilmoth, Mitcheltree & Moss
1999) between the continuum and free-molecular values, constant outside
1e-3 < Kn < 10 (the bridge has zero slope at both ends, so C_D is C1):

    x = clip(3/8 + log10(Kn)/8, 0, 1/2),   C_D = C_c + (C_fm - C_c) sin^2(pi x)

Heating at the stagnation point:

    continuum      q_c  = k sqrt(rho / R_n) v^3, Sutton-Graves with the N2
                   constant k = 1.7407e-4 SI (``heating.CONVECTIVE_SETS
                   ["sutton_graves_n2"]``), R_n = D/2 for the envelope
    free molecular q_fm = alpha rho v^3 / 2 (accommodation alpha)
    bridged        q    = (q_c^-2 + q_fm^-2)^(-1/2)   (``smooth_min``: tends to
                   the smaller of the two in either limit)
    radiative      q_r  = C R_n rho^1.22 f(V), C = 4.736e4 (Tauber & Sutton
                   1991, air, optically thin). The radius exponent is held at
                   its cap of 1 (linear in R_n), so for a tens-of-metres
                   envelope this is an upper bound; f(V) is their tabulated air
                   function, log-interpolated by a monotone cubic and extended
                   log-linearly outside 9-16 km/s.

Film temperature, radiative equilibrium from ``sides`` faces (an enveloping
film radiates from both): T = (q / (sides eps sigma))^(1/4)
(``heating.t_radiative_equilibrium``).

Every function is complex-step safe: interval selection and clamps use the
real part only.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from aerocapture.core import heating
from aerocapture.triton.atmosphere import M_N2

D_N2 = 3.7e-10                   # m, N2 hard-sphere diameter (kinetic theory, rounded)
KN_BRIDGE = (1e-3, 10.0)         # Kn range of the Wilmoth bridge
TS_C = 4.736e4                   # Tauber-Sutton constant (W/cm^2, SI rho, R_n in m)
TS_RHO_EXP = 1.22                # Tauber-Sutton density exponent
# Tauber & Sutton (1991) f(V) for air: V in m/s, f dimensionless.
TS_TABLE = (
    (9000, 1.5), (9250, 4.3), (9500, 9.7), (9750, 19.5), (10000, 35.0), (10250, 55.0),
    (10500, 81.0), (10750, 115.0), (11000, 151.0), (11500, 238.0), (12000, 359.0),
    (12500, 495.0), (13000, 660.0), (13500, 850.0), (14000, 1065.0), (14500, 1313.0),
    (15000, 1550.0), (15500, 1780.0), (16000, 2040.0))
CD_MODELS = ("wilmoth", "constant")


@dataclass(frozen=True)
class Ballute:
    """An enveloping ballute (inflated sphere around the probe).

    Attributes:
        name (str): Label used in result paths and tables.
        diameter_m (float): Envelope diameter, m; sets the drag area, the Kn
            length and the nose radius D/2.
        envelope_mass_kg (float): Envelope plus inflation-system mass, kg.
        cd_continuum (float): Continuum drag coefficient on the frontal area.
        cd_free_molecular (float): Free-molecular drag coefficient.
        cd_model (str): ``"wilmoth"`` (Kn bridge) or ``"constant"`` (``cd_continuum``).
        emissivity (float): Film emissivity.
        radiating_sides (int): Faces that reradiate the stagnation heat (2 for a film).
        t_film_max_k (float): Film radiative-equilibrium temperature limit, K.
    """
    name: str = "ead-70"
    diameter_m: float = 70.0
    envelope_mass_kg: float = 330.0
    cd_continuum: float = 0.9
    cd_free_molecular: float = 2.0
    cd_model: str = "wilmoth"
    emissivity: float = 0.85
    radiating_sides: int = 2
    t_film_max_k: float = 773.0

    @property
    def area_m2(self):
        """Return the frontal area pi D^2 / 4, m^2."""
        return math.pi * self.diameter_m ** 2 / 4.0

    @property
    def nose_radius_m(self):
        """Return the stagnation-point radius D / 2, m."""
        return 0.5 * self.diameter_m

    @property
    def cd_fm_effective(self):
        """Return the free-molecular C_D the bridge uses (``cd_continuum`` when constant)."""
        return self.cd_continuum if self.cd_model == "constant" else self.cd_free_molecular


@dataclass(frozen=True)
class Probe:
    """The probe/orbiter released from the ballute.

    Attributes:
        mass_kg (float): Mass after release, kg.
        cd_area_m2 (float): Drag area C_D A, m^2.
        nose_radius_m (float): Stagnation-point nose radius, m.
    """
    mass_kg: float = 200.0
    cd_area_m2: float = 2.0
    nose_radius_m: float = 0.5

    @property
    def beta_kg_m2(self):
        """Return the ballistic coefficient m / (C_D A), kg/m^2."""
        return self.mass_kg / self.cd_area_m2


def _real(x):
    """Return the real part of a possibly complex array."""
    return np.real(x)


def mean_free_path(rho):
    """Return the hard-sphere N2 mean free path in m at density ``rho`` (kg/m^3)."""
    return M_N2 / (math.sqrt(2.0) * math.pi * D_N2 ** 2 * rho)


def knudsen(rho, length_m):
    """Return the Knudsen number on ``length_m`` at density ``rho``."""
    return mean_free_path(rho) / length_m


def drag_coefficient(kn, cd_continuum, cd_free_molecular):
    """Return the Wilmoth-bridged drag coefficient at Knudsen number ``kn``."""
    x = 0.375 + np.log10(kn) / 8.0
    xr = _real(x)
    x = np.where(xr < 0.0, 0.0 * x, np.where(xr > 0.5, 0.0 * x + 0.5, x))
    return cd_continuum + (cd_free_molecular - cd_continuum) * np.sin(math.pi * x) ** 2


def q_continuum(rho, v, rn, model="sutton_graves_n2"):
    """Return the continuum stagnation heat rate, W/cm^2 (``heating.q_convective``)."""
    return heating.q_convective(rho, v, rn, model)


def q_free_molecular(rho, v, accommodation=1.0):
    """Return the free-molecular stagnation heat rate alpha rho v^3 / 2, W/cm^2."""
    return accommodation * 0.5 * rho * v ** 3 * 1e-4


def q_bridged(q_c, q_fm):
    """Return the smooth minimum (q_c^-2 + q_fm^-2)^(-1/2) of two heat rates."""
    return (q_c ** -2 + q_fm ** -2) ** -0.5


_V = np.array([v for v, _ in TS_TABLE], dtype=float)
_LNF = np.log([f for _, f in TS_TABLE])
_PCHIP = PchipInterpolator(_V, _LNF)
_COEF = _PCHIP.c                                  # (4, n-1), local power basis
_SLOPE_LO, _SLOPE_HI = float(_PCHIP(_V[0], 1)), float(_PCHIP(_V[-1], 1))


def tauber_sutton_lnf(v):
    """Return ln f(V) of the Tauber-Sutton air function at airspeed(s) ``v`` (m/s)."""
    v = np.asarray(v)
    vr = _real(v)
    i = np.clip(np.searchsorted(_V, vr) - 1, 0, len(_V) - 2)
    dx = v - _V[i]
    inner = ((_COEF[0, i] * dx + _COEF[1, i]) * dx + _COEF[2, i]) * dx + _COEF[3, i]
    lo = _LNF[0] + _SLOPE_LO * (v - _V[0])
    hi = _LNF[-1] + _SLOPE_HI * (v - _V[-1])
    return np.where(vr < _V[0], lo, np.where(vr > _V[-1], hi, inner))


def q_tauber_sutton(rho, v, rn):
    """Return the Tauber-Sutton radiative upper bound C R_n rho^1.22 f(V), W/cm^2."""
    return TS_C * rn * rho ** TS_RHO_EXP * np.exp(tauber_sutton_lnf(v))


def film_temperature(q, emissivity, sides):
    """Return the film radiative-equilibrium temperature in K for heat rate ``q`` (W/cm^2)."""
    return heating.t_radiative_equilibrium(q, emissivity, sides)
