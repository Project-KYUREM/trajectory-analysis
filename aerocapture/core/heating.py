"""Stagnation-point heating correlations for Neptune entry.

Inputs are free-stream density rho (kg/m^3), atmosphere-relative speed v (m/s)
and effective nose radius rn (m); heat rates are W/cm^2:

    convective   q_conv  = C rho^a v^b (RN_REF / rn)^0.5
    radiative    q_rad   = C rho^c (v / 1e4)^d (rn / RN_REF)^e
    coupling     q_rad,c = q_rad / (1 + kappa Gamma^0.7)
                 Gamma   = 2 q_rad / (rho v^3 / 2)     (Goulard number, q_rad in W/m^2)
    wall         T_eq    = (q / (n eps sigma_SB))^(1/4)  (q in W/m^2, n radiating sides)

Coefficients:

    vision_neptune  (C, a, b) = (2.24008e-7, 0.452130, 2.691800), RN_REF = 0.291 m;
                    radiative (C, c, d) = (8.125812e-3, 0.498140, 15.113).
                    NASA Vision Neptune study (2005, NNH04CC41C) fits as
                    implemented in AMAT ``vehicle.py``.
    sutton_graves   q = k sqrt(rho / rn) v^3, k = 6.556e-5 (SI) for H2-He,
                    AMAT's H2-He (Jupiter) value, rewritten against RN_REF.
    sutton_graves_n2  the same form with k = 1.7407e-4 (SI), AMAT's N2
                    (Titan) value, for nitrogen atmospheres.
    kappa = 3       Tauber & Wakefield (1971) value for air; no H2-He value is
                    established, so it is a parameter.
    e = 1           AMAT's nose-radius exponent; exposed because the
                    optically-thin radiative fit overstates heating far from
                    rn = 0.291 m.

Convective heating is cold-wall and non-ablating. All functions are
complex-step safe.
"""
SIGMA_SB = 5.670374419e-8      # W/m^2/K^4 (CODATA 2018)
RN_REF = 0.291                 # m, anchor radius of the Vision Neptune fits

# name -> (C, a, b) for q_conv = C rho^a v^b (RN_REF / rn)^0.5, W/cm^2
CONVECTIVE_SETS = {
    "vision_neptune": (2.24008e-7, 0.452130, 2.691800),
    "sutton_graves": (6.556e-9 / RN_REF ** 0.5, 0.5, 3.0),
    "sutton_graves_n2": (1.7407e-8 / RN_REF ** 0.5, 0.5, 3.0),
}

RADIATIVE_FIT = (8.125812e-3, 0.498140, 15.113)   # (C, c, d), W/cm^2


def q_convective(rho, v, rn, model="vision_neptune"):
    """Return the stagnation convective heat rate in W/cm^2 from ``CONVECTIVE_SETS[model]``."""
    c, a, b = CONVECTIVE_SETS[model]
    return c * rho ** a * v ** b * (RN_REF / rn) ** 0.5


def q_radiative_uncoupled(rho, v, rn, rn_exponent=1.0):
    """Return the stagnation radiative heat rate in W/cm^2, without radiative cooling."""
    c, a, d = RADIATIVE_FIT
    return c * rho ** a * (v / 1.0e4) ** d * (rn / RN_REF) ** rn_exponent


def goulard_number(q_rad, rho, v):
    """Return Gamma = 2 q_rad / (rho v^3 / 2) for ``q_rad`` in W/cm^2 and SI ``rho``, ``v``."""
    return 2.0 * q_rad * 1.0e4 / (0.5 * rho * v ** 3)


def q_radiative_coupled(q_rad_unc, rho, v, kappa=3.0):
    """Return the radiatively cooled heat rate ``q_rad_unc / (1 + kappa Gamma^0.7)``, W/cm^2."""
    return q_rad_unc / (1.0 + kappa * goulard_number(q_rad_unc, rho, v) ** 0.7)


def t_radiative_equilibrium(q, emissivity, sides=1):
    """Return the wall temperature in K at which ``sides`` reradiating faces balance ``q`` (W/cm^2)."""
    return (q * 1.0e4 / (sides * emissivity * SIGMA_SB)) ** 0.25
