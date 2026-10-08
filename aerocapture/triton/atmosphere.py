"""Triton's N2 atmosphere: a hydrostatic Bates model (or a user table) as a log-density spline.

Model ``bates``: pure N2 in hydrostatic equilibrium under Triton's point-mass
gravity g(r) = mu_T / r^2, with the Bates temperature profile

    T(z) = T_inf - (T_inf - T_0) exp(-z / L),
    d ln p / dz = -m_N2 g(R_T + z) / (k_B T(z)),     rho = m_N2 p / (k_B T),

integrated from the surface pressure p_0 and tabulated from 0 to
``TABLE_TOP_KM`` every ``TABLE_STEP_KM``. Defaults (``config.Config``):
p_0 = 16 ubar and T_0 = 38 K (Strobel & Zhu 2017; Benne et al. 2022;
occultations 14-16 ubar 1989-2022, Sicardy et al. 2024), T_inf = 92 K
(Triton-3 thermosphere 90-102 K), L = 161.6 km (fitted so p(1026 km) =
2.8e-9 Pa, the top of the Benne et al. 2022 model). Independent checks:
n_N2(575 km) = 4.0 +- 0.4e8 cm^-3 (Voyager UVS, Benne et al. 2024) and
p(47 km) = 1.18 +- 0.03 ubar (2017 occultation, Marques Oliveira et al. 2022).

Model ``table``: two whitespace columns, altitude (km) and density (kg/m^3).

Either way ``atm_density_scale`` multiplies the density, and the result is an
``atmosphere.Atmosphere`` (log-density Akima spline, log-linear extrapolation)
that ``atmosphere.DensityComp(atmosphere=...)`` evaluates with analytic partials.
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from aerocapture.core import atmosphere
from aerocapture.core import system

K_B = 1.380649e-23                       # J/K (SI exact)
M_N2 = 28.0134e-3 / 6.02214076e23        # kg, N2 molecular mass (IUPAC atomic weights)
TABLE_TOP_KM = 1100.0                    # km, top of the tabulated profile
TABLE_STEP_KM = 0.5                      # km, table spacing
MODELS = ("bates", "table")


def bates_profile(p_surface_ubar, t_surface_k, t_inf_k, t_scale_km, top_km=TABLE_TOP_KM,
                  step_km=TABLE_STEP_KM):
    """Return the hydrostatic Bates N2 profile on a regular altitude grid.

    Args:
        p_surface_ubar (float): Surface pressure, microbar.
        t_surface_k (float): Surface temperature T_0, K.
        t_inf_k (float): Thermospheric temperature T_inf, K.
        t_scale_km (float): Bates scale L, km.
        top_km (float, optional): Top of the grid, km. Defaults to ``TABLE_TOP_KM``.
        step_km (float, optional): Grid spacing, km. Defaults to ``TABLE_STEP_KM``.

    Returns:
        dict: ``h_km``, ``t_k``, ``p_pa``, ``n_cm3`` (number density) and
        ``rho_kg_m3`` arrays.
    """
    h = np.arange(0.0, top_km + 0.5 * step_km, step_km)
    mu, r_t = system.MU_TRITON * 1e9, system.R_TRITON * 1e3      # SI

    def temp(z_km):
        return t_inf_k - (t_inf_k - t_surface_k) * np.exp(-z_km / t_scale_km)

    def rhs(z_km, lnp):
        g = mu / (r_t + z_km * 1e3) ** 2
        return [-M_N2 * g / (K_B * temp(z_km)) * 1e3]

    sol = solve_ivp(rhs, (0.0, h[-1]), [np.log(p_surface_ubar * 0.1)], t_eval=h,
                    method="DOP853", rtol=1e-12, atol=1e-12)
    p = np.exp(sol.y[0])
    t = temp(h)
    n = p / (K_B * t)
    return dict(h_km=h, t_k=t, p_pa=p, n_cm3=n * 1e-6, rho_kg_m3=n * M_N2)


def load_table(path):
    """Return ``(h_km, rho_kg_m3)`` from a two-column table, sorted by altitude.

    Raises:
        FileNotFoundError: If the table is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Triton atmosphere table {path} not found")
    data = np.loadtxt(path, comments="#")
    h, idx = np.unique(data[:, 0], return_index=True)
    return h, data[idx, 1]


def build(model="bates", table_path=None, p_surface_ubar=16.0, t_surface_k=38.0, t_inf_k=92.0,
          t_scale_km=161.6, density_scale=1.0):
    """Return the Triton ``atmosphere.Atmosphere`` and a record of how it was built.

    Args:
        model (str, optional): ``"bates"`` or ``"table"``. Defaults to "bates".
        table_path (str, optional): Table for ``"table"``. Defaults to None.
        p_surface_ubar (float, optional): Bates surface pressure, microbar. Defaults to 16.0.
        t_surface_k (float, optional): Bates surface temperature, K. Defaults to 38.0.
        t_inf_k (float, optional): Bates thermospheric temperature, K. Defaults to 92.0.
        t_scale_km (float, optional): Bates scale L, km. Defaults to 161.6.
        density_scale (float, optional): Density multiplier. Defaults to 1.0.

    Returns:
        tuple: (atmosphere.Atmosphere, dict record with the model inputs and
        reference densities).

    Raises:
        ValueError: If ``model`` is unknown or ``table`` has no path.
    """
    if model == "bates":
        prof = bates_profile(p_surface_ubar, t_surface_k, t_inf_k, t_scale_km)
        h, rho = prof["h_km"], prof["rho_kg_m3"]
        rec = dict(model=model, p_surface_ubar=p_surface_ubar, t_surface_k=t_surface_k,
                   t_inf_k=t_inf_k, t_scale_km=t_scale_km,
                   p_1026km_pa=float(np.interp(1026.0, h, prof["p_pa"])),
                   n_575km_cm3=float(np.interp(575.0, h, prof["n_cm3"])),
                   p_47km_ubar=float(np.interp(47.0, h, prof["p_pa"])) * 10.0)
    elif model == "table":
        if table_path is None:
            raise ValueError("atm_model='table' needs atm_table_path")
        h, rho = load_table(table_path)
        rec = dict(model=model, table_path=str(table_path))
    else:
        raise ValueError(f"atm_model must be one of {MODELS}, not {model!r}")
    atm = atmosphere.Atmosphere.from_arrays(h, rho * density_scale)
    rec["density_scale"] = density_scale
    rec["rho_kg_m3_at_km"] = {f"{z:g}": float(atm.density(z)) for z in (0, 50, 100, 200, 400,
                                                                         600, 800, 1000)}
    return atm, rec
