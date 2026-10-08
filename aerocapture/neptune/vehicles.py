"""Entry vehicles: the few numbers the trajectory, heating and limits see.

The equations of motion see the vehicle only through the ballistic coefficient
``beta = m / (C_D S)``, the lift-to-drag ratio and the effective nose radius;
those are what a trade varies. The TPS type, emissivity, ``t_wall_max_k``,
``q_max_w_cm2`` and ``g_max`` set the path limits; mass is report-only.

The presets hold PLACEHOLDER numbers, not sourced designs. Tune one with
``dataclasses.replace(PRESETS["loftid"], beta_kg_m2=60.0)``.
"""
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class Vehicle:
    """One entry vehicle.

    Attributes:
        name (str): Label used in result paths and tables.
        beta_kg_m2 (float): Ballistic coefficient m / (C_D S), kg/m^2.
        lift_to_drag (float): Trim L/D magnitude; the bank angle points it.
        nose_radius_m (float): Effective stagnation-point nose radius, m.
        mass_kg (float, optional): Entry mass, kg; report-only.
        tps (str): ``"ablative"`` or ``"non_ablative"``; only the latter
            enforces ``t_wall_max_k``.
        emissivity (float): Surface emissivity of the radiative-equilibrium
            temperature.
        t_wall_max_k (float, optional): Radiative-equilibrium temperature
            limit, K (non-ablative TPS only; None: no limit).
        q_max_w_cm2 (float, optional): Peak total heat-rate limit, W/cm^2
            (None: no limit).
        g_max (float): Peak sensed-deceleration limit, Earth g.
    """
    name: str
    beta_kg_m2: float
    lift_to_drag: float
    nose_radius_m: float
    mass_kg: Optional[float] = None
    tps: str = "ablative"
    emissivity: float = 0.85
    t_wall_max_k: Optional[float] = None
    q_max_w_cm2: Optional[float] = None
    g_max: float = 20.0


PRESETS = {
    # mid-L/D slender body in the spirit of the 2004-2006 NASA Neptune studies
    "ellipsled": Vehicle("ellipsled", beta_kg_m2=400.0, lift_to_drag=0.8,
                         nose_radius_m=0.5, mass_kg=2200.0),
    # rigid 70 deg sphere-cone with a CG offset (heritage blunt body)
    "heritage-70deg": Vehicle("heritage-70deg", beta_kg_m2=200.0, lift_to_drag=0.4,
                              nose_radius_m=0.5, mass_kg=2000.0),
    # LOFTID-style inflatable: low beta, large nose, modest L/D, flexible TPS
    "loftid": Vehicle("loftid", beta_kg_m2=40.0, lift_to_drag=0.2,
                      nose_radius_m=1.5, mass_kg=1500.0, tps="non_ablative"),
}

DEFAULT = "heritage-70deg"


def get(name, **overrides):
    """Return preset ``name`` with every non-None keyword override applied.

    Args:
        name (str): Key of ``PRESETS``.
        **overrides: ``Vehicle`` fields to replace; None keeps the preset value.

    Returns:
        Vehicle: The preset with the overrides applied.

    Raises:
        ValueError: If ``name`` is not a preset.
    """
    if name not in PRESETS:
        raise ValueError(f"unknown vehicle {name!r}; presets: {sorted(PRESETS)}")
    return replace(PRESETS[name], **{k: v for k, v in overrides.items() if v is not None})
