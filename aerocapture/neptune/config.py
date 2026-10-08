"""Every tunable input of one Neptune aerocapture solve, in one dataclass.

``guess``, ``problem`` and ``case`` read their inputs only from a ``Config``.
Physical constants from SPICE or the literature (GM, J2, rotation rate,
Triton's orbit) live in ``core/system.py``. Units are in the field names (km,
s, deg, km/s, W/cm^2, K). The field groups follow the CONFIG cell of
``run_neptune.py``, which spells out every field with its meaning and tag
([physics], [numerics], [guess], [margin]); every [margin] field is listed in
``ASSUMPTIONS`` with its unit and the reason for its value. Validation, the
record and the echo come from ``core.config_base.ConfigBase``.
"""
from dataclasses import dataclass, field
from typing import Optional

from aerocapture.core import heating
from aerocapture.core import system
from aerocapture.core.config_base import Assumption, ConfigBase
from aerocapture.neptune import vehicles

TRITON_MODES = ("none", "inbound", "outbound")
DIRECTIONS = ("free", "radec", "from_run")
OBJECTIVES = ("qpeak", "heat", "dv", "g")

BOTH_MESSAGE = (
    "triton_mode='both' (an inbound and an outbound Triton flyby on the same pass) is "
    "geometrically impossible: between the inbound and outbound crossings of Triton's orbit "
    "the spacecraft sweeps ~260-325 deg around Neptune while Triton advances only ~40-100 deg "
    "in the same sense, a mismatch of >= ~205 deg for v_inf 3-20 km/s and any exit apoapsis. "
    "Tilting the trajectory plane cannot help: both encounter points would have to lie on the "
    "line of nodes with Triton's orbit, i.e. 0 or 180 deg apart. Use 'inbound' or 'outbound'.")


def default_segments():
    """Return the default number of Radau segments per phase."""
    return dict(approach=20, triton_in=20, coast_in=25, atmosphere=40, coast_mid=20,
                triton_out=24)


@dataclass
class Config(ConfigBase):
    """Every input of one Neptune aerocapture solve; ``run_neptune.py`` documents each field."""
    # arrival [physics]
    vinf_kms: float = 5.0
    direction: str = "free"
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    from_run: str = "capture/EJN"
    point: str = "p13116"
    epoch: str = system.DEFAULT_EPOCH
    opt_direction: Optional[bool] = None
    opt_t_peri: Optional[bool] = None
    t_peri_window_periods: float = 0.5
    r0_km: float = 1.0e6
    # Triton [physics; triton_window_km is a phase boundary, numerics]
    triton_mode: str = "inbound"
    triton_window_km: float = 30000.0
    # target orbit [physics]
    ra_target_km: Optional[float] = None
    ra_mode: str = "max"
    match_triton_plane: bool = False
    # vehicle [physics]
    vehicle: vehicles.Vehicle = field(default_factory=lambda: vehicles.PRESETS[vehicles.DEFAULT])
    # atmosphere and heating [physics]
    interface_alt_km: float = system.ENTRY_ALTITUDE
    conv_model: str = "vision_neptune"
    # objective and control limits [physics]
    objective: str = "qpeak"
    bank_min_deg: float = -180.0
    bank_max_deg: float = 180.0
    bank_rate_max_degs: float = 15.0
    # margins and assumptions [margin] (one ASSUMPTIONS entry each)
    triton_alt_min_km: float = 300.0
    triton_clear_km: float = 15000.0
    triton_exit_km: float = 100000.0
    ra_target_hill: float = 0.1
    min_inclination_deg: float = 90.0
    rp_raise_km: float = system.R_NEPTUNE + 3000.0
    rad_coupling: str = "goulard"
    rad_kappa: float = 3.0
    rad_rn_exp: float = 1.0
    heat_weight: float = 0.0
    sim_tol: float = 0.01
    # transcription and solver [numerics]
    order: int = 3
    segments: dict = field(default_factory=default_segments)
    r_split_km: float = 2.0e5
    max_iter: int = 500
    tol: float = 1e-5
    constr_viol_tol: float = 1e-6
    print_level: int = 5
    linear_solver: str = "mumps"
    # initial guess [guess]
    guess_inclination_deg: Optional[float] = 92.0
    guess_turn_fraction: float = 0.7
    guess_banks_deg: tuple = (0.0, 45.0, 75.0, 90.0, 105.0, 135.0)
    guess_capture_fraction: float = 1.0
    # output
    tag: Optional[str] = None

    CHOICES = dict(triton_mode=TRITON_MODES, direction=DIRECTIONS, objective=OBJECTIVES,
                   ra_mode=("max", "equal"), rad_coupling=("goulard", "none"),
                   conv_model=tuple(heating.CONVECTIVE_SETS),
                   **{"vehicle.tps": ("ablative", "non_ablative")})

    ASSUMPTIONS = (
        Assumption("triton_alt_min_km", "km", "margin",
                   "Triton flyby floor above the surface; Triton's atmosphere is not modelled "
                   "(rho ~3e-9 kg/m^3 at 300 km, drag dv < 1 m/s); binds at the inbound optimum, "
                   "so a lower floor buys a little more Triton dv"),
        Assumption("triton_clear_km", "km", "margin",
                   "keeps the non-Triton vacuum phases beyond Triton's SOI (11 970 km) so the "
                   "Triton phase is the only close pass; normally inactive"),
        Assumption("triton_exit_km", "km", "model",
                   "outbound reads the final orbit this far from Triton; Triton's residual pull "
                   "biases the Neptune-frame apoapsis ~1.5 % high (ra_mode='equal' overshoots "
                   "by that much); larger shrinks the bias, lengthens the phase"),
        Assumption("ra_target_hill", "R_Hill", "model",
                   "two-body capture target when ra_target_km is None (0.1 -> 1.16e7 km); the "
                   "unmodelled solar tide is ~0.1 % of Neptune's pull there (~8 % at 0.5); the "
                   "exit is then ~25 m/s below escape"),
        Assumption("min_inclination_deg", "deg", "requirement",
                   "final orbit polar or retrograde; the free optimum sits on this bound "
                   "(a polar orbit minimises airspeed), so it is normally active"),
        Assumption("rp_raise_km", "km", "requirement",
                   "periapsis-raise target (R_N + 3000 km) of the reported two-body raise dv "
                   "and of objective 'dv'; not a constraint"),
        Assumption("rad_coupling", "", "model",
                   "Goulard radiative cooling applied to the Vision Neptune radiative fit "
                   "(whether the fit already includes it is unconfirmed); 'none' is "
                   "conservative; both are always reported, limits use this one"),
        Assumption("rad_kappa", "", "model",
                   "Goulard kappa, the air value (no H2-He value is established); coupling cuts "
                   "q_total ~1 % at R_n 0.3 m, 22-67 % at R_n 1-3 m and 30-33 km/s"),
        Assumption("rad_rn_exp", "", "model",
                   "nose-radius exponent of the radiative fit (optically thin, anchored at "
                   "0.291 m): q_rad grows linearly with R_n, an upper bound for large noses"),
        Assumption("heat_weight", "per J/cm^2", "weight",
                   "heat-load term added to the objective; 0 leaves the objective pure"),
        Assumption("sim_tol", "", "check",
                   "largest relative final-apoapsis error of the simulate re-integration "
                   "before the run is flagged"),
    )

    # ------------------------------------------------------------ derived values
    @property
    def ra_target(self):
        """Return the apoapsis target, km (``ra_target_km`` if set, else the Hill fraction)."""
        if self.ra_target_km is not None:
            return self.ra_target_km
        return self.ra_target_hill * system.R_HILL

    @property
    def r_interface(self):
        """Return the entry/exit interface radius, km."""
        return system.R_NEPTUNE + self.interface_alt_km

    @property
    def coupled(self):
        """Return True when the Goulard radiative coupling is selected."""
        return self.rad_coupling == "goulard"

    def derived(self):
        """Return the resolved apoapsis target and the interface radius, km."""
        return dict(ra_target_resolved_km=self.ra_target, r_interface_km=self.r_interface)

    def auto_opt_flags(self):
        """Return the automatic ``(opt_direction, opt_t_peri)``.

        With Triton in play, a free direction optimises the v-inf RA/Dec and a
        fixed direction optimises the periapsis time (Triton's phase). In
        ``none`` mode both stay at the guess: rotations about the pole and
        shifts along the orbit are near-null directions that stall IPOPT.

        Returns:
            tuple: (bool, bool).
        """
        triton = self.triton_mode != "none"
        free = self.direction == "free"
        return free and triton, (not free) and triton

    def vehicle_params(self):
        """Return the vehicle and heating parameters the dynamics functions take."""
        v = self.vehicle
        return dict(beta=v.beta_kg_m2, ld=v.lift_to_drag, rn=v.nose_radius_m,
                    kappa=self.rad_kappa, coupled=1.0 if self.coupled else 0.0,
                    rn_exp=self.rad_rn_exp, eps=v.emissivity, conv_model=self.conv_model)

    # ------------------------------------------------------------ checks
    def warnings(self):
        """Return the input sanity warnings (printed with the echo and saved)."""
        out = []
        frac = self.ra_target / system.R_HILL
        if frac > 0.1 + 1e-9:
            out.append(f"target apoapsis {self.ra_target:.4g} km is {frac:.2f} R_Hill: solar tides "
                       "(not modelled) are ~0.1% of Neptune's pull at 0.1 R_Hill and ~8% at "
                       "0.5 R_Hill")
        rn = self.vehicle.nose_radius_m
        if not heating.RN_REF / 3 <= rn <= 3 * heating.RN_REF:
            out.append(f"nose radius {rn} m is far from the {heating.RN_REF} m anchor of "
                       "the radiative fit; treat q_rad as an upper bound")
        if self.triton_mode == "outbound" and self.ra_mode == "equal":
            out.append(f"outbound reads the final orbit {self.triton_exit_km:.0f} km from Triton, "
                       "where Triton's residual pull biases the Neptune-frame apoapsis ~1.5% "
                       "high; ra_mode='equal' overshoots the target by about that much "
                       "(raise triton_exit_km to shrink it)")
        if (self.guess_inclination_deg is not None and not self.match_triton_plane
                and self.guess_inclination_deg < self.min_inclination_deg):
            out.append(f"guess_inclination_deg {self.guess_inclination_deg} is below "
                       f"min_inclination_deg {self.min_inclination_deg}")
        return out

    def validate(self):
        """Check the categorical inputs (``CHOICES``) and the target apoapsis.

        Raises:
            ValueError: If ``triton_mode`` is ``"both"``, an option string is
                unknown, or the target apoapsis is below Neptune's surface.
        """
        if self.triton_mode == "both":
            raise ValueError(BOTH_MESSAGE)
        super().validate()

    def check(self):
        """Check that the target apoapsis lies above Neptune's surface.

        Raises:
            ValueError: If the target apoapsis is below Neptune's surface.
        """
        if self.ra_target <= system.R_NEPTUNE:
            raise ValueError("target apoapsis below Neptune's surface")
