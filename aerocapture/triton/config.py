"""Every tunable input of one Triton ballute-capture solve, in one dataclass.

``guess``, ``problem`` and ``case`` read their inputs only from a ``Config``.
Physical constants (GM, radii, J2, Triton's orbit) live in ``core/system.py``;
model constants (N2 molecular data, the Tauber-Sutton table, table spacing)
are cited in ``atmosphere`` and ``ballute``. Units are in the field names. The
field groups follow the CONFIG cell of ``run_triton.py``, which spells out
every field with its meaning and tag ([physics], [numerics], [guess],
[margin]); every [margin] field is listed in ``ASSUMPTIONS`` with its unit and
the reason for its value. Validation, the record and the echo come from
``core.config_base.ConfigBase``.
"""
from dataclasses import dataclass, field
from typing import Optional

from aerocapture.core import heating
from aerocapture.core import system
from aerocapture.core.config_base import Assumption, ConfigBase
from aerocapture.triton import atmosphere
from aerocapture.triton import ballute as ballute_mod
from aerocapture.triton import frame

DIRECTIONS = ("free", "radec", "from_run")
OBJECTIVES = ("qpeak", "heat", "dv", "g")
SCOPES = ("full", "triton_only")
SENSES = ("retrograde", "prograde", "any")
ENCOUNTERS = ("co", "counter")
KOZAI_LOW_DEG, KOZAI_HIGH_DEG = 39.2, 140.8     # Kozai-Lidov unstable band of i_T (Kozai 1962)
HILL_STABLE = {"prograde": 0.49, "retrograde": 0.93}   # Domingos et al. 2006, circular orbits


def default_segments():
    """Return the default number of Radau segments per phase."""
    return dict(approach=20, flyby_in=15, flyby_out=15, cruise=20, triton_approach=15,
                ballute=40, probe=25, capture_coast=15, final=40)


@dataclass
class Config(ConfigBase):
    """Every input of one Triton capture solve; ``run_triton.py`` documents each field."""
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
    # Neptune flyby [physics]
    rp_min_rn: float = 1.10
    rp_max_rn: float = 1.25
    ring_keepout_km: tuple = (38000.0, 85000.0)
    neptune_dv_max_kms: float = 0.0
    # Triton approach [physics; frame_switch_km is a frame boundary, numerics]
    frame_switch_km: float = 30000.0
    interface_alt_km: float = 1000.0
    tcm_max_kms: float = 0.0
    # Triton atmosphere [physics]
    atm_model: str = "bates"
    atm_table_path: Optional[str] = None
    p_surface_ubar: float = 16.0
    t_surface_k: float = 38.0
    t_inf_k: float = 92.0
    t_scale_km: float = 161.6
    atm_rotation: bool = True
    # ballute and probe [physics]
    ballute: ballute_mod.Ballute = field(default_factory=ballute_mod.Ballute)
    probe: ballute_mod.Probe = field(default_factory=ballute_mod.Probe)
    # release and limits [physics]
    release_mode: str = "free"
    g_max: float = 25.0
    q_max_w_cm2: Optional[float] = None
    # heating [physics]
    conv_model: str = "sutton_graves_n2"
    bridging: str = "smooth_min"
    rad_model: str = "tauber_sutton_air"
    # final Triton orbit [physics]
    ra_mode: str = "max"
    final_sense: str = "retrograde"
    burn_mode: str = "free"
    burn_dv_max_kms: float = 0.3
    final_coast_h: float = 24.0
    # objective [physics]
    objective: str = "qpeak"
    # margins and assumptions [margin] (one ASSUMPTIONS entry each)
    h_min_km: float = 10.0
    rp_final_alt_km: float = 1000.0
    final_rp_margin_km: float = 50.0
    ra_max_hill: float = 0.3
    final_ra_margin: float = 0.05
    i_retro_min_deg: float = 141.0
    i_pro_max_deg: float = 39.0
    atm_density_scale: float = 1.0
    fm_accommodation: float = 1.0
    rad_in_limits: bool = False
    heat_weight: float = 0.0
    dv_weight: float = 1.0e-3
    stability_days: float = 120.0
    sim_tol: float = 0.01
    sim_rp_tol_km: float = 5.0
    # run [physics]
    scope: str = "full"
    # transcription and solver [numerics]
    order: int = 3
    segments: dict = field(default_factory=default_segments)
    r_split_km: float = 2.0e5
    tcm_homotopy_kms: float = 0.2
    max_iter: int = 500
    tol: float = 1e-5
    constr_viol_tol: float = 1e-6
    print_level: int = 5
    linear_solver: str = "mumps"
    # initial guess [guess]
    guess_corridor_fraction: float = 0.5
    guess_plane: str = "triton"
    guess_encounter: str = "co"
    guess_ra_fraction: float = 0.92
    guess_rp_margin_km: float = 50.0
    # output
    tag: Optional[str] = None

    CHOICES = dict(direction=DIRECTIONS, objective=OBJECTIVES, scope=SCOPES,
                   final_sense=SENSES, ra_mode=("max", "equal"),
                   release_mode=("free", "none"), burn_mode=("free", "tangential"),
                   bridging=("smooth_min", "continuum"), rad_model=("tauber_sutton_air",),
                   atm_model=atmosphere.MODELS, guess_plane=("triton",),
                   guess_encounter=ENCOUNTERS, conv_model=tuple(heating.CONVECTIVE_SETS),
                   **{"ballute.cd_model": ballute_mod.CD_MODELS})

    ASSUMPTIONS = (
        Assumption("h_min_km", "km", "margin",
                   "altitude floor through the ballute and probe passes (Triton's relief is "
                   "~1 km); active only on steep passes"),
        Assumption("rp_final_alt_km", "km", "margin",
                   "final periapsis floor 50-150 km above Triton's exobase (850-950 km) so drag "
                   "does not decay the orbit; the stability check enforces it"),
        Assumption("final_rp_margin_km", "km", "margin",
                   "the optimiser holds rp >= floor + this: Neptune's tide moves the osculating "
                   "rp ~-10 km over 120 d; costs a slightly larger raise burn"),
        Assumption("ra_max_hill", "R_H", "margin",
                   "final apoapsis limit (0.3 -> 4379 km), well inside the circular-orbit "
                   "stability limits 0.93 R_H retrograde / 0.49 R_H prograde"),
        Assumption("final_ra_margin", "", "margin",
                   "the optimiser holds ra <= (1 - this) * limit: the tide adds ~+60 km of "
                   "distance over 120 d; costs a slightly deeper, hotter capture"),
        Assumption("i_retro_min_deg", "deg", "margin",
                   "retrograde floor 0.2 deg outside the Kozai band (140.8 deg); circular-orbit "
                   "theory, not sufficient at e >~ 0.25"),
        Assumption("i_pro_max_deg", "deg", "margin",
                   "prograde ceiling 0.2 deg inside the Kozai band's lower edge (39.2 deg)"),
        Assumption("atm_density_scale", "", "model",
                   "multiplier on the Bates N2 density (1 = nominal; dispersion cases use "
                   "0.5-1.5); the corridor moves with it"),
        Assumption("fm_accommodation", "", "model",
                   "free-molecular energy accommodation; 1 absorbs all incident energy "
                   "(conservative)"),
        Assumption("rad_in_limits", "", "model",
                   "the Tauber-Sutton air radiation is an optically thin upper bound far "
                   "outside its fit range (LAURA found N2 radiation minimal): False reports it "
                   "without constraining it"),
        Assumption("heat_weight", "per J/cm^2", "weight",
                   "heat-load term added to the objective; 0 leaves the objective pure"),
        Assumption("dv_weight", "per m/s", "weight",
                   "burn term that pins the burn and capture-orbit plane (0 lets IPOPT drift "
                   "to a plane-change burn onto the Kozai edge); moves peak q < 2 %"),
        Assumption("stability_days", "d", "check",
                   "horizon of the post-solve tidal propagation; spans the 60-180 d Kozai "
                   "growth time"),
        Assumption("sim_tol", "", "check",
                   "largest relative final-apoapsis error of the simulate re-integration "
                   "before the run is flagged"),
        Assumption("sim_rp_tol_km", "km", "check",
                   "largest final-periapsis error of the simulate re-integration before the "
                   "run is flagged"),
    )

    # ------------------------------------------------------------ derived values
    @property
    def r_interface(self):
        """Return the Triton entry/exit interface radius, km."""
        return system.R_TRITON + self.interface_alt_km

    @property
    def rp_final(self):
        """Return the smallest final periapsis radius about Triton, km."""
        return system.R_TRITON + self.rp_final_alt_km

    @property
    def rp_band_km(self):
        """Return the Neptune periapsis radius band (min, max), km."""
        return self.rp_min_rn * system.R_NEPTUNE, self.rp_max_rn * system.R_NEPTUNE

    def ra_max_km(self, triton):
        """Return the largest final apoapsis radius about Triton, km (``ra_max_hill`` R_H)."""
        return self.ra_max_hill * frame.hill_radius(triton)

    def ra_design_km(self, triton):
        """Return the apoapsis bound the optimiser applies to the final orbit, km."""
        return (1.0 - self.final_ra_margin) * self.ra_max_km(triton)

    @property
    def rp_design(self):
        """Return the periapsis bound the optimiser applies to the final orbit, km."""
        return self.rp_final + self.final_rp_margin_km

    def inclination_bounds_deg(self):
        """Return the (lower, upper) bound on i_T in deg for ``final_sense``."""
        return {"retrograde": (self.i_retro_min_deg, 180.0), "prograde": (0.0, self.i_pro_max_deg),
                "any": (0.0, 180.0)}[self.final_sense]

    def derived(self):
        """Return the derived radii, bands and ballute numbers recorded with the inputs."""
        b, p = self.ballute, self.probe
        total = b.envelope_mass_kg + p.mass_kg
        return dict(r_interface_km=self.r_interface, rp_final_km=self.rp_final,
                    rp_design_km=self.rp_design, rp_band_km=self.rp_band_km,
                    i_t_band_deg=self.inclination_bounds_deg(),
                    ballute_derived=dict(area_m2=b.area_m2, nose_radius_m=b.nose_radius_m,
                                         total_mass_kg=total,
                                         beta_continuum_kg_m2=total / (b.cd_continuum * b.area_m2),
                                         probe_beta_kg_m2=p.beta_kg_m2))

    def auto_opt_flags(self):
        """Return the automatic ``(opt_direction, opt_t_peri)``.

        A free direction optimises the v-inf RA/Dec and holds the periapsis
        time; a fixed direction optimises the periapsis time (Triton's phase).

        Returns:
            tuple: (bool, bool).
        """
        free = self.direction == "free"
        return free, not free

    def pass_params(self, phase):
        """Return the BalluteEOM parameters of the ``ballute`` or ``probe`` phase.

        Args:
            phase (str): ``"ballute"`` (envelope + probe) or ``"probe"``.

        Returns:
            dict: ``mass`` (kg), ``area`` (m^2), ``length`` and ``rn`` (m),
            ``cd_c``, ``cd_fm``.
        """
        b, p = self.ballute, self.probe
        if phase == "ballute":
            return dict(mass=b.envelope_mass_kg + p.mass_kg, area=b.area_m2, length=b.diameter_m,
                        rn=b.nose_radius_m, cd_c=b.cd_continuum, cd_fm=b.cd_fm_effective)
        return dict(mass=p.mass_kg, area=p.cd_area_m2, length=2.0 * p.nose_radius_m,
                    rn=p.nose_radius_m, cd_c=1.0, cd_fm=1.0)

    def eom_options(self, phase):
        """Return the BalluteEOM options (heating models, film data) of a pass phase."""
        b = self.ballute
        return dict(rotating=self.atm_rotation, conv_model=self.conv_model,
                    bridging=self.bridging, fm_accommodation=float(self.fm_accommodation),
                    rad_in_limits=bool(self.rad_in_limits) and phase == "ballute",
                    emissivity=float(b.emissivity), sides=int(b.radiating_sides))

    def make_atmosphere(self):
        """Return ``(atmosphere.Atmosphere, record)`` of Triton's atmosphere."""
        return atmosphere.build(self.atm_model, self.atm_table_path, self.p_surface_ubar,
                                self.t_surface_k, self.t_inf_k, self.t_scale_km,
                                self.atm_density_scale)

    # ------------------------------------------------------------ checks
    def warnings(self):
        """Return the input sanity warnings (printed with the echo and saved)."""
        out = []
        stable = HILL_STABLE.get(self.final_sense)
        if stable is not None and self.ra_max_hill > stable:
            out.append(f"ra_max_hill {self.ra_max_hill} exceeds the {self.final_sense} stability "
                       f"limit ~{stable} R_H for circular orbits (Domingos et al. 2006)")
        lo, hi = self.inclination_bounds_deg()
        if lo < KOZAI_HIGH_DEG and hi > KOZAI_LOW_DEG:
            out.append(f"final i_T band [{lo:g}, {hi:g}] deg overlaps the Kozai-Lidov unstable "
                       f"band {KOZAI_LOW_DEG}-{KOZAI_HIGH_DEG} deg (e grows in ~60-180 d)")
        if self.rp_final_alt_km < 950.0:
            out.append(f"rp_final_alt_km {self.rp_final_alt_km} is below Triton's exobase "
                       "(~850-950 km); the final orbit decays")
        if self.release_mode == "none":
            out.append("release_mode='none': a retained ballute has a ~0.1 deg entry corridor")
        if self.rad_in_limits:
            out.append("rad_in_limits: the Tauber-Sutton air bound (linear in R_n) is "
                       "constrained; NIAC LAURA solutions found N2 radiation minimal")
        if self.tcm_max_kms > 0.0:
            out.append(f"tcm_max_kms {self.tcm_max_kms}: a post-flyby TCM is allowed in the "
                       "final solution")
        if self.neptune_dv_max_kms > 0.0:
            out.append(f"neptune_dv_max_kms {self.neptune_dv_max_kms}: a Neptune periapsis "
                       "burn is allowed")
        if self.atm_density_scale != 1.0:
            out.append(f"atmosphere density scaled by {self.atm_density_scale}")
        if self.direction != "free":
            out.append("fixed v-inf direction: the plane rotates about the asymptote to reach "
                       "Triton; the ring constraint may be infeasible")
        rn = self.ballute.nose_radius_m
        if rn > 1.0:
            out.append(f"Tauber-Sutton radiative heating at R_n = {rn:g} m is far outside its "
                       "0.1-3 m fit range; it is reported as an upper bound only")
        return out

    def check(self):
        """Check the numeric orderings and ranges.

        Raises:
            ValueError: If a band is empty, a margin or fraction is out of range,
                the frame switch lies inside the interface, or objective ``dv``
                has no burn to minimise.
        """
        if not 1.0 < self.rp_min_rn < self.rp_max_rn:
            raise ValueError("need 1 < rp_min_rn < rp_max_rn")
        if not self.ring_keepout_km[0] < self.ring_keepout_km[1]:
            raise ValueError("ring_keepout_km must be (inner, outer) with inner < outer")
        if not 0.0 <= self.final_ra_margin < 1.0 or self.final_rp_margin_km < 0.0:
            raise ValueError("final margins must be non-negative (ra margin below 1)")
        if not 0.0 < self.guess_corridor_fraction < 1.0:
            raise ValueError("guess_corridor_fraction must lie in (0, 1)")
        if self.frame_switch_km <= self.r_interface:
            raise ValueError("frame_switch_km must exceed the interface radius")
        if self.objective == "dv" and self.burn_dv_max_kms <= 0.0:
            raise ValueError("objective 'dv' needs burn_dv_max_kms > 0")
