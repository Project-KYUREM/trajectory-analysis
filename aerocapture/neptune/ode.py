"""OpenMDAO ODE systems for the Dymos phases.

Both ODEs integrate the Cartesian NEQ state ``x y z vx vy vz`` (km, km/s) so
every phase links on the same names, and both evaluate the models in
``dynamics.py``:

    VacuumODE      Neptune point mass + J2 + Triton (circular ephemeris in time).
    AtmosphereODE  alt -> rho (``atmosphere.DensityComp``) -> AeroEOM: point
                   mass + J2 + lift and drag on the atmosphere-relative
                   velocity, stagnation heating and g-load (no Triton).

``TritonPeriapsisComp`` gives the periapsis of the Triton-centred osculating
conic at one node, for the between-node flyby altitude bound.

Each ODE also outputs the osculating final-orbit quantities (``final_orbit``),
because the last phase is the atmosphere pass (``none``, ``inbound``) or the
Triton window (``outbound``):

    energy, capture   km^2/s^2     cos_inc     h_z / |h|
    h_p, h_q          h_hat . p, h_hat . q for Triton's in-plane axes (p, q)
    dv_raise          km/s         r, rdotv    km, km^2/s

VacuumODE, AeroEOM and TritonPeriapsisComp take complex-step partials (the
ODEs with coloring); ``core.ode.AltitudeComp`` and DensityComp have analytic
partials because the density spline is real-valued.
"""
import numpy as np
import openmdao.api as om

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core import ode as core_ode
from aerocapture.core import system

STATES = ("x", "y", "z", "vx", "vy", "vz")
ORBIT_OUTPUTS = (("energy", "km**2/s**2"), ("capture", "km**2/s**2"), ("cos_inc", None),
                 ("h_p", None), ("h_q", None), ("dv_raise", "km/s"), ("r", "km"),
                 ("rdotv", "km**2/s"))


def _declare_orbit_options(comp):
    """Declare the final-orbit options shared by both ODEs."""
    comp.options.declare("ra_max", types=float, default=0.1 * system.R_HILL,
                         desc="largest acceptable apoapsis radius, km")
    comp.options.declare("rp_target", types=float, default=system.R_NEPTUNE + 3000.0,
                         desc="periapsis radius after the raise burn, km")
    comp.options.declare("triton_plane", default=(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
                         desc="(p, q) in-plane unit vectors of Triton's orbit")


def _add_state_io(comp, nn):
    """Add the state inputs, their rates and the final-orbit outputs."""
    core_ode.add_state_io(comp, nn, STATES)
    core_ode.add_outputs(comp, nn, ORBIT_OUTPUTS)


def _write_rates_and_orbit(outputs, r, v, a, opts):
    """Write the state rates and the final-orbit outputs."""
    core_ode.write_rates(outputs, STATES, v, a)
    orb = dynamics.final_orbit(r, v, opts["ra_max"], opts["rp_target"])
    for name in ("energy", "capture", "cos_inc", "dv_raise"):
        outputs[name] = orb[name]
    h = np.cross(r, v)
    hm = dynamics.norm(h)
    p, q = opts["triton_plane"]
    outputs["h_p"] = h @ p / hm
    outputs["h_q"] = h @ q / hm
    outputs["r"] = dynamics.norm(r)
    outputs["rdotv"] = dynamics.dot(r, v)


class VacuumODE(om.ExplicitComponent):
    """Vacuum equations of motion with altitude and Triton-distance outputs.

    Options ``triton`` (``TritonOrbit`` or None to leave Triton out) and
    ``j2`` (bool) select the force model. Outputs ``alt`` (km), ``d_triton``
    (km) and ``d_triton_dot`` (km/s); without Triton, ``d_triton`` is 1e9 km.
    """

    def initialize(self):
        """Declare options."""
        self.options.declare("num_nodes", types=int)
        self.options.declare("triton", default=None, allow_none=True)
        self.options.declare("j2", types=bool, default=True)
        _declare_orbit_options(self)

    def setup(self):
        """Declare inputs, outputs and colored complex-step partials."""
        nn = self.options["num_nodes"]
        self.add_input("time", shape=(nn,), units="s")
        _add_state_io(self, nn)
        core_ode.add_outputs(self, nn, (("alt", "km"), ("d_triton", "km"),
                                        ("d_triton_dot", "km/s")))
        core_ode.declare_cs(self)

    def compute(self, inputs, outputs):
        """Evaluate the rates and diagnostics."""
        tr = self.options["triton"]
        r, v = core_ode.state_arrays(inputs, STATES)
        t = inputs["time"]
        j2 = system.J2_NEPTUNE if self.options["j2"] else 0.0
        _, a = dynamics.vacuum_rhs(t, r, v, tr, j2=j2)
        _write_rates_and_orbit(outputs, r, v, a, self.options)
        outputs["alt"] = outputs["r"] - system.R_NEPTUNE
        if tr is None:
            outputs["d_triton"] = 1e9 + 0.0 * t
            outputs["d_triton_dot"] = 0.0 * t
        else:
            d = r - tr.position(t)
            dm = dynamics.norm(d)
            outputs["d_triton"] = dm
            outputs["d_triton_dot"] = dynamics.dot(d, v - tr.velocity(t)) / dm


class AeroEOM(om.ExplicitComponent):
    """Atmospheric-flight equations of motion, heating and load diagnostics.

    The vehicle inputs ``beta`` (kg/m^2), ``ld`` and ``rn`` (m) are static
    Dymos parameters, so a trade changes them without rebuilding the problem.
    ``q_peak`` and ``g_peak`` are the slack variables of the min-max
    objectives: ``q_excess = q_total - q_peak``, ``g_excess = g_load - g_peak``.
    ``Q_dot`` (= ``q_total``) integrates into the heat load.

    Options ``conv_model``, ``coupled``, ``kappa`` and ``rn_exp`` select the
    heating model (``heating.py``); ``emissivity`` sets ``t_radeq``.
    """

    def initialize(self):
        """Declare options."""
        self.options.declare("num_nodes", types=int)
        self.options.declare("conv_model", default="vision_neptune")
        self.options.declare("coupled", types=bool, default=True)
        self.options.declare("kappa", types=float, default=3.0)
        self.options.declare("rn_exp", types=float, default=1.0)
        self.options.declare("emissivity", types=float, default=0.85)
        _declare_orbit_options(self)

    def setup(self):
        """Declare inputs, outputs and colored complex-step partials."""
        nn = self.options["num_nodes"]
        _add_state_io(self, nn)
        self.add_input("sigma", shape=(nn,), units="rad")
        self.add_input("rho", shape=(nn,), units="kg/m**3")
        self.add_input("beta", shape=(1,), units="kg/m**2", val=200.0)
        self.add_input("ld", shape=(1,), val=0.4)
        self.add_input("rn", shape=(1,), units="m", val=0.5)
        self.add_input("g_peak", shape=(1,), val=20.0)
        self.add_input("q_peak", shape=(1,), units="W/cm**2", val=1000.0)
        core_ode.add_outputs(self, nn, (
            ("Q_dot", "W/cm**2"), ("g_load", None), ("g_excess", None), ("q_excess", "W/cm**2"),
            ("q_dyn", "kPa"), ("q_conv", "W/cm**2"), ("q_rad", "W/cm**2"),
            ("q_rad_unc", "W/cm**2"), ("q_total", "W/cm**2"), ("q_total_unc", "W/cm**2"),
            ("t_radeq", "K"), ("v_rel", "km/s"), ("fpa_rel", "rad")))
        core_ode.declare_cs(self)

    def compute(self, inputs, outputs):
        """Evaluate the rates and diagnostics."""
        o = self.options
        r, v = core_ode.state_arrays(inputs, STATES)
        params = dict(beta=inputs["beta"][0], ld=inputs["ld"][0], rn=inputs["rn"][0],
                      kappa=o["kappa"], coupled=1.0 if o["coupled"] else 0.0,
                      rn_exp=o["rn_exp"], eps=o["emissivity"], conv_model=o["conv_model"])
        _, a, diag = dynamics.atmosphere_rhs(None, r, v, inputs["sigma"], inputs["rho"], params)
        _write_rates_and_orbit(outputs, r, v, a, o)
        for name in ("g_load", "q_dyn", "q_conv", "q_rad", "q_rad_unc", "q_total",
                     "q_total_unc", "t_radeq", "v_rel"):
            outputs[name] = diag[name]
        outputs["Q_dot"] = diag["q_total"]
        outputs["g_excess"] = diag["g_load"] - inputs["g_peak"][0]
        outputs["q_excess"] = diag["q_total"] - inputs["q_peak"][0]
        vrel = dynamics.relative_velocity(r, v)
        outputs["fpa_rel"] = np.arctan(dynamics.dot(r, vrel) / dynamics.norm(np.cross(r, vrel)))


class AtmosphereODE(om.Group):
    """Atmospheric ODE: altitude -> DensityComp -> AeroEOM, all promoted."""

    def initialize(self):
        """Declare options; ``aero_options`` are passed to AeroEOM."""
        self.options.declare("num_nodes", types=int)
        self.options.declare("aero_options", types=dict, default={})

    def setup(self):
        """Add the subsystems."""
        nn = self.options["num_nodes"]
        self.add_subsystem("alt", core_ode.AltitudeComp(num_nodes=nn, radius=system.R_NEPTUNE),
                           promotes=["*"])
        self.add_subsystem("density", atmosphere.DensityComp(num_nodes=nn), promotes=["*"])
        self.add_subsystem("eom", AeroEOM(num_nodes=nn, **self.options["aero_options"]),
                           promotes=["*"])


class TritonPeriapsisComp(om.ExplicitComponent):
    """Periapsis radius (km) of the Triton-centred osculating conic of one NEQ state."""

    def initialize(self):
        """Declare the Triton ephemeris option."""
        self.options.declare("triton", desc="system.TritonOrbit")

    def setup(self):
        """Declare the scalar state and time inputs and the periapsis output."""
        core_ode.add_scalar_inputs(self, STATES, core_ode.STATE_UNITS)
        self.add_input("time", units="s")
        self.add_output("rp", units="km")
        core_ode.declare_cs(self, coloring=False)

    def compute(self, inputs, outputs):
        """Evaluate rp = h^2 / (mu_T (1 + e)) from the state relative to Triton."""
        tr, t = self.options["triton"], inputs["time"][0]
        r = core_ode.scalar_vector(inputs, STATES[:3]) - tr.position(t)
        v = core_ode.scalar_vector(inputs, STATES[3:]) - tr.velocity(t)
        h = np.cross(r, v)
        h2 = np.sum(h * h)
        energy = 0.5 * np.sum(v * v) - system.MU_TRITON / np.sqrt(np.sum(r * r))
        e = np.sqrt(1.0 + 2.0 * energy * h2 / system.MU_TRITON ** 2)
        outputs["rp"] = h2 / (system.MU_TRITON * (1.0 + e))
