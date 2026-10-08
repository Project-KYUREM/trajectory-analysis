"""OpenMDAO ODEs of the Triton-frame phases and the boundary components that join phases.

Triton-frame phases integrate ``xt yt zt vxt vyt vzt`` (km, km/s, frame TC of
``frame``); the Neptune phases reuse ``aerocapture.neptune.ode.VacuumODE`` on the NEQ
states ``x y z vx vy vz``. The two name sets never link by accident; the frame
switch is the explicit ``FrameMatchComp``.

    TritonVacuumODE      translated NEQ gravity (Triton + Neptune point mass,
                         J2 and tide); outputs alt, r, r.v and the osculating
                         Triton orbit (energy, capture metric, rp, cos i_T,
                         periapsis-raise dv).
    TritonAtmosphereODE  AltitudeComp -> atmosphere.DensityComp (Triton
                         table) -> BalluteEOM: the same gravity plus drag on
                         the airspeed relative to the co-rotating atmosphere,
                         Kn-bridged C_D, bridged stagnation heating, the
                         Tauber-Sutton bound, film temperature, g-load.

Boundary components (scalar inputs taken from phase timeseries nodes):

    FrameMatchComp  TC state minus (NEQ state - Triton ephemeris) at one time.
    ImpulseComp     v_after - v_before - dv for an impulsive burn ("free":
                    3-vector dv; "tangential": scalar along v_before).
    RingNodeComp    inverse radii u = (1 + e_vec . N)/p of the two equatorial
                    nodes of the osculating Neptune conic (N the ascending-node
                    unit vector), ordered near/far by a fixed sign.

Every component except the altitude (``core.ode.AltitudeComp``, analytic)
takes complex-step partials; the ODEs add coloring.
"""
import numpy as np
import openmdao.api as om

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core import ode as core_ode
from aerocapture.core import system
from aerocapture.triton import ballute
from aerocapture.triton import frame as tf

STATES_T = ("xt", "yt", "zt", "vxt", "vyt", "vzt")
STATES_N = ("x", "y", "z", "vx", "vy", "vz")
ORBIT_OUTPUTS = (("energy", "km**2/s**2"), ("capture", "km**2/s**2"), ("rp", "km"),
                 ("cos_inc", None), ("dv_raise", "km/s"), ("r", "km"), ("rdotv", "km**2/s"),
                 ("alt", "km"))


def _declare_common(comp):
    """Declare the options shared by the Triton-frame ODE components."""
    comp.options.declare("num_nodes", types=int)
    comp.options.declare("triton", desc="system.TritonOrbit")
    comp.options.declare("j2", types=bool, default=True)
    comp.options.declare("ra_max", types=float, default=4379.0,
                         desc="largest acceptable Triton apoapsis radius, km")
    comp.options.declare("rp_target", types=float, default=system.R_TRITON + 1000.0,
                         desc="periapsis radius after the raise burn, km")


def _add_state_io(comp, nn, with_alt_output=True):
    """Add time, the TC state inputs, their rates and the orbit outputs."""
    comp.add_input("time", shape=(nn,), units="s")
    core_ode.add_state_io(comp, nn, STATES_T)
    core_ode.add_outputs(comp, nn, [o for o in ORBIT_OUTPUTS if o[0] != "alt" or with_alt_output])


def state_arrays(inputs, names=STATES_T):
    """Return the ``(nn, 3)`` position and velocity arrays of the named state inputs."""
    return core_ode.state_arrays(inputs, names)


def _write(outputs, r, v, a, opts, with_alt=True):
    """Write the rates and the osculating Triton-orbit outputs."""
    core_ode.write_rates(outputs, STATES_T, v, a)
    orb = tf.orbit(r, v, opts["triton"], opts["ra_max"], opts["rp_target"])
    for name in ("energy", "capture", "rp", "cos_inc", "dv_raise"):
        outputs[name] = orb[name]
    rm = dynamics.norm(r)
    outputs["r"] = rm
    outputs["rdotv"] = dynamics.dot(r, v)
    if with_alt:
        outputs["alt"] = rm - system.R_TRITON


class TritonVacuumODE(om.ExplicitComponent):
    """Vacuum flight in the Triton frame (``frame.accel_translated``; ``tide=False``: Kepler)."""

    def initialize(self):
        """Declare options."""
        _declare_common(self)
        self.options.declare("tide", types=bool, default=True)

    def setup(self):
        """Declare inputs, outputs and colored complex-step partials."""
        _add_state_io(self, self.options["num_nodes"])
        core_ode.declare_cs(self)

    def compute(self, inputs, outputs):
        """Evaluate the rates and orbit outputs."""
        o = self.options
        r, v = state_arrays(inputs)
        if o["tide"]:
            a = tf.accel_translated(inputs["time"], r, o["triton"],
                                    j2=system.J2_NEPTUNE if o["j2"] else 0.0)
        else:
            a = -tf.MU_T * r / dynamics.norm(r)[:, None] ** 3 + 0.0 * inputs["time"][:, None]
        _write(outputs, r, v, a, o)


class BalluteEOM(om.ExplicitComponent):
    """Drag-only atmospheric flight of a ballute or probe in the Triton frame.

    Static Dymos parameters (changed in place by trades): ``mass`` (kg),
    ``area`` (drag reference area, m^2), ``length`` (Kn reference, m), ``rn``
    (stagnation radius, m), ``cd_c`` and ``cd_fm`` (bridge end values; equal
    values give a constant C_D), and the min-max slacks ``q_peak`` (W/cm^2)
    and ``g_peak``. ``q_heat`` (bridged convective, plus the Tauber-Sutton
    bound when ``rad_in_limits``) drives the film temperature, the heat-rate
    limit, the slack ``q_excess = q_heat - q_peak`` and ``Q_dot``.
    """

    def initialize(self):
        """Declare options."""
        _declare_common(self)
        self.options.declare("rotating", types=bool, default=True)
        self.options.declare("conv_model", default="sutton_graves_n2")
        self.options.declare("bridging", default="smooth_min", values=("smooth_min", "continuum"))
        self.options.declare("fm_accommodation", types=float, default=1.0)
        self.options.declare("rad_in_limits", types=bool, default=False)
        self.options.declare("emissivity", types=float, default=0.85)
        self.options.declare("sides", types=int, default=2)

    def setup(self):
        """Declare inputs, outputs and colored complex-step partials."""
        nn = self.options["num_nodes"]
        _add_state_io(self, nn, with_alt_output=False)
        self.add_input("rho", shape=(nn,), units="kg/m**3")
        for name, units, val in (("mass", "kg", 530.0), ("area", "m**2", 3848.0),
                                 ("length", "m", 70.0), ("rn", "m", 35.0), ("cd_c", None, 0.9),
                                 ("cd_fm", None, 2.0), ("q_peak", "W/cm**2", 10.0),
                                 ("g_peak", None, 25.0)):
            self.add_input(name, shape=(1,), units=units, val=val)
        core_ode.add_outputs(self, nn, (
            ("Q_dot", "W/cm**2"), ("g_load", None), ("g_excess", None), ("q_excess", "W/cm**2"),
            ("q_dyn", "Pa"), ("kn", None), ("cd", None), ("beta", "kg/m**2"),
            ("q_conv", "W/cm**2"), ("q_fm", "W/cm**2"), ("q_bridged", "W/cm**2"),
            ("q_rad", "W/cm**2"), ("q_heat", "W/cm**2"), ("t_film", "K"), ("v_rel", "km/s")))
        core_ode.declare_cs(self)

    def compute(self, inputs, outputs):
        """Evaluate the rates, heating and load diagnostics."""
        o = self.options
        r, v = state_arrays(inputs)
        rho = inputs["rho"]
        p = {n: inputs[n][0] for n in ("mass", "area", "length", "rn", "cd_c", "cd_fm",
                                       "q_peak", "g_peak")}
        a = tf.accel_translated(inputs["time"], r, o["triton"],
                                j2=system.J2_NEPTUNE if o["j2"] else 0.0)
        diag = aero(r, v, rho, p, o["triton"], o["rotating"], o["conv_model"], o["bridging"],
                    o["fm_accommodation"], o["rad_in_limits"], o["emissivity"], o["sides"])
        a = a - diag.pop("drag")[:, None] * diag.pop("u")
        _write(outputs, r, v, a, o, with_alt=False)
        for name, val in diag.items():
            outputs[name] = val
        outputs["Q_dot"] = diag["q_heat"]
        outputs["q_excess"] = diag["q_heat"] - p["q_peak"]
        outputs["g_excess"] = diag["g_load"] - p["g_peak"]


def aero(r, v, rho, p, triton, rotating, conv_model, bridging, accommodation, rad_in_limits,
         emissivity, sides):
    """Return the drag and heating diagnostics of states in the Triton atmosphere.

    Args:
        r (np.ndarray): TC positions, km, ``(n, 3)``.
        v (np.ndarray): TC velocities, km/s, ``(n, 3)``.
        rho (np.ndarray): Density, kg/m^3.
        p (dict): ``mass``, ``area``, ``length``, ``rn``, ``cd_c``, ``cd_fm``.
        triton (system.TritonOrbit): Ephemeris (spin axis and rate).
        rotating (bool): Co-rotating atmosphere.
        conv_model (str): ``heating.CONVECTIVE_SETS`` key of the continuum heating.
        bridging (str): ``"smooth_min"`` or ``"continuum"``.
        accommodation (float): Free-molecular energy accommodation.
        rad_in_limits (bool): Add the Tauber-Sutton bound to ``q_heat``.
        emissivity (float): Film emissivity.
        sides (int): Radiating faces.

    Returns:
        dict: ``drag`` (km/s^2), ``u`` (airspeed unit vectors) and the
        ``BalluteEOM`` diagnostic outputs.
    """
    vrel = dynamics.relative_velocity(r, v, tf.omega_vec(triton, rotating))
    vm = dynamics.norm(vrel)
    v_ms = vm * 1e3
    kn = ballute.knudsen(rho, p["length"])
    cd = ballute.drag_coefficient(kn, p["cd_c"], p["cd_fm"])
    beta = p["mass"] / (cd * p["area"])
    drag = 0.5 * rho * v_ms ** 2 / beta * 1e-3
    q_c = ballute.q_continuum(rho, v_ms, p["rn"], conv_model)
    q_fm = ballute.q_free_molecular(rho, v_ms, accommodation)
    q_b = ballute.q_bridged(q_c, q_fm) if bridging == "smooth_min" else q_c
    q_r = ballute.q_tauber_sutton(rho, v_ms, p["rn"])
    q_heat = q_b + (q_r if rad_in_limits else 0.0 * q_r)
    return dict(drag=drag, u=vrel / vm[:, None], g_load=drag / system.G0,
                q_dyn=0.5 * rho * v_ms ** 2, kn=kn, cd=cd, beta=beta, q_conv=q_c, q_fm=q_fm,
                q_bridged=q_b, q_rad=q_r, q_heat=q_heat,
                t_film=ballute.film_temperature(q_heat, emissivity, sides), v_rel=vm)


class TritonAtmosphereODE(om.Group):
    """Triton atmospheric ODE: altitude -> density -> BalluteEOM, all promoted."""

    def initialize(self):
        """Declare options; ``eom_options`` go to BalluteEOM."""
        self.options.declare("num_nodes", types=int)
        self.options.declare("atmosphere", desc="atmosphere.Atmosphere of Triton")
        self.options.declare("eom_options", types=dict, default={})

    def setup(self):
        """Add the subsystems."""
        nn = self.options["num_nodes"]
        self.add_subsystem("alt", core_ode.AltitudeComp(num_nodes=nn, radius=system.R_TRITON,
                                                        positions=STATES_T[:3]), promotes=["*"])
        self.add_subsystem("density", atmosphere.DensityComp(
            num_nodes=nn, atmosphere=self.options["atmosphere"]), promotes=["*"])
        self.add_subsystem("eom", BalluteEOM(num_nodes=nn, **self.options["eom_options"]),
                           promotes=["*"])


# ----------------------------------------------------------- boundary components
class FrameMatchComp(om.ExplicitComponent):
    """Residuals ``TC state - (NEQ state - Triton state)`` at one time (6 outputs, km and km/s)."""

    def initialize(self):
        """Declare the ephemeris option."""
        self.options.declare("triton", desc="system.TritonOrbit")

    def setup(self):
        """Declare the NEQ and TC state inputs, the time and the residuals."""
        units = core_ode.STATE_UNITS
        core_ode.add_scalar_inputs(self, STATES_N, units)
        core_ode.add_scalar_inputs(self, STATES_T, units)
        self.add_input("time", units="s")
        for n, u in zip(("dx", "dy", "dz", "dvx", "dvy", "dvz"), units):
            self.add_output(n, units=u)
        core_ode.declare_cs(self, coloring=False)

    def compute(self, inputs, outputs):
        """Evaluate the residuals."""
        tr, t = self.options["triton"], inputs["time"][0]
        rn = core_ode.scalar_vector(inputs, STATES_N)
        rt = core_ode.scalar_vector(inputs, STATES_T)
        ref = np.concatenate([tr.position(t), tr.velocity(t)])
        for i, n in enumerate(("dx", "dy", "dz", "dvx", "dvy", "dvz")):
            outputs[n] = rt[i] - (rn[i] - ref[i])


class ImpulseComp(om.ExplicitComponent):
    """Velocity-jump residuals ``v_after - v_before - dv`` and the burn magnitude (km/s)."""

    def initialize(self):
        """Declare the burn mode."""
        self.options.declare("mode", default="free", values=("free", "tangential"))

    def setup(self):
        """Declare the velocity inputs, the burn and the residuals."""
        for side in ("b", "a"):
            core_ode.add_scalar_inputs(self, [f"v{c}_{side}" for c in "xyz"], ("km/s",) * 3)
        n_dv = 3 if self.options["mode"] == "free" else 1
        self.add_input("dv", shape=(n_dv,), units="km/s")
        for c in "xyz":
            self.add_output(f"res_{c}", units="km/s")
        self.add_output("dv_mag", units="km/s")
        core_ode.declare_cs(self, coloring=False)

    def compute(self, inputs, outputs):
        """Evaluate the residuals and |dv|."""
        vb = core_ode.scalar_vector(inputs, [f"v{c}_b" for c in "xyz"])
        va = core_ode.scalar_vector(inputs, [f"v{c}_a" for c in "xyz"])
        dv = inputs["dv"]
        if self.options["mode"] == "tangential":
            dv = dv[0] * vb / np.sqrt(vb @ vb)
        for i, c in enumerate("xyz"):
            outputs[f"res_{c}"] = va[i] - vb[i] - dv[i]
        outputs["dv_mag"] = np.sqrt(dv @ dv + 1e-12)


class RingNodeComp(om.ExplicitComponent):
    """Inverse radii (1/km) of the near and far equatorial nodes of the osculating Neptune conic.

    Option ``sign`` (+1 or -1) is the sign of e_vec . N at the guess, so
    ``u_near = (1 + sign e_vec.N)/p`` is the node nearer periapsis. A node is
    clear of the rings when u >= 1/r_in or u <= 1/r_out; u < 0 means the node
    lies beyond the asymptote and is never reached.
    """

    def initialize(self):
        """Declare the node-ordering sign and the central GM."""
        self.options.declare("sign", types=float, default=1.0)
        self.options.declare("mu", types=float, default=system.MU_NEPTUNE)

    def setup(self):
        """Declare the state inputs and node outputs."""
        core_ode.add_scalar_inputs(self, STATES_N, core_ode.STATE_UNITS)
        self.add_output("u_near", units="1/km")
        self.add_output("u_far", units="1/km")
        core_ode.declare_cs(self, coloring=False)

    def compute(self, inputs, outputs):
        """Evaluate the node inverse radii."""
        r = core_ode.scalar_vector(inputs, STATES_N[:3])
        v = core_ode.scalar_vector(inputs, STATES_N[3:])
        u_asc, u_desc = node_inverse_radii(r, v, self.options["mu"])
        s = self.options["sign"]
        outputs["u_near"] = u_asc if s > 0 else u_desc
        outputs["u_far"] = u_desc if s > 0 else u_asc


def node_inverse_radii(r, v, mu=system.MU_NEPTUNE):
    """Return ``(u_ascending, u_descending)`` = (1 +- e_vec . N)/p of the osculating conic, 1/km."""
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / mu - r / np.sqrt(r @ r)
    n = np.array([-h[1], h[0], 0.0 * h[0]])            # z x h
    n = n / np.sqrt(n @ n)
    p = (h @ h) / mu
    en = e_vec @ n
    return (1.0 + en) / p, (1.0 - en) / p
