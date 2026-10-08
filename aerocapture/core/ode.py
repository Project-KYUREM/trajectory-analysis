"""Shared building blocks of the OpenMDAO ODE and boundary components.

Every mission integrates six Cartesian states, three positions in km and three
velocities in km/s, under its own names (``x y z vx vy vz`` in NEQ,
``xt yt zt vxt vyt vzt`` in the Triton frame). The helpers here declare those
inputs and their rates, pack them into ``(nn, 3)`` arrays and write the rates
back, so each ODE component only states its physics. ``AltitudeComp`` is the
altitude above a reference sphere with analytic partials, which the
real-valued density spline downstream needs.
"""
import numpy as np
import openmdao.api as om

STATE_UNITS = ("km",) * 3 + ("km/s",) * 3


def add_state_io(comp, nn, names):
    """Add the six state inputs and their ``<name>_dot`` rate outputs to a component.

    Args:
        comp (om.ExplicitComponent): Component being set up.
        nn (int): Number of nodes.
        names (tuple): The six state names, positions first.
    """
    for name in names[:3]:
        comp.add_input(name, shape=(nn,), units="km")
        comp.add_output(f"{name}_dot", shape=(nn,), units="km/s")
    for name in names[3:]:
        comp.add_input(name, shape=(nn,), units="km/s")
        comp.add_output(f"{name}_dot", shape=(nn,), units="km/s**2")


def add_outputs(comp, nn, outputs):
    """Add one ``(nn,)`` output per ``(name, units)`` pair."""
    for name, units in outputs:
        comp.add_output(name, shape=(nn,), units=units)


def state_arrays(inputs, names):
    """Return the ``(nn, 3)`` position and velocity arrays of the named state inputs."""
    r = np.stack([inputs[n] for n in names[:3]], axis=-1)
    v = np.stack([inputs[n] for n in names[3:]], axis=-1)
    return r, v


def write_rates(outputs, names, v, a):
    """Write the state rates: position rates from ``v``, velocity rates from ``a``."""
    for i, name in enumerate(names[:3]):
        outputs[f"{name}_dot"] = v[:, i]
    for i, name in enumerate(names[3:]):
        outputs[f"{name}_dot"] = a[:, i]


def declare_cs(comp, coloring=True):
    """Declare complex-step partials of every output, colored when ``coloring`` is True."""
    comp.declare_partials("*", "*", method="cs")
    if coloring:
        comp.declare_coloring(wrt="*", method="cs", show_summary=False)


def add_scalar_inputs(comp, names, units):
    """Add one scalar input per name with the matching units."""
    for n, u in zip(names, units):
        comp.add_input(n, units=u)


def scalar_vector(inputs, names):
    """Return the scalar inputs ``names`` as one 1-D array (complex during complex step)."""
    return np.array([inputs[n][0] for n in names])


class AltitudeComp(om.ExplicitComponent):
    """Altitude ``|r| - radius`` in km of the named position inputs, with analytic partials."""

    def initialize(self):
        """Declare the node count, reference radius and position names."""
        self.options.declare("num_nodes", types=int)
        self.options.declare("radius", types=float, desc="reference radius, km")
        self.options.declare("positions", default=("x", "y", "z"), desc="position input names")

    def setup(self):
        """Declare inputs, output and diagonal partials."""
        nn = self.options["num_nodes"]
        ar = np.arange(nn)
        self.add_output("alt", shape=(nn,), units="km")
        for name in self.options["positions"]:
            self.add_input(name, shape=(nn,), units="km")
            self.declare_partials("alt", name, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        """Evaluate the altitude."""
        rm = np.sqrt(sum(inputs[n] ** 2 for n in self.options["positions"]))
        outputs["alt"] = rm - self.options["radius"]

    def compute_partials(self, inputs, partials):
        """Evaluate d alt / d r = r_hat."""
        rm = np.sqrt(sum(inputs[n] ** 2 for n in self.options["positions"]))
        for name in self.options["positions"]:
            partials["alt", name] = inputs[name] / rm
