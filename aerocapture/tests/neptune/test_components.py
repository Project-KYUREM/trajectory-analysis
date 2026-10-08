"""check_partials on the Neptune-mission ODEs, and a Dymos coast against Kepler."""
import math

import dymos as dm
import numpy as np
import openmdao.api as om
import pytest

from aerocapture.core import atmosphere
from aerocapture.core import guess
from aerocapture.core import ode as core_ode
from aerocapture.core import system
from aerocapture.neptune import ode
from aerocapture.tests.helpers import component_problem
from aerocapture.tests.helpers import worst_rel_error


def _set_states(p, r, v):
    """Set the six state inputs from (n, 3) arrays."""
    for i, n in enumerate("xyz"):
        p[n] = r[:, i]
    for i, n in enumerate(("vx", "vy", "vz")):
        p[n] = v[:, i]


def test_vacuum_ode_partials(epoch):
    """VacuumODE complex-step partials match finite differences."""
    nn = 4
    rng = np.random.default_rng(0)
    p = component_problem(ode.VacuumODE(num_nodes=nn, triton=epoch.triton))
    _set_states(p, rng.normal(3e5, 5e4, (nn, 3)), rng.normal(2, 1, (nn, 3)))
    p["time"] = rng.normal(0, 1e4, nn)
    p.run_model()
    assert worst_rel_error(p) < 1e-4


@pytest.mark.parametrize("aero_options", [{}, dict(coupled=False, conv_model="sutton_graves")])
def test_atmosphere_ode_partials(aero_options):
    """AtmosphereODE partials (analytic and complex step) match finite differences."""
    nn = 4
    rng = np.random.default_rng(1)
    p = component_problem(ode.AtmosphereODE(num_nodes=nn, aero_options=aero_options))
    r = rng.normal(size=(nn, 3))
    r *= (system.R_NEPTUNE + 300.0) / np.linalg.norm(r, axis=1)[:, None]
    _set_states(p, r, rng.normal(0, 14, (nn, 3)))
    p["sigma"] = rng.normal(0, 1, nn)
    p.run_model()
    assert worst_rel_error(p) < 1e-3     # fd error on rho**0.45 dominates


def test_density_and_altitude_partials():
    """AltitudeComp and DensityComp partials hold inside and outside the table."""
    nn = 5
    p = om.Problem(reports=False)
    p.model.add_subsystem("alt", core_ode.AltitudeComp(num_nodes=nn, radius=system.R_NEPTUNE),
                          promotes=["*"])
    p.model.add_subsystem("rho", atmosphere.DensityComp(num_nodes=nn), promotes=["*"])
    p.setup(force_alloc_complex=True)
    rhat = np.random.default_rng(2).normal(size=(nn, 3))
    rhat /= np.linalg.norm(rhat, axis=1)[:, None]
    r = rhat * (system.R_NEPTUNE + np.array([-200.0, 0.0, 400.0, 2500.0, 5000.0]))[:, None]
    for i, n in enumerate("xyz"):
        p[n] = r[:, i]
    p.run_model()
    assert worst_rel_error(p) < 1e-5


def test_dymos_coast_matches_kepler():
    """A J2-off, Triton-off Dymos coast reproduces the analytic Kepler orbit."""
    mu = system.MU_NEPTUNE
    a, e = 200000.0, 0.8
    n = math.sqrt(mu / a ** 3)
    t_end = 0.3 * 2 * math.pi / n

    def kepler(t):
        """Return the Kepler state at time t."""
        m = n * t
        big_e = m
        for _ in range(50):
            big_e -= (big_e - e * math.sin(big_e) - m) / (1 - e * math.cos(big_e))
        edot = n / (1 - e * math.cos(big_e))
        b = a * math.sqrt(1 - e ** 2)
        return [a * (math.cos(big_e) - e), b * math.sin(big_e), 0.0,
                -a * math.sin(big_e) * edot, b * math.cos(big_e) * edot, 0.0]

    tt = np.linspace(0, t_end, 400)
    ends = guess.segment_ends(guess.Arc(tt, np.array([kepler(t) for t in tt])), 40)
    p = om.Problem(reports=False)
    ph = dm.Phase(ode_class=ode.VacuumODE, ode_init_kwargs=dict(triton=None, j2=False),
                  transcription=dm.Radau(num_segments=40, order=3, segment_ends=ends,
                                         solve_segments="forward"))
    ph.set_time_options(fix_initial=True, fix_duration=True, units="s", targets=["time"])
    for nme in "xyz":
        ph.add_state(nme, rate_source=f"{nme}_dot", units="km", fix_initial=True)
    for nme in ("vx", "vy", "vz"):
        ph.add_state(nme, rate_source=f"{nme}_dot", units="km/s", fix_initial=True)
    ph.add_timeseries_output("energy")
    p.model.add_subsystem("ph", ph)
    p.setup(force_alloc_complex=True)
    ph.set_time_val(0.0, t_end)
    tg = np.linspace(0, t_end, 7)          # coarse seed; Newton converges the collocation
    seed = np.array([kepler(t) for t in tg])
    for k, nme in enumerate(("x", "y", "z", "vx", "vy", "vz")):
        ph.set_state_val(nme, seed[:, k], time_vals=tg)
    p.run_model()
    x_ref, y_ref = kepler(t_end)[:2]
    x = p.get_val("ph.timeseries.x")[-1, 0]
    y = p.get_val("ph.timeseries.y")[-1, 0]
    assert math.hypot(x - x_ref, y - y_ref) < 1.0       # km, over ~170 000 km of arc
    energy = p.get_val("ph.timeseries.energy").ravel()
    assert abs(energy[-1] - energy[0]) / abs(energy[0]) < 1e-5
    assert np.ptp(energy) / abs(energy.mean()) < 1e-3     # interior collocation nodes


def test_triton_periapsis_partials(epoch):
    """TritonPeriapsisComp partials match finite differences near a Triton flyby."""
    tr = epoch.triton
    t = 5.0e4
    p = component_problem(ode.TritonPeriapsisComp(triton=tr))
    r = tr.position(t) + np.array([2000.0, 1500.0, -500.0])
    v = tr.velocity(t) + np.array([-3.0, 6.0, 1.0])
    for n, val in zip(ode.STATES, np.r_[r, v]):
        p[n] = val
    p["time"] = t
    p.run_model()
    assert worst_rel_error(p) < 1e-5
