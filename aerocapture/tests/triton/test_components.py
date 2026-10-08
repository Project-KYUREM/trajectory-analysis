"""check_partials on every Triton-mission component, and a Triton-frame Dymos coast against Kepler."""
import math

import dymos as dm
import numpy as np
import openmdao.api as om
import pytest

from aerocapture.core import guess as cg
from aerocapture.core import system
from aerocapture.tests.helpers import component_problem
from aerocapture.tests.helpers import worst_rel_error
from aerocapture.triton import atmosphere
from aerocapture.triton import ode


def _set_tc(p, nn, alt_lo, alt_hi, speed, seed):
    """Set random Triton-frame states at altitudes in [alt_lo, alt_hi] km."""
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(nn, 3))
    r *= (system.R_TRITON + rng.uniform(alt_lo, alt_hi, nn))[:, None] / np.linalg.norm(
        r, axis=1)[:, None]
    v = rng.normal(size=(nn, 3)) * speed
    for i, n in enumerate(ode.STATES_T[:3]):
        p[n] = r[:, i]
    for i, n in enumerate(ode.STATES_T[3:]):
        p[n] = v[:, i]
    p["time"] = rng.uniform(0, 2e5, nn)


def test_triton_vacuum_ode_partials(epoch):
    """TritonVacuumODE complex-step partials match finite differences."""
    p = component_problem(ode.TritonVacuumODE(num_nodes=4, triton=epoch.triton))
    _set_tc(p, 4, 500, 3000, 1.0, 0)
    p.run_model()
    assert worst_rel_error(p) < 1e-4


@pytest.mark.parametrize("opts", [{}, dict(rad_in_limits=True, bridging="continuum",
                                           rotating=False)])
def test_triton_atmosphere_ode_partials(epoch, opts):
    """TritonAtmosphereODE (altitude, density, BalluteEOM) partials match finite differences."""
    atm, _ = atmosphere.build()
    p = component_problem(ode.TritonAtmosphereODE(
        num_nodes=4, atmosphere=atm, eom_options=dict(triton=epoch.triton, **opts)))
    _set_tc(p, 4, 40, 400, 5.0, 1)
    p.run_model()
    assert worst_rel_error(p) < 1e-3


def test_frame_match_partials(epoch):
    """FrameMatchComp partials match finite differences."""
    p = component_problem(ode.FrameMatchComp(triton=epoch.triton))
    rng = np.random.default_rng(2)
    for n in ode.STATES_N[:3]:
        p[n] = rng.normal(3e5, 1e4)
    for n in ode.STATES_N[3:] + ode.STATES_T[3:]:
        p[n] = rng.normal(0, 3)
    for n in ode.STATES_T[:3]:
        p[n] = rng.normal(0, 2e4)
    p["time"] = 9.3e4
    p.run_model()
    assert worst_rel_error(p) < 1e-5


@pytest.mark.parametrize("mode", ["free", "tangential"])
def test_impulse_partials(mode):
    """ImpulseComp partials match finite differences in both burn modes."""
    p = component_problem(ode.ImpulseComp(mode=mode))
    rng = np.random.default_rng(3)
    for side in "ba":
        for c in "xyz":
            p[f"v{c}_{side}"] = rng.normal(0, 1)
    p["dv"] = rng.normal(0, 0.1, 3 if mode == "free" else 1)
    p.run_model()
    assert worst_rel_error(p) < 1e-5


def test_ring_node_partials():
    """RingNodeComp partials match finite differences."""
    p = component_problem(ode.RingNodeComp(sign=1.0))
    vals = dict(x=20000.0, y=15000.0, z=3000.0, vx=-10.0, vy=14.0, vz=4.0)
    for k, v in vals.items():
        p[k] = v
    p.run_model()
    assert worst_rel_error(p) < 1e-5


def test_dymos_triton_coast_matches_kepler(epoch):
    """A tide-free Dymos coast about Triton reproduces the analytic Kepler orbit within 0.1 km."""
    mu = system.MU_TRITON
    a, e = 3300.0, 0.3
    n = math.sqrt(mu / a ** 3)
    t_end = 0.6 * 2 * math.pi / n

    def kepler(t):
        """Return the Kepler state at time t."""
        big_e = n * t
        for _ in range(50):
            big_e -= (big_e - e * math.sin(big_e) - n * t) / (1 - e * math.cos(big_e))
        edot = n / (1 - e * math.cos(big_e))
        b = a * math.sqrt(1 - e ** 2)
        return [a * (math.cos(big_e) - e), b * math.sin(big_e), 0.0,
                -a * math.sin(big_e) * edot, b * math.cos(big_e) * edot, 0.0]

    tt = np.linspace(0, t_end, 400)
    ends = cg.segment_ends(cg.Arc(tt, np.array([kepler(t) for t in tt])), 30,
                           lambda arc: np.linalg.norm(arc.y[:, :3], axis=1) ** -1.5)
    p = om.Problem(reports=False)
    ph = dm.Phase(ode_class=ode.TritonVacuumODE,
                  ode_init_kwargs=dict(triton=epoch.triton, tide=False),
                  transcription=dm.Radau(num_segments=30, order=3, segment_ends=ends,
                                         solve_segments="forward"))
    ph.set_time_options(fix_initial=True, fix_duration=True, units="s", targets=["time"])
    for nme in ode.STATES_T[:3]:
        ph.add_state(nme, rate_source=f"{nme}_dot", units="km", fix_initial=True)
    for nme in ode.STATES_T[3:]:
        ph.add_state(nme, rate_source=f"{nme}_dot", units="km/s", fix_initial=True)
    p.model.add_subsystem("ph", ph)
    p.setup(force_alloc_complex=True)
    ph.set_time_val(0.0, t_end)
    tg = np.linspace(0, t_end, 9)
    seed = np.array([kepler(t) for t in tg])
    for k, nme in enumerate(ode.STATES_T):
        ph.set_state_val(nme, seed[:, k], time_vals=tg)
    p.run_model()
    x_ref, y_ref = kepler(t_end)[:2]
    x = p.get_val("ph.timeseries.xt")[-1, 0]
    y = p.get_val("ph.timeseries.yt")[-1, 0]
    assert math.hypot(x - x_ref, y - y_ref) < 0.1
