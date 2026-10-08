"""Shared solve and guess helpers: active box-bound detection, segment clustering."""
import numpy as np
import openmdao.api as om

from aerocapture.core import guess as cg
from aerocapture.core import solve


def _box_problem(b_t, b_r, t_peri):
    """Return a set-up problem with arrival-style design variables at the given values."""
    p = om.Problem(reports=False)
    ivc = p.model.add_subsystem("arrival_dv", om.IndepVarComp())
    for n in ("b_t", "b_r", "t_peri"):
        ivc.add_output(n, val=0.0, units="km")
    p.model.add_subsystem("f", om.ExecComp("y = b_t**2 + b_r**2 + t_peri**2"))
    for n in ("b_t", "b_r", "t_peri"):
        p.model.connect(f"arrival_dv.{n}", f"f.{n}")
        p.model.add_design_var(f"arrival_dv.{n}", lower=-1e4, upper=2e4, ref=1e3)
    p.model.add_objective("f.y")
    p.setup()
    p.set_val("arrival_dv.b_t", b_t)
    p.set_val("arrival_dv.b_r", b_r)
    p.set_val("arrival_dv.t_peri", t_peri)
    p.final_setup()
    return p


def test_active_box_bounds_lists_only_box_variables_at_a_bound():
    """A box variable at its bound is listed with its side; a stated bound (t_peri) is not."""
    p = _box_problem(2e4, -1e4 + 1e-5, 2e4)
    assert solve.active_box_bounds(p) == ["arrival.b_t_upper", "arrival.b_r_lower"]
    p = _box_problem(1.99e4, -0.5e4, 0.0)
    assert solve.active_box_bounds(p) == []


def test_box_labels():
    """Design-variable names map to phase.var and arrival.var labels."""
    assert solve._label("traj.approach.states:x") == "approach.x"
    assert solve._label("traj.atmosphere.parameters:q_peak") == "atmosphere.q_peak"
    assert solve._label("traj.coast_in.t_initial") == "coast_in.t_initial"
    assert solve._label("arrival_dv.dec") == "arrival.dec"


def test_segment_ends_follow_the_weight():
    """Segment ends span [-1, 1], increase, and crowd where the weight is large."""
    t = np.linspace(0.0, 1.0, 501)
    y = np.zeros((501, 6))
    y[:, 0] = 1.0 + 9.0 * t                  # radius grows tenfold: segments crowd at t = 0
    ends = cg.segment_ends(cg.Arc(t, y), 10)
    assert ends[0] == -1.0 and ends[-1] == 1.0 and np.all(np.diff(ends) > 0)
    assert np.diff(ends)[0] < np.diff(ends)[-1]
    assert np.allclose(cg.segment_ends(cg.Arc(t, y), 4, "uniform"), [-1, -0.5, 0, 0.5, 1])
