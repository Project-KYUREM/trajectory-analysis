"""check_partials on the shared OpenMDAO components."""
from aerocapture.core import arrival
from aerocapture.core import system
from aerocapture.tests.helpers import component_problem
from aerocapture.tests.helpers import worst_rel_error


def test_arrival_partials():
    """ArrivalComp complex-step partials match finite differences."""
    p = component_problem(arrival.ArrivalComp())
    p["vinf"], p["ra"], p["dec"] = 6.0, 0.7, -0.2
    p["b_t"], p["b_r"], p["t_peri"] = 40000.0, -60000.0, 1000.0
    p.run_model()
    assert worst_rel_error(p) < 1e-5


def test_triton_arrival_partials():
    """ArrivalComp about Triton (mu option) partials match finite differences."""
    p = component_problem(arrival.ArrivalComp(r0=30000.0, mu=system.MU_TRITON))
    p["vinf"], p["ra"], p["dec"] = 8.2, 1.5, 0.3
    p["b_t"], p["b_r"], p["t_peri"] = 1400.0, 300.0, 9.3e4
    p.run_model()
    assert worst_rel_error(p) < 1e-3     # fd noise on the ~1e-7 velocity/B partials
