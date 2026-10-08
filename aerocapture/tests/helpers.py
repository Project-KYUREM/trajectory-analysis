"""Helpers shared by the component tests: one-component problems and partials checks."""
import numpy as np
import openmdao.api as om


def worst_rel_error(p):
    """Return the largest relative error of the declared partials against central differences."""
    data = p.check_partials(method="fd", form="central", step=1e-7, step_calc="rel_avg",
                            compact_print=True, out_stream=None)
    worst = 0.0
    for comp in data.values():
        for d in comp.values():
            ref = np.abs(d["J_fd"]).max()
            if ref > 1e-10:
                worst = max(worst, np.abs(d["J_fwd"] - d["J_fd"]).max() / ref)
    return worst


def component_problem(comp):
    """Return a set-up problem around one component, every variable promoted."""
    p = om.Problem(reports=False)
    p.model.add_subsystem("c", comp, promotes=["*"])
    p.setup(force_alloc_complex=True)
    return p
