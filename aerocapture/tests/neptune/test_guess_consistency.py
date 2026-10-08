"""Consistency of the scipy guess dynamics with dynamics.py, and config validation."""
import math

import numpy as np
import pytest

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core import guess as cg
from aerocapture.core import system
from aerocapture.neptune import config
from aerocapture.neptune import guess


def test_scalar_rhs_matches_vectorised(epoch):
    """The guess generator's float RHS equals the vectorised dynamics."""
    p = config.Config().vehicle_params()
    rng = np.random.default_rng(1)
    atm = atmosphere.default()
    for _ in range(5):
        r = rng.normal(size=3)
        r *= (system.R_NEPTUNE + rng.uniform(200, 600)) / np.linalg.norm(r)
        v = rng.normal(size=3) * 10 + 15
        f = guess._f_atm(0.0, np.r_[r, v, 0.0], guess._bank_fn(0.7), p, atm)
        rho = atm.density(np.linalg.norm(r) - system.R_NEPTUNE)
        _, vdot, diag = dynamics.atmosphere_rhs(None, r[None], v[None], np.array([0.7]),
                                                np.atleast_1d(rho), p)
        assert np.allclose(f[3:6], vdot[0], rtol=1e-12, atol=1e-15)
        assert f[6] == pytest.approx(diag["q_total"][0], rel=1e-12)
        t = 1234.0
        _, av = dynamics.vacuum_rhs(t, r[None] * 10, v[None], epoch.triton)
        fv = cg.f_neq(t, np.r_[r * 10, v], epoch.triton)
        assert np.allclose(fv[3:], av[0], rtol=1e-12, atol=1e-18)


def test_flyby_matches_patched_conic(epoch):
    """A full-dynamics Triton flyby matches the patched-conic turn within 5 %."""
    cfg = config.Config(triton_mode="inbound", vinf_kms=5.0)
    g = guess.make_guess(guess.GuessSettings.from_config(cfg, epoch, verbose=False))
    arc = g.arcs["triton_in"]
    tr = epoch.triton
    t0, t1, y0 = arc.t[0], arc.t[-1], arc.y[0]
    dv_full = np.linalg.norm(cg.propagate(cg.f_neq, t0, y0, t1, (tr,)).y[3:, -1]
                             - cg.propagate(cg.f_neq, t0, y0, t1, (None,)).y[3:, -1])
    rel = arc.y[:, :3] - tr.position(arc.t)
    i = int(np.argmin(np.linalg.norm(rel, axis=1)))
    vrel = arc.y[i, 3:] - tr.velocity(arc.t[i])
    bp = dynamics.state_to_bplane(rel[i], vrel, arc.t[i], mu=system.MU_TRITON)
    b = math.hypot(bp["b_t"], bp["b_r"])
    e = math.sqrt(1 + (b * bp["vinf"] ** 2 / system.MU_TRITON) ** 2)
    assert dv_full == pytest.approx(2 * bp["vinf"] / e, rel=0.05)


def test_both_mode_rejected():
    """Config rejects triton_mode='both'."""
    with pytest.raises(ValueError, match="geometrically impossible"):
        config.Config(triton_mode="both").validate()
