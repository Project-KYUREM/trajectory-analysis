"""Unit checks of the Triton atmosphere, frame, ballute physics, patched conics and the guess."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from aerocapture.core import dynamics
from aerocapture.core import guess as cg
from aerocapture.core import system
from aerocapture.triton import atmosphere
from aerocapture.triton import ballute
from aerocapture.triton import config
from aerocapture.triton import frame as tf
from aerocapture.triton import guess as guess_mod
from aerocapture.triton import ode


# --------------------------------------------------------------- atmosphere --
def test_atmosphere_anchors():
    """The default Bates profile meets the Benne, Voyager UVS and 2017 occultation anchors."""
    prof = atmosphere.bates_profile(16.0, 38.0, 92.0, 161.6)
    h = prof["h_km"]
    assert np.interp(1026.0, h, prof["p_pa"]) == pytest.approx(2.8e-9, rel=0.01)
    assert 3.6e8 <= np.interp(575.0, h, prof["n_cm3"]) <= 4.4e8
    assert np.interp(47.0, h, prof["p_pa"]) * 10.0 == pytest.approx(1.18, rel=0.10)
    assert prof["p_pa"][0] == pytest.approx(1.6, rel=1e-10)
    assert np.all(np.diff(prof["rho_kg_m3"]) < 0)


def test_atmosphere_spline_exact_and_scaled():
    """The log-density spline hits every table node and the density scale multiplies it."""
    atm, rec = atmosphere.build()
    assert np.allclose(atm.density(atm.h_table), atm.rho_table, rtol=1e-12)
    atm2, _ = atmosphere.build(density_scale=1.5)
    assert atm2.density(123.4) == pytest.approx(1.5 * atm.density(123.4), rel=1e-12)
    assert rec["rho_kg_m3_at_km"]["100"] == pytest.approx(6.6e-7, rel=0.02)


# -------------------------------------------------------------------- frame --
def test_transforms_round_trip(epoch):
    """NEQ -> TC -> NEQ is the identity."""
    tr = epoch.triton
    r, v = np.array([3.5e5, 1e4, -2e4]), np.array([1.0, -3.0, 0.5])
    rho, w = tf.to_triton(1234.0, r, v, tr)
    r2, v2 = tf.to_neq(1234.0, rho, w, tr)
    assert np.allclose(r2, r, rtol=0, atol=1e-9) and np.allclose(v2, v, rtol=0, atol=1e-12)


def test_frame_consistency(epoch):
    """One state integrated 2 d in NEQ and in the Triton frame agrees to better than 1 m."""
    tr = epoch.triton
    t0, t1 = 0.0, 2 * 86400.0
    rho0 = np.array([4000.0, 1500.0, -800.0])
    w0 = np.array([0.1, -0.55, 0.2])
    r0, v0 = tf.to_neq(t0, rho0, w0, tr)
    kw = dict(method="DOP853", rtol=1e-13, atol=1e-10)
    neq = solve_ivp(cg.f_neq, (t0, t1), np.r_[r0, v0], args=(tr,), **kw)
    tc = solve_ivp(guess_mod.f_tc, (t0, t1), np.r_[rho0, w0], args=(tr,), **kw)
    rho_n, _ = tf.to_triton(t1, neq.y[:3, -1], neq.y[3:, -1], tr)
    assert np.linalg.norm(rho_n - tc.y[:3, -1]) < 1e-3


def test_scalar_rhs_matches_vectorised(epoch):
    """The guess's float right-hand sides equal the vectorised ODE models."""
    tr = epoch.triton
    cfg = config.Config()
    atm, _ = cfg.make_atmosphere()
    rng = np.random.default_rng(3)
    for phase in ("ballute", "probe"):
        model = guess_mod.PassModel(cfg, tr, atm, phase)
        for _ in range(4):
            r = rng.normal(size=3)
            r *= (system.R_TRITON + rng.uniform(20, 300)) / np.linalg.norm(r)
            v = rng.normal(size=3) * 3
            t = rng.uniform(0, 1e5)
            f = model.rhs(t, np.r_[r, v])
            rho = atm.density(np.linalg.norm(r) - system.R_TRITON)
            d = ode.aero(r[None], v[None], np.atleast_1d(rho), cfg.pass_params(phase), tr,
                                True, "sutton_graves_n2", "smooth_min", 1.0, False, 0.85, 2)
            a = tf.accel_translated(t, r[None], tr)[0] - d["drag"][0] * d["u"][0]
            assert np.allclose(f[3:], a, rtol=1e-11, atol=1e-16)
    y = np.r_[5000.0, 200.0, -10.0, 0.1, 0.4, 0.0]
    assert np.allclose(guess_mod.f_tc(77.0, y, tr)[3:],
                       tf.accel_translated(77.0, y[None, :3], tr)[0], rtol=1e-12)


def test_triton_constants(epoch):
    """Triton's Hill radius, surface rotation speed and orbit speed match the plan."""
    tr = epoch.triton
    assert tf.hill_radius(tr) == pytest.approx(14597.0, abs=5.0)
    assert tr.n * system.R_TRITON * 1e3 == pytest.approx(16.7, abs=0.1)
    assert tr.a * tr.n == pytest.approx(4.390, abs=0.002)


# ------------------------------------------------------------------ ballute --
def test_drag_coefficient_limits():
    """The Wilmoth bridge tends to C_c in the continuum and C_fm in free-molecular flow."""
    cd = ballute.drag_coefficient(np.array([1e-6, 1e-3, 10.0, 1e4]), 0.9, 2.0)
    assert cd[0] == pytest.approx(0.9) and cd[1] == pytest.approx(0.9)
    assert cd[2] == pytest.approx(2.0) and cd[3] == pytest.approx(2.0)
    mid = ballute.drag_coefficient(np.array([0.1]), 0.9, 2.0)[0]
    assert 0.9 < mid < 2.0


def test_heating_bridge_limits():
    """The bridged heat rate tends to Sutton-Graves (dense) and free-molecular (rarefied)."""
    v, rn = 8000.0, 35.0
    for rho, which in ((1e-5, "c"), (1e-14, "fm")):
        qc = ballute.q_continuum(rho, v, rn)
        qfm = ballute.q_free_molecular(rho, v)
        qb = ballute.q_bridged(qc, qfm)
        assert qb == pytest.approx(qc if which == "c" else qfm, rel=1e-3)
    assert ballute.q_continuum(1e-6, v, rn) == pytest.approx(
        1.7407e-4 * math.sqrt(1e-6 / rn) * v ** 3 * 1e-4, rel=1e-12)


def test_film_temperature_calibration():
    """3.5 W/cm^2 on a two-sided film with emissivity 0.85 gives 776 K (NIAC Pluto 520-550 C)."""
    assert ballute.film_temperature(3.5, 0.85, 2) == pytest.approx(776.0, abs=1.0)


def test_tauber_sutton():
    """The radiative bound hits the table nodes and gives ~19.8 W/cm^2 at 10 km/s, 5e-6, 35 m."""
    for v, f in ballute.TS_TABLE:
        assert math.exp(ballute.tauber_sutton_lnf(float(v))) == pytest.approx(f, rel=1e-12)
    assert ballute.q_tauber_sutton(5e-6, 1.0e4, 35.0) == pytest.approx(19.8, rel=0.01)
    h = 1e-30
    v = 9123.0
    cs = ballute.tauber_sutton_lnf(v + 1j * h).imag / h
    fd = (ballute.tauber_sutton_lnf(v + 0.01) - ballute.tauber_sutton_lnf(v - 0.01)) / 0.02
    assert cs == pytest.approx(fd, rel=1e-6)


# ----------------------------------------------------------- patched conics --
def test_patched_conic_table(epoch):
    """v_inf 5, rp 1.10 R_N: v_rel 8.21 / 9.10 / 9.91 km/s (co / polar / counter)."""
    sp = guess_mod.encounter_speeds(5.0, 1.10 * system.R_NEPTUNE, epoch.triton,
                                       system.R_TRITON + 1000.0)
    assert sp["co"]["v_rel"] == pytest.approx(8.21, abs=0.01)
    assert sp["polar"]["v_rel"] == pytest.approx(9.10, abs=0.01)
    assert sp["counter"]["v_rel"] == pytest.approx(9.91, abs=0.015)
    assert sp["co"]["v_entry"] == pytest.approx(8.28, abs=0.01)
    assert sp["crossing"]["turn_deg"] == pytest.approx(131.0, abs=0.5)
    assert sp["crossing"]["f_deg"] == pytest.approx(140.0, abs=0.5)


def test_brisby_lyne_and_direct_intercept(epoch):
    """Direct intercept: Brisby & Lyne 4.01 / 9.41 km/s at v_inf 5.44; 3.75 km/s at v_inf 5."""
    r_i = system.R_TRITON + 1000.0
    tr = epoch.triton
    assert guess_mod.direct_intercept(5.44, tr, r_i, 0.0)["v_entry"] == pytest.approx(4.01,
                                                                                         abs=0.01)
    assert guess_mod.direct_intercept(5.44, tr, r_i, 90.0)["v_entry"] == pytest.approx(9.41,
                                                                                          abs=0.01)
    assert guess_mod.direct_intercept(5.0, tr, r_i)["v_entry"] == pytest.approx(3.75, abs=0.01)


def test_ring_nodes_match_brute_force():
    """Analytic node radii equal the z-crossings of a propagated Neptune conic."""
    mu = system.MU_NEPTUNE
    r, v, t0 = dynamics.hyperbola_state(5.0, 0.3, 0.25, 70000.0, 45000.0, 0.0, 4.0e5)
    u_asc, u_desc = ode.node_inverse_radii(r, v)

    def f(t, y):
        return np.r_[y[3:], -mu * y[:3] / np.linalg.norm(y[:3]) ** 3]

    def zc(t, y):
        return y[2]
    sol = solve_ivp(f, (t0, -t0), np.r_[r, v], rtol=1e-12, atol=1e-6, events=zc,
                    method="DOP853")
    radii = sorted(np.linalg.norm(y[:3]) for y in sol.y_events[0])
    analytic = sorted(1.0 / u for u in (u_asc, u_desc) if u > 0)
    assert len(radii) == len(analytic) >= 1
    assert np.allclose(radii, analytic, rtol=1e-6)


# -------------------------------------------------------------------- guess --
def test_guess_corridor(epoch):
    """D 70 m, 530 kg at v_E 8.3 km/s: corridor about -52.0 .. -53.0 deg (plan prototype)."""
    cfg = config.Config()
    atm, _ = cfg.make_atmosphere()
    gen = guess_mod.Generator(cfg, epoch, atm, verbose=False)
    cor, g = gen.corridor(0.0, 8.3, 0.0)
    assert cor["gamma_shallow_deg"] == pytest.approx(-52.0, abs=0.3)
    assert cor["gamma_steep_deg"] == pytest.approx(-53.0, abs=0.3)
    assert cor["gamma_steep_deg"] < math.degrees(g) < cor["gamma_shallow_deg"]


def test_config_validation():
    """Unknown options and empty bands are rejected."""
    with pytest.raises(ValueError):
        config.Config(objective="nope").validate()
    with pytest.raises(ValueError):
        config.Config(rp_min_rn=1.3, rp_max_rn=1.2).validate()
    config.Config().validate()
