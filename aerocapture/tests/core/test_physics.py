"""Unit checks of the physics models in system, dynamics, atmosphere and heating."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core import heating
from aerocapture.core import system

MU = system.MU_NEPTUNE


# ------------------------------------------------------------------- system --
def test_triton_fit_residual(epoch):
    """The circular Triton model stays within 50 km of nep097 over +-6 d and is retrograde."""
    assert system.triton_residual_km(epoch.triton, epoch.et_ref, span_days=6.0) < 50.0
    assert 156.0 < math.degrees(math.acos(epoch.triton.h_hat[2])) < 159.0
    assert epoch.triton.period / 86400.0 == pytest.approx(5.877, abs=0.01)


def test_triton_velocity_is_position_derivative(epoch):
    """TritonOrbit.velocity is the time derivative of TritonOrbit.position."""
    tr, t, h = epoch.triton, 1.0e4, 1.0
    fd = (tr.position(t + h) - tr.position(t - h)) / (2 * h)
    assert np.allclose(tr.velocity(t), fd, rtol=1e-8)


def test_frame_and_epoch(epoch):
    """NEQ is orthonormal with z on Neptune's pole, and Epoch.iso returns UTC."""
    rot = system.neq_from_j2000(epoch.et_ref)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-14)
    assert rot[0, 2] == pytest.approx(0.0, abs=1e-14)          # x lies in the J2000 equator
    assert epoch.iso(0.0) == "2059-12-31T23:58:51"            # 2060-01-01 TDB = UTC + 69.18 s
    assert system.mjd2000_to_et(0.0) == pytest.approx(-43200.0, abs=1e-6)   # J2000 is noon


def test_hill_radius():
    """The Neptune Hill radius is about 1.16e8 km."""
    assert 1.15e8 < system.R_HILL < 1.17e8


# ------------------------------------------------------------------ gravity --
def _potential(r):
    """Return Neptune's point-mass + J2 potential (positive convention) at ``r``."""
    rm = np.linalg.norm(r)
    sin2 = (r[2] / rm) ** 2
    return MU / rm * (1.0 - system.J2_NEPTUNE * (system.R_J2 / rm) ** 2 * (1.5 * sin2 - 0.5))


def test_j2_acceleration_is_potential_gradient():
    """accel_neptune equals the gradient of the J2 potential (signs and terms)."""
    r = np.array([21000.0, -14000.0, 9000.0])
    h = 1e-3
    grad = np.array([(_potential(r + h * e) - _potential(r - h * e)) / (2 * h) for e in np.eye(3)])
    assert np.allclose(dynamics.accel_neptune(r), grad, rtol=1e-7)


def test_j2_sign():
    """An oblate Neptune pulls harder at the equator and weaker over the pole."""
    rm = 30000.0
    point = MU / rm ** 2
    eq = -dynamics.accel_neptune(np.array([rm, 0.0, 0.0]))[0]
    pole = -dynamics.accel_neptune(np.array([0.0, 0.0, rm]))[2]
    k = 1.5 * system.J2_NEPTUNE * (system.R_J2 / rm) ** 2
    assert eq == pytest.approx(point * (1 + k), rel=1e-14)
    assert pole == pytest.approx(point * (1 - 2 * k), rel=1e-14)


def test_triton_indirect_term(epoch):
    """At Neptune's centre the direct and indirect Triton terms cancel."""
    rt = epoch.triton.position(0.0)
    assert np.allclose(dynamics.accel_triton(np.zeros(3), rt), 0.0, atol=1e-20)


def test_point_mass_gravity_matches_kepler():
    """With J2 and Triton off, one orbital period returns to the initial state."""
    a, e = 60000.0, 0.5
    rp = a * (1 - e)
    y0 = np.array([rp, 0, 0, 0, math.sqrt(MU * (1 + e) / rp), 0])
    period = 2 * math.pi * math.sqrt(a ** 3 / MU)

    def f(t, y):
        return np.concatenate(dynamics.vacuum_rhs(t, y[:3], y[3:], j2=0.0))
    sol = solve_ivp(f, (0, period), y0, rtol=1e-12, atol=1e-9)
    assert np.allclose(sol.y[:, -1], y0, atol=1e-4)


# ------------------------------------------------------------- aerodynamics --
def test_relative_velocity_rotation_sense():
    """Flying against Neptune's rotation adds omega R to the airspeed, with it subtracts."""
    r = np.array([system.R_INTERFACE, 0.0, 0.0])
    w = system.OMEGA_NEPTUNE * system.R_INTERFACE
    assert dynamics.relative_velocity(r, np.array([0.0, 25.0, 0.0]))[1] == pytest.approx(25 - w)
    assert dynamics.relative_velocity(r, np.array([0.0, -25.0, 0.0]))[1] == pytest.approx(-25 - w)


def test_aero_accel_direction_and_magnitude():
    """Drag opposes the airspeed; sigma = 0 lifts up, pi down, pi/2 to the right."""
    r = np.array([[system.R_NEPTUNE + 300.0, 0.0, 0.0]])
    v = np.array([[-2.0, 25.0, 0.0]])
    rho, beta, ld = 1e-4, 200.0, 0.3
    vrel = dynamics.relative_velocity(r, v)[0]
    u = vrel / np.linalg.norm(vrel)
    drag = 0.5 * rho * (np.linalg.norm(vrel) * 1e3) ** 2 / beta * 1e-3
    for sigma, radial_sign, z_sign in ((0.0, 1, 0), (math.pi, -1, 0), (math.pi / 2, 0, -1)):
        a, vm, d = dynamics.aero_accel(r, v, np.array([sigma]), np.array([rho]), beta, ld)
        a = a[0]
        assert d[0] == pytest.approx(drag)
        assert a @ u == pytest.approx(-drag)
        lift = a + drag * u
        assert np.linalg.norm(lift) == pytest.approx(ld * drag)
        assert lift @ u == pytest.approx(0.0, abs=1e-15)
        assert np.sign(round(lift[0] / (ld * drag), 6)) == radial_sign
        assert np.sign(round(lift[2] / (ld * drag), 6)) == z_sign      # right of +y travel is -z


def test_g_load_and_dynamic_pressure():
    """g_load is the sensed aerodynamic acceleration over g0; q_dyn is in kPa."""
    r = np.array([[system.R_NEPTUNE + 300.0, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 25.0]])
    p = dict(beta=150.0, ld=0.25, rn=0.5, kappa=3.0, coupled=1.0, rn_exp=1.0, eps=0.85,
             conv_model="vision_neptune")
    _, vdot, diag = dynamics.atmosphere_rhs(None, r, v, np.array([0.4]), np.array([2e-5]), p)
    a_aero = vdot[0] - dynamics.accel_neptune(r)[0]
    assert diag["g_load"][0] == pytest.approx(np.linalg.norm(a_aero) / 9.80665e-3)
    assert diag["q_dyn"][0] == pytest.approx(0.5 * 2e-5 * (diag["v_rel"][0] * 1e3) ** 2 / 1e3)


# ------------------------------------------------------------------ heating --
def test_heating_coefficients_hand_computed():
    """Heat rates match the AMAT Vision Neptune and Sutton-Graves formulas at one point."""
    rho, v, rn = 1e-4, 25.0e3, 0.5
    assert heating.q_convective(rho, v, rn) == pytest.approx(
        2.24008e-7 * rho ** 0.452130 * v ** 2.691800 * math.sqrt(0.291 / rn), rel=1e-14)
    assert heating.q_radiative_uncoupled(rho, v, rn) == pytest.approx(
        8.125812e-3 * rho ** 0.498140 * 2.5 ** 15.113 * (rn / 0.291), rel=1e-14)
    assert heating.q_radiative_uncoupled(rho, v, rn, 0.5) == pytest.approx(
        8.125812e-3 * rho ** 0.498140 * 2.5 ** 15.113 * (rn / 0.291) ** 0.5, rel=1e-14)
    assert heating.q_convective(rho, v, rn, "sutton_graves") == pytest.approx(
        6.556e-5 * math.sqrt(rho / rn) * v ** 3 * 1e-4, rel=1e-14)


def test_goulard_limits():
    """Gamma uses consistent units; coupling vanishes as Gamma -> 0 or kappa = 0."""
    assert heating.goulard_number(1.0, 1.0, 10.0) == pytest.approx(2e4 / 500.0)
    rho, v = 1e-4, 30e3
    assert heating.q_radiative_coupled(100.0, rho, v, kappa=0.0) == 100.0
    tiny = 1e-12
    assert heating.q_radiative_coupled(tiny, rho, v) == pytest.approx(tiny, rel=1e-6)
    big = 1e7
    gamma = heating.goulard_number(big, rho, v)
    assert heating.q_radiative_coupled(big, rho, v) == pytest.approx(
        big / (3.0 * gamma ** 0.7), rel=1e-2)


def test_radiative_equilibrium_temperature():
    """eps sigma T^4 balances q."""
    t = heating.t_radiative_equilibrium(1000.0, 0.85)
    assert 0.85 * heating.SIGMA_SB * t ** 4 == pytest.approx(1000.0e4)


# --------------------------------------------------------------- atmosphere --
def test_atmosphere_exact_at_nodes_and_monotone():
    """The log-Akima spline hits every node, falls with altitude and never overshoots."""
    atm = atmosphere.default()
    h, rho = atm.h_table, atm.rho_table
    assert np.allclose(atm.density(h), rho, rtol=1e-12)
    hh = np.linspace(h[0], h[-1], 40001)[1:-1]
    rr = atm.density(hh)
    assert np.all(np.diff(rr) < 0)
    idx = np.searchsorted(h, hh) - 1
    assert np.all((rr >= rho[idx + 1] * (1 - 1e-9)) & (rr <= rho[idx] * (1 + 1e-9)))


def test_atmosphere_extrapolation_is_exponential_and_c1():
    """Outside the table ln rho is linear with the end slope, so rho stays C1 and monotone."""
    atm = atmosphere.default()
    for h_end, step in ((atm.h_table[0], -1.0), (atm.h_table[-1], 1.0)):
        rho0, slope0 = atm.density_and_slope(h_end)
        out = h_end + step * np.array([10.0, 100.0, 1000.0])
        assert np.allclose(atm.density(out), rho0 * np.exp(slope0 / rho0 * (out - h_end)),
                           rtol=1e-12)
        rho1, slope1 = atm.density_and_slope(h_end + step * 1e-6)
        assert slope1 == pytest.approx(slope0, rel=1e-5)
        assert np.all(np.diff(atm.density(np.sort(out))) < 0)


# ----------------------------------------------------------- conic geometry --
def test_hyperbola_roundtrip_and_periapsis_time():
    """state_to_bplane inverts hyperbola_state, and Kepler flight reaches periapsis at t_peri."""
    r, v, t0 = dynamics.hyperbola_state(7.0, 0.4, -0.3, 30000.0, -55000.0, 5000.0, 1.0e6)
    assert np.linalg.norm(r) == pytest.approx(1.0e6, abs=1e-6)
    bp = dynamics.state_to_bplane(r, v, t0)
    for k, ref in dict(vinf=7.0, ra=0.4, dec=-0.3, b_t=30000.0, b_r=-55000.0,
                       t_peri=5000.0).items():
        assert bp[k] == pytest.approx(ref, rel=1e-9, abs=1e-6)

    def f(t, y):
        return np.r_[y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3]

    def peri(t, y):
        return y[:3] @ y[3:]
    peri.terminal = True
    sol = solve_ivp(f, (t0, t0 + 1e7), np.r_[r, v], rtol=1e-12, atol=1e-9, events=peri)
    assert sol.t_events[0][0] == pytest.approx(5000.0, abs=1e-3)


def test_final_orbit_capture_metric():
    """capture <= 0 exactly when bound with ra <= ra_max; dv_raise is the vis-viva burn."""
    rp, ra_max, rp_target = 25000.0, 1e7, 27622.0
    for ra in (3e5, 1e7, 2e7):
        a = 0.5 * (ra + rp)
        vp = math.sqrt(MU * (2 / rp - 1 / a))
        orb = dynamics.final_orbit(np.array([rp, 0, 0.0]), np.array([0, vp, 0.0]), ra_max,
                                   rp_target)
        assert orb["ra"] == pytest.approx(ra, rel=1e-9)
        assert orb["rp"] == pytest.approx(rp, rel=1e-12)
        assert (orb["capture"] <= 1e-12) == (ra <= ra_max * (1 + 1e-12))
        va_new = math.sqrt(MU * (2 / ra - 2 / (ra + rp_target)))
        assert orb["dv_raise"] == pytest.approx(va_new - vp * rp / ra, rel=1e-9)
