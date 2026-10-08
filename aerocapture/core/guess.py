"""Shared numerics of the initial-guess generators: float dynamics, propagation, roots, shooting.

The guess generators integrate thousands of short arcs with scipy, so the
right-hand sides here work on plain floats rather than arrays. Each one
reproduces a vectorised model of ``dynamics.py`` to round-off (the mission
tests check this):

    accel_neptune_scalar   ``dynamics.accel_neptune`` (point mass + J2)
    f_neq                  ``dynamics.vacuum_rhs`` (Neptune + J2, optional Triton)

Also here: the ``Arc`` sample container every phase seed uses, dense-output
propagation, ``solve_ivp`` events, root scans along a dense solution, a
scaled Levenberg-Marquardt solver for the shooting problems, and the Radau
segment clustering ``segment_ends`` with its weight functions.

Units: km, s, km/s, rad. Times are seconds from the epoch.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

from aerocapture.core import system

MU_N, MU_T = system.MU_NEPTUNE, system.MU_TRITON
J2K = 1.5 * system.J2_NEPTUNE * system.R_J2 ** 2   # km^2, the J2 factor k r^2 of accel_neptune
RTOL, ATOL = 1e-10, 1e-7                          # vacuum propagation tolerances
SAMPLES = 1500                                    # samples per guess arc


@dataclass
class Arc:
    """Sampled trajectory piece: times (n,), states (n, 6), optional bank and heat load."""
    t: np.ndarray
    y: np.ndarray
    sigma: np.ndarray = None
    Q: np.ndarray = None


# ------------------------------------------------------------ float dynamics --
def accel_neptune_scalar(x, y, z, j2k=J2K):
    """Return Neptune point mass + J2 acceleration in km/s^2 on plain floats.

    Args:
        x (float): Position component, km.
        y (float): Position component, km.
        z (float): Position component along Neptune's pole, km.
        j2k (float, optional): ``1.5 J2 R_J2^2``, km^2; 0 drops J2. Defaults to ``J2K``.

    Returns:
        tuple: ``(ax, ay, az)``, km/s^2.
    """
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    k = j2k / r2
    zr2 = z * z / r2
    c = -MU_N / (r2 * r)
    f = 1.0 + k * (1.0 - 5.0 * zr2)
    return c * x * f, c * y * f, c * z * (1.0 + k * (3.0 - 5.0 * zr2))


def triton_state_scalar(triton, t):
    """Return Triton's NEQ position (km) and velocity (km/s) at time ``t`` (s) as tuples."""
    th = triton.theta0 + triton.n * t
    c, s = math.cos(th), math.sin(th)
    p, q, a, n = triton.p, triton.q, triton.a, triton.n
    pos = tuple(a * (c * p[i] + s * q[i]) for i in range(3))
    vel = tuple(a * n * (-s * p[i] + c * q[i]) for i in range(3))
    return pos, vel


def f_neq(t, y, triton=None, j2k=J2K):
    """Return the NEQ vacuum state derivative: Neptune + J2, plus Triton when given.

    Args:
        t (float): Time past the epoch, s.
        y (np.ndarray): State ``[x, y, z, vx, vy, vz]``, km and km/s.
        triton (system.TritonOrbit, optional): Third body; None leaves it out.
            Defaults to None.
        j2k (float, optional): ``1.5 J2 R_J2^2``, km^2. Defaults to ``J2K``.

    Returns:
        np.ndarray: ``[vx, vy, vz, ax, ay, az]``.
    """
    ax, ay, az = accel_neptune_scalar(y[0], y[1], y[2], j2k)
    if triton is not None:
        (tx, ty, tz), _ = triton_state_scalar(triton, t)
        dx, dy, dz = y[0] - tx, y[1] - ty, y[2] - tz
        d3 = (dx * dx + dy * dy + dz * dz) ** 1.5
        t3 = (tx * tx + ty * ty + tz * tz) ** 1.5
        ax -= MU_T * (dx / d3 + tx / t3)
        ay -= MU_T * (dy / d3 + ty / t3)
        az -= MU_T * (dz / d3 + tz / t3)
    return np.array([y[3], y[4], y[5], ax, ay, az])


def energy(y, mu=MU_N):
    """Return the two-body specific energy of a state about a body of GM ``mu``, km^2/s^2."""
    return 0.5 * (y[3:6] @ y[3:6]) - mu / math.sqrt(y[:3] @ y[:3])


# --------------------------------------------------------------- propagation --
def propagate(fun, t0, y0, t_end, args=(), events=(), rtol=RTOL, atol=ATOL):
    """Return the dense DOP853 solution of ``fun(t, y, *args)`` from ``t0`` to ``t_end``.

    Args:
        fun (callable): Right-hand side ``fun(t, y, *args)``.
        t0 (float): Start time, s.
        y0 (np.ndarray): Start state.
        t_end (float): End time, s; may lie before ``t0``.
        args (tuple, optional): Extra arguments of ``fun``. Defaults to ().
        events (sequence, optional): ``solve_ivp`` events. Defaults to ().
        rtol (float, optional): Relative tolerance. Defaults to ``RTOL``.
        atol (float, optional): Absolute tolerance. Defaults to ``ATOL``.

    Returns:
        scipy.integrate.OdeResult: Solution with ``sol`` (dense output).
    """
    return solve_ivp(fun, (t0, t_end), y0, args=args, method="DOP853", rtol=rtol, atol=atol,
                     events=events, dense_output=True)


def event(fn, terminal=True, direction=0):
    """Return ``fn`` marked as a ``solve_ivp`` event."""
    fn.terminal, fn.direction = terminal, direction
    return fn


def radius_event(radius, direction=0, terminal=True):
    """Return a ``solve_ivp`` event at distance ``radius`` (km) from the frame origin."""
    return event(lambda t, y, *_: math.sqrt(y[:3] @ y[:3]) - radius, terminal, direction)


def apsis_event(direction=0, terminal=False):
    """Return a ``solve_ivp`` event at each apsis about the frame origin (r . v = 0)."""
    return event(lambda t, y, *_: y[:3] @ y[3:6], terminal, direction)


def sample(sol, t_a, t_b, n=SAMPLES):
    """Return ``n`` samples of a dense solution between two times as an ``Arc``."""
    tt = np.linspace(t_a, t_b, n)
    return Arc(tt, sol.sol(tt).T[:, :6])


def find_roots(sol, g, t_a, t_b, n):
    """Return the roots of ``g(t, r)`` along a dense solution, in order from ``t_a`` to ``t_b``.

    Args:
        sol (scipy.integrate.OdeResult): Dense ``solve_ivp`` solution.
        g (callable): ``g(t (m,), r (m, 3)) -> (m,)``.
        t_a (float): Start of the scan, s.
        t_b (float): End of the scan, s.
        n (int): Scan samples; roots closer than one sample apart can be missed.

    Returns:
        list[float]: Root times, s.
    """
    ts = np.linspace(t_a, t_b, n)
    vals = g(ts, sol.sol(ts)[:3].T)
    f = lambda t: float(g(np.array([t]), sol.sol(t)[:3][None])[0])
    return [brentq(f, min(ts[i], ts[i + 1]), max(ts[i], ts[i + 1]))
            for i in np.where(np.diff(np.sign(vals)) != 0)[0]]


# ------------------------------------------------------------------- algebra --
def unit(v):
    """Return ``v`` normalised."""
    return v / np.linalg.norm(v)


def rotation(axis, angle):
    """Return the rotation matrix about unit ``axis`` by ``angle`` (Rodrigues)."""
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def wrap_angle(a):
    """Return the angle ``a`` (rad) wrapped to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def solve_scaled(fun, x0, scale, h=0.1):
    """Solve ``fun(x) = 0`` by Levenberg-Marquardt in scaled variables with forward differences.

    Args:
        fun (callable): Residual function of the physical variables.
        x0 (np.ndarray): Starting point.
        scale (np.ndarray): Physical size of one scaled unit per variable.
        h (float, optional): Finite-difference step in scaled units. Defaults to 0.1.

    Returns:
        scipy.optimize.OptimizeResult: With ``x`` in physical units.
    """
    x0, scale = np.asarray(x0, float), np.asarray(scale, float)
    f = lambda z: fun(x0 + z * scale)

    def jac(z):
        f0 = f(z)
        return np.array([(f(z + h * e) - f0) / h for e in np.eye(len(z))]).T
    res = least_squares(f, np.zeros_like(x0), jac=jac, method="lm", xtol=1e-12, ftol=1e-12,
                        max_nfev=200)
    res.x = x0 + res.x * scale
    return res


# -------------------------------------------------------- segment clustering --
def radius_weight(arc):
    """Return 1/|r| along an arc (dense segments near the frame origin)."""
    return 1.0 / np.linalg.norm(arc.y[:, :3], axis=1)


def density_weight(atm, radius):
    """Return a weight function ``(rho / rho_max)^0.25`` along an arc in ``atm``.

    Args:
        atm (atmosphere.Atmosphere): Density model of altitude in km.
        radius (float): Reference radius the altitude is measured from, km.

    Returns:
        callable: ``weight(arc)`` with one positive weight per sample.
    """
    def weight(arc):
        rho = atm.density(np.linalg.norm(arc.y[:, :3], axis=1) - radius)
        return (rho / rho.max()) ** 0.25
    return weight


def segment_ends(arc, num_segments, weight=radius_weight, blend=0.15):
    """Return Radau segment boundaries in [-1, 1], equally spaced in a weighted time.

    Args:
        arc (Arc): Guess arc of the phase.
        num_segments (int): Number of segments.
        weight (callable | str, optional): ``weight(arc)`` returning one
            positive weight per sample, or ``"uniform"``. Defaults to
            ``radius_weight``.
        blend (float, optional): Fraction of uniform spacing mixed in so no
            segment grows too long. Defaults to 0.15.

    Returns:
        np.ndarray: ``num_segments + 1`` increasing boundaries from -1 to 1.
    """
    if isinstance(weight, str) and weight == "uniform":
        return np.linspace(-1, 1, num_segments + 1)
    t = arc.t
    tau = (t - t[0]) / (t[-1] - t[0])
    w = np.asarray(weight(arc), dtype=float)
    w = (1 - blend) * w / np.mean(w) + blend
    s = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    ends = 2 * np.interp(np.linspace(0.0, 1.0, num_segments + 1), s / s[-1], tau) - 1
    ends[0], ends[-1] = -1.0, 1.0
    return ends
