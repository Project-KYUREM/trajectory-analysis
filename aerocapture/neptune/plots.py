"""Figures of the initial guess, the solved trajectory and the sweeps (matplotlib).

Every figure function returns its figure; ``save_all`` writes PNGs. Colours,
axis style and the collocated / simulated / guess line styles come from
``aerocapture.core.plotting`` (re-exported here as ``SERIES``, ``style``, ...).

    atmosphere_panels  every pass quantity versus time from entry, with limits
    corridor_figure    altitude versus atmosphere-relative speed
    system_view        3-D and Triton-plane views, auto-scaled to the whole
                       trajectory and the captured orbit, plus a zoom on
                       Triton's orbit
    triton_closeup     Triton-relative path through a Triton phase
    sweep_figure       peak heat rate, heat load and g versus v-inf per mode
    trade_figure       peak heat-rate terms per traded vehicle
"""
import math

import matplotlib.pyplot as plt
import numpy as np

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core import heating
from aerocapture.core import plotting as P
from aerocapture.core import system
from aerocapture.neptune import results

SERIES, INK, MUTED, GRID = P.SERIES, P.INK, P.MUTED, P.GRID
style, _sphere, save_all = P.style, P.sphere, P.save_all
PHASE_COLOR = {"approach": SERIES[0], "triton_in": SERIES[1], "coast_in": SERIES[2],
               "atmosphere": SERIES[7], "coast_mid": SERIES[3], "triton_out": SERIES[6]}
CAPTURE_COLOR = SERIES[4]
KM_SCALE = 1e3          # system views plot in 1000 km


# ------------------------------------------------------------------ series
def atmosphere_series(t, y, sigma, params, Q=None, sigma_rate=None):
    """Return every plotted pass quantity of one atmosphere arc.

    Args:
        t (np.ndarray): Times, s (n,).
        y (np.ndarray): NEQ states, km and km/s (n, 6).
        sigma (np.ndarray): Bank angle, rad (n,).
        params (dict): ``config.Config.vehicle_params()``.
        Q (np.ndarray, optional): Heat load, J/cm^2 (n,). Defaults to the
            cumulative integral of the total heat rate.
        sigma_rate (np.ndarray, optional): Bank rate, rad/s (n,). Defaults to
            finite differences of ``sigma``.

    Returns:
        dict: ``t`` (s from the first sample), ``alt`` (km), ``v`` and
        ``v_rel`` (km/s), ``g_load``, ``q_conv``, ``q_rad`` (selected
        coupling), ``q_rad_cpl``, ``q_rad_unc``, ``q_total`` (W/cm^2), ``Q``
        (kJ/cm^2), ``sigma`` and ``sigma_rate`` (deg, deg/s), ``q_dyn`` (kPa),
        ``rho`` (kg/m^3), ``fpa_rel`` (deg) and ``t_radeq`` (K).
    """
    t = np.asarray(t, dtype=float)
    r, v = y[:, :3], y[:, 3:6]
    rm = np.linalg.norm(r, axis=1)
    alt = rm - system.R_NEPTUNE
    rho = atmosphere.default().density(alt)
    _, _, diag = dynamics.atmosphere_rhs(None, r, v, sigma, rho, params)
    vrel = dynamics.relative_velocity(r, v)
    fpa = np.degrees(np.arcsin(np.sum(r * vrel, axis=1) / (rm * diag["v_rel"])))
    if Q is None:
        tu, qu = P.unique_times(t, diag["q_total"])
        Q = np.interp(t, tu, np.r_[0.0, np.cumsum(0.5 * (qu[1:] + qu[:-1]) * np.diff(tu))])
    if sigma_rate is None:
        sigma_rate = P.rate(t, sigma)
    q_rad_cpl = heating.q_radiative_coupled(diag["q_rad_unc"], rho, diag["v_rel"] * 1e3,
                                            params["kappa"])
    return dict(t=t - t[0], alt=alt, v=np.linalg.norm(v, axis=1), v_rel=diag["v_rel"],
                g_load=diag["g_load"], q_conv=diag["q_conv"], q_rad=diag["q_rad"],
                q_rad_cpl=q_rad_cpl, q_rad_unc=diag["q_rad_unc"], q_total=diag["q_total"],
                Q=np.asarray(Q) / 1e3, sigma=np.degrees(sigma),
                sigma_rate=np.degrees(sigma_rate), q_dyn=diag["q_dyn"], rho=rho, fpa_rel=fpa,
                t_radeq=diag["t_radeq"])


def _series_from_ts(ts, params):
    """Return ``atmosphere_series`` of an atmosphere timeseries dict (``results.timeseries``)."""
    if ts is None or "x" not in ts or "sigma" not in ts:
        return None
    y = np.stack([ts[n] for n in results.STATE_NAMES], 1)
    return atmosphere_series(ts["time"], y, ts["sigma"], params, ts.get("Q"),
                             ts.get("sigma_rate"))


def _series_from_arc(arc, params):
    """Return ``atmosphere_series`` of a guess atmosphere arc."""
    return atmosphere_series(arc.t, arc.y, arc.sigma, params, arc.Q)


def _legacy_series(ts):
    """Return the plottable subset of a timeseries dict that already holds the diagnostics."""
    if ts is None:
        return None
    t = np.asarray(ts["time"], dtype=float)
    out = dict(t=t - t[0])
    if "x" in ts:
        r = np.stack([ts["x"], ts["y"], ts["z"]], 1)
        out["alt"] = np.linalg.norm(r, axis=1) - system.R_NEPTUNE
        out["v"] = np.linalg.norm(np.stack([ts["vx"], ts["vy"], ts["vz"]], 1), axis=1)
    for k in ("alt", "v_rel", "g_load", "q_conv", "q_rad", "q_rad_unc", "q_total", "q_dyn",
              "t_radeq"):
        if k in ts:
            out[k] = np.asarray(ts[k])
    if "Q" in ts:
        out["Q"] = np.asarray(ts["Q"]) / 1e3
    if "sigma" in ts:
        out["sigma"] = np.degrees(ts["sigma"])
        out["sigma_rate"] = np.degrees(ts["sigma_rate"]) if "sigma_rate" in ts \
            else P.rate(t, out["sigma"])
    if "fpa_rel" in ts:
        out["fpa_rel"] = np.degrees(ts["fpa_rel"])
    return out


def limits_of(cfg, q_peak=None):
    """Return the path limits drawn on the pass panels.

    Args:
        cfg (config.Config): Inputs (vehicle limits and bank bounds).
        q_peak (float, optional): Solved peak-heat-rate slack, W/cm^2. Defaults to None.

    Returns:
        dict: ``g_max``, ``q_max``, ``q_peak``, ``t_wall_max`` (None unless the
        TPS is non-ablative), ``bank_min``, ``bank_max`` (deg) and
        ``bank_rate_max`` (deg/s).
    """
    v = cfg.vehicle
    return dict(g_max=v.g_max, q_max=v.q_max_w_cm2, q_peak=q_peak,
                t_wall_max=v.t_wall_max_k if v.tps == "non_ablative" else None,
                bank_min=cfg.bank_min_deg, bank_max=cfg.bank_max_deg,
                bank_rate_max=cfg.bank_rate_max_degs, rad_coupling=cfg.rad_coupling)


# ------------------------------------------------------------- pass figures
def atmosphere_panels(ts, title, sim=None, guess=None, params=None, limits=None):
    """Plot every atmosphere-pass quantity versus time from the entry interface.

    Panels: altitude, inertial and atmosphere-relative speed, sensed g, heat
    rates (convective, radiative coupled and uncoupled, total), heat load,
    bank angle, bank rate, dynamic pressure, density, atmosphere-relative
    flight-path angle and radiative-equilibrium temperature.

    Args:
        ts (dict): Atmosphere timeseries (``results.timeseries(ap)['atmosphere']``),
            or an ``atmosphere_series`` dict (has key ``t``).
        title (str): Figure title.
        sim (dict, optional): Simulated atmosphere timeseries, same forms as
            ``ts``; drawn dashed. Defaults to None.
        guess (dict, optional): Guess series (``atmosphere_series`` dict, or a
            ``core.guess.Arc``); drawn dotted. Defaults to None.
        params (dict, optional): ``Config.vehicle_params()``; without it only
            the quantities already in ``ts`` are drawn. Defaults to None.
        limits (dict, optional): ``limits_of(cfg)``. Defaults to None (no limits).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    def series(src):
        if src is None:
            return None
        if hasattr(src, "sigma") and hasattr(src, "y"):
            return _series_from_arc(src, params) if params else None
        if "t" in src:
            return src
        return _series_from_ts(src, params) if params else _legacy_series(src)

    sets = {"collocated": series(ts), "simulated": series(sim), "guess": series(guess)}
    if sets["collocated"] is None and sets["guess"] is not None:
        sets = {"initial guess": sets["guess"]}
    lim = limits or {}
    fig, axs, spare = P.panel_grid(11, ncols=4, title=title)
    ov = lambda ax, key, color=SERIES[0], label=None, **kw: P.overlay(ax, sets, "t", key, color,
                                                                         label, **kw)
    ax = axs[0]
    ov(ax, "alt")
    ax.set_ylabel("altitude (km)")
    ax = axs[1]
    ov(ax, "v", SERIES[0], "inertial")
    ov(ax, "v_rel", SERIES[1], "rel. to atmosphere")
    ax.set_ylabel("speed (km/s)")
    ax.legend(fontsize=7, loc="lower left")
    ax = axs[2]
    ov(ax, "g_load")
    ax.set_ylabel("sensed load (g)")
    P.limit_line(ax, lim.get("g_max"), f"g_max {lim.get('g_max')}")
    ax = axs[3]
    coupling = lim.get("rad_coupling", "selected")
    ov(ax, "q_conv", SERIES[0], "convective")
    ov(ax, "q_rad_cpl", SERIES[1], "radiative, Goulard")
    if sets.get("collocated") is not None and "q_rad_unc" in sets["collocated"]:
        ax.plot(sets["collocated"]["t"], sets["collocated"]["q_rad_unc"], color=SERIES[4],
                linewidth=1.2, label="radiative, uncoupled")
    else:
        ov(ax, "q_rad_unc", SERIES[4], "radiative, uncoupled")
    ov(ax, "q_total", SERIES[2], f"total ({coupling})")
    ax.set_ylabel("heat rate (W/cm²)")
    P.reference_line(ax, lim.get("q_peak"), f"q_peak {lim['q_peak']:.0f}"
                     if lim.get("q_peak") else "")
    P.limit_line(ax, lim.get("q_max"), f"q_max {lim.get('q_max')}")
    ax.legend(fontsize=6.5, loc="center right")
    ax = axs[4]
    ov(ax, "Q")
    ax.set_ylabel("heat load (kJ/cm²)")
    ax = axs[5]
    ov(ax, "sigma")
    ax.set_ylabel("bank angle (deg, 0 = lift up)")
    P.limit_line(ax, lim.get("bank_max"), f"bank max {lim.get('bank_max')}")
    P.limit_line(ax, lim.get("bank_min"), f"bank min {lim.get('bank_min')}", side="lower")
    ax = axs[6]
    ov(ax, "sigma_rate")
    ax.set_ylabel("bank rate (deg/s)")
    r = lim.get("bank_rate_max")
    if r is not None:
        P.limit_line(ax, r, f"+{r:g} deg/s")
        P.limit_line(ax, -r, f"-{r:g} deg/s", side="lower")
    ax = axs[7]
    ov(ax, "q_dyn")
    ax.set_ylabel("dynamic pressure (kPa)")
    ax = axs[8]
    ov(ax, "rho", mask=lambda y: np.maximum(y, 1e-30))
    ax.set_yscale("log")
    ax.set_ylabel("density (kg/m³)")
    ax = axs[9]
    ov(ax, "fpa_rel")
    ax.axhline(0.0, color=MUTED, linewidth=0.6)
    ax.set_ylabel("flight-path angle, rel. (deg)")
    ax = axs[10]
    ov(ax, "t_radeq")
    ax.set_ylabel("rad.-equilibrium T (K)")
    P.limit_line(ax, lim.get("t_wall_max"), f"t_wall_max {lim.get('t_wall_max')}")
    for a in axs:
        if not a.lines:
            a.text(0.5, 0.5, "not available", transform=a.transAxes, ha="center",
                   color=MUTED, fontsize=8)
    for a in axs[-4:]:
        a.set_xlabel("time from entry interface (s)")
    P.style_key(spare[0], P.present(sets), loc="center")
    return P.finish(fig)


def corridor_figure(sets, title, cfg=None):
    """Plot altitude versus atmosphere-relative speed through the pass (the entry corridor view).

    Args:
        sets (dict): {kind: ``atmosphere_series`` dict or None}.
        title (str): Figure title.
        cfg (config.Config, optional): Inputs; draws the interface altitude. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, xkey, xlab in ((axs[0], "v_rel", "atmosphere-relative speed (km/s)"),
                           (axs[1], "q_dyn", "dynamic pressure (kPa)")):
        P.overlay(ax, sets, xkey, "alt", SERIES[0])
        for kind, d in sets.items():
            if d is None:
                continue
            i = int(np.argmin(d["alt"]))
            ax.scatter(d[xkey][[0, i, -1]], d["alt"][[0, i, -1]], color=SERIES[0],
                       marker="o", s=14 if kind == "collocated" else 8, zorder=4,
                       alpha=P.kind_style(kind)["alpha"])
        if cfg is not None:
            P.reference_line(ax, cfg.interface_alt_km, "entry interface", x=0.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel("altitude (km)")
        P.style(ax)
    d = sets.get("collocated") or next(v for v in sets.values() if v is not None)
    i = int(np.argmin(d["alt"]))
    for k, lab in ((0, "entry"), (i, "minimum altitude"), (-1, "exit")):
        axs[0].annotate(lab, (d["v_rel"][k], d["alt"][k]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color=INK)
    P.style_key(axs[1], P.present(sets), loc="center right")
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------- trajectory views
def _phase_lines(ax, src, kind, proj, label):
    """Draw every phase of one source on ``ax`` through ``proj`` (positions -> plot coords)."""
    for name, (t, y) in src.items():
        ax.plot(*proj(y[:, :3]), color=PHASE_COLOR.get(name, INK), label=name if label else None,
                **P.kind_style(kind))


def capture_path(arcs):
    """Return the two-body captured orbit after the last phase, ``(path, apoapsis)``, km."""
    last = list(arcs.values())[-1][1][-1]
    return P.two_body_path(last[:3], last[3:6], system.MU_NEPTUNE, r_max=2e7)


def system_view(arcs, epoch, title, sim_arcs=None, t_mark=None, guess_arcs=None,
                capture=True, primary="collocated"):
    """Plot the trajectory phases, the captured orbit, Neptune and Triton's orbit in NEQ.

    Three panels: 3-D and Triton-plane projection, both auto-scaled to the
    whole trajectory and the two-body captured orbit after the last phase,
    and the Triton-plane projection zoomed on Triton's orbit.

    Args:
        arcs (dict): {phase: (t, y)} with y of shape (n, 6), km.
        epoch (system.Epoch): Epoch whose Triton orbit is drawn.
        title (str): Figure title.
        sim_arcs (dict, optional): Simulated {phase: (t, y)} drawn dashed. Defaults to None.
        t_mark (float, optional): Time at which Triton's position is marked, s.
            Defaults to None.
        guess_arcs (dict, optional): Guess {phase: (t, y)} drawn dotted. Defaults to None.
        capture (bool, optional): Draw the two-body orbit after the last phase.
            Defaults to True.
        primary (str, optional): Source kind of ``arcs`` in the line-style key
            (``"initial guess"`` for a guess-only figure). Defaults to "collocated".

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    s = KM_SCALE
    tr = epoch.triton
    p, q = np.asarray(tr.p), np.asarray(tr.q)
    torb = tr.position(np.linspace(0, tr.period, 400))
    sources = {primary: arcs, "simulated": sim_arcs, "guess": guess_arcs}
    sources = {k: v for k, v in sources.items() if v}
    cap, apo = capture_path(arcs) if capture else (None, None)
    fig = plt.figure(figsize=(16, 5.6))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    axz = fig.add_subplot(1, 3, 3)
    plane = lambda r: (r @ p / s, r @ q / s)
    for kind, src in sources.items():
        _phase_lines(ax3, src, kind, lambda r: (r.T / s), False)
        _phase_lines(ax2, src, kind, plane, kind == primary)
        _phase_lines(axz, src, kind, plane, False)
    if cap is not None:
        kw = dict(color=CAPTURE_COLOR, linewidth=1.0, linestyle="-.")
        ax3.plot(*(cap.T / s), label="captured orbit (two-body)", **kw)
        for a in (ax2, axz):
            a.plot(*plane(cap), label="captured orbit (two-body)" if a is ax2 else None, **kw)
        if apo is not None:
            ax3.scatter(*(apo / s), color=CAPTURE_COLOR, s=18)
            ax2.scatter(*plane(apo), color=CAPTURE_COLOR, s=18, zorder=5)
            ax2.annotate("apoapsis", plane(apo), xytext=(4, 4), textcoords="offset points",
                         fontsize=7, color=CAPTURE_COLOR)
    ax3.plot(*(torb.T / s), color=MUTED, linewidth=1, label="Triton orbit")
    P.sphere(ax3, system.R_NEPTUNE / s, P.NEPTUNE_COLOR)
    for a in (ax2, axz):
        a.plot(*plane(torb), color=MUTED, linewidth=1, label="Triton orbit" if a is ax2 else None)
        P.disk(a, system.R_NEPTUNE / s, P.NEPTUNE_COLOR)
    if t_mark is not None:
        rt = tr.position(np.array([t_mark]))[0]
        ax3.scatter(*(rt / s), color=INK, s=25)
        for a in (ax2, axz):
            a.scatter(*plane(rt), color=INK, s=25, zorder=5,
                      label="Triton at encounter" if a is ax2 else None)
    pts = [y[:, :3] for src in sources.values() for _, y in src.values()] + [torb, cap]
    P.equal_3d(ax3, [pp / s for pp in pts if pp is not None])
    P.equal_2d(ax2, [np.stack(plane(pp), 1) for pp in pts if pp is not None])
    zl = 1.3 * tr.a / s
    axz.set_xlim(-zl, zl)
    axz.set_ylim(-zl, zl)
    axz.set_aspect("equal")
    ax3.set_xlabel("x NEQ (1000 km)", fontsize=8)
    ax3.set_ylabel("y NEQ (1000 km)", fontsize=8)
    ax3.set_zlabel("z = Neptune pole (1000 km)", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.set_title("NEQ frame, whole trajectory", fontsize=9)
    for a, t_ in ((ax2, "projected on Triton's orbital plane"),
                  (axz, "same, zoomed on Triton's orbit")):
        a.set_xlabel("along Triton p (1000 km)")
        a.set_ylabel("along Triton q (1000 km)")
        a.set_title(t_, fontsize=9)
        P.style(a)
    ax2.legend(fontsize=7, loc="best")
    P.style_key(axz, list(sources), loc="upper right")
    fig.suptitle(title, color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.11, top=0.88, wspace=0.25)
    return fig


def triton_closeup(t, y, epoch, title, r_min=None, sim=None, guess=None):
    """Plot the Triton-relative path through a Triton phase.

    The path is projected on the plane of the Triton-relative motion (axes along
    the closest-approach vector and the relative velocity there, both taken
    from the collocated path).

    Args:
        t (np.ndarray): Times, s.
        y (np.ndarray): Neptune-centred states, (n, 6).
        epoch (system.Epoch): Epoch whose Triton ephemeris is used.
        title (str): Figure title.
        r_min (float, optional): Minimum allowed flyby radius drawn as a circle, km.
            Defaults to None.
        sim (tuple, optional): Simulated (t, y) drawn dashed. Defaults to None.
        guess (tuple, optional): Guess (t, y) drawn dotted. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    tr = epoch.triton
    rel = y[:, :3] - tr.position(t)
    i = int(np.argmin(np.linalg.norm(rel, axis=1)))
    e1 = rel[i] / np.linalg.norm(rel[i])
    vrel = y[i, 3:] - tr.velocity(t[i])
    e2 = vrel - (vrel @ e1) * e1
    e2 /= np.linalg.norm(e2)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    P.disk(ax, system.R_TRITON, P.TRITON_COLOR, 0.5, label="Triton")
    if r_min is not None:
        P.circle(ax, r_min, color=P.LIMIT, linewidth=1, linestyle="--",
                 label="minimum flyby altitude")
    srcs = {"collocated": (t, y), "simulated": sim, "guess": guess}
    for kind, src in srcs.items():
        if src is None:
            continue
        rr = src[1][:, :3] - tr.position(src[0])
        ax.plot(rr @ e1, rr @ e2, color=SERIES[1], label="spacecraft" if kind == "collocated"
                else None, **P.kind_style(kind))
    ax.scatter(rel[i] @ e1, rel[i] @ e2, color=INK, s=18, zorder=5)
    ax.annotate(f"closest approach\n{np.linalg.norm(rel[i]) - system.R_TRITON:.0f} km alt.",
                (rel[i] @ e1, rel[i] @ e2), xytext=(6, -14), textcoords="offset points",
                fontsize=7, color=INK)
    lim = 6 * np.linalg.norm(rel[i])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("along closest-approach vector (km)")
    ax.set_ylabel("along relative velocity at closest approach (km)")
    leg = ax.legend(fontsize=8, loc="upper left")
    ax.add_artist(leg)
    P.style_key(ax, [k for k, v in srcs.items() if v is not None], loc="lower right")
    ax.set_title(title, fontsize=10)
    P.style(ax)
    return P.finish(fig, rect=(0, 0, 1, 1))


# ---------------------------------------------------------------- top level
def _arcs_of_guess(guess):
    """Return ``{phase: (t, y)}`` of a guess."""
    return {k: (a.t, a.y) for k, a in guess.arcs.items()}


def guess_figures(guess, epoch, cfg=None):
    """Return (system view, atmosphere panels) of an initial guess.

    Args:
        guess (guess.Guess): Initial guess.
        epoch (system.Epoch): Epoch (Triton ephemeris).
        cfg (config.Config, optional): Inputs; adds the limit lines. Defaults to None.

    Returns:
        tuple: (system figure, atmosphere figure), with the same panels as the
        solution figures so they compare one to one.
    """
    arcs = _arcs_of_guess(guess)
    fig1 = system_view(arcs, epoch, f"initial guess ({guess.mode})", primary="initial guess")
    ser = _series_from_arc(guess.arcs["atmosphere"], guess.vehicle_params)
    fig2 = atmosphere_panels(None, f"initial guess atmosphere pass ({guess.mode})", guess=ser,
                             limits=limits_of(cfg) if cfg is not None else None)
    return fig1, fig2


def solution_figures(ap, summary=None, guess=None):
    """Return ``{name: figure}`` of a solved problem.

    Collocated curves are solid, simulated dashed and, when ``guess`` is
    given, the initial guess dotted.

    Args:
        ap (problem.AeroProblem): Solved (and simulated) problem.
        summary (dict, optional): ``results.summarize`` output (Triton
            encounter time). Defaults to None.
        guess (guess.Guess, optional): Initial guess to overlay. Defaults to None.

    Returns:
        dict: ``system``, ``atmosphere``, ``corridor`` and, with a Triton
        phase, ``triton``.
    """
    cfg = ap.cfg
    ts = results.timeseries(ap)
    try:
        sim = results.timeseries(ap, sim=True)
        sim = sim if all("x" in d for d in sim.values()) else None
    except Exception:          # no simulation recorded
        sim = None
    st = lambda d: (d["time"], np.stack([d[n] for n in results.STATE_NAMES], 1))
    arcs = {ph: st(d) for ph, d in ts.items()}
    sim_arcs = {ph: st(d) for ph, d in sim.items()} if sim else None
    guess_arcs = _arcs_of_guess(guess) if guess is not None else None
    tag = f"{cfg.triton_mode}, v_inf {cfg.vinf_kms:g} km/s, {cfg.vehicle.name}"
    t_mark = summary["triton"]["t_ca_s"] if summary and summary.get("triton") else None
    params = cfg.vehicle_params()
    for src, get in ((ts, ap.ts), (sim, ap.sim_ts)):
        if src is not None:
            try:
                src["atmosphere"]["sigma_rate"] = get("atmosphere", "sigma_rate")
            except Exception:  # control rate not in the timeseries
                pass
    try:
        q_peak = float(ap.p.get_val("traj.atmosphere.parameter_vals:q_peak")[0]) \
            if cfg.objective == "qpeak" else None
    except Exception:          # parameter not exposed
        q_peak = None
    sets = {"collocated": _series_from_ts(ts["atmosphere"], params),
            "simulated": _series_from_ts(sim["atmosphere"], params) if sim else None,
            "guess": _series_from_arc(guess.arcs["atmosphere"], params) if guess else None}
    figs = {"system": system_view(arcs, ap.epoch, f"trajectory ({tag})", sim_arcs, t_mark,
                                  guess_arcs),
            "atmosphere": atmosphere_panels(sets["collocated"], f"atmosphere pass ({tag})",
                                            sets["simulated"], sets["guess"], params,
                                            limits_of(cfg, q_peak)),
            "corridor": corridor_figure(sets, f"altitude-velocity corridor ({tag})", cfg)}
    for ph in ap.phase_names:
        if ph.startswith("triton"):
            figs["triton"] = triton_closeup(
                *arcs[ph], ap.epoch, f"Triton flyby, {ph} ({tag})",
                system.R_TRITON + cfg.triton_alt_min_km,
                sim=sim_arcs.get(ph) if sim_arcs else None,
                guess=guess_arcs.get(ph) if guess_arcs else None)
    return figs


# ------------------------------------------------------------ sweeps, trades
def sweep_figure(rows, modes, title, rad_coupling="goulard"):
    """Plot peak total heat rate, heat load and peak g versus v-inf, one line per Triton mode.

    Args:
        rows (list): Sweep table rows (``results.table_row``); unconverged rows are skipped.
        modes (list): Triton modes, one line each.
        title (str): Figure title.
        rad_coupling (str, optional): Radiative coupling named in the heat-rate
            label. Defaults to "goulard".

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    keys = (("peak_q_total_w_cm2", 1.0, f"peak total heat rate, {rad_coupling} (W/cm²)"),
            ("heat_load_j_cm2", 1e-3, "heat load (kJ/cm²)"),
            ("peak_g", 1.0, "peak sensed load (g)"))
    for k, mode in enumerate(modes):
        d = sorted((r for r in rows if r.get("mode") == mode and r.get("converged")),
                   key=lambda r: r["vinf_kms"])
        for ax, (key, sc, _) in zip(axs, keys):
            ax.plot([r["vinf_kms"] for r in d], [r[key] * sc for r in d], "o-",
                    color=SERIES[k], linewidth=2, markersize=4, label=mode)
    for ax, (_, _, lab) in zip(axs, keys):
        ax.set_xlabel("hyperbolic excess speed v_inf (km/s)")
        ax.set_ylabel(lab)
        P.style(ax)
    axs[0].legend(fontsize=8, title="Triton mode", title_fontsize=8)
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.93))


def trade_figure(rows, title):
    """Plot the peak convective and radiative heat rate of every converged trade case.

    Args:
        rows (list): Trade table rows (``results.table_row``).
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ok = [r for r in rows if r.get("converged")]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(ok)), 5.2))
    labels = [f"{r['vehicle']}, {r['mode']}, {r['vinf_kms']:g} km/s\n"
              f"β {r['beta_kg_m2']:g} kg/m², L/D {r['ld']:g}, Rn {r['rn_m']:g} m" for r in ok]
    x = np.arange(len(ok))
    ax.bar(x - 0.27, [r["peak_q_conv_w_cm2"] for r in ok], width=0.27, color=SERIES[0],
           label="convective")
    ax.bar(x, [r["peak_q_rad_coupled_w_cm2"] for r in ok], width=0.27, color=SERIES[1],
           label="radiative, Goulard-coupled")
    ax.bar(x + 0.27, [r["peak_q_total_w_cm2"] for r in ok], width=0.27, color=SERIES[2],
           label="total (selected coupling)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("peak stagnation heat rate (W/cm²)")
    ax.legend(fontsize=8)
    P.style(ax)
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.93))
