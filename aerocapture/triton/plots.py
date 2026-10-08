"""Figures of the Triton capture guess and solution (matplotlib).

Every figure function returns its figure; ``save_all`` writes PNGs and closes
the figures. Colours, axis style and the collocated / simulated / guess line
styles come from ``aerocapture.core.plotting``.

    system_view       Neptune system on Triton's orbit plane and in 3-D,
                      auto-scaled to the whole trajectory and Triton's orbit.
    neptune_flyby     the Neptune flyby from Neptune's pole and in the
                      meridian plane, with the rings, the keep-out band and
                      the ring-plane crossings.
    triton_view       Triton-frame approach, pass, capture coast and final orbit.
    pass_panels       altitude, airspeed, heat-rate terms, film temperature,
                      g-load, Kn, C_D, ballistic coefficient, dynamic
                      pressure, density and flight-path angle through the pass.
    corridor_figure   altitude versus airspeed through the pass.
    stability_figure  distance, osculating rp / ra and i_T over the stability run.
"""
import matplotlib.pyplot as plt
import numpy as np

from aerocapture.core import dynamics
from aerocapture.core import plotting as P
from aerocapture.core import system
from aerocapture.triton import frame as tf
from aerocapture.triton import ode

SERIES, INK, MUTED = P.SERIES, P.INK, P.MUTED
PHASE_COLOR = {"approach": SERIES[0], "flyby_in": SERIES[1], "flyby_out": SERIES[2],
               "cruise": SERIES[3], "triton_approach": SERIES[4], "ballute": SERIES[7],
               "probe": SERIES[5], "capture_coast": SERIES[6], "final": "#8a5a2b"}
PHASE_ORDER = list(PHASE_COLOR)
RINGS_KM = {"Galle": 41900.0, "Le Verrier": 53200.0, "Arago": 57200.0, "Adams": 62930.0}
TRITON_PHASES = ("triton_approach", "ballute", "probe", "capture_coast", "final")
PASS_PHASES = ("ballute", "probe")
KM_SCALE = 1e3          # Neptune-system views plot in 1000 km

_circle, _disk = P.circle, P.disk


def arcs_of(tp, sim=False):
    """Return ``{phase: (t, y)}`` of a solved problem (NEQ or TC states per phase)."""
    get = tp.sim_ts if sim else tp.ts
    out = {}
    for ph in tp.phase_names:
        cols = [get(ph, n) for n in tp._states(ph)]
        if any(c is None for c in cols):
            return None
        out[ph] = (get(ph, "time"), np.stack(cols, axis=1))
    return out


def _neq(ph, t, y, tr):
    """Return NEQ positions of a phase (Triton-frame phases are shifted by Triton's position)."""
    return y[:, :3] + tr.position(t) if ph in TRITON_PHASES else y[:, :3]


def ring_plane_crossings(arcs):
    """Return the NEQ positions (k, 3) where the Neptune phases cross Neptune's equator."""
    pts = []
    for ph, (t, y) in arcs.items():
        if ph in TRITON_PHASES:
            continue
        t, y = P.unique_times(t, y)
        z = y[:, 2]
        for i in np.where(np.sign(z[:-1]) * np.sign(z[1:]) < 0)[0]:
            f = z[i] / (z[i] - z[i + 1])
            pts.append(y[i, :3] + f * (y[i + 1, :3] - y[i, :3]))
    return np.array(pts).reshape(-1, 3)


def _sources(arcs, sim_arcs, guess_arcs, primary="collocated"):
    """Return the non-empty ``{kind: arcs}`` of the three sources."""
    out = {primary: arcs, "simulated": sim_arcs, "guess": guess_arcs}
    rank = lambda ph: PHASE_ORDER.index(ph) if ph in PHASE_ORDER else len(PHASE_ORDER)
    return {k: dict(sorted(v.items(), key=lambda kv: rank(kv[0]))) for k, v in out.items() if v}


# ------------------------------------------------------------ Neptune system
def system_view(arcs, epoch, cfg, title, sim_arcs=None, guess_arcs=None, primary="collocated"):
    """Plot the whole trajectory with Triton's orbit on Triton's orbital plane and in 3-D.

    Both panels are auto-scaled to every phase and Triton's orbit; the
    Triton-frame phases are shifted into NEQ by Triton's ephemeris.

    Args:
        arcs (dict): {phase: (t, y)}; Triton-phase states are Triton-centred.
        epoch (system.Epoch): Epoch (Triton ephemeris).
        cfg (config.Config): Inputs (unused; kept for a uniform signature).
        title (str): Figure title.
        sim_arcs (dict, optional): Simulated arcs drawn dashed. Defaults to None.
        guess_arcs (dict, optional): Guess arcs drawn dotted. Defaults to None.
        primary (str, optional): Source kind of ``arcs`` in the line-style key.
            Defaults to "collocated".

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    tr = epoch.triton
    p, q = np.asarray(tr.p), np.asarray(tr.q)
    s = KM_SCALE
    srcs = _sources(arcs, sim_arcs, guess_arcs, primary)
    fig = plt.figure(figsize=(14, 6.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    plane = lambda r: (r @ p / s, r @ q / s)
    pts = []
    for kind, src in srcs.items():
        for ph, (t, y) in src.items():
            r = _neq(ph, t, y, tr)
            pts.append(r)
            c = PHASE_COLOR.get(ph, INK)
            ax1.plot(*plane(r), color=c, label=ph if kind == primary else None,
                     **P.kind_style(kind))
            ax3.plot(*(r.T / s), color=c, **P.kind_style(kind))
    torb = tr.position(np.linspace(0, tr.period, 400))
    pts.append(torb)
    ax1.plot(*plane(torb), color=MUTED, linewidth=1, label="Triton orbit")
    ax3.plot(*(torb.T / s), color=MUTED, linewidth=1)
    P.disk(ax1, system.R_NEPTUNE / s, P.NEPTUNE_COLOR)
    P.sphere(ax3, system.R_NEPTUNE / s, P.NEPTUNE_COLOR)
    if "triton_approach" in arcs:
        t_enc = arcs["triton_approach"][0][-1]
        rt = tr.position(np.array([t_enc]))[0]
        ax1.scatter(*plane(rt), color=INK, s=25, zorder=5, label="Triton at entry")
        ax3.scatter(*(rt / s), color=INK, s=25)
    P.equal_2d(ax1, [np.stack(plane(r), 1) for r in pts])
    P.equal_3d(ax3, [r / s for r in pts])
    ax1.set_title("projected on Triton's orbital plane", fontsize=9)
    ax1.set_xlabel("along Triton p (1000 km)")
    ax1.set_ylabel("along Triton q (1000 km)")
    P.style(ax1)
    ax1.legend(fontsize=7, loc="upper left", ncol=2)
    ax3.set_xlabel("x NEQ (1000 km)", fontsize=8)
    ax3.set_ylabel("y NEQ (1000 km)", fontsize=8)
    ax3.set_zlabel("z = Neptune pole (1000 km)", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.set_title("NEQ frame", fontsize=9)
    P.style_key(fig, list(srcs), loc="lower right")
    fig.suptitle(title, color=INK)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.1, top=0.9, wspace=0.15)
    return fig


def neptune_flyby(arcs, cfg, title, sim_arcs=None, guess_arcs=None, primary="collocated"):
    """Plot the Neptune flyby against the rings: pole view and meridian (cylindrical) view.

    Args:
        arcs (dict): {phase: (t, y)}; only the Neptune (NEQ) phases are drawn.
        cfg (config.Config): Inputs (ring keep-out band, periapsis band).
        title (str): Figure title.
        sim_arcs (dict, optional): Simulated arcs drawn dashed. Defaults to None.
        guess_arcs (dict, optional): Guess arcs drawn dotted. Defaults to None.
        primary (str, optional): Source kind of ``arcs`` in the line-style key.
            Defaults to "collocated".

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    s = KM_SCALE
    r_in, r_out = cfg.ring_keepout_km
    srcs = _sources(arcs, sim_arcs, guess_arcs, primary)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    P.annulus(ax1, r_in / s, r_out / s, P.LIMIT, label="ring keep-out band")
    ax2.axvspan(r_in / s, r_out / s, color=P.LIMIT, alpha=0.12, linewidth=0)
    for name, rr in RINGS_KM.items():
        P.circle(ax1, rr / s, color=MUTED, linewidth=0.6)
        ax2.plot([rr / s, rr / s], [-0.02 * r_out / s, 0.02 * r_out / s], color=MUTED,
                 linewidth=1.5)
        ax2.annotate(name, (rr / s, 0.02 * r_out / s), rotation=90, fontsize=6, color=MUTED,
                     ha="center", va="bottom")
    P.disk(ax1, system.R_NEPTUNE / s, P.NEPTUNE_COLOR)
    th = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ax2.fill(system.R_NEPTUNE / s * np.cos(th), system.R_NEPTUNE / s * np.sin(th),
             color=P.NEPTUNE_COLOR, alpha=0.35, linewidth=0)
    for kind, src in srcs.items():
        for ph, (t, y) in src.items():
            if ph in TRITON_PHASES:
                continue
            c = PHASE_COLOR.get(ph, INK)
            ax1.plot(y[:, 0] / s, y[:, 1] / s, color=c, label=ph if kind == primary else None,
                     **P.kind_style(kind))
            ax2.plot(np.hypot(y[:, 0], y[:, 1]) / s, y[:, 2] / s, color=c, **P.kind_style(kind))
    for kind, src in srcs.items():
        cr = ring_plane_crossings(src)
        if not len(cr):
            continue
        rc = np.hypot(cr[:, 0], cr[:, 1])
        main = kind == primary
        ax1.scatter(cr[:, 0] / s, cr[:, 1] / s, color=INK, s=28 if main else 12, zorder=6,
                    marker="o" if main else "x",
                    label="ring-plane crossing" if main else None)
        ax2.scatter(rc / s, np.zeros_like(rc), color=INK, s=28 if main else 12, zorder=6,
                    marker="o" if main else "x")
        if main:
            for x, y_, r in zip(cr[:, 0] / s, cr[:, 1] / s, rc):
                ax1.annotate(f"{r / 1e3:.1f}k km", (x, y_), xytext=(5, 5),
                             textcoords="offset points", fontsize=7, color=INK)
    lim = 1.3 * r_out / s
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.set_title("from Neptune's pole (dots: equator crossings)", fontsize=9)
    ax1.set_xlabel("x NEQ (1000 km)")
    ax1.set_ylabel("y NEQ (1000 km)")
    ax1.legend(fontsize=7, loc="upper left")
    ax2.axhline(0.0, color=MUTED, linewidth=0.6)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(-0.5 * lim, 0.5 * lim)
    ax2.set_aspect("equal")
    ax2.set_title("meridian view: distance from the pole axis vs height above the ring plane",
                  fontsize=9)
    ax2.set_xlabel("cylindrical radius √(x²+y²) (1000 km)")
    ax2.set_ylabel("z above Neptune's equator (1000 km)")
    for a in (ax1, ax2):
        P.style(a)
    P.style_key(ax2, list(srcs), loc="upper right")
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------- Triton frame
def triton_view(arcs, epoch, cfg, title, sim_arcs=None, guess_arcs=None, primary="collocated"):
    """Plot the Triton-frame phases on Triton's orbital plane: whole capture and pass close-up.

    Args:
        arcs (dict): {phase: (t, y)}; Triton-phase states are Triton-centred.
        epoch (system.Epoch): Epoch (Triton ephemeris).
        cfg (config.Config): Inputs (interface, rp floor, ra limit).
        title (str): Figure title.
        sim_arcs (dict, optional): Simulated arcs drawn dashed. Defaults to None.
        guess_arcs (dict, optional): Guess arcs drawn dotted. Defaults to None.
        primary (str, optional): Source kind of ``arcs`` in the line-style key.
            Defaults to "collocated".

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    tr = epoch.triton
    p, q = np.asarray(tr.p), np.asarray(tr.q)
    srcs = _sources(arcs, sim_arcs, guess_arcs, primary)
    ra_max = cfg.ra_max_km(tr)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    for k, (ax, lim) in enumerate(zip(axs, (1.15 * ra_max,
                                            system.R_TRITON + 1.6 * cfg.interface_alt_km))):
        P.disk(ax, system.R_TRITON, P.TRITON_COLOR, 0.5, label="Triton" if k == 0 else None)
        P.circle(ax, cfg.r_interface, color=MUTED, linewidth=0.8, linestyle=":",
                 label="entry interface" if k == 0 else None)
        P.circle(ax, cfg.rp_final, color=P.LIMIT, linewidth=0.8, linestyle="-.",
                 label="final rp floor" if k == 0 else None)
        P.circle(ax, ra_max, color=P.LIMIT, linewidth=0.8, linestyle="--",
                 label="final ra limit" if k == 0 else None)
        for kind, src in srcs.items():
            for ph, (t, y) in src.items():
                if ph not in TRITON_PHASES:
                    continue
                ax.plot(y[:, :3] @ p, y[:, :3] @ q, color=PHASE_COLOR[ph],
                        label=ph if kind == primary and k == 0 else None, **P.kind_style(kind))
        for ph, mark in (("ballute", "release"), ("capture_coast", "burn")):
            if ph in arcs and (ph != "ballute" or "probe" in arcs):
                y = arcs[ph][1][-1]
                ax.scatter(y[:3] @ p, y[:3] @ q, color=INK, s=30, zorder=5)
                ax.annotate(mark, (y[:3] @ p, y[:3] @ q), xytext=(4, 4),
                            textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("along Triton p (km)")
        ax.set_ylabel("along Triton q (km)")
        P.style(ax)
    axs[0].legend(fontsize=7, loc="upper left", ncol=2)
    axs[0].set_title("Triton frame: approach, capture and final orbit", fontsize=9)
    axs[1].set_title("atmosphere pass close-up", fontsize=9)
    P.style_key(axs[1], list(srcs), loc="upper right")
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------- pass
def pass_series(parts, cfg, triton, atm):
    """Return the concatenated pass diagnostics of the ballute and probe arcs of one source.

    Args:
        parts (dict): {``"ballute"`` / ``"probe"``: (t, y)} with TC states.
        cfg (config.Config): Inputs (vehicle and heating options).
        triton (system.TritonOrbit): Ephemeris (spin axis).
        atm (atmosphere.Atmosphere): Triton atmosphere.

    Returns:
        dict or None: ``ode.aero`` diagnostics plus ``time``, ``t`` (s from the
        first sample), ``alt`` (km), ``rho`` (kg/m^3), ``fpa_rel`` (deg) and
        ``q_dyn`` (Pa), and ``t_release`` (s from the first sample, None
        without a probe phase); None when no pass arc is given.
    """
    chunks = []
    for ph in PASS_PHASES:
        if ph not in parts or parts[ph] is None:
            continue
        t, y = parts[ph]
        r, v = y[:, :3], y[:, 3:6]
        rm = np.linalg.norm(r, axis=1)
        rho = atm.density(rm - system.R_TRITON)
        opts = cfg.eom_options(ph)
        d = ode.aero(r, v, rho, cfg.pass_params(ph), triton, opts["rotating"],
                     opts["conv_model"], opts["bridging"], opts["fm_accommodation"],
                     opts["rad_in_limits"], opts["emissivity"], opts["sides"])
        vrel = dynamics.relative_velocity(r, v, tf.omega_vec(triton, cfg.atm_rotation))
        d = {k: np.asarray(val) for k, val in d.items() if k != "u"}
        d.update(time=np.asarray(t, dtype=float), alt=rm - system.R_TRITON, rho=rho,
                 fpa_rel=np.degrees(np.arcsin(np.sum(r * vrel, axis=1) / (rm * d["v_rel"]))))
        chunks.append((ph, d))
    if not chunks:
        return None
    out = {k: np.concatenate([d[k] for _, d in chunks]) for k in chunks[0][1]}
    out["t"] = out["time"] - out["time"][0]
    out["t_release"] = (chunks[0][1]["time"][-1] - out["time"][0]
                        if len(chunks) > 1 else None)
    return out


def _as_series(ts):
    """Return a pass-diagnostics dict with the relative time ``t`` (added when missing)."""
    if ts is None:
        return None
    if "t" not in ts:
        ts = dict(ts)
        ts["t"] = np.asarray(ts["time"]) - ts["time"][0]
    return ts


def limits_of(cfg):
    """Return the pass limits drawn on the panels: film T, g, minimum altitude, heat rate."""
    return dict(t_film_max=cfg.ballute.t_film_max_k, g_max=cfg.g_max, h_min=cfg.h_min_km,
                q_max=cfg.q_max_w_cm2)


def pass_panels(ts, title, t_release=None, sim=None, guess=None, limits=None, t_window=None):
    """Plot the pass diagnostics versus time from the interface.

    Args:
        ts (dict): Ballute + probe diagnostics (``pass_series`` output, or the
            ``ode.aero`` names plus ``time`` and ``alt``).
        title (str): Figure title.
        t_release (float, optional): Release time from entry, s. Defaults to
            ``ts["t_release"]`` when present.
        sim (dict, optional): Simulated diagnostics, drawn dashed. Defaults to None.
        guess (dict, optional): Guess diagnostics, drawn dotted. Defaults to None.
        limits (dict, optional): ``limits_of(cfg)``. Defaults to None (no limits).
        t_window (tuple, optional): (start, end) time range shown, s from the
            interface. Defaults to None (the whole pass).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    sets = {"collocated": _as_series(ts), "simulated": _as_series(sim),
            "guess": _as_series(guess)}
    if sets["collocated"] is None:
        sets = {"initial guess": sets["guess"]}
    main = next(v for v in sets.values() if v is not None)
    if t_release is None:
        t_release = main.get("t_release")
    lim = limits or {}
    fig, axs, spare = P.panel_grid(11, ncols=4, title=title)
    pos = lambda y: np.maximum(y, 1e-12)
    ov = lambda ax, key, color=SERIES[0], label=None, **kw: P.overlay(ax, sets, "t", key, color,
                                                                         label, **kw)
    ov(axs[0], "alt")
    axs[0].set_ylabel("altitude (km)")
    P.limit_line(axs[0], lim.get("h_min"), f"h_min {lim.get('h_min')} km", side="lower")
    ov(axs[1], "v_rel")
    axs[1].set_ylabel("airspeed (km/s)")
    ax = axs[2]
    for k, (name, lab) in enumerate((("q_conv", "continuum (SG N2)"), ("q_fm", "free-molecular"),
                                     ("q_bridged", "bridged"),
                                     ("q_rad", "Tauber-Sutton bound"))):
        ov(ax, name, SERIES[k], lab, mask=lambda y: np.maximum(y, 1e-8))
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_ylabel("stagnation heat rate (W/cm²)")
    P.limit_line(ax, lim.get("q_max"), f"q_max {lim.get('q_max')}")
    ax.legend(fontsize=6.5, loc="lower center")
    ov(axs[3], "t_film")
    axs[3].set_ylabel("film temperature (K)")
    P.limit_line(axs[3], lim.get("t_film_max"), f"t_film_max {lim.get('t_film_max'):g} K"
                 if lim.get("t_film_max") else "")
    ov(axs[4], "g_load")
    axs[4].set_ylabel("sensed load (g)")
    P.limit_line(axs[4], lim.get("g_max"), f"g_max {lim.get('g_max')}")
    ov(axs[5], "kn", mask=pos)
    axs[5].set_yscale("log")
    axs[5].set_ylabel("Knudsen number")
    ov(axs[6], "cd")
    axs[6].set_ylabel("drag coefficient")
    ov(axs[7], "beta", mask=pos)
    axs[7].set_yscale("log")
    axs[7].set_ylabel("ballistic coefficient (kg/m²)")
    ov(axs[8], "q_dyn")
    axs[8].set_ylabel("dynamic pressure (Pa)")
    ov(axs[9], "rho", mask=lambda y: np.maximum(y, 1e-30))
    axs[9].set_yscale("log")
    axs[9].set_ylabel("density (kg/m³)")
    ov(axs[10], "fpa_rel")
    axs[10].axhline(0.0, color=MUTED, linewidth=0.6)
    axs[10].set_ylabel("flight-path angle, rel. (deg)")
    for a in axs:
        P.event_line(a, t_release, "release" if a is axs[0] else None)
        if not a.lines:
            a.text(0.5, 0.5, "not available", transform=a.transAxes, ha="center",
                   color=MUTED, fontsize=8)
    for a in axs[-4:]:
        a.set_xlabel("time from interface (s)")
    if t_window is not None:
        axs[0].set_xlim(*t_window)
    P.style_key(spare[0], P.present(sets), loc="center")
    return P.finish(fig)


def corridor_figure(sets, title, cfg=None):
    """Plot altitude versus airspeed through the pass, with the release point marked.

    Args:
        sets (dict): {kind: ``pass_series`` dict or None}.
        title (str): Figure title.
        cfg (config.Config, optional): Inputs; draws the interface and h_min. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    P.overlay(ax, sets, "v_rel", "alt", SERIES[0])
    main = next(v for v in sets.values() if v is not None)
    for k, lab in ((0, "entry"), (int(np.argmin(main["alt"])), "minimum altitude"),
                   (-1, "exit")):
        ax.scatter(main["v_rel"][k], main["alt"][k], color=SERIES[0], s=16, zorder=5)
        ax.annotate(lab, (main["v_rel"][k], main["alt"][k]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=INK)
    if main.get("t_release") is not None:
        i = int(np.searchsorted(main["t"], main["t_release"]))
        i = min(i, len(main["t"]) - 1)
        ax.scatter(main["v_rel"][i], main["alt"][i], color=INK, marker="D", s=22, zorder=6)
        ax.annotate("release", (main["v_rel"][i], main["alt"][i]), xytext=(4, -10),
                    textcoords="offset points", fontsize=7, color=INK)
    if cfg is not None:
        P.reference_line(ax, cfg.interface_alt_km, "entry interface", x=0.5)
        P.limit_line(ax, cfg.h_min_km, f"h_min {cfg.h_min_km:g} km", side="lower")
    ax.set_xlabel("airspeed relative to the atmosphere (km/s)")
    ax.set_ylabel("altitude (km)")
    P.style(ax)
    P.style_key(ax, P.present(sets), loc="center")
    ax.set_title(title, fontsize=10, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 1))


def stability_figure(series, summary, title):
    """Plot distance, osculating rp / ra and i_T over the stability propagation."""
    st = summary["stability"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    d = series["t_days"]
    ax1.plot(d, series["r_km"], color=MUTED, linewidth=0.3, label="distance")
    ax1.plot(d, series["rp_km"], color=SERIES[0], linewidth=1, label="osculating rp")
    ax1.plot(d, series["ra_km"], color=SERIES[1], linewidth=1, label="osculating ra")
    ax1.set_ylim(0, None)
    P.limit_line(ax1, st["floor_km"], f"rp floor {st['floor_km']:.0f} km", side="lower")
    P.limit_line(ax1, st["ra_limit_km"], f"ra limit {st['ra_limit_km']:.0f} km")
    ax1.set_ylabel("radius from Triton's centre (km)")
    ax1.legend(fontsize=7, loc="lower left")
    ax2.plot(d, series["i_t_deg"], color=SERIES[0], linewidth=1)
    ax2.ticklabel_format(axis="y", useOffset=False)
    ax2.set_ylabel("inclination to Triton's orbit plane (deg)")
    for a in (ax1, ax2):
        a.set_xlabel("days after the final phase")
        P.style(a)
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.94))


# ---------------------------------------------------------------- top level
def _pass_ts(tp):
    """Return the collocated pass diagnostics of a solved problem and the release time."""
    ts = pass_series(_pass_parts(arcs_of(tp)), tp.cfg, tp.epoch.triton, tp.atm)
    return ts, ts["t_release"]


def _pass_parts(arcs):
    """Return the ballute and probe arcs of an ``arcs`` dict."""
    return {ph: arcs[ph] for ph in PASS_PHASES if arcs and ph in arcs}


def solution_figures(tp, summary, guess=None):
    """Return ``{name: figure}`` of a solved problem.

    Collocated curves are solid, simulated dashed and, when ``guess`` is
    given, the initial guess dotted.

    Args:
        tp (problem.TritonProblem): Solved (and simulated) problem.
        summary (dict): ``results.summarize`` output (stability series).
        guess (guess.TritonGuess, optional): Initial guess to overlay. Defaults to None.

    Returns:
        dict: ``system`` and ``neptune_flyby`` (full scope only), ``triton``,
        ``pass``, ``pass_ballute`` (with a release), ``corridor`` and ``stability`` (when the summary holds the series).
    """
    arcs = arcs_of(tp)
    try:
        sim = arcs_of(tp, sim=True)
    except Exception:          # no simulation recorded
        sim = None
    garcs = {k: (a.t, a.y) for k, a in guess.arcs.items()} if guess is not None else None
    if garcs is not None:
        garcs = {k: v for k, v in garcs.items() if k in tp.phase_names}
    cfg, tr = tp.cfg, tp.epoch.triton
    tag = f"{tp.scope}, v_inf {cfg.vinf_kms:g} km/s, {cfg.ballute.name}"
    figs = {}
    if tp.full:
        figs["system"] = system_view(arcs, tp.epoch, cfg, f"trajectory ({tag})", sim, garcs)
        figs["neptune_flyby"] = neptune_flyby(arcs, cfg, f"Neptune flyby and rings ({tag})",
                                              sim, garcs)
    figs["triton"] = triton_view(arcs, tp.epoch, cfg, f"Triton capture ({tag})", sim, garcs)
    sets = {kind: pass_series(_pass_parts(src), cfg, tr, tp.atm) if src else None
            for kind, src in (("collocated", arcs), ("simulated", sim), ("guess", garcs))}
    figs["pass"] = pass_panels(sets["collocated"], f"ballute and probe pass ({tag})",
                               sim=sets["simulated"], guess=sets["guess"],
                               limits=limits_of(cfg))
    t_rel = sets["collocated"]["t_release"]
    if t_rel is not None:
        figs["pass_ballute"] = pass_panels(
            sets["collocated"], f"ballute pass, to twice the release time ({tag})",
            sim=sets["simulated"], guess=sets["guess"], limits=limits_of(cfg),
            t_window=(0.0, 2.0 * t_rel))
    figs["corridor"] = corridor_figure(sets, f"altitude-velocity, Triton pass ({tag})", cfg)
    if "_stability_series" in summary:
        figs["stability"] = stability_figure(summary["_stability_series"], summary,
                                             f"{cfg.stability_days:g}-day stability ({tag})")
    return figs


def guess_figures(guess, epoch, cfg, atm):
    """Return ``{name: figure}`` of an initial guess, with the panels of the solution figures."""
    arcs = {k: (a.t, a.y) for k, a in guess.arcs.items()}
    kind = "initial guess"
    figs = {}
    if any(ph not in TRITON_PHASES for ph in arcs):
        figs["guess_system"] = system_view(arcs, epoch, cfg, "initial guess", primary=kind)
        figs["guess_neptune_flyby"] = neptune_flyby(arcs, cfg, "initial guess, Neptune flyby",
                                                    primary=kind)
    figs["guess_triton"] = triton_view(arcs, epoch, cfg, "initial guess, Triton frame",
                                       primary=kind)
    ser = pass_series(_pass_parts(arcs), cfg, epoch.triton, atm)
    figs["guess_pass"] = pass_panels(None, "initial guess pass", guess=ser, limits=limits_of(cfg))
    figs["guess_corridor"] = corridor_figure({kind: ser}, "initial guess, Triton pass", cfg)
    return figs


def save_all(figs, out_dir):
    """Save each figure as ``<name>.png`` in ``out_dir``, close it, and return the paths."""
    return P.save_all(figs, out_dir, close=True)


# ------------------------------------------------------------ sweeps, trades
def sweep_figure(rows, encounters, title):
    """Plot Triton entry speed, peak bridged heat rate and peak film T versus v-inf.

    Args:
        rows (list): Sweep table rows (``results.table_row``); unconverged rows are skipped.
        encounters (list): Encounter senses, one line each.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ok = [r for r in rows if r.get("converged")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    keys = (("entry_speed_kms", "Triton entry speed (km/s)"),
            ("peak_q_bridged_w_cm2", "peak bridged heat rate (W/cm²)"),
            ("peak_t_film_k", "peak film temperature (K)"))
    for k, enc in enumerate(encounters):
        d = sorted((r for r in ok if r.get("encounter") == enc), key=lambda r: r["vinf_kms"])
        for ax, (key, _) in zip(axs, keys):
            ax.plot([r["vinf_kms"] for r in d], [r[key] for r in d], "o-", color=SERIES[k],
                    linewidth=2, markersize=4, label=enc)
    direct = {r["vinf_kms"]: r["direct_entry_speed_kms"] for r in ok
              if r.get("direct_entry_speed_kms") is not None}
    axs[0].plot(sorted(direct), [direct[v] for v in sorted(direct)], "s--",
                color=MUTED, markersize=4, label="direct intercept")
    for ax, (_, lab) in zip(axs, keys):
        ax.set_xlabel("Neptune v_inf (km/s)")
        ax.set_ylabel(lab)
        P.style(ax)
    axs[0].legend(fontsize=8, title="encounter", title_fontsize=8)
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.93))


def trade_figure(rows, title, xkey="diameter_m", xlabel="ballute diameter (m)"):
    """Plot peak bridged heat rate, peak film temperature and peak g against a traded input.

    Args:
        rows (list): Trade table rows (``results.table_row``).
        title (str): Figure title.
        xkey (str, optional): Row key on the x axis. Defaults to "diameter_m".
        xlabel (str, optional): x-axis label with units. Defaults to "ballute diameter (m)".

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ok = sorted((r for r in rows if r.get("converged")), key=lambda r: r[xkey])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, lab in zip(axs, ("peak_q_bridged_w_cm2", "peak_t_film_k", "peak_g"),
                            ("peak bridged heat rate (W/cm²)", "peak film temperature (K)",
                             "peak sensed load (g)")):
        ax.plot([r[xkey] for r in ok], [r[key] for r in ok], "o-", color=SERIES[0],
                linewidth=2, markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lab)
        P.style(ax)
    fig.suptitle(title, color=INK)
    return P.finish(fig, rect=(0, 0, 1, 0.93))
