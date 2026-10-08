"""Shared matplotlib primitives of the mission figures.

One colour order, one axis style and one line-style convention for every
figure of both missions:

    collocated  solid, 2 px     the solved (collocated) trajectory
    simulated   dashed, 1.2 px  the scipy re-integration of the solution
    guess       dotted, thin    the initial guess, drawn lighter

A quantity keeps its colour from ``SERIES`` in every source, so the line style
alone says which source a curve comes from. ``overlay`` draws one quantity
from up to three sources with that convention and ``style_key`` adds the
figure-level legend of the line styles. Limits are drawn by ``limit_line`` as
a red dashed line with the infeasible side shaded.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LIMIT = SERIES[7]
NEPTUNE_COLOR, TRITON_COLOR = SERIES[0], MUTED

KIND_STYLE = {
    "collocated": dict(linestyle="-", linewidth=2.0, alpha=1.0),
    "simulated": dict(linestyle="--", linewidth=1.2, alpha=1.0),
    "guess": dict(linestyle=":", linewidth=1.3, alpha=0.65),
}
KIND_ORDER = ("collocated", "simulated", "guess")


def style(ax):
    """Apply the shared grid, spine and label colours to an axis."""
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def kind_style(kind):
    """Return the line-style keywords of a source kind (unknown kinds draw as collocated)."""
    return KIND_STYLE.get(kind, KIND_STYLE["collocated"])


# ------------------------------------------------------------------- figures
def panel_grid(n, ncols=4, panel_size=(3.6, 2.5), sharex=True, title=None):
    """Return a figure with ``n`` styled panels on a grid and the unused cells.

    Args:
        n (int): Number of panels wanted.
        ncols (int, optional): Columns of the grid. Defaults to 4.
        panel_size (tuple, optional): (width, height) of one panel, inches.
            Defaults to (3.6, 2.5).
        sharex (bool, optional): Share the x axis between panels. Defaults to True.
        title (str, optional): Figure title. Defaults to None.

    Returns:
        tuple: ``(fig, axes, spare)``: the figure, the ``n`` panels in row-major
        order and the leftover axes (switched off, free for a legend).
    """
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols,
                                                   panel_size[1] * nrows + 0.6),
                            sharex=sharex, squeeze=False)
    flat = list(axs.ravel())
    for ax in flat:
        style(ax)
    spare = flat[n:]
    for ax in spare:
        ax.axis("off")
    if sharex:
        for c in range(ncols):
            col = [axs[r, c] for r in range(nrows) if r * ncols + c < n]
            if col:
                col[-1].xaxis.set_tick_params(labelbottom=True)
    if title:
        fig.suptitle(title, color=INK)
    return fig, flat[:n], spare


def finish(fig, rect=(0, 0, 1, 0.97)):
    """Apply ``tight_layout`` within ``rect`` (room for the suptitle) and return the figure."""
    fig.tight_layout(rect=rect)
    return fig


def save_all(figs, out_dir, close=False):
    """Save each figure as ``<name>.png`` in ``out_dir`` and return the paths.

    Args:
        figs (dict): {name: matplotlib.figure.Figure}.
        out_dir (str or Path): Output directory, created if missing.
        close (bool, optional): Close each figure after saving. Defaults to False.

    Returns:
        list: The written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=130)
        paths.append(path)
        if close:
            plt.close(fig)
    return paths


# ------------------------------------------------------------------- overlays
def overlay(ax, sets, xkey, ykey, color, label=None, scale=1.0, mask=None):
    """Draw one quantity from every available source with the shared line styles.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        sets (dict): {kind: {name: array} or None}; kinds in ``KIND_STYLE`` get
            their style, any other kind is drawn solid. A None source, or one
            missing ``xkey`` or ``ykey``, is skipped.
        xkey (str): Name of the x series.
        ykey (str): Name of the y series.
        color (str): Line colour (the same for every source).
        label (str, optional): Legend label, attached to the first drawn line
            only. Defaults to None.
        scale (float, optional): Factor applied to y. Defaults to 1.0.
        mask (callable, optional): ``mask(y)`` returning y with unplottable
            values replaced (e.g. ``np.maximum(y, 1e-8)`` on log axes).
            Defaults to None.

    Returns:
        bool: True if at least one source was drawn.
    """
    drawn = False
    for kind, d in sets.items():
        if d is None or d.get(xkey) is None or d.get(ykey) is None:
            continue
        y = np.asarray(d[ykey], dtype=float) * scale
        if mask is not None:
            y = mask(y)
        ax.plot(d[xkey], y, color=color, label=None if drawn else label, **kind_style(kind))
        drawn = True
    return drawn


def style_key(fig_or_ax, kinds, loc="upper right", **kw):
    """Add the legend of the line styles (collocated / simulated / guess) present.

    Args:
        fig_or_ax (Figure or Axes): Where the legend goes.
        kinds (iterable): Source kinds drawn, in any order.
        loc (str, optional): Legend location. Defaults to "upper right".
        **kw: Extra ``legend`` keywords.

    Returns:
        matplotlib.legend.Legend or None: The legend (None when nothing was drawn).
    """
    kinds = list(kinds)
    order = [k for k in KIND_ORDER if k in kinds] + [k for k in kinds if k not in KIND_ORDER]
    if not order:
        return None
    labels = {"collocated": "collocated (solution)", "simulated": "simulated (scipy)",
              "guess": "initial guess"}
    handles = [Line2D([], [], color=INK, **kind_style(k)) for k in order]
    kw = dict(dict(fontsize=8, frameon=True, framealpha=0.85, edgecolor="none"), **kw)
    return fig_or_ax.legend(handles, [labels.get(k, k) for k in order], loc=loc, **kw)


def present(sets):
    """Return the kinds of ``sets`` that hold data."""
    return [k for k, d in sets.items() if d is not None]


def limit_line(ax, value, label, side="upper", color=LIMIT, show="auto"):
    """Draw a horizontal limit with its label and shade the infeasible side.

    Call it after the data are plotted: the y range may be widened to show the
    limit, and the shading then fills to the axis edge. A limit too far from
    the data to widen the range is written as an "off scale" note instead.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        value (float or None): Limit value in the axis units; None draws nothing.
        label (str): Text written next to the line.
        side (str, optional): ``"upper"`` (values above are infeasible) or
            ``"lower"``. Defaults to "upper".
        color (str, optional): Line and shading colour. Defaults to ``LIMIT``.
        show (bool or str, optional): True always widens the y range to the
            limit, False never does, ``"auto"`` widens it when the limit lies
            within twice the data span (a factor 3 on log axes). Defaults to "auto".
    """
    if value is None or not np.isfinite(value):
        return
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    inside = lo <= value <= hi
    if show == "auto" and not inside:
        if log:
            show = lo / 3.0 <= value <= hi * 3.0
        else:
            span = hi - lo
            show = lo - 2 * span <= value <= hi + 2 * span
    if not show and not inside:
        if show is False:
            return
        above = value > hi
        ax.annotate(f"{label} (off scale)", xy=(1.0, 1.0 if above else 0.0),
                    xycoords="axes fraction", xytext=(-3, -3 if above else 3),
                    textcoords="offset points", ha="right", va="top" if above else "bottom",
                    fontsize=7, color=color)
        return
    if log:
        lo, hi = min(lo, value / 1.5), max(hi, value * 1.5)
    else:
        pad = 0.06 * (max(hi, value) - min(lo, value) or abs(value) or 1.0)
        lo, hi = min(lo, value - pad), max(hi, value + pad)
    ax.axhline(value, color=color, linestyle="--", linewidth=1.0)
    if side == "upper":
        ax.axhspan(value, hi, color=color, alpha=0.08, linewidth=0)
    else:
        ax.axhspan(lo, value, color=color, alpha=0.08, linewidth=0)
    ax.set_ylim(lo, hi)
    ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"),
                xytext=(-3, 2 if side == "upper" else -2), textcoords="offset points",
                ha="right", va="bottom" if side == "upper" else "top", fontsize=7, color=color)


def reference_line(ax, value, label, color=MUTED, x=0.0):
    """Draw a horizontal reference value (not a limit), labelled at axes fraction ``x``."""
    if value is None or not np.isfinite(value):
        return
    ax.axhline(value, color=color, linestyle="-.", linewidth=0.9)
    ha = "left" if x < 0.3 else ("right" if x > 0.7 else "center")
    ax.annotate(label, xy=(x, value), xycoords=("axes fraction", "data"),
                xytext=(3 if ha == "left" else -3 if ha == "right" else 0, 2),
                textcoords="offset points", ha=ha, va="bottom", fontsize=7, color=color)


def event_line(ax, x, label=None, color=MUTED):
    """Draw a vertical event marker (release, closest approach) with an optional label."""
    if x is None:
        return
    ax.axvline(x, color=color, linestyle="--", linewidth=0.9)
    if label:
        ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=7, color=color)


# ----------------------------------------------------------------- geometry
def sphere(ax, radius, color, alpha=0.35, center=(0.0, 0.0, 0.0)):
    """Draw a translucent sphere on a 3-D axis."""
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    c = np.asarray(center, dtype=float)
    ax.plot_surface(c[0] + radius * np.cos(u) * np.sin(v), c[1] + radius * np.sin(u) * np.sin(v),
                    c[2] + radius * np.cos(v), color=color, alpha=alpha, linewidth=0)


_sphere = sphere


def circle(ax, radius, center=(0.0, 0.0), **kw):
    """Draw a circle outline of ``radius`` about ``center`` on a 2-D axis."""
    th = np.linspace(0, 2 * math.pi, 300)
    ax.plot(center[0] + radius * np.cos(th), center[1] + radius * np.sin(th), **kw)


def disk(ax, radius, color, alpha=0.35, center=(0.0, 0.0), label=None):
    """Draw a filled disk of ``radius`` about ``center`` on a 2-D axis."""
    th = np.linspace(0, 2 * math.pi, 200)
    ax.fill(center[0] + radius * np.cos(th), center[1] + radius * np.sin(th), color=color,
            alpha=alpha, linewidth=0, label=label)


def annulus(ax, r_in, r_out, color, alpha=0.12, label=None):
    """Shade the ring between ``r_in`` and ``r_out`` about the origin."""
    th = np.linspace(0, 2 * math.pi, 300)
    ax.fill(np.r_[r_out * np.cos(th), r_in * np.cos(th[::-1])],
            np.r_[r_out * np.sin(th), r_in * np.sin(th[::-1])], color=color, alpha=alpha,
            linewidth=0, label=label)


def _extent(points, ndim):
    """Return (centre, half-width) of the bounding box of a list of (n, ndim) arrays."""
    pts = [np.asarray(p, dtype=float).reshape(-1, ndim) for p in points if p is not None]
    pts = np.vstack([p[np.all(np.isfinite(p), axis=1)] for p in pts])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return 0.5 * (lo + hi), 0.5 * float(np.max(hi - lo))


def equal_2d(ax, points, pad=0.08, min_half=None):
    """Set equal-aspect limits on a 2-D axis that enclose every point.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        points (list): Arrays of shape (n, 2), any of them None.
        pad (float, optional): Margin as a fraction of the half-width. Defaults to 0.08.
        min_half (float, optional): Smallest half-width, in data units. Defaults to None.
    """
    c, h = _extent(points, 2)
    h = max(h * (1 + pad), min_half or 0.0)
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_aspect("equal")


def equal_3d(ax, points, pad=0.05):
    """Set equal-aspect cubic limits on a 3-D axis that enclose every point.

    Args:
        ax (mpl_toolkits.mplot3d.Axes3D): Target axis.
        points (list): Arrays of shape (n, 3), any of them None.
        pad (float, optional): Margin as a fraction of the half-width. Defaults to 0.05.
    """
    c, h = _extent(points, 3)
    h *= 1 + pad
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    ax.set_box_aspect((1, 1, 1))


def two_body_path(r, v, mu, n=600, to_apoapsis=False, r_max=None):
    """Return positions along the osculating two-body conic of one state, for drawing.

    Args:
        r (np.ndarray): Position, km (3,).
        v (np.ndarray): Velocity, km/s (3,).
        mu (float): Gravitational parameter, km^3/s^2.
        n (int, optional): Samples. Defaults to 600.
        to_apoapsis (bool, optional): Stop at the next apoapsis instead of
            closing one revolution. Defaults to False.
        r_max (float, optional): Largest radius drawn on an unbound conic, km.
            Defaults to 10 times the starting radius.

    Returns:
        tuple: ``(path, apoapsis)``: positions (n, 3) in km from the state
        onward, and the apoapsis position (3,), or None when unbound.
    """
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    h = np.cross(r, v)
    hm = np.linalg.norm(h)
    e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
    e = float(np.linalg.norm(e_vec))
    p_ax = e_vec / e if e > 1e-10 else r / np.linalg.norm(r)
    q_ax = np.cross(h / hm, p_ax)
    semi_latus = hm ** 2 / mu
    f0 = math.atan2(r @ q_ax, r @ p_ax) % (2 * math.pi)
    if e < 1.0:
        f1 = math.pi if f0 < math.pi else 3 * math.pi
        if not to_apoapsis:
            f1 = f0 + 2 * math.pi
        apo = -semi_latus / (1 - e) * p_ax
    else:
        r_max = r_max or 10.0 * float(np.linalg.norm(r))
        f_lim = math.acos(max(-1.0, min(1.0, (semi_latus / r_max - 1) / e)))
        f0s = f0 if f0 < math.pi else f0 - 2 * math.pi
        f0, f1 = f0s, max(f0s, f_lim)
        apo = None
    f = np.linspace(f0, f1, n)
    rad = semi_latus / (1 + e * np.cos(f))
    path = rad[:, None] * (np.cos(f)[:, None] * p_ax + np.sin(f)[:, None] * q_ax)
    return path, apo


def unique_times(t, *arrays):
    """Return ``t`` and ``arrays`` with repeated times (segment boundaries) removed."""
    t = np.asarray(t, dtype=float)
    tu, idx = np.unique(t, return_index=True)
    return (tu,) + tuple(np.asarray(a)[idx] for a in arrays)


def rate(t, x):
    """Return dx/dt by finite differences, tolerant of repeated (segment-boundary) times."""
    tu, xu = unique_times(t, x)
    if len(tu) < 2:
        return np.zeros_like(np.asarray(x, dtype=float))
    return np.interp(t, tu, np.gradient(xu, tu))
