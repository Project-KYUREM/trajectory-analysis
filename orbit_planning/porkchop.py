"""Render a static two-panel porkchop PNG from a raw ``pareto.json``.

Example: ``python orbit_planning/porkchop.py --model capture --sequence EVEEJN
--theme dark``. Each marker is one trajectory the search found, ringed when
non-dominated. A multi-flyby trajectory has more free parameters than its two
dates, so many costs share one launch/arrival pair.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import mission
import paths

# Single-hue sequential ramp (blue). The reader hunts for LOW cost, so the
# cheap end must be the salient one: dark on a light surface, light on a dark
# surface. The colourbar label states the direction in force.
_BLUE_DARK_TO_LIGHT = ['#0d366b', '#104281', '#184f95', '#1c5cab', '#256abf',
                       '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef',
                       '#9ec5f4', '#b7d3f6']
CMAPS = {
    'light': LinearSegmentedColormap.from_list('cost_light', _BLUE_DARK_TO_LIGHT),
    'dark': LinearSegmentedColormap.from_list('cost_dark', _BLUE_DARK_TO_LIGHT[::-1]),
}
DIRECTION = {'light': 'darker is cheaper', 'dark': 'brighter is cheaper'}

# Source of truth for these values is docs/style.css (:root tokens); keep in sync.
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#17161c', ink2='#52514e', grid='#e4e3df', accent='#eb6834'),
    'dark': dict(surface='#16161a', ink='#f3f2ef', ink2='#b7b6b0', grid='#34333a', accent='#f0824f'),
}


def _style(theme):
    """Apply a ``THEMES`` entry to matplotlib's rcParams and return its palette."""
    t = THEMES[theme]
    matplotlib.rcParams.update({
        'figure.facecolor': t['surface'], 'axes.facecolor': t['surface'],
        'savefig.facecolor': t['surface'], 'text.color': t['ink'],
        'axes.labelcolor': t['ink2'], 'xtick.color': t['ink2'], 'ytick.color': t['ink2'],
        'axes.edgecolor': t['grid'], 'grid.color': t['grid'], 'font.size': 10,
    })
    return t


def plot(data, theme='light', clip_pct=85.0):
    """Render the porkchop and cost/time-of-flight trade figure for one raw result.

    Args:
        data (dict): A loaded ``pareto.json``.
        theme (str, optional): ``'light'`` or ``'dark'``. Defaults to 'light'.
        clip_pct (float, optional): Percentile at which the colour scale is
            clipped, so the high-cost tail does not flatten the ramp.
            Defaults to 85.0.

    Returns:
        tuple[matplotlib.figure.Figure, dict]: The figure and
        ``{'n_over', 'vmax', 'vmin'}`` describing the colour scale.
    """
    t, cmap = _style(theme), CMAPS[theme]
    pts = data['points']
    cost = np.array([p['cost'] for p in pts])
    tof_yr = np.array([p['tof_days'] for p in pts]) / 365.25
    to_num = lambda key: np.array([mdates.date2num(mission.mjd2000_to_datetime(p[key])) for p in pts])
    launch, arrival = to_num('launch_mjd2000'), to_num('arrival_mjd2000')
    front = np.array([p['pareto'] for p in pts], dtype=bool)
    units, label = data.get('cost_units', 'm/s'), data.get('cost_label', 'cost')

    vmin, vmax = float(cost.min()), float(np.percentile(cost, clip_pct))
    n_over = int((cost > vmax).sum())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, (ax_pc, ax_pf) = plt.subplots(1, 2, figsize=(14.5, 6.2), layout='constrained',
                                       gridspec_kw=dict(width_ratios=[1.35, 1]))

    ax_pc.scatter(launch[~front], arrival[~front], c=cost[~front], cmap=cmap, norm=norm,
                  s=13, linewidths=0, alpha=0.55, zorder=2)
    sc = ax_pc.scatter(launch[front], arrival[front], c=cost[front], cmap=cmap, norm=norm,
                       s=52, linewidths=1.6, edgecolors=t['accent'], zorder=3)
    ax_pc.set_title(f"{data['sequence']} launch/arrival porkchop", color=t['ink'], fontsize=12,
                    pad=10, loc='left')
    ax_pc.set_xlabel('Launch date')
    ax_pc.set_ylabel(f"{data['labels'][-1]} arrival date")
    for axis in (ax_pc.xaxis, ax_pc.yaxis):   # spans differ by 10x; let each adapt
        loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
        axis.set_major_locator(loc)
        axis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    cb = fig.colorbar(sc, ax=ax_pc, pad=0.02, fraction=0.045, aspect=32,
                      extend='max' if n_over else 'neither', extendrect=True, extendfrac=0.025)
    cb.set_label(f"{label} ({units})  —  {DIRECTION[theme]}", color=t['ink2'])
    cb.outline.set_edgecolor(t['grid'])

    ax_pf.scatter(tof_yr[~front], cost[~front], c=cost[~front], cmap=cmap, norm=norm,
                  s=13, linewidths=0, alpha=0.45, zorder=2)
    order = np.argsort(tof_yr[front])
    fx, fy = tof_yr[front][order], cost[front][order]
    gaps = np.diff(fx)   # never draw a segment across a real hole in the front
    breaks = np.where(gaps > max(4 * np.median(gaps), 0.75))[0] if len(gaps) else []
    for seg_x, seg_y in zip(np.split(fx, breaks + 1), np.split(fy, breaks + 1)):
        if len(seg_x) > 1:
            ax_pf.plot(seg_x, seg_y, color=t['accent'], lw=2, zorder=3, solid_capstyle='round')
    ax_pf.scatter(tof_yr[front], cost[front], c=cost[front], cmap=cmap, norm=norm,
                  s=52, linewidths=1.6, edgecolors=t['accent'], zorder=4)
    ax_pf.set_title('Cost / time-of-flight trade', color=t['ink'], fontsize=12, pad=10, loc='left')
    ax_pf.set_xlabel('Total time of flight (years)')
    ax_pf.set_ylabel(f"{label} ({units})")
    ax_pf.set_ylim(vmin * 0.92, vmax * 1.04)

    for ax in (ax_pc, ax_pf):
        ax.grid(True, lw=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=cmap(0.25),
                      markeredgecolor=t['accent'], markeredgewidth=1.6, label='Pareto-optimal'),
               Line2D([], [], marker='o', ls='', markersize=5.5, markerfacecolor=cmap(0.7),
                      markeredgecolor='none', label='Dominated')]
    ax_pf.legend(handles=handles, frameon=False, loc='upper right', labelcolor=t['ink2'])
    fig.suptitle(' → '.join(data['labels']), color=t['ink2'], fontsize=10, ha='left', x=0.01)
    if n_over:
        fig.supxlabel(f"Colour scale clipped at the {clip_pct:.0f}th percentile ({vmax:,.0f} {units}); "
                      f"{n_over} of {len(cost)} points exceed it and are drawn in the end colour.",
                      color=t['ink2'], fontsize=8.5, ha='left', x=0.01)
    return fig, dict(n_over=n_over, vmax=vmax, vmin=vmin)


def front_table(data, rows=10):
    """Format a sampled text table of the front.

    Args:
        data (dict): A loaded ``pareto.json``.
        rows (int, optional): Approximate number of rows, spread across the
            front. Defaults to 10.

    Returns:
        str: Multi-line table of time of flight, cost, launch and arrival dates.
    """
    front = sorted((p for p in data['points'] if p['pareto']), key=lambda p: p['tof_days'])
    step = max(1, len(front) // rows)
    lines = [f"{'TOF (yr)':>9}  {'cost':>10}  {'Launch':>12}  {'Arrival':>12}"]
    for p in front[::step]:
        lines.append(f"{p['tof_days'] / 365.25:9.2f}  {p['cost']:10.0f}  "
                     f"{mission.mjd2000_to_datetime(p['launch_mjd2000']):%Y-%m-%d}  "
                     f"{mission.mjd2000_to_datetime(p['arrival_mjd2000']):%Y-%m-%d}")
    return '\n'.join(lines)


def main():
    """Render one run's porkchop PNG and print its front table."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=paths.DEFAULT_MODELS[0])
    parser.add_argument('--sequence', default=paths.DEFAULT_SEQUENCES[0])
    parser.add_argument('--data', type=Path, default=None, help='explicit pareto.json path')
    parser.add_argument('--theme', choices=tuple(THEMES), default='light')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    data_path = args.data or paths.results_path(args.model, args.sequence.upper(), 'pareto.json')
    if not data_path.is_file():
        raise FileNotFoundError(f"no Pareto data at {data_path}; run pareto.py first")
    data = json.loads(data_path.read_text())
    fig, _ = plot(data, theme=args.theme)
    out = args.out or data_path.with_name(f"porkchop_{args.theme}.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}\n\n{front_table(data)}")


if __name__ == '__main__':
    main()
