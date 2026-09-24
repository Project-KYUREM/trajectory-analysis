"""Porkchop + Pareto-front plot for the multi-objective EVEEJN search.

    python orbit_planning/pareto.py          # produces pareto.json
    python orbit_planning/porkchop.py        # renders porkchop.png

This is a *sampled* porkchop, not a contoured one, and the distinction is real.
A classic porkchop contours DV over a departure x arrival grid because a
ballistic two-body transfer is fully determined by those two dates. An MGA-1DSM
trajectory is not: it has 23 free parameters, so many different DVs share the
same pair of dates and no single-valued surface over (launch, arrival) exists.
What is plotted is the population NSGA-II actually found, with the
non-dominated subset marked.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import mission
from pareto import PARETO_PATH

# Single-hue sequential ramp (blue, steps 150->700). The reader is hunting for
# LOW DV, so the cheap end must be the salient one -- which means the ramp is
# oriented against the surface, not flipped between themes by reflex:
#   light surface -> cheap = dark   (dark ink stands out on near-white)
#   dark surface  -> cheap = light  (the same dark navy would disappear)
# The colorbar label states the direction in force, so neither reading is
# left to guess.
_BLUE_DARK_TO_LIGHT = ['#0d366b', '#104281', '#184f95', '#1c5cab', '#256abf',
                       '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef',
                       '#9ec5f4', '#b7d3f6']
CMAPS = {
    'light': LinearSegmentedColormap.from_list('dv_light', _BLUE_DARK_TO_LIGHT),
    'dark': LinearSegmentedColormap.from_list('dv_dark', _BLUE_DARK_TO_LIGHT[::-1]),
}
CB_LABEL = {'light': 'Total ΔV (m/s)  —  darker is cheaper',
            'dark': 'Total ΔV (m/s)  —  brighter is cheaper'}

THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e',
                  grid='#e4e3df', accent='#eb6834'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7',
                 grid='#383835', accent='#d95926'),
}


def _style(theme):
    t = THEMES[theme]
    matplotlib.rcParams.update({
        'figure.facecolor': t['surface'], 'axes.facecolor': t['surface'],
        'savefig.facecolor': t['surface'],
        'text.color': t['ink'], 'axes.labelcolor': t['ink2'],
        'xtick.color': t['ink2'], 'ytick.color': t['ink2'],
        'axes.edgecolor': t['grid'], 'grid.color': t['grid'],
        'font.size': 10,
    })
    return t


def plot(data, theme='light', clip_pct=85.0):
    t = _style(theme)
    cmap = CMAPS[theme]
    pts = data['points']

    dv = np.array([p['dv_ms'] for p in pts])
    tof_yr = np.array([p['tof_days'] for p in pts]) / 365.25
    launch = np.array([mdates.date2num(mission.mjd2000_to_datetime(p['launch_mjd2000'])) for p in pts])
    arrival = np.array([mdates.date2num(mission.mjd2000_to_datetime(p['arrival_mjd2000'])) for p in pts])
    front = np.array([p['pareto'] for p in pts], dtype=bool)

    # The tail of the population runs to absurd DV; clipping the *colour* scale
    # keeps the interesting region legible. Every point is still drawn, and the
    # colourbar carries an arrow to say the scale is clipped.
    vmin = float(dv.min())
    vmax = float(np.percentile(dv, clip_pct))
    n_over = int((dv > vmax).sum())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, (ax_pc, ax_pf) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), layout='constrained',
        gridspec_kw=dict(width_ratios=[1.35, 1]))

    # ---------------------------------------------------------- porkchop ----
    ax_pc.scatter(launch[~front], arrival[~front], c=dv[~front], cmap=cmap,
                  norm=norm, s=13, linewidths=0, alpha=0.55, zorder=2)
    sc = ax_pc.scatter(launch[front], arrival[front], c=dv[front], cmap=cmap,
                       norm=norm, s=52, linewidths=1.6, edgecolors=t['accent'], zorder=3)

    ax_pc.set_title('EVEEJN launch/arrival porkchop', color=t['ink'],
                    fontsize=12, pad=10, loc='left')
    ax_pc.set_xlabel('Launch date')
    ax_pc.set_ylabel('Neptune arrival date')
    # Launch and arrival spans differ by an order of magnitude (a few years vs
    # two decades), so let the locator adapt per axis instead of fixing a base.
    for axis in (ax_pc.xaxis, ax_pc.yaxis):
        loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
        axis.set_major_locator(loc)
        axis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    # extendrect + a small extendfrac: the default triangular cap balloons into
    # a wedge that reads as part of the data range.
    cb = fig.colorbar(sc, ax=ax_pc, pad=0.02, fraction=0.045, aspect=32,
                      extend='max' if n_over else 'neither',
                      extendrect=True, extendfrac=0.025)
    cb.set_label(CB_LABEL[theme], color=t['ink2'])
    cb.outline.set_edgecolor(t['grid'])

    # ------------------------------------------------------ pareto front ----
    ax_pf.scatter(tof_yr[~front], dv[~front], c=dv[~front], cmap=cmap,
                  norm=norm, s=13, linewidths=0, alpha=0.45, zorder=2)
    fx, fy = tof_yr[front], dv[front]
    order = np.argsort(fx)
    fx, fy = fx[order], fy[order]
    # Break the line wherever the front has a real hole, so the plot never
    # implies solutions between two widely separated knees.
    gaps = np.diff(fx)
    brk = np.where(gaps > max(4 * np.median(gaps), 0.75))[0]
    for seg_x, seg_y in zip(np.split(fx, brk + 1), np.split(fy, brk + 1)):
        if len(seg_x) > 1:
            ax_pf.plot(seg_x, seg_y, color=t['accent'], lw=2, zorder=3,
                       solid_capstyle='round')
    ax_pf.scatter(tof_yr[front], dv[front], c=dv[front], cmap=cmap, norm=norm,
                  s=52, linewidths=1.6, edgecolors=t['accent'], zorder=4)

    ax_pf.set_title('ΔV / time-of-flight trade', color=t['ink'],
                    fontsize=12, pad=10, loc='left')
    ax_pf.set_xlabel('Total time of flight (years)')
    ax_pf.set_ylabel('Total ΔV (m/s)')
    ax_pf.set_ylim(vmin * 0.92, vmax * 1.04)

    for ax in (ax_pc, ax_pf):
        ax.grid(True, lw=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=cmap(0.25),
               markeredgecolor=t['accent'], markeredgewidth=1.6, label='Pareto-optimal'),
        Line2D([], [], marker='o', ls='', markersize=5.5, markerfacecolor=cmap(0.7),
               markeredgecolor='none', label='Dominated'),
    ]
    leg = ax_pf.legend(handles=handles, frameon=False, loc='upper right',
                       labelcolor=t['ink2'])
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])

    seq_txt = ' → '.join(data['sequence'])
    fig.suptitle(seq_txt, color=t['ink2'], fontsize=10, ha='left', x=0.01)

    if n_over:
        fig.supxlabel(
            f'Colour scale clipped at the {clip_pct:.0f}th percentile ({vmax:,.0f} m/s); '
            f'{n_over} of {len(dv)} points exceed it and are drawn in the end colour.',
            color=t['ink2'], fontsize=8.5, ha='left', x=0.01)

    return fig, {'n_over': n_over, 'vmax': vmax, 'vmin': vmin}


def front_table(data, rows=10):
    """Text view of the front -- the chart's accessible counterpart."""
    front = sorted((p for p in data['points'] if p['pareto']), key=lambda p: p['tof_days'])
    step = max(1, len(front) // rows)
    out = [f"{'TOF (yr)':>9}  {'DV (m/s)':>10}  {'Launch':>12}  {'Arrival':>12}"]
    for p in front[::step]:
        out.append(f"{p['tof_days'] / 365.25:9.2f}  {p['dv_ms']:10.0f}  "
                   f"{mission.mjd2000_to_datetime(p['launch_mjd2000']):%Y-%m-%d}  "
                   f"{mission.mjd2000_to_datetime(p['arrival_mjd2000']):%Y-%m-%d}")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=PARETO_PATH)
    ap.add_argument('--theme', choices=('light', 'dark'), default='light')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"No Pareto data at {args.data}. Run pareto.py first.")
    data = json.loads(args.data.read_text())

    fig, info = plot(data, theme=args.theme)
    out = args.out or mission.HERE / f"porkchop_{args.theme}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    print(f"\n{front_table(data)}")


if __name__ == '__main__':
    main()
