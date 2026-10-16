"""Build each model's combined Pareto front across every published sequence.

For one model, the front points of all its published runs are pooled and the
points no other sequence beats on both cost and time of flight are flagged:
for every trip time, the cheapest trajectory any sequence offers. The site's
"Best by trip time" tab plots it from ``docs/data/<model>/fronts.json``.

It reads only the published runs under ``docs/data`` (via ``sitedata``), not
``results/``, and needs no SPICE kernels, so it runs anywhere::

    python orbit_planning/fronts.py              # every model in index.json
    python orbit_planning/fronts.py --models dfd

``publish.py`` runs it at the end of every publish.
"""
import argparse
import json

import paths
import sitedata

# Background (non-winning) front points kept per run: enough to show each
# sequence's front shape without shipping thousands of chemical-model points.
BACKGROUND_PER_RUN = 150


def global_front(points):
    """Flag the points no other point beats on both cost and time of flight.

    Args:
        points (list[dict]): Records with ``cost`` and ``tof_days``; each gets
            ``best`` set in place.

    Returns:
        int: Number of points on the combined front.
    """
    order = sorted(range(len(points)), key=lambda i: (points[i]['tof_days'], points[i]['cost']))
    best_cost = float('inf')
    for p in points:
        p['best'] = False
    for i in order:
        if points[i]['cost'] < best_cost:
            points[i]['best'] = True
            best_cost = points[i]['cost']
    return sum(p['best'] for p in points)


def _thin(points, keep):
    """Keep at most ``keep`` points, evenly spread over time of flight."""
    if len(points) <= keep:
        return points
    points = sorted(points, key=lambda p: p['tof_days'])
    step = (len(points) - 1) / (keep - 1)
    return [points[round(k * step)] for k in range(keep)]


def build_model(model, runs):
    """Write ``docs/data/<model>/fronts.json`` from the model's published clouds.

    Args:
        model (str): Model name.
        runs (list[dict]): The model's ``index.json`` entries.

    Returns:
        dict or None: Summary (``model``, ``n_runs``, ``n_points``, ``n_best``,
        ``winners``), or ``None`` if no run had a front.
    """
    pooled, used = [], []
    for run in runs:
        cloud_path = paths.DOCS_DIR / run['path']
        if not cloud_path.is_file():
            continue
        _meta, points, _detail = sitedata.read_run(cloud_path.parent)
        front = [dict(seq=run['sequence'], id=p['id'], cost=p['cost'], tof_days=p['tof_days'],
                      launch_mjd2000=p['launch_mjd2000'])
                 for p in points if p.get('pareto')]
        if front:
            pooled.extend(front)
            used.append(dict(sequence=run['sequence'], labels=run['labels'], n_front=len(front)))
    if not pooled:
        return None

    n_best = global_front(pooled)
    by_seq = {}
    for p in pooled:
        by_seq.setdefault(p['seq'], []).append(p)
    kept = []
    for seq_points in by_seq.values():
        kept += [p for p in seq_points if p['best']]
        kept += _thin([p for p in seq_points if not p['best']], BACKGROUND_PER_RUN)

    first = runs[0]
    out = dict(model=model, cost_label=first['cost_label'], cost_units=first['cost_units'],
               runs=used, points=[{**p, 'cost': round(p['cost'], 3), 'tof_days': round(p['tof_days'], 3),
                                   'launch_mjd2000': round(p['launch_mjd2000'], 4)} for p in kept])
    target = paths.DOCS_DATA_DIR / model / 'fronts.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, separators=(',', ':')))
    winners = sorted({p['seq'] for p in pooled if p['best']})
    return dict(model=model, n_runs=len(used), n_points=len(kept), n_best=n_best, winners=winners)


def build_all(models=None, log=print):
    """Build ``fronts.json`` for every model in ``docs/data/index.json``.

    Args:
        models (list[str], optional): Only these models. Defaults to None (all).
        log (Callable[[str], None], optional): Printer. Defaults to print.

    Returns:
        list[dict]: One ``build_model`` summary per model built.
    """
    index_path = paths.DOCS_DATA_DIR / 'index.json'
    if not index_path.is_file():
        log(f"no {index_path}; run publish.py first")
        return []
    by_model = {}
    for run in json.loads(index_path.read_text()):
        by_model.setdefault(run['model'], []).append(run)
    summaries = []
    for model, runs in sorted(by_model.items()):
        if models and model not in models:
            continue
        summary = build_model(model, runs)
        if summary is None:
            log(f"fronts {model}: no front points")
            continue
        summaries.append(summary)
        log(f"fronts {model}: {summary['n_runs']} sequences, {summary['n_best']} points on the "
            f"combined front, won by {', '.join(summary['winners'])}")
    return summaries


def main():
    """Rebuild the combined fronts from the published clouds."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default=None, help='comma list; default: every published model')
    args = parser.parse_args()
    build_all(paths.csv_list(args.models) if args.models else None)


if __name__ == '__main__':
    main()
