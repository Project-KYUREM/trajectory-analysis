"""Compact raw results into the static site payload under ``docs/data/``.

Each run becomes ``docs/data/<model>/<sequence>/cloud.json``: every point of
the cloud, a ``trajectory.describe()`` for each point flagged ``detail``, and
the ECLIPJ2000 to EME2000 rotation. ``planets.json`` (orbit polylines shared
by all runs) and ``index.json`` (the gallery manifest) are written alongside
(runs published earlier stay listed unless ``--replace-index``), then
``fronts.py`` builds each model's combined front (``fronts.json``).
Floats are rounded to a per-key precision.
"""
import argparse
import json

import numpy as np

import fronts
import mission
import models
import paths
import trajectory

# Decimal places per JSON key; lists inherit the key of their parent.
_PRECISION = {
    'r_eclip_m': 0, 'v_eclip_ms': 4, 'vinf_vec_eclip_ms': 4, 'xyz': 0,
    'epoch_mjd2000': 6, 'dt_days': 6, 'launch_mjd2000': 4,
    'cost': 3, 'tof_days': 3,
}


def compact(obj, key=None):
    """Round every float in a nested structure to its key's precision.

    Args:
        obj (Any): Float, dict, list or tuple to compact; other values pass
            through unchanged.
        key (str, optional): JSON key ``obj`` sits under, looked up in
            ``_PRECISION``; list items inherit it. Defaults to None.

    Returns:
        Any: A copy of ``obj`` with its floats rounded.
    """
    if isinstance(obj, float):
        return round(obj, _PRECISION.get(key, 4))
    if isinstance(obj, dict):
        return {k: compact(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [compact(v, key) for v in obj]
    return obj


def discover_runs(model_filter=None, sequence_filter=None):
    """List the runs that have a raw ``pareto.json`` and a decodable model.

    Args:
        model_filter (list[str], optional): Model names to keep; ``None`` keeps
            all. Defaults to None.
        sequence_filter (list[str], optional): Sequence strings to keep; ``None``
            keeps all. Defaults to None.

    Returns:
        list[tuple[str, str]]: Sorted ``(model, sequence)`` pairs.
    """
    runs = []
    for raw in sorted(paths.RESULTS_DIR.glob('*/*/pareto.json')):
        model, sequence = raw.parent.parent.name, raw.parent.name
        if model not in models.MODELS or models.get(model).decode is None:
            continue
        if model_filter and model not in model_filter:
            continue
        if sequence_filter and sequence not in sequence_filter:
            continue
        runs.append((model, sequence))
    return runs


def select_detail(points, per_bin=2, bins=14):
    """Flag in place which points get a full trajectory description.

    Every front and polished point is flagged, plus the ``per_bin`` cheapest
    dominated points in each time-of-flight bin.

    Args:
        points (list[dict]): Compact point records with ``pareto``, ``polished``,
            ``tof_days`` and ``cost``; each gets a ``detail`` flag.
        per_bin (int, optional): Dominated points to flag per bin. Defaults to 2.
        bins (int, optional): Number of equal-width time-of-flight bins.
            Defaults to 14.
    """
    for p in points:
        p['detail'] = p['pareto'] or p['polished']
    tofs = np.array([p['tof_days'] for p in points])
    edges = np.linspace(tofs.min(), tofs.max(), bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        dominated = sorted((p for p in points if lo <= p['tof_days'] <= hi and not p['pareto']),
                           key=lambda p: p['cost'])
        for p in dominated[:per_bin]:
            p['detail'] = True


def planet_orbits(labels, samples=160):
    """Sample one heliocentric orbit polyline per body.

    Args:
        labels (Iterable[str]): Body names to include.
        samples (int, optional): Points per orbit. Defaults to 160.

    Returns:
        dict: ``{label: {'period_days': float, 'xyz': list}}`` with positions in
        metres, ECLIPJ2000.
    """
    letters = {b['label']: letter for letter, b in mission.BODIES.items()}
    t0 = mission.LAUNCH_WINDOW[0].mjd2000
    orbits = {}
    for label in sorted(set(labels)):
        planet = mission.make_planet(letters[label])
        period = mission.orbital_period(planet) / 86400.0
        orbits[label] = dict(period_days=period, xyz=[
            [float(c) for c in planet.eph(mission.pk.epoch(t0 + period * k / samples, 'mjd2000'))[0]]
            for k in range(samples + 1)])
    return orbits


def publish_run(model_name, sequence):
    """Compact one raw result into its ``cloud.json``.

    Args:
        model_name (str): Model of the run.
        sequence (str): Sequence of the run.

    Returns:
        dict: The run's ``index.json`` entry.

    Raises:
        ValueError: If the raw file's ``schema`` differs from
            ``paths.RESULT_SCHEMA``.
    """
    raw = json.loads(paths.results_path(model_name, sequence, 'pareto.json').read_text())
    if raw.get('schema') != paths.RESULT_SCHEMA:
        raise ValueError(f"result schema {raw.get('schema')} != {paths.RESULT_SCHEMA}")
    model = models.get(model_name)
    # Constrained models (dfd) carry infeasible individuals; they are not
    # trajectories, so they never reach the site.
    points = [dict(id=f"p{i:05d}", cost=p['cost'], tof_days=p['tof_days'],
                   launch_mjd2000=p['launch_mjd2000'], pareto=p['pareto'],
                   polished=p.get('polished', False), x=p['x'])
              for i, p in enumerate(raw['points']) if p.get('feasible', True)]
    if not points:
        raise ValueError("no feasible point in the run")
    select_detail(points)

    detail = {}
    for p in points:
        if p['detail']:
            try:
                detail[p['id']] = trajectory.describe(model_name, sequence, p['x'])
            except Exception as exc:  # keep the run publishable if one point fails
                detail[p['id']] = {'error': str(exc)}
        del p['x']

    cloud = dict(model=model_name, sequence=sequence, labels=raw['labels'],
                 cost_label=model.label, cost_units=model.units, blurb=model.blurb,
                 settings=raw.get('settings', {}),
                 frames=dict(eclip_to_eme=trajectory.ECLIP_TO_EME.tolist()),
                 points=points, detail=detail)
    (paths.docs_run_dir(model_name, sequence) / 'cloud.json').write_text(
        json.dumps(compact(cloud), separators=(',', ':')))

    front = [p for p in points if p['pareto']]
    best = min(front, key=lambda p: p['cost'])
    return dict(model=model_name, sequence=sequence, labels=raw['labels'],
                cost_label=model.label, cost_units=model.units,
                n_points=len(points), n_front=len(front), n_detail=len(detail),
                best_cost=best['cost'], best_tof_years=best['tof_days'] / 365.25,
                path=f"data/{model_name}/{sequence}/cloud.json")


def merge_index(new_entries, replace=False):
    """Combine this publish's runs with the runs already in ``index.json``.

    Publishing from a machine that holds only some of the raw results (a
    cluster running one sweep) must not drop runs published earlier from
    elsewhere: their ``cloud.json`` files are still in ``docs/data``.

    Args:
        new_entries (list[dict]): Index entries published in this call.
        replace (bool, optional): Drop every earlier entry instead (the index
            then lists only this call's runs). Defaults to False.

    Returns:
        tuple[list[dict], int]: The merged index, sorted by model then
        sequence, and how many earlier entries were kept.
    """
    merged = {(e['model'], e['sequence']): e for e in new_entries}
    kept = 0
    index_path = paths.DOCS_DATA_DIR / 'index.json'
    if not replace and index_path.is_file():
        for old in json.loads(index_path.read_text()):
            key = (old['model'], old['sequence'])
            if key not in merged and (paths.DOCS_DIR / old['path']).is_file():
                merged[key] = old
                kept += 1
    return [merged[k] for k in sorted(merged)], kept


def main():
    """Publish every discovered run, optionally filtered, and the shared manifests."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default=None, help='comma list; default: every run found')
    parser.add_argument('--sequences', default=None, help='comma list; default: every run found')
    parser.add_argument('--replace-index', action='store_true', dest='replace_index',
                        help='list only the runs published now; by default earlier published '
                             'runs whose cloud.json is still in docs/data stay on the site')
    args = parser.parse_args()

    runs = discover_runs(paths.csv_list(args.models) if args.models else None,
                         paths.csv_list(args.sequences, upper=True) if args.sequences else None)
    if not runs:
        print(f"no results under {paths.RESULTS_DIR}; run batch.py or pareto.py first")
        return

    published = []
    for model_name, sequence in runs:
        try:
            entry = publish_run(model_name, sequence)
        except ValueError as exc:
            print(f"skip {model_name}/{sequence}: {exc}; re-run pareto.py for it")
            continue
        published.append(entry)
        print(f"published {model_name}/{sequence}: {entry['n_points']} pts, {entry['n_front']} front, "
              f"{entry['n_detail']} detailed, best {entry['best_cost']:.0f} {entry['cost_units']}")

    paths.DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    index, kept = merge_index(published, replace=args.replace_index)
    labels = {label for entry in index for label in entry['labels']}
    (paths.DOCS_DATA_DIR / 'planets.json').write_text(
        json.dumps(compact(planet_orbits(labels)), separators=(',', ':')))
    (paths.DOCS_DATA_DIR / 'index.json').write_text(json.dumps(index, indent=2))
    print(f"\nwrote index.json ({len(index)} runs: {len(published)} published now, {kept} kept from "
          f"earlier publishes) and planets.json ({len(labels)} bodies)")
    fronts.build_all()   # docs/data/<model>/fronts.json: the "Best by trip time" tab


if __name__ == '__main__':
    main()
