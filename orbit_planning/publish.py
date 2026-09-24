"""Compact raw results (results/<model>/<sequence>/pareto.json) into the static
site payload (docs/data/...). The browser never touches SPICE or 36 MB clouds.

Per run it writes docs/data/<model>/<sequence>/cloud.json:
    cost_label/units          what the trade axis means for this model
    points[]                  the whole cloud, compact: id, cost, tof, launch,
                              arrival, pareto flag  (for the scatter)
    detail{id: describe()}    full mission description + propagatable arcs, only
                              for selectable points (front + a few dominated per
                              TOF bin -- the blob is context, not clickable)
    planets[]                 heliocentric orbit polylines for 3D context

and refreshes docs/data/index.json (the gallery manifest).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import mission
import models
import trajectory

DOCS_DATA = (mission.HERE / '..' / 'docs' / 'data').resolve()


def _planet_orbits(sequence, samples=160):
    """One orbit polyline per distinct body, heliocentric ECLIPJ2000 (metres)."""
    seen, orbits = set(), []
    planets, labels = mission.make_sequence(sequence)
    t0 = mission.LAUNCH_WINDOW[0]
    for pl, lab in zip(planets, labels):
        if lab in seen:
            continue
        seen.add(lab)
        period_d = mission.orbital_period(pl) / 86400.0
        xs = []
        for k in range(samples + 1):
            r, _ = pl.eph(mission.pk.epoch(t0.mjd2000 + period_d * k / samples, 'mjd2000'))
            xs.append([float(c) for c in r])
        orbits.append({'label': lab, 'period_days': period_d, 'xy': xs})
    return orbits


def _select_detail_ids(points, per_bin=2, bins=14):
    """Front points (all) + a few cheapest dominated per TOF bin, so a user can
    click into representative dominated trajectories without exploding the
    payload."""
    ids = {p['id'] for p in points if p['pareto']}
    tofs = np.array([p['tof_days'] for p in points])
    if len(tofs):
        edges = np.linspace(tofs.min(), tofs.max(), bins + 1)
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            inb = [p for p in points if lo <= p['tof_days'] <= hi and not p['pareto']]
            inb.sort(key=lambda p: p['cost'])
            ids.update(p['id'] for p in inb[:per_bin])
    return ids


def publish_run(model, sequence):
    raw_path = mission.RESULTS_DIR / model / sequence / 'pareto.json'
    if not raw_path.is_file():
        return None
    raw = json.loads(raw_path.read_text())
    spec = models.get(model)

    points = []
    for i, p in enumerate(raw['points']):
        points.append({
            'id': f"p{i:05d}",
            'cost': float(p['dv_ms']),
            'tof_days': float(p['tof_days']),
            'tof_years': float(p['tof_days']) / 365.25,
            'launch_mjd2000': float(p['launch_mjd2000']),
            'arrival_mjd2000': float(p['arrival_mjd2000']),
            'pareto': bool(p['pareto']),
            '_x': p['x'],
        })

    detail_ids = _select_detail_ids(points)
    detail = {}
    for p in points:
        if p['id'] in detail_ids and p['_x'] is not None:
            try:
                detail[p['id']] = trajectory.describe(model, sequence, p['_x'])
            except Exception as exc:
                detail[p['id']] = {'error': str(exc)}
    for p in points:
        p.pop('_x', None)

    front = [p for p in points if p['pareto']]
    best = min(front, key=lambda p: p['cost']) if front else None
    cloud = {
        'model': model, 'sequence': sequence,
        'labels': mission.make_sequence(sequence)[1],
        'cost_label': spec['label'], 'cost_units': spec['units'],
        'blurb': spec['blurb'],
        'settings': raw.get('settings', {}),
        'points': points,
        'detail': detail,
        'planets': _planet_orbits(sequence),
    }
    outdir = DOCS_DATA / model / sequence
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'cloud.json').write_text(json.dumps(cloud, separators=(',', ':')))

    return {
        'model': model, 'sequence': sequence,
        'labels': cloud['labels'],
        'cost_label': spec['label'], 'cost_units': spec['units'],
        'n_points': len(points), 'n_front': len(front), 'n_detail': len(detail),
        'best_cost': best['cost'] if best else None,
        'best_tof_years': best['tof_years'] if best else None,
        'path': f"data/{model}/{sequence}/cloud.json",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', default='capture,ballistic')
    ap.add_argument('--sequences', default='EVEEJN,EVEJN,EVJN,EVVEJN,EJN')
    args = ap.parse_args()

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    index = []
    for m in [x.strip() for x in args.models.split(',') if x.strip()]:
        for s in [x.strip().upper() for x in args.sequences.split(',') if x.strip()]:
            summary = publish_run(m, s)
            if summary:
                index.append(summary)
                print(f"published {m}/{s}: {summary['n_points']} pts, "
                      f"{summary['n_front']} front, {summary['n_detail']} detailed, "
                      f"best {summary['best_cost']:.0f} {summary['cost_units']}")
            else:
                print(f"skip {m}/{s}: no results")
    (DOCS_DATA / 'index.json').write_text(json.dumps(index, indent=2))
    print(f"\nwrote {DOCS_DATA / 'index.json'} ({len(index)} runs)")


if __name__ == '__main__':
    main()
