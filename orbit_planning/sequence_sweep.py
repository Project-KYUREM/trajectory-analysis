"""Sweep an array of gravity-assist sequences for one model and rank them.

Edit ``SEQUENCES`` below (a plain Python list) or pass ``--sequences`` on the
command line; each sequence is searched in its own process by ``pareto.py``
(or ``optimize.py`` with ``--mode champion``), exactly as ``batch.py`` does,
and finished sequences are skipped on a rerun. A ranking table is then written
to ``results/<model>/sequence_sweep.{csv,json}``.

Examples::

    # the list below, Direct Fusion Drive model, one launch window
    python orbit_planning/sequence_sweep.py

    # an explicit array, swept over synodic launch slices, 32 polished points each
    python orbit_planning/sequence_sweep.py --model dfd \\
        --sequences EJN EJSN EJUN EVEJN --sweep --polish 32

    # only re-rank what is already under results/ (no search)
    python orbit_planning/sequence_sweep.py --summary-only --max-years 10

    # many sequences in parallel on a big machine: scripts/run_dfd_sweep.sh

Options this script does not recognise (``--sweep``, ``--islands``,
``--polish``, ``--polish-maxeval``, ...) are forwarded to the search script.
"""
import argparse
import csv
import json
from pathlib import Path

import batch
import paths

# The array of sequences to sweep. Letters are mission.BODIES keys
# (V Venus, E Earth, M Mars, J Jupiter, S Saturn, U Uranus, N Neptune);
# Earth first, the destination (Neptune) last.
SEQUENCES = list(paths.DFD_SEQUENCES)

MODEL = 'dfd'

SCRIPTS = {
    'pareto': (Path(__file__).resolve().with_name('pareto.py'), 'pareto.json'),
    'champion': (Path(__file__).resolve().with_name('optimize.py'), 'champion.json'),
}


def summarise(model, sequence, filename, max_years=None):
    """Summarise one sequence's result file into a ranking row.

    Args:
        model (str): Model name.
        sequence (str): Sequence string.
        filename (str): ``pareto.json`` or ``champion.json``.
        max_years (float, optional): Only points at or under this total flight
            time (cruise plus capture spiral) count for ``best_*``. Defaults to
            None.

    Returns:
        dict or None: The row, or ``None`` when the file is missing.
    """
    path = paths.results_path(model, sequence, filename)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if 'points' in raw:
        points = raw['points']
    else:   # champion.json: a single point, scored without a time objective
        points = [dict(cost=raw['cost'], tof_days=float('nan'), launch_mjd2000=raw['x'][0],
                       feasible=raw.get('feasible', True), pareto=True)]
    feasible = [p for p in points if p.get('feasible', True)]
    capped = [p for p in feasible if max_years is None or p['tof_days'] <= max_years * 365.25]
    best = min(capped, key=lambda p: p['cost'], default=None)
    fastest = min(feasible, key=lambda p: p['tof_days'], default=None)
    wet = raw.get('assumptions', {}).get('wet_mass_kg')
    row = dict(sequence=sequence, labels='-'.join(raw.get('labels', [])),
               n_points=len(points), n_feasible=len(feasible),
               n_front=sum(1 for p in points if p.get('pareto')),
               best_cost=None, best_tof_years=None, best_launch_mjd2000=None,
               best_final_mass_kg=None, fastest_tof_years=None, fastest_cost=None,
               cost_units=raw.get('cost_units'), swept=raw.get('settings', {}).get('swept'))
    if best:
        row.update(best_cost=round(best['cost'], 1),
                   best_tof_years=round(best['tof_days'] / 365.25, 3),
                   best_launch_mjd2000=round(best['launch_mjd2000'], 2),
                   best_final_mass_kg=round(wet - best['cost'], 1) if wet else None)
    if fastest:
        row.update(fastest_tof_years=round(fastest['tof_days'] / 365.25, 3),
                   fastest_cost=round(fastest['cost'], 1))
    return row


def write_table(model, rows, max_years=None):
    """Print the ranking and write it as CSV and JSON next to the runs.

    Args:
        model (str): Model name.
        rows (list[dict]): ``summarise`` rows.
        max_years (float, optional): The time cap the rows were built with,
            recorded in the JSON. Defaults to None.

    Returns:
        pathlib.Path: The CSV path.
    """
    rows = sorted(rows, key=lambda r: (r['best_cost'] is None, r['best_cost'] or 0.0))
    cap = f" (best within {max_years:g} yr)" if max_years else ""
    print(f"\nsequence ranking, model {model}{cap}")
    print(f"{'sequence':>10}  {'feasible':>8}  {'best cost':>10}  {'@ yr':>6}  "
          f"{'final kg':>9}  {'fastest yr':>10}  {'@ cost':>9}")
    fmt = lambda v, f: format(v, f) if v is not None else '-'
    for r in rows:
        print(f"{r['sequence']:>10}  {r['n_feasible']:>8}  {fmt(r['best_cost'], '10.0f'):>10}  "
              f"{fmt(r['best_tof_years'], '6.2f'):>6}  {fmt(r['best_final_mass_kg'], '9.0f'):>9}  "
              f"{fmt(r['fastest_tof_years'], '10.2f'):>10}  {fmt(r['fastest_cost'], '9.0f'):>9}")
    out_dir = paths.RESULTS_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'sequence_sweep.csv'
    if rows:
        with csv_path.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / 'sequence_sweep.json').write_text(
        json.dumps(dict(model=model, max_years=max_years, rows=rows), indent=2))
    print(f"wrote {csv_path} and sequence_sweep.json")
    return csv_path


def main():
    """Search every sequence of the array that is not done yet, then rank them."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='unrecognised options are forwarded to the search script')
    parser.add_argument('--model', default=MODEL, help='model to search (default: %(default)s)')
    parser.add_argument('--sequences', nargs='+', default=SEQUENCES, metavar='SEQ',
                        help='sequences to sweep, space separated (default: SEQUENCES in this file)')
    parser.add_argument('--mode', choices=sorted(SCRIPTS), default='pareto',
                        help='pareto: cost/time front per sequence; champion: cheapest only')
    parser.add_argument('--force', action='store_true', help='rerun sequences that already have results')
    parser.add_argument('--summary-only', action='store_true', dest='summary_only',
                        help='skip the searches and only rank existing results')
    parser.add_argument('--max-years', type=float, default=None, dest='max_years',
                        help='rank by the cheapest point within this total flight time')
    parser.add_argument('--build-tables', action='store_true', dest='build_tables',
                        help='only build (or load) the model\'s cached capture table (DFD models), then exit; '
                             'run this once before launching searches in parallel')
    args, forwarded = parser.parse_known_args()

    if args.build_tables:
        import models   # loads SPICE; the rest of this script does not need it
        models.build(args.model, 'EN')
        print(f"{args.model}: ready (capture tables cache under {paths.RESULTS_DIR / 'cache'})")
        return

    sequences = [s.strip().upper() for s in args.sequences if s.strip()]
    script, filename = SCRIPTS[args.mode]
    print(f"sequence sweep: model {args.model}, {len(sequences)} sequences {sequences}, "
          f"{args.mode} mode; forwarding {forwarded or 'nothing'}\n")
    if not args.summary_only:
        batch.run_pairs([(args.model, s) for s in sequences], forwarded, args.force,
                        script=script, filename=filename)

    rows = [r for s in sequences if (r := summarise(args.model, s, filename, args.max_years))]
    if not rows:
        print(f"no results for {args.model} under {paths.RESULTS_DIR}")
        return
    write_table(args.model, rows, args.max_years)


if __name__ == '__main__':
    main()
