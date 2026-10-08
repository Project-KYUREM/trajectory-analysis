"""Run ``pareto.py`` for every (model, sequence) pair in a matrix.

Example: ``python orbit_planning/batch.py --models capture,ballistic
--sequences EVEEJN,EVEJN --sweep``. Options this script does not recognise are
forwarded to ``pareto.py`` unchanged, and each pair runs in its own process.
A pair whose ``pareto.json`` already holds the same kind of run (swept or
single window) is skipped unless ``--force`` is given.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import paths

PARETO = Path(__file__).resolve().with_name('pareto.py')


def _existing_run(path):
    """Describe a previous result so a skip reports what it found.

    Args:
        path (pathlib.Path): The pair's ``pareto.json``.

    Returns:
        dict or None: ``None`` if the file is absent, else
        ``{'swept': bool, 'summary': str}`` built from the file's ``settings``.
    """
    if not path.is_file():
        return None
    try:
        settings = json.loads(path.read_text()).get('settings', {})
    except (OSError, ValueError):
        return {'swept': False, 'summary': 'unreadable result'}
    polish = settings.get('polish', {}).get('n_polished', 0)
    kind = 'swept' if settings.get('swept') else 'single window'
    return {'swept': bool(settings.get('swept')),
            'summary': f"{kind}, {settings.get('islands')} islands x pop {settings.get('pop')} x "
                       f"{settings.get('rounds')}x{settings.get('gen')} gen, {polish} polished, "
                       f"{(settings.get('wall_seconds') or 0) / 60:.0f} min"}


def main():
    """Run ``pareto.py`` for each (model, sequence) pair not already done."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='unrecognised options are forwarded to pareto.py')
    parser.add_argument('--models', default=','.join(paths.DEFAULT_MODELS))
    parser.add_argument('--sequences', default=','.join(paths.DEFAULT_SEQUENCES))
    parser.add_argument('--force', action='store_true', help='rerun pairs that already have results')
    args, forwarded = parser.parse_known_args()

    pairs = [(m, s) for m in paths.csv_list(args.models)
             for s in paths.csv_list(args.sequences, upper=True)]
    print(f"batch: {len(pairs)} (model, sequence) pairs; forwarding {forwarded or 'nothing'}\n")

    want_sweep = '--sweep' in forwarded
    t_all, done = time.time(), 0
    for model, sequence in pairs:
        existing = _existing_run(paths.results_path(model, sequence, 'pareto.json'))
        if existing and not args.force:
            # A quick unswept result must not satisfy a request for a sweep:
            # "exists" is only a reason to skip when it is the same kind of run.
            if existing['swept'] == want_sweep:
                print(f"  skip {model}/{sequence} (exists: {existing['summary']})")
                continue
            print(f"  redo {model}/{sequence} (exists but {existing['summary']}; "
                  f"{'sweep' if want_sweep else 'single window'} requested)")
        print(f"\n===== {model}/{sequence} =====", flush=True)
        t0 = time.time()
        # Output streams straight through: a swept run takes an hour and its
        # per-slice progress is the only sign of life.
        result = subprocess.run([sys.executable, str(PARETO), '--model', model,
                                 '--sequence', sequence, *forwarded])
        status = 'ok' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
        print(f"===== {model}/{sequence}: {status} in {time.time() - t0:.0f}s =====", flush=True)
        done += result.returncode == 0
    print(f"\nbatch done: {done}/{len(pairs)} run in {(time.time() - t_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
