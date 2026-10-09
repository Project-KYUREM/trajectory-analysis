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


def run_pairs(pairs, forwarded, force=False, script=PARETO, filename='pareto.json'):
    """Run a search script once per (model, sequence) pair, each in its own process.

    Args:
        pairs (list[tuple[str, str]]): ``(model, sequence)`` pairs, in order.
        forwarded (list[str]): Extra CLI options passed to every run.
        force (bool, optional): Rerun pairs whose result file already holds the
            same kind of run. Defaults to False.
        script (pathlib.Path, optional): ``pareto.py`` or ``optimize.py``.
            Defaults to ``pareto.py``.
        filename (str, optional): The result file the script writes, checked
            for the skip. Defaults to ``pareto.json``.

    Returns:
        dict: ``{(model, sequence): status}`` with status ``ok``, ``skipped``
        or ``failed``.
    """
    want_sweep = '--sweep' in forwarded
    t_all, status = time.time(), {}
    for model, sequence in pairs:
        existing = _existing_run(paths.results_path(model, sequence, filename))
        if existing and not force:
            # A quick unswept result must not satisfy a request for a sweep:
            # "exists" is only a reason to skip when it is the same kind of run.
            if existing['swept'] == want_sweep:
                print(f"  skip {model}/{sequence} (exists: {existing['summary']})")
                status[(model, sequence)] = 'skipped'
                continue
            print(f"  redo {model}/{sequence} (exists but {existing['summary']}; "
                  f"{'sweep' if want_sweep else 'single window'} requested)")
        print(f"\n===== {model}/{sequence} =====", flush=True)
        t0 = time.time()
        # Output streams straight through: a swept run takes an hour and its
        # per-slice progress is the only sign of life.
        result = subprocess.run([sys.executable, str(script), '--model', model,
                                 '--sequence', sequence, *forwarded])
        ok = result.returncode == 0
        print(f"===== {model}/{sequence}: {'ok' if ok else f'FAILED (exit {result.returncode})'} "
              f"in {time.time() - t0:.0f}s =====", flush=True)
        status[(model, sequence)] = 'ok' if ok else 'failed'
    done = sum(v == 'ok' for v in status.values())
    print(f"\nbatch done: {done}/{len(pairs)} run in {(time.time() - t_all) / 60:.1f} min")
    return status


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
    run_pairs(pairs, forwarded, args.force)


if __name__ == '__main__':
    main()
