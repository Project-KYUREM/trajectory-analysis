"""Run ``pareto.py`` over a matrix of (model, sequence) pairs.

    python orbit_planning/batch.py --models capture,ballistic \
        --sequences EVEEJN,EVEJN --sweep --polish 24 --islands 20

Every option this script does not recognise is forwarded to ``pareto.py``
unchanged, so the batch owns only the matrix, skipping and tallying.
Resumable: a pair whose ``pareto.json`` already exists is skipped unless
``--force``. Each pair is its own process (the mp_island / spawn guard lives
in ``pareto.py``). This module stays kernel-free on purpose.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import paths

PARETO = Path(__file__).resolve().with_name('pareto.py')


def main():
    """CLI: run pareto.py for each (model, sequence) pair not yet done."""
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

    t_all, done = time.time(), 0
    for model, sequence in pairs:
        if paths.results_path(model, sequence, 'pareto.json').is_file() and not args.force:
            print(f"  skip {model}/{sequence} (exists)")
            continue
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
