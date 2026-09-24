"""Run the Pareto search across many (model, sequence) pairs into
results/<model>/<sequence>/pareto.json.

    python orbit_planning/batch.py --models capture,ballistic \
        --sequences EVEEJN,EVEJN,EVJN,EVVEJN,EJN --sweep

Resumable: a pair whose pareto.json already exists is skipped unless --force.
Each pair is a separate process (the mp_island/spawn guard lives in pareto.py),
so this driver just shells out and tallies. NEP is intentionally excluded --
it is a GMAT seed exporter, not an optimiser (see PLAN.md).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', default='capture,ballistic')
    ap.add_argument('--sequences', default='EVEEJN,EVEJN,EVJN,EVVEJN,EJN')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--islands', type=int, default=20)
    ap.add_argument('--pop', type=int, default=100)
    ap.add_argument('--gen', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--force', action='store_true', help='rerun even if results exist')
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    seqs = [s.strip().upper() for s in args.sequences.split(',') if s.strip()]
    pairs = [(m, s) for m in models for s in seqs]
    print(f"batch: {len(pairs)} (model, sequence) pairs\n")

    t_all = time.time()
    done = 0
    for m, s in pairs:
        out = HERE / '..' / 'results' / m / s / 'pareto.json'
        out = out.resolve()
        if out.is_file() and not args.force:
            print(f"  skip {m}/{s} (exists)")
            continue
        cmd = [sys.executable, str(HERE / 'pareto.py'),
               '--model', m, '--sequence', s,
               '--islands', str(args.islands), '--pop', str(args.pop),
               '--gen', str(args.gen), '--rounds', str(args.rounds)]
        if args.sweep:
            cmd.append('--sweep')
        print(f"  run  {m}/{s} ...", flush=True)
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"       FAILED ({time.time() - t:.0f}s):\n"
                  + "\n".join('       ' + ln for ln in r.stderr.splitlines()[-6:]))
            continue
        tail = [ln for ln in r.stdout.splitlines() if 'front' in ln or 'wrote' in ln]
        print(f"       ok ({time.time() - t:.0f}s)  " + (tail[-1].strip() if tail else ''))
        done += 1

    print(f"\nbatch done: {done}/{len(pairs)} run in {(time.time() - t_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
