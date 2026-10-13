#!/bin/bash
# flux: -N 1
# flux: --exclusive
# flux: -t 24h
# flux: --job-name=traj-sweep
# flux: --output=sweep.{{id}}.log
set -euo pipefail
cd /scratch/user/juandurante/git/trajectory-analysis
export PYTHON=/home/juandurante/miniconda3/envs/orbital_env/bin/python
export MODELS="dfd dfd-1mw capture ballistic"
export WORKERS=8
export JOBS=$(( $(nproc) / WORKERS ))     # 128 cores -> 16 sequences at once
echo "node $(hostname): $(nproc) cores, JOBS=$JOBS x WORKERS=$WORKERS"
./scripts/run_dfd_sweep.sh
