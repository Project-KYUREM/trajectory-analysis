#!/bin/bash
# flux: -N 1
# flux: --exclusive
# flux: -t 24h
# flux: --job-name=traj-sweep
# flux: --output=sweep.{{id}}.log
set -euo pipefail
cd /scratch/user/juandurante/git/trajectory-analysis
export PYTHON=/home/juandurante/miniconda3/envs/orbital_env/bin/python
export MODELS="chem"
export WORKERS=8
export PYTHONUNBUFFERED=1                 # live per-slice progress in results/<model>/logs/
export JOBS=$(( $(nproc) / WORKERS ))     # 128 cores -> 16 sequences at once
echo "node $(hostname): $(nproc) cores, JOBS=$JOBS x WORKERS=$WORKERS"

# read the SPICE kernels from node-local RAM, not /scratch
export KYUREM_DATA_DIR=/dev/shm/kyurem_kernels_$USER
mkdir -p "$KYUREM_DATA_DIR"
cp data/de440.bsp data/naif0012.tls data/pck00011.tpc data/gm_de440.tpc data/nep097.bsp "$KYUREM_DATA_DIR"/
trap 'rm -rf "$KYUREM_DATA_DIR"' EXIT

./scripts/run_dfd_sweep.sh
