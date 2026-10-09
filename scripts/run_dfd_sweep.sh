#!/usr/bin/env bash
# Sweep the 77 Earth -> Neptune swing-by sequences with the Direct Fusion Drive
# (low-thrust) model on one many-core machine, then rank them.
#
# Each sequence is one `pareto.py` run (NSGA-II archipelago over every synodic
# launch slice, then MBH + IPOPT polish of the front). JOBS sequences run at
# once, each with WORKERS processes, so JOBS x WORKERS should be about the core
# count. Finished sequences are skipped on a rerun (pareto.json already there),
# so an interrupted sweep resumes where it stopped; FORCE=1 redoes them.
#
# Usage (from anywhere; conda env `orbital_env` active):
#   scripts/run_dfd_sweep.sh                    # defaults below, 16 x 8 = 128 cores
#   JOBS=32 WORKERS=4 scripts/run_dfd_sweep.sh  # more sequences at once, smaller archipelagos
#   MODELS="dfd dfd-1mw" scripts/run_dfd_sweep.sh
#   SEQUENCES="EJN EVEJN" scripts/run_dfd_sweep.sh   # a subset instead of the 77
#   DRY_RUN=1 scripts/run_dfd_sweep.sh          # print the commands only
#
# On a SLURM node, wrap it: sbatch -N1 -c128 --wrap "scripts/run_dfd_sweep.sh".
#
# Outputs: results/<model>/<sequence>/pareto.json, one log per sequence in
# results/<model>/logs/, and the ranking in results/<model>/sequence_sweep.{csv,json}.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"

# ---- parallel layout -------------------------------------------------------
JOBS="${JOBS:-16}"          # sequences running at once
WORKERS="${WORKERS:-8}"     # islands and polish processes per sequence

# ---- search settings (forwarded to pareto.py; see `pareto.py --help`) -------
MODELS="${MODELS:-dfd}"
POP="${POP:-100}"           # individuals per island (NSGA-II: multiple of 4)
GEN="${GEN:-200}"           # generations per round
ROUNDS="${ROUNDS:-10}"      # migration rounds per launch slice
POLISH="${POLISH:-24}"      # front points MBH + IPOPT polished per sequence
POLISH_STOP="${POLISH_STOP:-5}"
SEED="${SEED:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA="${EXTRA:-}"          # any other pareto.py options, e.g. EXTRA="--polish-maxeval 1500"

# ---- the 77 swing-by sequences ---------------------------------------------
# Flybys only (Table 1 of the supplied sequence study: at most 5 assists; only
# Venus and Earth repeat back to back, Venus up to 3x and Earth up to 2x).
# Earth is prepended and Neptune appended below, so "VEJ" runs as EVEJN.
FLYBYS=(
  V     E     M     J     S     VV    VE
  VM    VJ    VS    EE    EM    EJ    ES
  MJ    MS    JS    VVV   VVE   VVM   VVJ
  VVS   VEE   VEM   VEJ   VES   VMJ   VMS
  VJS   EEM   EEJ   EES   EMJ   EMS   EJS
  MJS   VVVE  VVVM  VVVJ  VVVS  VVEE  VVEM
  VVEJ  VVES  VVMJ  VVMS  VVJS  VEEM  VEEJ
  VEES  VEMJ  VEMS  VEJS  VMJS  EEMJ  EEMS
  EEJS  EMJS  VVVEE VVVEM VVVEJ VVVES VVVMJ
  VVVMS VVVJS VVEEM VVEEJ VVEES VVEMJ VVEMS
  VVEJS VVMJS VEEMJ VEEMS VEEJS VEMJS EEMJS
)

if [[ -n "${SEQUENCES:-}" ]]; then
  read -r -a SEQS <<< "$SEQUENCES"
else
  SEQS=()
  for f in "${FLYBYS[@]}"; do SEQS+=("E${f}N"); done
fi
# Longest first: they take the longest (more legs, more genes), so starting
# them early keeps the cores busy at the end of the sweep.
mapfile -t SEQS < <(printf '%s\n' "${SEQS[@]}" | awk '{ print length, $0 }' | sort -k1,1nr -k2 | cut -d' ' -f2)

echo "sweep: ${#SEQS[@]} sequences x models [${MODELS}] | ${JOBS} at once x ${WORKERS} workers" \
     "| pop ${POP} gen ${GEN} rounds ${ROUNDS} polish ${POLISH}"

# One thread per process: pygmo/IPOPT workers already fill the cores, and
# BLAS/OpenMP threads on top would oversubscribe them.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {   # run_one <model> <sequence>
  local model="$1" seq="$2"
  local out="$ROOT/results/$model/$seq/pareto.json"
  local log="$ROOT/results/$model/logs/$seq.log"
  if [[ -f "$out" && "$FORCE" != 1 ]]; then
    echo "skip  $model/$seq (done)"
    return 0
  fi
  local cmd=("$PY" "$ROOT/orbit_planning/pareto.py" --model "$model" --sequence "$seq" --sweep
             --islands "$WORKERS" --polish-workers "$WORKERS" --pop "$POP" --gen "$GEN"
             --rounds "$ROUNDS" --polish "$POLISH" --polish-stop "$POLISH_STOP" --seed "$SEED"
             $EXTRA)
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "${cmd[*]}"
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  local t0=$SECONDS
  echo "start $model/$seq"
  if "${cmd[@]}" > "$log" 2>&1; then
    echo "done  $model/$seq in $(( (SECONDS - t0) / 60 )) min ($(grep -m1 'cheapest on front' "$log" || echo 'no feasible point'))"
  else
    echo "FAIL  $model/$seq (exit $?, see $log)"
  fi
}
export -f run_one
export ROOT PY WORKERS POP GEN ROUNDS POLISH POLISH_STOP SEED FORCE DRY_RUN EXTRA

for model in $MODELS; do
  # Build the model's capture table once, before the parallel searches load it.
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PY" "$ROOT/orbit_planning/sequence_sweep.py" --model "$model" --build-tables
  fi
  printf '%s\n' "${SEQS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$0" "$1"' "$model" {}
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PY" "$ROOT/orbit_planning/sequence_sweep.py" --model "$model" --summary-only --sequences "${SEQS[@]}"
  fi
done

echo "all done. Publish to the site with: $PY $ROOT/orbit_planning/publish.py --models ${MODELS// /,}"
