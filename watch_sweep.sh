#!/usr/bin/env bash
# Live progress of a sequence sweep. Run from the repo root:
#   bash watch_sweep.sh                      # chem, 77 sequences, every 5 s
#   MODELS="dfd dfd-1mw" bash watch_sweep.sh
MODELS="${MODELS:-chem}" TOTAL="${TOTAL:-77}" EVERY="${EVERY:-5}" STALE_MIN="${STALE_MIN:-30}"

bar() {  # bar <done> <total> <width>
  local d=$1 t=$2 w=${3:-40} n
  n=$(( t > 0 ? d * w / t : 0 )); (( n > w )) && n=$w
  printf '[%s%s] %3d%%' "$(printf '%*s' "$n" '' | tr ' ' '#')" \
         "$(printf '%*s' $(( w - n )) '' | tr ' ' '.')" $(( t > 0 ? d * 100 / t : 0 ))
}

declare -A START_DONE
T0=$(date +%s)
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT
while true; do
  {
    now=$(date +%s)
    echo "sweep progress  $(date '+%F %T')   (refresh ${EVERY}s, Ctrl-C to quit)"
    for m in $MODELS; do
      done=$(ls results/"$m"/*/pareto.json 2>/dev/null | wc -l)
      : "${START_DONE[$m]:=$done}"
      echo
      printf '%-8s ' "$m"; bar "$done" "$TOTAL" 40; printf '  %d/%d sequences done' "$done" "$TOTAL"
      gained=$(( done - START_DONE[$m] )); elapsed=$(( now - T0 ))
      if (( gained > 0 && done < TOTAL )); then
        printf '   ETA ~%d min' $(( (TOTAL - done) * elapsed / gained / 60 ))
      fi
      echo
      running=0
      for log in results/"$m"/logs/*.log; do
        [[ -e $log ]] || continue
        seq=$(basename "$log" .log)
        [[ -f results/$m/$seq/pareto.json ]] && continue
        running=$(( running + 1 ))
        n=$(sed -n 's/^sweeping \([0-9]*\) launch slices.*/\1/p' "$log" | head -1)
        s=$(grep -c '^  slice ' "$log")
        age=$(( (now - $(stat -c %Y "$log")) / 60 ))
        if grep -q '^polishing' "$log"; then
          phase="polishing $(sed -n 's/^polishing \([0-9]*\) of.*/\1/p' "$log" | head -1) front points"
          printf '  %-8s ' "$seq"; bar 1 1 20
        elif [[ -n $n ]]; then
          phase="slice $s/$n"
          printf '  %-8s ' "$seq"; bar "$s" "$n" 20
        else
          phase="starting"
          printf '  %-8s ' "$seq"; bar 0 1 20
        fi
        printf '  %-28s log %3d min old' "$phase" "$age"
        (( age >= STALE_MIN )) && printf '  <- stalled or failed? tail %s' "$log"
        echo
      done
      (( running == 0 && done < TOTAL )) && echo "  (nothing running)"
    done
  } > "$TMP"     # draw off-screen, then swap: no flicker
  clear; cat "$TMP"
  sleep "$EVERY"
done
