#!/bin/sh
# Deletion test readout, chained after B2 (user-approved 2026-10-26). Waits until night_B2.sh logs B2_natural_test_DONE
# (gives up if B2 failed or was skipped), requires AC power, then runs b2_deletion.py on eval and natural_test (resumable).
S=v2_reflective/e23/stage_B2.txt; LOG=v2_reflective/e23/night_B2.log
echo "$(date) night_DEL waiting for B2" >> $LOG
until grep -q "B2_natural_test_DONE" $S 2>/dev/null; do
  grep -qE "B2_.*FAILED|SKIPPED" $S 2>/dev/null && { echo "DEL not started: B2 did not finish $(date)" >> $S; exit 1; }
  sleep 120
done
pmset -g batt | grep -q "AC Power" || { echo "DEL SKIPPED: no AC power $(date)" >> $S; exit 1; }
echo "DEL started $(date)" >> $S
for sp in eval natural_test; do
  caffeinate -is .venv/bin/python v2_reflective/e23/b2_deletion.py $sp >> v2_reflective/e23/b2_deletion_$sp.out 2>&1 && echo "DEL_${sp}_DONE $(date)" >> $S || { echo "DEL_${sp}_FAILED $(date)" >> $S; exit 1; }
done
