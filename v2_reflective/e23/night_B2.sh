#!/bin/sh
# Overnight B2 readout (declared LOG.md 2026-10-26): wait until START, require AC power and low swap, keep the Mac awake,
# then run the Explainer + Critic loop on eval and natural_test (resumable). Steps appended to stage_B2.txt. Run from the repo root.
START=${START:-"22:30"}; S=v2_reflective/e23/stage_B2.txt; LOG=v2_reflective/e23/night_B2.log
echo "$(date) night_B2 scheduled for $START" >> $LOG
while [ "$(date +%H:%M)" \< "$START" ]; do sleep 60; done
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do                       # up to 2 h waiting for the charger
  pmset -g batt | grep -q "AC Power" && break; echo "$(date) on battery, waiting" >> $LOG; sleep 600
done
pmset -g batt | grep -q "AC Power" || { echo "B2 night run SKIPPED: no AC power by $(date)" >> $S; exit 1; }
SW=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); echo "$(date) start, swap used ${SW}M" >> $LOG
[ "${SW%.*}" -lt 6000 ] || { echo "B2 night run SKIPPED: swap ${SW}M at $(date)" >> $S; exit 1; }
echo "B2 night run started $(date)" >> $S
for sp in eval natural_test; do
  caffeinate -is .venv/bin/python v2_reflective/e23/b2_run.py $sp >> v2_reflective/e23/b2_run_$sp.out 2>&1 && echo "B2_${sp}_DONE $(date)" >> $S || { echo "B2_${sp}_FAILED $(date)" >> $S; exit 1; }
done
