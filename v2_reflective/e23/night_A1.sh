#!/bin/sh
# Overnight A1 run (user-approved 2026-10-25): wait until START, require AC power and low swap, keep the Mac awake, resume extraction,
# then run the declared A1 gate (dev2 + natural_dev only; no test read). Every step is appended to stage_A1.txt. Run from the repo root.
START=${START:-"22:30"}; S=v2_reflective/e23/stage_A1.txt; LOG=v2_reflective/e23/night_A1.log
echo "$(date) night_A1 scheduled for $START" >> $LOG
while [ "$(date +%H:%M)" \< "$START" ]; do sleep 60; done
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do                       # up to 2 h waiting for the charger
  pmset -g batt | grep -q "AC Power" && break; echo "$(date) on battery, waiting" >> $LOG; sleep 600
done
pmset -g batt | grep -q "AC Power" || { echo "A1 night run SKIPPED: no AC power by $(date)" >> $S; exit 1; }
SW=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); echo "$(date) start, swap used ${SW}M" >> $LOG
[ "${SW%.*}" -lt 6000 ] || { echo "A1 night run SKIPPED: swap ${SW}M at $(date)" >> $S; exit 1; }
echo "A1 night run started $(date)" >> $S
caffeinate -is sh v2_reflective/e23/queue_A1.sh || exit 1         # -i -s: no idle/system sleep while on AC (lid must stay open)
echo "$(date) extraction done; running gate" >> $LOG
caffeinate -is .venv/bin/python v2_reflective/e23/gate_A1.py > v2_reflective/e23/gate_A1.out 2>&1 && echo "A1_gate_DONE $(date)" >> $S || echo "A1_gate_FAILED $(date)" >> $S
