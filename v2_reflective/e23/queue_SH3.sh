#!/bin/sh
# SH3 adoption reruns (declared LOG.md 2026-11-10). Run from the repo root:  sh v2_reflective/e23/queue_SH3.sh
# Resumable: each finished step writes "<STEP>_DONE" to stage_SH3.txt and is skipped next time; b2_run, b2_deletion and q4 also resume
# inside a step. Needs AC power and swap < 6 GB; keeps the Mac awake. Stops at the first failure. Estimated ~14 h in total.
set -u
E=v2_reflective/e23; S=$E/stage_SH3.txt; PY=.venv/bin/python; touch $S
done_() { grep -q "^$1_DONE" $S; }
mark() { echo "$1_DONE $(date)" >> $S; echo "== $1 done $(date)"; }
fail() { echo "$1_FAILED $(date)" >> $S; echo "!! $1 FAILED, see $E/$2"; exit 1; }
pmset -g batt | grep -q "AC Power" || { echo "On battery: plug in the charger and start again."; exit 1; }
SW=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); [ "${SW%.*}" -lt 6000 ] || { echo "Swap ${SW}M in use: close apps or restart, then start again."; exit 1; }
pgrep -f "b2_run.py|b2_deletion.py|q4_robustness.py" >/dev/null && { echo "A run is already going."; exit 1; }
caffeinate -is -w $$ &

# 0. Archive the two-agent (SH) outputs that the reruns overwrite. Copy only; the old explanation folder is renamed, not deleted.
if ! done_ ARCHIVE; then
  A=$E/sh2_archive; mkdir -p $A/figures
  for f in gate_B1b.json gate_B1b.out s1_selective.json s1_selective.out q1_conformal.json q1_conformal.out o1_ordinal.json o1_ordinal.out \
           q5_breakdown.json q5_breakdown.out hard_cases.json hard_cases.out analysis_C.out figures_deck.out results_frozen.md b2_report.json b2_report.out \
           x1_explain_compare.json x1_explain_compare.out q4_robustness.json q4_robustness.out b2_run_eval.out b2_run_natural_test.out; do
    [ -e $E/$f ] && cp -n $E/$f $A/$f
  done
  cp -n paper/figures/e23/*.png $A/figures/
  for f in $E/i1_more_rounds_*.out; do [ -e "$f" ] && mv -n "$f" $A/; done        # I1 (max 4 rewrites) is now the default inside b2_run
  [ -d $E/b2 ] && [ ! -e $E/b2_sh2 ] && mv $E/b2 $E/b2_sh2 && mkdir -p $E/b2
  mark ARCHIVE
fi

run() {  # run STEP LOGFILE command...
  st=$1; lf=$2; shift 2; done_ $st && return 0; echo "== $st start $(date)"
  "$@" > $E/$lf 2>&1 || fail $st $lf; mark $st
}
run DEBCHECK deb_local_check.out $PY $E/deb_local_check.py
run REFLECTOR persist_sh3.out $PY $E/persist_sh3.py
run ROUTER gate_B1b.out env TOL=0 OUT=gate_B1b $PY $E/gate_B1.py
run HANDOFF s1_selective.out $PY $E/s1_selective.py
run RANGES q1_conformal.out $PY $E/q1_conformal.py
run ORDINAL o1_ordinal.out $PY $E/o1_ordinal.py
run BANDS q5_breakdown.out $PY $E/q5_breakdown.py
run HARD hard_cases.out $PY $E/hard_cases.py
run ANALYSIS analysis_C.out $PY $E/analysis_C.py
run FIGURES figures_deck.out $PY $E/figures_deck.py
run FREEZE1 freeze_tables.out $PY $E/freeze_tables.py
# Explainer + Critic loop on all 3,000 test reviews (~9 h), then the deletion test (~4-5 h).
run B2_EVAL b2_run_eval.out $PY $E/b2_run.py eval
run B2_NATURAL b2_run_natural_test.out $PY $E/b2_run.py natural_test
run DEL_EVAL b2_deletion_eval.out $PY $E/b2_deletion.py eval
run DEL_NATURAL b2_deletion_natural_test.out $PY $E/b2_deletion.py natural_test
run B2REPORT b2_report.out $PY $E/b2_report.py
run X1 x1_explain_compare.out $PY $E/x1_explain_compare.py
run Q4 q4_robustness.out $PY $E/q4_robustness.py
run HUMANSHEET human_check_make.out $PY $E/human_check.py make
run FREEZE2 freeze_tables.out $PY $E/freeze_tables.py
run COLLECT collect_results.out $PY $E/collect_results.py
echo "ALL SH3 RERUNS DONE $(date)" | tee -a $S
