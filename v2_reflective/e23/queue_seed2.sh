#!/bin/bash
# LoRA seed 2 (arm R2): train then score dev/eval/natural_test. Resumable: skips training if adapter exists; eval_p1.py resumes. Markers p1/stage_seed2.txt
cd "$(dirname "$0")/../.."
P=.venv/bin/python; S=v2_reflective/e23/p1
[ -f $S/adapter_R2/adapters.safetensors ] || { $P -m mlx_lm lora --config $S/config_R2.yaml >> $S/train_R2.log 2>&1 && echo R2_TRAIN_DONE >> $S/stage_seed2.txt || { echo R2_TRAIN_FAILED >> $S/stage_seed2.txt; exit 1; }; }
for X in dev eval natural_test; do
  $P v2_reflective/e23/eval_p1.py R2 $X >> $S/eval_R2_$X.log 2>&1 && echo R2_${X}_DONE >> $S/stage_seed2.txt || { echo R2_${X}_FAILED >> $S/stage_seed2.txt; exit 1; }
done
$P v2_reflective/e23/seed2_report.py > v2_reflective/e23/seed2_report.out 2>&1; echo seed2_finished >> $S/stage_seed2.txt
