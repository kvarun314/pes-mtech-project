#!/bin/bash
# Arm R10: train (resume-safe via adapter check) then score dev only. Gate decided manually. Markers p1/stage_R10.txt
cd "$(dirname "$0")/../.."
P=.venv/bin/python; S=v2_reflective/e23/p1
[ -f $S/adapter_R10/adapters.safetensors ] && grep -q R10_TRAIN_DONE $S/stage_R10.txt 2>/dev/null || { $P -m mlx_lm lora --config $S/config_R10.yaml >> $S/train_R10.log 2>&1 && echo R10_TRAIN_DONE >> $S/stage_R10.txt || { echo R10_TRAIN_FAILED >> $S/stage_R10.txt; exit 1; }; }
$P v2_reflective/e23/eval_p1.py R10 dev >> $S/eval_R10_dev.log 2>&1 && echo R10_dev_DONE >> $S/stage_R10.txt || echo R10_dev_FAILED >> $S/stage_R10.txt
