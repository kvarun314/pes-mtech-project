#!/bin/bash
cd /Users/varunk/Documents/PES_Project_Phase1/pes-mtech-project
P=.venv/bin/python
for A in P R; do
  $P -m mlx_lm lora --config v2_reflective/e23/p1/config_$A.yaml > v2_reflective/e23/p1/train_$A.log 2>&1
  $P v2_reflective/e23/eval_p1.py $A dev > v2_reflective/e23/p1/eval_${A}_dev.log 2>&1
  echo ${A}_DONE >> v2_reflective/e23/p1/stage.txt
done
echo FINISHED >> v2_reflective/e23/p1/stage.txt
