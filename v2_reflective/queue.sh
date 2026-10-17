#!/bin/bash
# waits for the pair-adjudicator training to exit, then: Gate 0 -> Phase 1 answer-only retrain -> eval. Logs next to each step.
cd /Users/varunk/Documents/PES_Project_Phase1/pes-mtech-project
while kill -0 12642 2>/dev/null; do sleep 30; done
PY=.venv/bin/python
$PY v2_reflective/gate0.py > v2_reflective/gate0.log 2>&1; touch v2_reflective/gate0.done
$PY -m mlx_lm lora --config v2_reflective/p1_config.yaml > v2_reflective/p1_train.log 2>&1
$PY v2_reflective/eval_p1_answeronly.py > v2_reflective/p1_eval.log 2>&1; touch v2_reflective/p1.done
