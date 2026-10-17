#!/bin/bash
cd /Users/varunk/Documents/PES_Project_Phase1/pes-mtech-project
llama-server -m .scratch/gguf/Meta-Llama-3-8B.Q4_K_M.gguf --lora .scratch/phase1_lora.gguf -c 8192 --parallel 1 --port 8090 > v2_reflective/llama-server.log 2>&1 &
SP=$!
until curl -s localhost:8090/health | grep -q ok; do sleep 3; done
P=.venv/bin/python
$P v2_reflective/probs_zeroshot.py .scratch/kaggle_dev_rows.jsonl v2_reflective/zs_dev_probs.jsonl
$P v2_reflective/probs_zeroshot.py .scratch/kaggle_eval_rows.jsonl v2_reflective/zs_eval_probs.jsonl
echo EVAL_DONE > v2_reflective/zs_stage.txt
$P v2_reflective/probs_zeroshot.py v2_reflective/paper_protocol_rows.jsonl v2_reflective/zs_paper_probs.jsonl
kill $SP; echo FINISHED > v2_reflective/zs_stage.txt
