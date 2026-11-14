#!/bin/bash
cd /Users/varunk/Documents/PES_Project_Phase1/pes-mtech-project
until grep -q FINISHED v2_reflective/zs_stage.txt; do sleep 10; done
llama-server -m .scratch/gguf/Meta-Llama-3-8B.Q4_K_M.gguf --lora .scratch/phase1_lora.gguf -c 8192 --parallel 1 --port 8090 > v2_reflective/llama-server.log 2>&1 &
SP=$!
until curl -s localhost:8090/health | grep -q ok; do sleep 3; done
for s in dev eval natural_test; do .venv/bin/python v2_reflective/probs_zeroshot.py v2_reflective/e23/$s.jsonl v2_reflective/e23/zs_${s}_probs.jsonl; echo ${s}_DONE >> v2_reflective/e23/stage.txt; done
kill $SP; echo FINISHED >> v2_reflective/e23/stage.txt
