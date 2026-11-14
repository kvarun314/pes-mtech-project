#!/bin/bash
cd /Users/varunk/Documents/PES_Project_Phase1/pes-mtech-project
M=.scratch/gguf/Meta-Llama-3-8B.Q4_K_M.gguf
llama-server -m $M --lora .scratch/phase1_lora.gguf -c 8192 --parallel 1 --port 8090 > v2_reflective/llama-server.log 2>&1 &
SP=$!
until curl -s localhost:8090/health | grep -q ok; do sleep 3; done
.venv/bin/python v2_reflective/phase1_final.py > v2_reflective/phase1_final.log 2>&1
kill $SP
echo FINISHED >> v2_reflective/phase1_final.log
