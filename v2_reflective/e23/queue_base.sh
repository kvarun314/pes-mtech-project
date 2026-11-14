#!/bin/bash
# Base LLaMA-3-8B (Q4, no LoRA) zero-shot first-digit probs on dev, eval, natural_test. Resumable; stage markers in p1/stage_base.txt; logs p1/eval_base_<split>.log
cd "$(dirname "$0")/../.."
for S in dev eval natural_test; do
  .venv/bin/python v2_reflective/e23/eval_p1.py base $S >> v2_reflective/e23/p1/eval_base_$S.log 2>&1 && echo base_${S}_DONE >> v2_reflective/e23/p1/stage_base.txt || { echo base_${S}_FAILED >> v2_reflective/e23/p1/stage_base.txt; exit 1; }
done
