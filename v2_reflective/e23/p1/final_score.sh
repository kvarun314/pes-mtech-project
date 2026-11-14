#!/bin/bash
cd /Users/varunk/Documents/PES_Project_Phase1/pes-mtech-project
for S in eval natural_test; do .venv/bin/python v2_reflective/e23/eval_p1.py R $S > v2_reflective/e23/p1/eval_R_$S.log 2>&1; echo R_${S}_DONE >> v2_reflective/e23/p1/stage2.txt; done
