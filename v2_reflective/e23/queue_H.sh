#!/bin/sh
# Phase H GPU queue (declared LOG.md 2026-10-24). Run from repo root. Each step resumable.
S=v2_reflective/e23/stage_H.txt
.venv/bin/python v2_reflective/e23/eval_p1.py R dev2 >> v2_reflective/e23/p1/eval_R_dev2.log 2>&1 && echo H_analyst_dev2_DONE >> $S || { echo H_analyst_FAILED >> $S; exit 1; }
.venv/bin/python v2_reflective/e23/probe_agent3.py >> v2_reflective/e23/probe_agent3.out 2>&1 && echo H_extract_DONE >> $S || { echo H_extract_FAILED >> $S; exit 1; }
