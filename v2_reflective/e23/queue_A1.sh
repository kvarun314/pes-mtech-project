#!/bin/sh
# Phase A1 GPU queue (declared LOG.md 2026-10-24 night). Run from repo root. Resumable: rerun to continue.
S=v2_reflective/e23/stage_A1.txt
[ -f v2_reflective/e23/fit3.jsonl ] || .venv/bin/python v2_reflective/e23/build_A1.py >> v2_reflective/e23/probe_agent4.out 2>&1 || { echo A1_build_FAILED >> $S; exit 1; }
.venv/bin/python v2_reflective/e23/probe_agent4.py >> v2_reflective/e23/probe_agent4.out 2>&1 && echo A1_extract_DONE >> $S || { echo A1_extract_FAILED >> $S; exit 1; }
