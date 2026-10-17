#!/bin/bash
# Snapshot of everything needed to reproduce/restore results (code, splits, probs, adapters, logs, outputs, paper, progress). Not a git repo, so this is the history.
# usage: v2_reflective/backup_artifacts.sh   -> ../results_archive/backups/pes_YYYYmmdd_HHMM.tar.gz + .sha256   (copy the file to a second location too: external drive or cloud)
set -e
cd "$(dirname "$0")/.."
OUT=../results_archive/backups; mkdir -p $OUT; T=$OUT/pes_$(date +%Y%m%d_%H%M).tar.gz
tar --exclude='v2_reflective/llama-server.log' --exclude='v2_reflective/mlx_model' --exclude='*.pyc' --exclude='__pycache__' \
    -czf $T v2_reflective paper docs src tests pyproject.toml CLAUDE.md README.md progress 2>/dev/null || true
shasum -a 256 $T > $T.sha256; ls -lh $T; cat $T.sha256
