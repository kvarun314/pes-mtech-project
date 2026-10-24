#!/bin/bash
# Snapshot of everything needed to reproduce/restore results (code, splits, probs, adapters, logs, outputs, paper, progress). Not a git repo, so this is the history.
# usage: v2_reflective/backup_artifacts.sh   -> ../results_archive/backups/pes_YYYYmmdd_HHMM.tar.gz + .sha256   (copy the file to a second location too: external drive or cloud)
set -e
cd "$(dirname "$0")/.."
OUT=../results_archive/backups; mkdir -p $OUT; T=$OUT/pes_$(date +%Y%m%d_%H%M).tar.gz
tar --exclude='v2_reflective/llama-server.log' --exclude='v2_reflective/mlx_model' --exclude='*.pyc' --exclude='__pycache__' \
    -czf $T v2_reflective paper docs src tests pyproject.toml CLAUDE.md README.md progress 2>/dev/null || true
shasum -a 256 $T > $T.sha256; ls -lh $T; cat $T.sha256
# Off-disk copy to Google Drive (copy only, never overwrite, only into PES_backups). Run verify_backup.sh first; skip if Drive is not mounted.
GD="$HOME/Library/CloudStorage/GoogleDrive-kvarun314@gmail.com/My Drive/PES_backups"
if [ -d "$GD" ]; then cp -n "$T" "$T.sha256" "$GD/" && echo "copied to Drive: $GD/$(basename $T)"; else echo "Drive folder not found, NOT copied off-disk"; fi
