#!/bin/bash
# Proves a backup is restorable: checksum matches, archive lists cleanly, extracts to a scratch dir, key files exist and match the live copies byte for byte.
# usage: v2_reflective/verify_backup.sh [backup.tar.gz]   (default = newest). Exit code 0 only if every check passes.
cd "$(dirname "$0")/.."
B=${1:-$(ls -t ../results_archive/backups/pes_*.tar.gz | head -1)}; fail=0
shasum -a 256 -c "$B.sha256" >/dev/null 2>&1 && echo "OK checksum" || { echo "FAIL checksum"; fail=1; }
tar -tzf "$B" >/dev/null 2>&1 && echo "OK archive readable" || { echo "FAIL archive unreadable"; fail=1; }
X=$(mktemp -d); tar -xzf "$B" -C "$X" 2>/dev/null
for f in v2_reflective/LOG.md v2_reflective/e23/phase2_e23.out v2_reflective/e23/p1/probs_R_eval.jsonl v2_reflective/e23/p1/adapter_R/adapters.safetensors v2_reflective/e23/eval.jsonl v2_reflective/e23/natural_dev.jsonl; do
  if [ ! -f "$X/$f" ]; then echo "FAIL missing in backup: $f"; fail=1
  elif [ "$f" != "v2_reflective/LOG.md" ] && ! cmp -s "$X/$f" "$f"; then echo "FAIL differs from live: $f"; fail=1; fi
done
rm -rf "$X"; [ $fail = 0 ] && echo "BACKUP VERIFIED: $B" || echo "BACKUP NOT SAFE: $B"; exit $fail
