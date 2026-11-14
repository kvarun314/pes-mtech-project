# Contingency and backup plan (2026-10-18). Read with DAILY_PLAN.md. Every backup layer has a check and a fallback.

## A. Data safety: layers (no single point of failure)
| Layer | What | Check | If it fails |
|---|---|---|---|
| 1 Live files | project folder (a git repo; remote `origin` = github kvarun314/pes-mtech-project) | `LOG.md` entry per experiment | restore from layer 2 |
| 2 Local snapshot | `v2_reflective/backup_artifacts.sh` -> `../results_archive/backups/pes_*.tar.gz` + `.sha256` (first one 2026-10-18, 353MB) | `v2_reflective/verify_backup.sh` (checksum, readable, extracts, key files byte-identical to live; exit 0 only if all pass) | re-run the backup; if verify still fails keep the previous good snapshot and tell Claude |
| 3 Off-disk copy | the same `.tar.gz` + `.sha256` copied to a DIFFERENT device or account (external drive, iCloud Drive, or Google Drive) | run `shasum -a 256 -c` on the copy; `verify_backup.sh <copy>` | use the other destination; a same-disk copy does NOT count as protection against disk failure |
| 4 Git remote | commit after each logged experiment on branch `e23-leakfree-oct2026` and push (user runs `! git push`, the auto-mode classifier blocks Claude pushes); tag `freeze-nov1` at the freeze. Pushed 2026-10-18 (`a1ec9fa`, verified `HEAD == origin/e23-leakfree-oct2026`). Excluded from git (see `.gitignore`): review-text data, `.scratch`, old adapters, secrets, so git alone is NOT a full backup | `git fetch` then compare `git rev-parse HEAD origin/<branch>` | layers 2-3 still hold |
| 5 Reproduction recipe | frozen splits are regenerable: `build_splits.py` (seed 2023), `build_natural_dev.py`; evidence model is deterministic (reproduction diff 0.00e+00); LoRA adapters are NOT bit-reproducible, so adapter files are the irreplaceable items | rerun `evid_natural_dev.py` sanity check | retrain arm R locally (config in `p1/config_R.yaml`, time from `p1/train_R.log`) and re-score; treat as a new run and log it |

Irreplaceable items to protect first: `e23/p1/adapter_R/`, `e23/p1/adapter_P/`, all `probs_*.jsonl`, `LOG.md`, `*.out`, `evid_train.jsonl`/`pool.jsonl` (rebuilding the pool needs a re-stream of 2.5M review rows).

Rhythm: snapshot + verify (1) after every experiment that adds results, (2) every Friday, (3) before Nov 1 freeze, (4) before submission. The ledger has a "backup verified" line; Claude refuses to call a week done without a verified snapshot and a recorded off-disk copy.
Fail-safes on the backup itself: `verify_backup.sh` must pass before an older snapshot may be deleted; keep at least the last 3 snapshots plus the freeze snapshot forever; copies are never deleted on the same day they are replaced.

## B. Technical failures
| Failure | Detect | Backup action |
|---|---|---|
| Laptop sleeps / session dies mid-job | log stops growing; `stage*.txt` marker missing | all long jobs append and resume; use `caffeinate -i`; rerun the same command |
| MLX crash or Metal OOM | traceback in `.log` | close other apps; lower prompt cap (reviews already truncated to 1,200 chars); reload and resume; fall back to `llama-server` GGUF path (`.scratch/gguf`, proven for Phase 1) |
| Disk full | `df -h` < 20GB | delete old snapshots only after verify passes; drop `llama-server.log`; move scratch to the external drive |
| Corrupt/partial jsonl line | json decode error in a script | resume logic skips complete ids; truncate the last partial line and rerun |
| Package break (pip/venv) | import error | `.venv` pinned in `pyproject.toml`; recreate venv; system python3 has python-pptx for deck |
| Adapter file lost or corrupt | load error | restore from layer 2/3; else retrain (see layer 5) |

## C. Scientific failures (decide in advance, so no panic later)
| If | Then |
|---|---|
| Natural-prior stacker removes the natural_test gain (C1-C4 not above Evidence alone) | paper angle becomes B: leak audit + preprocessing ablation + honest result that a classical evidence model carries Phase 2; Phase 2 reported as "modest complement", no "agentic wins" claim |
| Gate 0 (evidence-grounded adjudicator) fails | report as a negative result with numbers; keep routing (P2-1) as cost-saving analysis only; skip the adjudicator run (Oct 30) and use the day for the LoRA seed-2 variance |
| Base LLaMA without LoRA beats arm R | report it: LoRA on 2K rows adds nothing; Phase 1 contribution becomes the protocol/leak audit, not the adapter |
| Encoder baseline beats everything | report as the strongest baseline; frame the contribution as audit + the agent-layer cost/benefit analysis, not SOTA accuracy |
| Variance shows stack-minus-evidence CI includes 0 on both splits | state that the stack is not reliably better than evidence alone; do not claim improvement |
| Results disagree between dev CV and eval | trust the pre-declared protocol, report the discrepancy; never re-tune on eval |
| Adapter R seed-2 differs by more than the claimed gains | report adapter variance and widen all claims accordingly |
Rule: a negative result is a valid paper section. The paper never claims more than the logged numbers.

## D. Schedule and writing failures
| Failure | Backup |
|---|---|
| Missed days | catch-up slots and queue bundling in DAILY_PLAN.md; hard stops listed there |
| Numbers not frozen by Nov 1 | freeze whatever is complete on Nov 2, mark missing items as not run, cut order: seed 2, 20K encoder, extra ablations |
| Draft 1 not ready Nov 8 | send a skeleton with results tables and figures plus the sections written so far; supervisor feedback on structure still helps |
| Similarity or AI score above target | rewrite flagged passages in own words; remove any scaffold text; check citations and quotations; rerun the institution checker; leave 3 days for this (Nov 16-18) |
| Supervisor asks for a new experiment after the freeze | log justification; only if it fits Nov 9-15; otherwise list as future work |
| Illness / lost days late in November | submit the thesis/paper with the frozen results and a clearly marked limitations section; deck is already built (v5) and rebuilds in one command |

## E. Cost/compute fallbacks
Local first. If a step cannot finish locally: shrink (subset, smaller encoder), then split across nights, then tell the user it needs Colab with a GPU-hour estimate. Colab credits are 0 as of 2026-10-09; no plan step depends on them.
