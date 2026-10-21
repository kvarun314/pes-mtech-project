# Daily job ledger (updated by Claude on "day N done" / "skip day N"). Status: todo | running | done | skipped | moved
| Date | Job | Status | Note |
|---|---|---|---|
| Oct 18 | natural_dev analyst scoring finish; run stack_week1.py | done | scored 500/500; stack_week1.out + prior_baseline.out; see LOG 2026-10-18 (natural_test exact gains are mostly class-prior; report macro-F1) |
| Oct 19 | base LLaMA (no LoRA) dev + eval | DONE (moved earlier, overnight Oct 19-20) | arm `base` in eval_p1.py |
| Oct 20 | base LLaMA natural_test | DONE | ran in same overnight queue |
| Oct 21 | base_vs_lora.py + make_figures.py | DONE (Oct 20, auto) | figures in paper/figures/e23 |
| Oct 22 | Gate 0 build prompts | DONE (Oct 20) | gate0_e23.py |
| Oct 23 | Gate 0 run | DONE (Oct 20) | |
| Oct 24 | Gate 0 score + verdict | DONE (Oct 20): FAIL, LLM 54.2 vs stack 73.3 | hard stop met |
| Oct 25 | rest / catch-up + week_01 note | todo | catch-up slot |
| Oct 26 | DistilBERT 2K | DONE (Oct 21): enc2k eval 52.8 vs LoRA 54.0 ns | |
| Oct 27 | baselines table | DONE (in encoder_2k.out, Oct 21) | |
| Oct 28 | DistilBERT 20K overnight | todo | first to cut if behind |
| Oct 29 | variance run | DONE (Oct 20): stack-Evidence +2.7+/-1.3 eval, +4.3+/-1.1 natural_test | hard stop met |
| Oct 30 | adjudicator run (if Gate 0 passed) or LoRA seed 2 | DONE (Oct 21): stack stable across seeds; single-analyst natural_test exact seed-unstable (53.4 vs 74.8) | queue_seed2.sh, seed2_report.out |
| Oct 31 | frozen tables + figures | DRAFT DONE (Oct 21), rerun Nov 1 with seed 2 | results_frozen.md |
| Nov 1 | freeze + angle decision + deck rebuild | todo | hard stop; catch-up in the morning |
| Nov 2–7 | one paper section per day (user writes) | todo | |
| Nov 8 | draft 1 to supervisor + similarity check #1 | todo | hard stop; catch-up slot |
| Nov 9–14 | supervisor fixes; git init; archive; docs; thesis | todo | |
| Nov 15 | draft 2 + similarity check #2 | todo | catch-up slot |
| Nov 16–20 | buffer, final checks, submission | todo | submit by Nov 20 |

## Backup ledger (update after each snapshot)
| Date | Snapshot | verify_backup.sh | Off-disk copy (where) |
|---|---|---|---|
| 2026-10-18 | `../results_archive/backups/pes_20261018_0007.tar.gz` | PASSED | NOT YET (needs user: destination?) |
| 2026-10-18 | git branch `e23-leakfree-oct2026` `a1ec9fa` (code, logs, probs, small adapters; no review text) | `HEAD == origin` verified | pushed to GitHub (code only; data/.scratch not included) |
| 2026-10-20 | `../results_archive/backups/pes_20261020_1842.tar.gz` | PASSED | NOT YET (needs user: destination?) |
| 2026-10-20 | git `f6c4cff` pushed to origin/e23-leakfree-oct2026 | HEAD==origin verified | GitHub |
| 2026-10-21 | git commit 190816a (local only): push REJECTED, contains 255MB enc2k_model (>100MB GitHub limit). Fix pending user: see below | NOT pushed | last pushed = b565b4d |
