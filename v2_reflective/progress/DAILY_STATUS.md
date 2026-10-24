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
| Oct 31 | frozen tables + figures | REGENERATED Oct 24 with Phase H rows (SH main); rerun Nov 1 | results_frozen.md, freeze_tables.out |
| Nov 1 | freeze + angle decision + deck rebuild | deck v6 + paper results tables DONE Oct 24 (early); freeze still Nov 1 | hard stop; catch-up in the morning |
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
| 2026-10-22 | `../results_archive/backups/pes_20261022_1502.tar.gz` | see sha256 | NOT YET |
| 2026-10-24 | `../results_archive/backups/pes_20261024_1641.tar.gz` (after Phase H readout, before variance_H) | PASSED (re-run Oct 24 evening) | NOT YET |
| 2026-10-24 | `../results_archive/backups/pes_20261024_2301.tar.gz` (after variance_H + tables) | PASSED | NOT YET |
| 2026-10-24 | `../results_archive/backups/pes_20261024_2306.tar.gz` (after deck v6 + paper section; .pptx files live outside the archive, rebuild with build_deck_e23.py) | PASSED | Google Drive `My Drive/PES_backups/` (sha256 match on Drive copy; both files confirmed in Drive web, Oct 24 night) |
| 2026-10-24 | git 3b67b86 pushed to origin/e23-leakfree-oct2026 (freeze prep, deck builder, paper section, notes) | HEAD==origin verified | GitHub |
| 2026-10-21 | git e8c7ae7 (4 commits squashed; enc2k_model gitignored) pushed to origin/e23-leakfree-oct2026 | HEAD==origin verified | GitHub |

## Improvement-plan jobs (IMPROVEMENT_PLAN.md, added Oct 21)
| Job | Status | Note |
|---|---|---|
| A1 metrics_extra | DONE Oct 21 | polarity gain larger than 5-class |
| B1 MiniLM embedding agent | DONE Oct 21 (5 min) | weaker than TF-IDF |
| B2 char n-gram agent | DONE Oct 21 | |
| B3 dev-gated stack | DONE Oct 21: FAIL all | agents dropped |
| C1 LoRA-10K smoke test | DONE Oct 21: 4 it/min, 3.4 h projected | no Colab |
| C2 LoRA-10K train+score dev | RUNNING (started 21:55 Oct 21) | queue_R10.sh, p1/stage_R10.txt; gate dev>=58 |
| D1/D2 Reflector fit | DONE Oct 22 (see Phase D) | |
| C R10 gate | DONE Oct 22: FAIL (dev 56.4 < 58) | eval/natural_test not scored |
| F LLaMA hidden-state probe | DONE Oct 22: gate FAIL | probe_agent.py, probe_gate_F |
| Polarity CIs | DONE Oct 22 | polarity_ci.out: significant on both splits/seeds |
| D pooled-dev Reflector fit | DONE Oct 22: gate PASS, NOT adopted (prior shift) | phase_D_gate.out |
- 2026-10-22 Phase F probe gate: FAIL (natural_dev .803 vs .804); logged in LOG.md
- 2026-10-22 Phase D: gate PASS (dev), readout shows prior shift only; NOT adopted. Logged.
- 2026-10-24 Phase F2 multi-layer probe: gate FAIL (dev +.005). Probe line closed.
- 2026-10-24 Phase G2 (500K Evidence): gate FAIL; E6 alone better, stack not. Logged.
- 2026-10-24 Phase H: gate PASS, readout eval 67.5 / natural_test 75.7 (macro-F1 .675/.612). New main config SH = P3+E6b. Logged.
- 2026-10-24 Phase H variance (5 seeds, E6b fixed): SH eval 65.8±0.6, natural_test 75.4±0.8; gain over each single agent 5/5 seeds. Logged.
- 2026-10-24 freeze_tables.py extended with Phase H rows; tables regenerated.
- 2026-10-24 late: deck v6 + shared paper results section (both tex) done for Oct 25 presentation. Logged.
