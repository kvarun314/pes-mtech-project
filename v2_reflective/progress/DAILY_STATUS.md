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
- 2026-10-26/27 B2: Explainer + Critic loop on all 3,000 test reviews; faithful 60.5/63.4% -> 88.9/88.8%, 793 fixed / 0 broken; deletion test passes. Battery pause overnight, resumed with nothing lost. Logged.
- 2026-10-28 Task 11/12: B2 report, paper tables, agent audit, X1 (Phase 1 vs Phase 2 explanations), H1 (hard cases), final demo notebook (check_demo PASS), deck at 25 slides. Logged.
- 2026-10-28 I1 PASS: Critic max 4 rewrites -> faithful 90.6 / 91.1%, 0 broken; now the default. I2 (learned Hand-off) FAIL, I3 (fixed-form fallback) FAIL. Logged.
- 2026-10-28 Improvement round (plan 2026-10-28-phase2-v3-improve-plan.md): Q1 star ranges PASS (person needed for 16% / 2% instead of 58% / 44%); Q4 robustness (flat text flips 11.7% vs Phase 1 29.7%; no title -5.6); Q5 gain positive in every year/popularity band; Q2 dropped; Q3/Q3b second judge FAIL (constant verdict). Deck, paper, notes updated. Logged.
- 2026-10-28 Disk: 11 old backup tarballs moved to Google Drive after Drive confirmed each upload (user's request); free 14 -> 33 GB; newest tarball kept locally.
- Pending (user): human check of 50 explanations (`human_check.py score` after marking). Next dates: freeze 1 Nov 1, freeze 2 Nov 5, paper draft Nov 8.
- 2026-11-02 Round R: O1 kappa .888 / .881 vs Phase 1 .824 / .779; G1 gradient-boosted Reflector FAIL; N1 NB-SVM FAIL. Main stack unchanged (67.5 / 75.7). D1 with the user on Colab. Freeze 1 (due Nov 1) not done yet.
- 2026-11-05 D1 DeBERTa: gate PASS; eval +0.0, natural_test +2.3 (78.0); alone 66.5 / 75.75. Adoption pending the user. Freeze 2 due today; Freeze 1 still not done.
- 2026-11-05 D2 DeBERTa seeds 0, 1, 2: gain eval +0.1 ± 0.3 (not stable), natural_test +2.2 ± 0.2 (stable). Selects nothing; adoption pending the user. 14 GB of unused local models copied to Drive `PES_backups/local_models/` (MD5 matched Drive metadata), then the local copies deleted at the user's OK.
- 2026-11-10 SH3 declared: user adopted DeBERTa as the third agent (option B). Code and rerun queue ready.
- 2026-11-12 SH3 reruns done (queue_SH3.sh, ~16 h): main stack 67.5 / 78.0; Critic loop 62.3/63.65% -> 90.5/91.6%, 0 broken; Router gate FAIL with three agents, Router off; no-title flips 15.7% (limitation). Two stalls: iCloud evicted .venv files twice, and the base LLaMA had to be copied back from Drive for the deletion test. User: iCloud is not used; Drive is the only storage. Logged.
- 2026-11-12 SH3 write-up: EXPERIMENTS_SUMMARY, README, paper tables and FACT lines, deck (25 slides) + speaker notes, Presenter_Notes Part 0, demo notebook, results index updated. Demo Router fix: tau used only when gate_B1b.json says pass.
- Pending (user): score the new human-check sheet `v2_reflective/e23/b2/human_check_50.csv`; paper prose; turn off iCloud Documents sync. Submission Nov 20.
- 2026-11-14 Deck: slide 23 now shows a worked explanation example from the live demo (review, evidence pack, three drafts with Critic feedback; read at build time from the gitignored demo cache, no review text committed). Phase 1 reframed everywhere in the deck, notes and demo as "the reference paper we chose, reproduced and explored", not "the baseline" (user).
- 2026-11-14 Storage: turning off iCloud "Desktop & Documents" hid the files from the Mac, so the user turned it back on. Instead the project is being copied to `~/PES_Project_Phase1` (outside iCloud, .venv rebuilt there); the old folder stays until the user OKs deleting it.
- TO DO (Claude): paper references. Add many more real, verifiable references (LLM sentiment, LoRA / QLoRA, LLaMA-3, DeBERTa, stacking and calibration, conformal prediction, selective prediction, faithfulness / ERASER deletion tests, self-refine and critic loops, LLM-as-judge, multi-agent LLM systems, Amazon Reviews 2023). Each with DOI or arXiv link; PDF link where openly available. Verify every entry exists before adding; no invented citations. Bib entries only, no prose.
- TO DO (Claude): rebuild Final_Demo.ipynb from the new folder (build_final_demo.py needs .venv; the current .venv is evicted by iCloud), rerun check_demo.py and paper/integrity/audit.py there.
- Pending (user): score `v2_reflective/e23/b2/human_check_50.csv`; paper prose. Submission Nov 20.
- 2026-11-14 Project moved out of iCloud: copied to `~/PES_Project_Phase1` (8,877 files, identical by rsync dry-run, 0 cloud-only files). New `.venv` with the same 161 package versions; Jupyter kernel `pes-mtech` re-registered to it; model symlinks made relative; old run scripts repointed. Tests 168 passed. check_demo: 10/10 PAPER numbers match, but runtime 335 s (limit 180 s) with heavy swap from other open apps. The old `~/Documents/PES_Project_Phase1` is kept until the user OKs deleting it.
- 2026-11-14 Conference paper: full draft content written (Claude-drafted at the user's request; the user rewrites it). 45 verified references. audit.py: 100% AI-drafted until the user's rewrite is committed.
- TO DO (user): before the presentation, quit other apps (Teams, Chrome, Notion, Spotify, MySQL) or restart the Mac, then run `check_demo.py` the day before; the demo needs ~10 GB free for the two LLaMA models.
