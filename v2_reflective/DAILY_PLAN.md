# Daily plan Oct 18 – Nov 20, 2026 (one runnable job per day)

All local (M1 16GB). Colab is flagged explicitly (none expected). Run commands from `pes-mtech-project/`. Prefix long jobs with `caffeinate -i` so the laptop does not sleep (the natural_dev scoring slowed badly overnight because of sleep). Every job is resumable or short. After each job: tell Claude "day N done" and it will read the output, append a dated entry to `LOG.md` and update `progress/week_NN.md`. Scripts marked (Claude writes) are written the day before they are needed, tested on a small slice first.

Rule: dev only for choices; eval and natural_test are read once per pre-declared config. Numbers freeze Nov 1.

## Week 1 (Oct 18–25): confound, features, routing, base-LLaMA
| Day | Job (you run) | Time | Output (Claude logs) |
|---|---|---|---|
| Sun Oct 18 | Finish natural_dev analyst scoring: `caffeinate -i .venv/bin/python v2_reflective/e23/eval_p1.py R natural_dev >> v2_reflective/e23/p1/eval_R_natural_dev.log 2>&1` (resumes where it stopped; skip if `probs_R_natural_dev.jsonl` has 500 rows). Then `.venv/bin/python v2_reflective/e23/stack_week1.py \| tee v2_reflective/e23/stack_week1.out` | 20 min + 1 min | `stack_week1.out`, `stack_week1_results.json` (C0–C4, S1/P2-1/P2-3) |
| Mon Oct 19 | Base LLaMA, no LoRA, zero-shot on dev + eval (Claude adds an `--noadapter` option to `eval_p1.py`) | ~55 min | `p1/probs_base_dev.jsonl`, `probs_base_eval.jsonl` |
| Tue Oct 20 | Same on natural_test (2,000) | ~75 min | `probs_base_natural_test.jsonl` |
| Wed Oct 21 | `e23/base_vs_lora.py` (Claude writes): base vs arm R vs stack, paired CIs. Also `e23/make_figures.py` (confusion matrices, per-class recall, main bar chart) | 2 min | `base_vs_lora.out`, `paper/figures/e23/*.png` |
| Thu Oct 22 | Adjudicator Gate 0, step 1: build prompts from disagreement rows of dev + natural_dev with the Evidence model's top n-grams (Claude writes `e23/gate0_build.py`) | 2 min | `e23/gate0_prompts.jsonl` |
| Fri Oct 23 | Gate 0, step 2: local LLM (Q4 base, optionally arm R adapter) answers the two-candidate question; resumable (`e23/gate0_run.py`) | ~30–60 min | `e23/gate0_answers.jsonl` |
| Sat Oct 24 | Gate 0 score (`e23/gate0_score.py`): pairwise accuracy vs the stack on the same rows. Verdict pass/fail vs the criterion written in PROJECT_ROADMAP | 1 min | `gate0_e23.out`; LOG verdict |
| Sun Oct 25 | Rest day / buffer. Claude writes `progress/week_01.md` | – | week_01 note |

## Week 2 (Oct 26–Nov 1): baselines, variance, decision, freeze
| Day | Job | Time | Output |
|---|---|---|---|
| Mon Oct 26 | DistilBERT fine-tune on the same 2,000 Electronics rows as the LoRA, score dev/eval/natural_test (`e23/encoder_2k.py`, Claude writes; MPS) | ~30–60 min | `probs_enc2k_*.jsonl`, `encoder_2k.out` |
| Tue Oct 27 | Compare encoder-2K vs arm R vs base vs evidence vs stack; table for the paper | 1 min | `baselines_table.out` |
| Wed Oct 28 | DistilBERT on a 20K-row subset overnight (`encoder_20k.py`). If projected > 10 h, stop and ask; the 100K full run would be the Colab candidate, only if you want it | ≤ 8 h | `encoder_20k.out` |
| Thu Oct 29 | Variance: 5 resamples of the evidence-training set + stacker refits; spread of stack-minus-evidence on eval and natural_test (`e23/variance.py`, Claude writes) | ~20–40 min | `variance.out` |
| Fri Oct 30 | If Gate 0 passed: run the adjudicator once on eval + natural_test disagreement rows. If it failed: optional second LoRA seed for arm R (local, time per `p1/train_R.log`) to estimate adapter variance | 1–2 h | `adjudicator_eval.out` or `p1/adapter_R_seed2/` |
| Sat Oct 31 | Assemble the frozen results table and figures from the logs (`e23/freeze_tables.py`) | 2 min | `paper/figures/e23/`, `results_frozen.md` |
| Sun Nov 1 | **Numbers freeze + angle decision with Claude.** Rebuild deck: `/opt/homebrew/bin/python3 v2_reflective/build_deck_e23.py` | 1 min | `progress/week_02.md`, deck v5 refresh |

## Week 3 (Nov 2–8): you write, Claude supplies facts
One section per day written by you, from `results_frozen.md` and the outline. Claude supplies tables, figures, sourced facts, reference checks, never final prose.
Mon intro + related work outline · Tue method (agents, splits, protocol) · Wed experimental setup · Thu results · Fri leak audit + ablations · Sat discussion, limitations, negative results · Sun assemble draft 1 → supervisor + similarity check #1 (log the % in `progress/`).

## Week 4 (Nov 9–15): feedback, thesis, reproducibility
Mon–Tue supervisor fixes · Wed tag the freeze in git (you run the push), one-command-per-experiment README, pin dependencies · Thu archive bundle with checksums (adapters, probs, traces, `.out` files, LOG.md) · Fri docs cleanup (CLAUDE.md, README, summaries) · Sat thesis sections as required · Sun draft 2 + similarity check #2.

## Week 5 (Nov 16–20): buffer only
Proofread, final similarity and AI-disclosure checks, final PDFs, submission, viva rehearsal with the deck. Nov 21–22 spare.

## If a day slips
Shift the next job; do not skip Gate 0, the variance run or the Nov 1 freeze. Drop the 20K encoder and the second LoRA seed first. Never run an experiment after Nov 1 without a LOG.md justification.

## Missed days and catch-up (user, 2026-10-18: not every day is possible, make up lost time later)
- **Ledger:** `progress/DAILY_STATUS.md` lists every job with a status (todo / done / skipped / moved). Claude updates it whenever you say "day N done" or "skip day N", and at the start of each session it reads the ledger first and reports what is behind.
- **Slipped jobs do not vanish:** a missed job keeps its place in order (later jobs that depend on it wait) and is marked `moved`. Independent jobs can run out of order.
- **Catch-up slots already reserved:** Sundays (Oct 25, Nov 1 morning, Nov 8, Nov 15) and the buffer days Nov 16–20 are first used for moved jobs, before any rest.
- **Batching:** jobs with no human step in between are chained into one overnight queue (Claude writes `e23/queue_catchup.sh`, resumable, logs each stage to `stage_catchup.txt`). Typical bundle: base-LLaMA dev+eval+natural_test (~2.5 h) or encoder 2K + variance. So one lazy evening can recover 2–3 missed days.
- **Hard stops (cannot slip past):** Gate 0 verdict and the variance run before Nov 1 (numbers freeze); draft 1 to the supervisor by Nov 8; submission by Nov 20. If the ledger shows these at risk, Claude says so immediately and proposes what to cut (order: second LoRA seed, 20K encoder, extra ablations) instead of silently shrinking scope.
- **Writing weeks:** if a section day is missed, the missed section moves to the next free day; draft 1 date does not move, so sections can be shorter first drafts.
