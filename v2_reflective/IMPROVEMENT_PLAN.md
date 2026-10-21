# Improvement plan (written 2026-10-21, before any of these runs). Goal: raise the stack and strengthen Phase 1 honestly, with kill criteria. Local only; Colab is NOT needed (see Colab section).

## Why: what limits us today
1. Evidence (plain TF-IDF) carries the stack; Analyst adds a small, robust +1.5..+4.0. 2. The Analyst is data-starved (2,000 rows vs Evidence 100,000). 3. 5-class exact has a label-noise ceiling (within-1 already ~95%). Plan targets 1, 2, 3.

## Rules (unchanged)
Choices on dev only (dev 500 balanced + natural_dev 500). eval/natural_test read once per declared config. Every experiment: declared in LOG.md BEFORE running, resumable script, `.out` file, ledger row, backup, push. Count of declared configs is reported (multiple-testing note). No result is dropped because it is bad.

## Phase A — metrics that suit the task (Oct 22, ~1 h, zero compute)
A1 `e23/metrics_extra.py`: from stored probs recompute for Analyst, Evidence, base, DistilBERT, stacks: 3-class polarity (1-2 neg / 3 neu / 4-5 pos) accuracy + macro-F1, binary (excl. 3), within-1, MAE, ordinal expected-rating MAE. Output `metrics_extra.out`. Gate: none (reporting only). Expected: polarity high 80s.

## Phase B — stronger second agent (Oct 22-24, local CPU/MPS, ~1-2 h)
B1 Embedding agent: all-MiniLM-L6-v2 (cached) sentence embeddings (title+text, 256 tokens) for evid_train 100K + dev + natural_dev + eval + natural_test; LR on embeddings. Est. embedding 100K ~ 25-40 min on M1. Resumable in chunks (`e23/embed_agent.py`, `emb_*.npy`).
B2 Char/word n-gram agent: char_wb 2-5 TF-IDF + LR (CPU, ~20 min) as a lexically different view.
B3 Stack: Analyst + TF-IDF + (B1) + (B2) log-probs, LR Reflector. 
GATE (dev only): 5-fold CV on dev: add an agent only if 3/4-agent stack beats the current [A+E] stack by >= +1.0 exact AND macro-F1 does not drop, and the same direction on natural_dev CV. Pass -> read out once on eval + natural_test and add to the ladder. Fail -> log, report as negative, drop.

## Phase C — fix the data-starved Analyst: LoRA-10K (Oct 23-25, local MLX, overnight)
Measured speed: 400 iters x 12 rows took ~2 h 15 m (adapter timestamps), i.e. ~2,100 rows/h plus scoring ~2.1 h for dev+eval+natural_test.
C1 Smoke test: 20 iters, record it/s and peak memory; project time.
C2 Run: arm R10 = 10,000 raw balanced rows (2,000/star, same pool, disjoint from every eval split), same hyperparameters, ONE epoch (~800 iters at batch 12 = ~4.5 h), seed 42. Score dev first (~18 min).
GATE (dev): R10 dev exact >= 58 (R: 55.2; margin ~ +/-4.4, so this is a direction) -> score eval + natural_test (~1.8 h) and add to ladder. Else stop after dev, log "more data does not rescue the LoRA", keep as a finding.
Time guard: if C1 projects > 9 h training, cut to 6,000 rows (500 iters). If even that > 9 h: tell the user (possible Colab case, see below).

## Phase D — better Reflector fit (Oct 25-26, minutes)
D1 Fit the Reflector on pooled dev + natural_dev (1,000 rows, balanced+natural mix with prior reweight options pre-declared as before) with 5-fold CV for dev-only reporting. D2 ordinal decoding (expected-rating rounding vs argmax) on dev CV. Keep only if dev CV gain >= +0.5 and no loss of macro-F1.

## Phase E — combine and freeze (Oct 27-31, Nov 1)
Best dev-gated combination only (not eval-gated), read out once, added to `results_frozen.md`, figures, deck, outline. Nov 1 angle decision: if any of B/C passed -> "improved stack + audit"; else angle B as in PAPER_OUTLINE.md (still publishable-quality honest study).

## Schedule (ahead of the Oct 22 -> Nov 1 window)
Oct 22 review + A1 + B1 embed start | Oct 23 B2, B3 dev gate, C1 smoke test | Oct 23-24 night C2 training | Oct 24-25 C scoring, D1/D2 | Oct 26-28 read-outs, baselines refresh | Oct 29-31 frozen tables | Nov 1 freeze. Slack: 3 days; missed days follow DAILY_PLAN catch-up rules.

## Colab policy (user, 2026-10-21: only if REQUIRED, no credits bought for nothing)
Nothing above needs Colab. Embedding, n-gram, LoRA-10K and scoring all fit M1 16GB (hours, not days). Claude will say "Colab required" only if: (a) C1 projects > 9 h even at 6,000 rows, or (b) a result is blocked and no local route exists. Even then Claude states why, estimated GPU hours, and the cheaper local alternative first.

## Fail-safes
Every phase writes resumable files; rerunning the queue resumes. Fallback if everything fails: current frozen results (PAPER_OUTLINE.md angle B). Backup + push after each phase. Do-not-do: more LLM-adjudicator prompts, more Reflector features, bigger encoder baselines.
