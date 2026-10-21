# v2_reflective — separate track (existing Phase 1/2 code, results, docs stay untouched)

Goal: Phase 1 + Phase 2 reach high accuracy with an AGENTIC ORCHESTRATION design where
SELF-REFLECTION is a core feature. Every step, number, and failure is logged here.

## Rules
- Nothing in src/, results/, docs/, colab/ is modified. v2 code lives only in v2_reflective/.
- Fit on dev set (`.scratch/kaggle_dev_rows.jsonl`, 500 rows). Eval set (1000 rows) only for final read-out.
- Report all of: 5-class exact, within-1, MAE, 3-class, binary. No metric swapping without saying so.
- Negative results get logged too.

## Baseline carried in (from docs/PROGRESS.md, 2026-10-09)
- Phase 1 text_only and Phase 2 full_graph: 60.1% exact, 92.0% within-1, 79.2% 3-class, 90.8% binary (Kaggle 1000 balanced).
- Phase 2 Critic fired on 0.4% of rows (needs 2 unanimous dissenting votes) -> structurally dead.
- Calibration, kNN-RAG vote, retrieved few-shot: all no help. Errors = adjacent-star confusion (2/3/4).

## Log
### 2026-10-10 — setup
- Local run C (MLX, 4,000 examples, r16, 8 layers) CRASHED: `RuntimeError: [METAL] Command buffer
  execution failed: Internal Error`. Checkpoint at iter 1000 exists:
  `.scratch/mlx_adapter_C_r16_4k/0001000_adapters.safetensors`. No eval of it yet.

### 2026-10-10 — Phase 1, iteration-1000 checkpoint of 4,000-example run (MLX, 300 eval rows)
- 58.00% exact, 95.33% within-1, vs 59.67% for the committed adapter on the same 300 rows. More data did NOT beat baseline
  at this checkpoint (run crashed at iter ~1000 of 4000, so this is partial evidence only). Colab credits not justified yet.

### 2026-10-10 — Phase 1, paper-protocol evaluation (in progress)
- Why: the paper (`itmconf_dai2024_04021.pdf`) defines 1-2 negative / 3 neutral / 4-5 positive, tests on "4,000 reviews randomly
  selected from the remaining data" (natural, skewed distribution), and its classical baselines (DT 78.6, SVM 87.2, NB 87.6) are
  implausible for balanced 5-class. Its Table 3 also has an impossible row (DT F1 0.974 > precision 0.769 and recall 0.821).
  So 92.6% is probably not 5-class exact accuracy and is not comparable to our balanced 60.1%.
- Test set: `v2_reflective/paper_protocol_rows.jsonl`, 4,000 random Kaggle rows, seed 2024, natural distribution
  {1: 8.8%, 2: 5.6%, 3: 7.1%, 4: 14.8%, 5: 63.8%}, disjoint from the balanced eval and dev sets.
  Caveat: the committed adapter's training rows were never recorded, so overlap is excluded only statistically.
- Model: committed adapter (paper Table 1 config) via llama.cpp, Q4_K_M base, first-token P(1..5) probe (argmax matched stored
  predictions on 999/1000 earlier). Scorer: `v2_reflective/score.py`. Trivial baselines are reported next to every metric.
- Phase 2 plan written: v2_reflective/PHASE2_PLAN.md (Gate 0 = adjudicator must beat ~60% pairwise on top-2 pairs).

### 2026-10-10 — Phase 2 direction chosen: local-only fine-tuned pair adjudicator (user decision)
- Finding: the earlier local MLX runs (A/B/C) trained on `{"text": prompt+answer}`, so loss covered the whole prompt (rubric,
  instruction, review) and only ~1 answer token was the real signal. C's train loss sat flat at ~1.0–1.07 from iter 100 to 1200.
  `mlx_lm` 0.32 supports `mask_prompt` with prompt/completion data. This is a plausible reason more data did not help locally.
  To test for Phase 1 later: rerun the 400-example config with mask_prompt (compare to 59.67% on the same 300 rows).
- Built `v2_reflective/build_pair_data.py` -> `pair_data/` (1,444 train / 56 valid): "is the star rating A or B?" for adjacent pairs
  (1/2, 2/3, 3/4, 4/5: 300 each) and one-apart pairs (1/3, 2/4, 3/5: 100 each), both display orders, rows disjoint from eval,
  dev and paper-protocol sets. Config: `pair_config.yaml` (r=16, 8 layers, completion-only loss, 1,500 iters, batch 2).
- Training starts when the Phase 1 paper-protocol probe finishes (GPU contention; earlier Metal crash).

### 2026-10-10 — Phase 1 paper-protocol result (4,000 natural-distribution rows, committed adapter, Q4 llama.cpp)
- 5-class exact 70.70% (always-5 baseline 63.78%); within-1 91.67%, MAE 0.422; weighted F1 0.721, macro-F1 0.548.
- Polarity 3-class (1-2 / 3 / 4-5) 84.97% (always-positive 78.60%); weighted F1 0.865, macro-F1 0.698.
- Binary, 3-star dropped: 88.57% (always-positive 84.58%).
- Read: well above the trivial baselines but only +7 points (exact) and +6 (polarity) over them. Paper's 92.6% not reproduced; the
  gap to it is mostly protocol, since the skew inflates every number. Balanced 5-class stays the honest difficulty measure (60.1%).

### 2026-10-11 — Phase 1 prior-shift correction (offline, free, from the 4,000-row probabilities)
- Stacking multinomial LR on log P(1..5), 5-fold CV on the 4,000 natural rows: exact 79.9% (base 70.7%), polarity 92.0% (base 85.0%),
  within-1 95.2%, weighted F1 0.781 (0.721), macro-F1 0.625 (0.548).
- It is mostly a PRIOR correction: the adapter was trained balanced, test is 63.8% 5-star. 5-star recall 0.80 -> 0.95; 1-star 0.89 -> 0.78.
  Fit on natural rows and applied to the balanced eval it LOWERS exact (60.2% -> 57.9%). So it is not better discrimination.
- Use only for a skewed deployment distribution, fit on rows disjoint from the test rows, and disclose. Not a capability claim.

### 2026-10-11 — Phase 1 reframed: reproduce the paper's results; Phase 2 is the add-on (user direction)
- Classical baselines, paper Table 3 configs (DT+CountVec, SVM sigmoid g=1 + TF-IDF, MNB a=0.2), 16K natural-distribution training rows,
  tested on the SAME 4,000 paper-protocol rows as the LLaMA run (`v2_reflective/classical_paper_protocol.py`):
  | | 5-class exact | 5-class -> polarity | trained on polarity | paper Table 3 |
  | DecisionTree | 58.5% | 75.3% | 76.3% | 78.6% |
  | SVM | 70.0% | 84.5% | 84.9% | 87.2% |
  | MultinomialNB | 67.4% | 83.1% | 82.6% | 87.6% |
  Trivial baselines on the same rows: always-5 63.8% (5-class), always-positive 78.6% (polarity).
- Read: the paper's baseline numbers are far closer to POLARITY accuracy on a natural-distribution test (we get 76-85%, they report
  79-88%, with 8x more training data in their K=5 setup) than to 5-class accuracy. Supports the reading that 92.6% is a polarity-style number.
- LLaMA adapter at the same protocol: polarity 85.0% raw (92.0% with prior-shift correction, see above). Raw LLaMA is about level with
  SVM (84.5-84.9%); the paper's claim that LLaMA beats SVM/NB by 4-5 points is NOT reproduced at our scale.
- Confidence check for the "hallucination" angle: the adapter is NOT overconfident. On 4,000 rows no prediction reaches 0.9 probability;
  at >=0.7 (18% of rows) it is wrong 5.3% of the time, and only 3 rows are 2+ stars off. So confident-wrong errors are not the story;
  the generated CoT/justification text (invented details, rating-vs-reasoning contradictions, sarcasm) is the thing to inspect next.

### 2026-10-11 — Deck: added "Reproducing the Paper's Claims" slide (slide 8 of 20) with all caveats; run card saved (`PHASE1_RUN_CARD.md`)

### 2026-10-11 — Phase 2 proof of concept started (user: Phase 2 must beat Phase 1 on the things Phase 1 is weak at)
- Pair-adjudicator training launched (`pair_config.yaml`, 1,500 iters, completion-only loss). First attempt failed: base tokenizer has no
  chat template ("Cannot use chat template functions..."). Fix: `v2_reflective/mlx_model/` = symlinked weights + a plain-concatenation
  chat_template, so only loss masking is used.
- Phase 1 weaknesses we will test Phase 2 against (each with a metric, reported win OR loss):
  1. Adjacent-star confusion -> exact accuracy and per-star recall on the balanced 1,000 eval and the 4,000 paper-protocol set.
  2. Weak minority classes -> macro-F1 (Phase 1: 0.548 on the paper-protocol set).
  3. Reflection almost never fires (0.4%) -> gate coverage and fixed/regressed counts.
  4. Reasoning text that contradicts or invents -> consistency check on generated justifications (200-row sample).
  5. Phase 2 regressing vs Phase 1 (Run 2) -> paired fixed vs regressed with a significance test.
  Latency will be worse (extra calls on gated rows); it is reported as a cost, not hidden.

### 2026-10-11 — Does more training data help? Learning curve on the 4,000-row paper-protocol test (CPU, natural-distribution training rows)
| train rows | NB 5-class / polarity | TF-IDF(1-2gram) LogReg 5-class / polarity |
| 1,000 | 63.0 / 78.3 | 64.6 / 79.2 |
| 4,000 | 65.1 / 80.9 | 68.1 / 82.3 |
| 16,000 | 66.6 / 81.9 | 72.6 / 86.1 |
| 64,000 | 68.4 / 83.1 | 76.4 / 88.9 |
| 128,000 | 69.2 / 83.6 | 78.7 / 90.4 |
- Steady gain per 4x data (LogReg: about +4 exact, +3 polarity). Task is data-hungry; polarity 90.4% at the paper's 128K scale is near its 92.6%.
- LLaMA adapter (2-4K balanced training rows): 70.7 / 85.0 raw, 79.9 / 92.0 with class-balance correction. At 4K rows it beats LogReg (68.1 / 82.3);
  the 128K LogReg beats raw LLaMA. Caveat: the classical models train on natural-distribution data, so part of their gain is learning the 64% 5-star prior.
- Local MLX evidence (400 vs 4,000 examples) showed no gain, but it was partial (iter 1000/4000), loss was on prompt tokens, base was 4-bit.

### 2026-10-11 — CORRECTION to the "answer-only loss" idea for Phase 1
- Read `phase1/data/dataset.py` and `colab/llama_sentiment_baseline_train.ipynb` (line 578): the ORIGINAL Phase 1 training already masks prompt
  tokens (`labels = [-100] * prompt_len + input_ids[prompt_len:]`), i.e. it already trains on the answer only.
- So "answer-only loss" is NOT a new lever over the committed adapter. It only explains why the earlier LOCAL MLX runs (text format, loss on all
  tokens) were weaker than the Colab-trained adapter. The queued 2,000-row run (`p1_config.yaml`) is therefore a LOCAL REPRODUCTION test
  (can MLX Q4 + answer-only reach the committed adapter's ~59.7% on the 300 rows?), not an improvement test. A gain over 59.7% would be a surprise.
- Also found: `phase1/models/lora.py` defaults target_modules to q,k,v,o but the committed adapter uses q,v only (documented in CODE_EXPLAINED.md).

## 2026-10-11 — Gate 0 FAILED, Phase 1 local retrain worse, recalibration works
- Gate 0: adjudicator picks right 51.9-57.5% vs Phase 1 55.6-67.2% on same pairs. Simulated Phase 2 exact 50.8-57.8% vs 59.4% base. Never beats Phase 1.
- Cause checked: scoring code NOT buggy. Same scoring on 150 TRAIN pairs = 64% (adjacent 60%), valid 59%. Model underfit (1,444 pairs). The logged "val loss 0.195" was misleading: completion " 5" = 2 tokens, space is trivial, real digit loss ~0.39 vs 0.69 chance.
- Phase 1 local answer-only retrain (2,000 rows, ~1 epoch): 55.33% exact vs 59.67% committed adapter, 300 rows. 2★ 0.24, 4★ 0.29 recall.
- NEW: learned recalibration of Phase 1's 5 probs (multinomial LR on log-probs, fit on 500 dev rows, CPU, no training of LLaMA).
  Tested ONCE on 1,000 held-out balanced eval rows (`recalibrate_eval.py`), 2 configs both reported:
  base 60.2% exact / within1 92.1% / 2★ recall .34 / 4★ .47
  logprobs C=1: 65.2% / 93.3% / MAE .429 / recall [.74 .52 .64 .64 .72], paired bootstrap delta CI [+2.4, +7.6]
  logprobs+log(len) C=10: 66.5% / 93.1% / MAE .416 / recall [.76 .54 .61 .67 .76], delta CI [+3.5, +9.0]
  Trade-off: 1★ recall .86 -> .74-.76. Balanced-set result; fit on balanced dev so it assumes balanced class prior (same caveat as prior-shift).

## 2026-10-11 — Phase 2 reflective pipeline (final form) + analysis
- Feature check (`feature_cv.py`, dev CV only): TextBlob polarity/subjectivity/exclaims/contrast words add nothing; log(len) +~1pt. Reflector = LR on log-probs + log-len, C=10.
- `reflective.py`: Analyst (Phase 1 probs) -> gate (margin<th, th chosen on dev CV: 0.2; gate choices differ <0.5pt, within noise) -> Reflector (calibrator) -> trace. Loop cap 1 (deterministic).
- `phase2_analysis.py` (output `phase2_analysis.out`, `phase2_trace.json`), 1,000 held-out balanced eval rows, touched once for this config:
  exact 60.2 -> 65.2%, within1 92.1 -> 93.6, MAE .503 -> .425, 3-class 79.3 -> 81.1, binary 90.8 -> 93.8
  delta +5.0 [95% CI +2.5, +7.4], fixed 105 / regressed 55, McNemar p=9.5e-5. Reflects 51% of rows, changes 19%.
  Recall 1* .86->.79, 2* .34->.48, 3* .59->.61, 4* .47->.61, 5* .75->.76. Gain concentrated in margin<0.2 bucket (.49->.59).
- Caveats: errors remain ~82% adjacent-star (unchanged); 1* recall falls; balanced-prior assumption; 55 regressions are mostly mixed/hedged reviews; calibrator fit on only 500 dev rows (more dev rows is the cheap next lever).

## 2026-10-11 — Phase 1 close-out: label leak is worth ~8 points (300 balanced eval rows, committed adapter, paired bootstrap vs A)
- A  ANALYST_PROMPT + one-shot chosen with rating != true rating (how every reported number was produced): 59.7% exact, within1 94.0%, MAE .467
- B  same prompt, one-shot chosen at random independent of truth: 52.0%  (-7.7 [-13.3,-2.0])  <- the leak is real and large
- C  training-format build_prompt + random shot: 52.3% (-7.3 [-13.7,-0.7]); B vs C ~ same => prompt-wording mismatch costs ~0 (+0.3)
- D  ANALYST_PROMPT zero-shot (no example): 58.3% (-1.3 [-7.0,+4.3]), within1 94.7%, MAE .473  <- best leak-free config
- Mechanism: a random example anchors the prediction toward the example's own star; excluding the true star from the pool removes one wrong anchor option. Zero-shot has no anchor.
- Consequence: the 70.7%/60.1% headline and the Phase 2 calibrator (+5.0) were built on leaky probs. Re-run all probs zero-shot and re-fit/re-test before any claim. (Queued.)
- Hallucination study (150 CoT generations, `p1final_H_generations.jsonl`): 0 unparseable, 41% wrong star, 16 severe misses (>=2 stars), 7 outputs run on into an invented "Review:/Example:" turn (the model fabricates a new, glowing review: "I love this hot chocolate...") and in the severe misses the model rates a 2-star complaint 3 while its own text says "mix of positive and negative", i.e. rationale is a canned template tied to the digit, not evidence from the review. Description never mismatches the digit (0/150) because it is a fixed template per star.

## 2026-10-11 — RETRACTION: Phase 2 calibrator gain was a leak artifact
- Zero-shot (leak-free) probs, dev 500 / eval 1000 (`zs_dev_probs.jsonl`, `zs_eval_probs.jsonl`): Phase 1 = 59.2% exact (within1 91.8%, MAE .502).
- Same calibrator re-fit on zero-shot dev probs, tested once on zero-shot eval: 59.3% exact, delta +0.1 [-2.3,+2.5], fixed 80 / regressed 79, McNemar p=1 (`phase2_analysis_zs.out`).
  within1 91.8 -> 94.2, MAE .502 -> .479, polarity 76.3 -> 79.5, binary 88.8 -> 91.9; recall 2* .30->.46, 4* .41->.58, but 3* .72->.54, 5* .76->.65. Only redistributes errors.
- The earlier +5.0 (leaky probs) is withdrawn. Reason: the leaky prompt's distortions were systematic and learnable; recalibration only reshuffles the same information, so on clean probs there is nothing to gain.
- Lesson: a post-hoc recalibrator adds no information. A real gain needs a second independent signal.

## 2026-10-11 — Second independent signal: evidence model + LLaMA stack (`stack_experiment.py`, leak-free probs)
- Evidence model: TF-IDF(1-2gram, 300K feats) + LogisticRegression, trained on 100K class-balanced Kaggle reviews (20K/star, dev/eval/paper ids excluded). CPU only.
- Dev 5-fold CV exact: LLaMA 58.0, evidence alone 66.8, stack 68.2. Eval (1,000 held-out, touched once, 2 pre-declared configs):
  LLaMA 59.2 | evidence alone 65.5 | LLaMA+evidence stack 67.6 [vs LLaMA +8.4, CI +5.3..+11.6; fixed 174 / regressed 90] | +len 67.6
  stack vs evidence-alone: +2.1 [-0.3,+4.5] (not significant). Evidence alone vs LLaMA: +6.3.
  within1 93.8, MAE .402, recall 1* .81 2* .59 3* .61 4* .64 5* .73 (more even than any earlier config).
- HONEST READ: a 100K-row bag-of-words model beats the 2K-row LLaMA+LoRA on balanced 5-class exact. Most of the stack gain is the evidence model, not LLaMA. This matches the learning-curve finding (classical scales with data). LLaMA adds a small, non-significant +2.

## 2026-10-11 — DATASET DECISION: Amazon Reviews 2023, Electronics only (user). Kaggle = original paper's data (adapter trained on it; reproduction/appendix only)
- Frozen Electronics splits (`v2_reflective/e23/`, seed 2023, first 2.5M review rows streamed, reservoir sampled, disjoint by id AND exact text):
  natural_test 2,000 (1:189 2:83 3:150 4:277 5:1301) | eval 1,000 balanced | dev 500 balanced | evid_train 100,000 balanced (20K/star).
  Phase 1 adapter is trained on Kaggle => Electronics is a transfer setting; state in paper.
- Preprocessing ablation on the EVIDENCE model (`e23/ablate_preproc.py`; validation = 10K held-out raw balanced Electronics rows, +-1.0 pt at 95%):
  A raw 90K 51.2 | B TextBlob-DQC 68K 47.7 | C random 10K 47.6 | E DQC+random 10K 44.3 | D DQC+VGST 10K 44.5 | F DQC then re-balance 33K 47.6 | G raw balanced 33K 50.1
  => TextBlob DQC HURTS (-2.5 at equal size, -3.5 full): it deletes 52% of 1* and 67% of 2* (negative reviews contain positive words) and drops mixed reviews. VGST adds nothing (D vs E +0.2, within noise) and selects very long reviews (156 vs 64 words). Best: raw balanced, no filtering.
  Decision: Phase 2 evidence model trains on raw balanced data. TextBlob and VGST are reported as ablations, not used. Test sets never filtered.

## 2026-10-11 — Kaggle paper-protocol re-score, leak-free zero-shot (appendix / reproduction only)
- 4,000 natural rows (seed 2024), committed Kaggle adapter, ANALYST_PROMPT zero-shot: 69.10% exact (always-5 63.78%), polarity 85.88% (always-pos 78.60%), binary 90.31%, within1 93.45%, MAE .383, macro-F1 .549 (5-class), weighted-F1 .709. Was 70.7% / 85.0% with the leaky one-shot.
- USER DECISION: Phase 1 adapter must be TRAINED on Amazon 2023 Electronics (not Kaggle). Kaggle-adapter scoring of Electronics stopped (dev 500 done, eval partial 503/1000; files e23/zs_*_probs.jsonl are a partial transfer diagnostic, not paper results).

## 2026-10-13 — Phase 1 trained on Electronics 2023 (paper Table 1 hyperparameters; MLX QLoRA on Q4 base; zero-shot prompt; 2,000 rows; 400 updates = 3 epochs at effective batch 12)
- Dev (500 balanced, touched for arm choice only; 95% margin ~ +-4.4):
  Arm P (paper pipeline: TextBlob+stratified+VGST+neutral x2): 52.2% exact, within1 92.0, MAE .566, polarity 69.2; recall [.73 .17 .66 .49 .56]
  Arm R (raw balanced 2,000): 55.2% exact, within1 94.2, MAE .520, polarity 75.0; recall [.69 .37 .45 .69 .56]
  R > P by 3.0 pts (not significant on 500 rows) but same direction as the evidence-model ablation (TextBlob hurts). P training set: 772 of 2,000 rows are 3*, only 263/284 are 2*/1*; avg 119 words (VGST length bias) vs 61.
- Adapters: v2_reflective/e23/p1/adapter_{P,R}; probs_{P,R}_dev.jsonl. Eval / natural_test not yet scored.
- Arm chosen on dev: R (raw balanced). Scoring R once on e23 eval + natural_test (arm P not scored on eval/natural; reported as dev-only ablation).

## 2026-10-16 — Phase 1 arm R scored on e23 eval + natural_test; Phase 2 stack (Analyst R + Evidence) read out ONCE
- Scoring: `e23/eval_p1.py R eval` (1,000/1,000, done) and `R natural_test`. First natural_test run died at 1,634/2,000 (script overwrote its output, no resume). Fix: `eval_p1.py` now appends, skips done review_ids, logs progress/timing to `p1/eval_R_natural_test.log`. Killed 2nd run at 580 rows, resumed to 2,000/2,000 (~2,950s). Rows were not re-scored, so no mixed-config risk (same adapter_R, same prompt).
- Phase 2 (`e23/phase2_e23.py`, output `e23/phase2_e23.out`, traces `e23/phase2_trace_{eval,natural_test}.json`): Analyst = arm R probs; Evidence = TF-IDF-LR on 100K raw balanced e23 rows; Reflector = LR stacker (C=1) fit on 500 balanced dev rows; run once on both splits, 2 pre-declared configs (A+E, A+E+len).
- eval (1,000 balanced): Analyst 54.0 | Evidence 62.4 | stack 63.9 (within1 95.5, MAE .411, macroF1 .638). Stack vs Analyst +9.9 [+6.7,+13.0] (fixed 183/regressed 84, p=1.3e-9); stack vs Evidence +1.5 [-1.2,+4.1] (p=0.3, NOT significant). +len: 63.5.
- natural_test (2,000, natural skew 5*=65%): Analyst 53.4 | Evidence 69.3 | stack 73.4 (within1 96.2, MAE .320, macroF1 .577). vs Analyst +20.0 [+17.8,+22.2]; vs Evidence +4.0 [+2.4,+5.8] (fixed 197/regressed 116, p=5.5e-6). +len: 73.2.
- Agents disagree on 44% of rows (both splits). eval: analyst right 32%, evidence 51%, stack 55%. natural: analyst 25%, evidence 61%, stack 67%.
- HONEST READ: same pattern as Kaggle. Evidence model carries most of the gain; Analyst (LLaMA, 2K rows) is weaker alone (54.0 vs 62.4) but adds a significant gain only on natural_test (+4.0), not on balanced eval (+1.5, ns). Caveats: stacker fit on balanced dev, so natural_test gain mixes in a class-prior effect (stack recall 1*-4* roughly equal to evidence; 5* up .77->.81); natural_test macroF1 .577 is the fairer headline for imbalance. Dev-only arm choice (R over P) still within noise (+-4.4).

## 2026-10-17 — Paper: results draft written (not yet merged into main.tex)
- User chose paper first. Wrote `paper/results_paper_e23.md`: Electronics-only leak-free results, stack vs analyst/evidence with CIs, honest reading, preprocessing ablation, leak retraction, Kaggle as Appendix A. Every number copied from LOG entries / `e23/phase2_e23.out`; nothing new computed. main.tex untouched (still Run 1 narrative).
- Open: confusion matrices + per-class recall figures from `e23/phase2_trace_*.json`; decide target paper (thesis main.tex vs conference tex); optional Electronics hallucination study.

## 2026-10-17 — Plan documented: `v2_reflective/PAPER_PLAN.md`
- Honest assessment: evidence TF-IDF beats LLaMA; stack vs evidence +1.5 ns on balanced eval; natural_test gain confounded by balanced-dev stacker fit. Plan S1-S6 (natural-prior dev stacker, base-LLaMA + encoder baselines, variance of the +1.5, figures, angle decision + merge, optional deck/notebook).

## 2026-10-17 — Whole-project roadmap: `v2_reflective/PROJECT_ROADMAP.md`
- User asked for everything, not only the paper. Workstreams W1-W10 (experiments, paper, thesis, conference paper, deck, notebook, reproducibility, archive, docs cleanup, submission), order of work, risks, open questions (deadlines/venue unknown). Linked from `PAPER_PLAN.md` / memory.

## 2026-10-17 — Deadline set: end of Nov 2026. Dated week-by-week schedule added to `PROJECT_ROADMAP.md`; weekly progress notes in `v2_reflective/progress/week_NN.md` (week_00 written).

## 2026-10-17 — Hallucination study on Electronics adapter R (150 balanced eval rows, zero-shot training-format prompt, 120-token greedy) `e23/halluc_p1.py`, output `e23/halluc_p1.out`, generations `e23/p1/halluc_R_generations.jsonl`
- See `.out` for the exact counts. Qualitative: adapter outputs a canned rubric sentence tied to the digit and repeats it ("... (rating N).! N. ..."); no review-specific reasoning. Matches the Kaggle-adapter finding.
- Writing-integrity policy (plagiarism/AI content <5%) added to PROJECT_ROADMAP.md.

## 2026-10-17 — Deadline clarified: "November 3rd weekend" read as 3rd weekend of Nov = Nov 21-22 (finish by Fri Nov 20). Schedule in PROJECT_ROADMAP.md compressed to 5 weeks, numbers freeze Nov 1, W4/W6 dropped unless required. Reading to be confirmed with user.

## 2026-10-17 — User: Phase 2 is the focus. PROJECT_ROADMAP.md updated: Phase 2 candidates P2-1..P2-4 (disagreement routing, evidence-grounded adjudicator with Gate 0 on e23 dev, better reflector features, ablation ladder) with kill criteria; Week 1-2 re-prioritised. Prior LLM adjudicator failed Gate 0 on Kaggle (LOG 2026-10-11), so P2-2 only proceeds if it passes on e23 dev.

## 2026-10-17 — Deck updated: `Agentic_LLaMA3_Presentation_v5_electronics.pptx` (25 slides) built by `v2_reflective/build_deck_e23.py` from `..._FINAL_v3_aligned.pptx` (original untouched)
- Slides 1-9 kept (title/dataset text edited). New slides 10-16: leak audit, Electronics protocol, Phase 2 results, preprocessing ablation (native chart), hallucination study, Phase 2 focus (P2-1..P2-4), roadmap to Nov 21-22. Slides 17-25 = old Run 1 / audit / v2-plan slides, eyebrow relabelled "HISTORICAL (SUPERSEDED)"; old conclusion slide dropped.
- Numbers copied from LOG / `e23/phase2_e23.out` / `e23/halluc_p1.out` only. Rendered via soffice and visually checked (slides 1, 10, 12, 13, 16, 17).
- Known gaps: slides 5-9 still describe the Kaggle pipeline and the old 4-agent design (kept as method background, not results); deck must be REBUILT (not hand-edited) after the Nov 1 numbers freeze, then Week 4 deck task becomes a refresh.

## 2026-10-17 — Week 1 started: natural-prior dev set (S1) + Reflector variants (P2-1, P2-3). PRE-DECLARED BEFORE ANY OUTCOME:
- `e23/build_natural_dev.py` -> `e23/natural_dev.jsonl`: 500 rows from the unused natural pool (953 candidates after disjointness by id and exact text from natural_test/eval/dev/evid_train), seed 2024. Rating mix 1:42 2:19 3:35 4:70 5:334. Frozen splits untouched. Output `e23/build_natural_dev.out`.
- Evidence probs: `e23/evid_natural_dev.py` refits the identical evidence model; reproduction check vs stored eval probs max |diff| = 0.00e+00 (`e23/evid_natural_dev.out`). Analyst probs: `eval_p1.py R natural_dev` running (resumable, `p1/eval_R_natural_dev.log`).
- `e23/stack_week1.py` (output to be teed to `e23/stack_week1.out`): five configs declared in advance, ALL reported on eval + natural_test once: C0 A+E fit balanced dev (reference); C1 A+E fit natural_dev; C2 A+E balanced-dev reweighted to natural prior; C3 A+E+[margin, entropy, disagree] fit natural_dev; C4 routing (agree rows keep shared label, disagree rows use a stacker fit on natural_dev disagree rows). Dev-only 5-fold CV printed first. With 5 configs the p-values are uncorrected; treat marginal ones cautiously.
- Compute policy added to PROJECT_ROADMAP.md: local by default, flag Colab needs.

## 2026-10-18 — Daily plan written: `v2_reflective/DAILY_PLAN.md` (one runnable job per day Oct 18 – Nov 20, local only, `caffeinate -i` for long jobs). Natural_dev analyst scoring was at 146/500 after the overnight sleep; resume command in the plan.

## 2026-10-18 — Catch-up rules added to DAILY_PLAN.md; ledger `progress/DAILY_STATUS.md` created (missed jobs keep order, marked `moved`; Sundays and Nov 16-20 reserved for catch-up; overnight queue bundling; hard stops: Gate 0 + variance before Nov 1, draft 1 Nov 8, submit Nov 20).

## 2026-10-18 — Backup + contingency plan: `v2_reflective/CONTINGENCY.md`; scripts `backup_artifacts.sh` and `verify_backup.sh`. First snapshot `../results_archive/backups/pes_20261018_0007.tar.gz` (353MB) created and verification PASSED (checksum, readable, extracts, key files byte-identical). Off-disk copy still missing (needs user destination); backup ledger added to `progress/DAILY_STATUS.md`.

## 2026-10-18 — Week 1 Day 1 results: natural-prior stackers (S1), features (P2-3), routing (P2-1). Outputs `e23/stack_week1.out`, `e23/stack_week1_results.json`, `e23/prior_baseline.out`. natural_dev analyst scoring finished (500/500, resumed after overnight sleep).
- natural_dev (500, prior 1:.084 2:.038 3:.07 4:.14 5:.668) 5-fold CV: Analyst 53.8 | Evidence 71.4 | stack A+E 79.2 exact but macroF1 .517, 2* recall 0.00; +feats 78.8 / .529.
- Read-out on eval (balanced) / natural_test (all 5 declared configs, uncorrected p):
  C0 A+E bal-dev (reference): 63.9 / 73.5 exact, macroF1 .638 / .578
  C1 A+E natural-dev: 60.3 / 78.4, macroF1 .575 / .541 (2* recall .26 / .12)
  C2 bal-dev reweighted to natural prior: 59.0 / 79.0, macroF1 .554 / .560
  C3 +[margin, entropy, disagree] natural-dev: 59.4 / 78.3, macroF1 .564 / .547
  C4 routing (agree rows keep shared label, disagree rows stacker): 61.3 / 74.2, macroF1 .606 / .569; vs C0 on natural_test +0.8 [-0.9,+2.4] ns; fewer regressions (56 vs Evidence).
- MISSING BASELINE found and run (`prior_baseline.py`, declared after seeing the above): Evidence alone with the same natural-prior correction (prior from natural_dev only) gets 76.3% exact on natural_test (raw 69.3), macroF1 .517; Analyst prior-adjusted 75.5%. On eval the prior correction hurts (Evidence 62.4 -> 53.2).
- CONCLUSIONS (honest): (1) The large natural_test exact gains of C1-C3 (+9 over raw Evidence) are mostly class-prior correction: prior-adjusted Evidence alone gets 76.3, so the real stacker gain on top of equal prior correction is about +2.7 (C2 79.0 vs 76.3, CI not yet computed). (2) Natural-prior fits trade away minority-class recall (2* recall .12-.16) and lose on balanced eval (-3.6 to -4.9 vs C0, p<0.03): they inflate exact, they hurt macroF1. Exact accuracy on a 67%-five-star set rewards the prior; macro-F1/balanced metrics are the honest headline for natural_test. (3) C0's earlier +4.1 over raw Evidence on natural_test came with macroF1 .578 vs .541 (and .517 for prior-adjusted Evidence): a gain in class-balanced quality survives. (4) Extra features (C3 vs C1) add nothing; routing (C4) ties C0 and cuts regressions but not accuracy.
- Consequences for the paper: report natural_test with macro-F1, per-class recall and prior-adjusted baselines beside exact; do not headline natural_test exact. Phase 2 claim stays "modest complement" unless the Oct 22-24 adjudicator Gate 0 or the baselines change it. TODO Oct 21: paired CI of C2 vs prior-adjusted Evidence on natural_test.

## 2026-10-18 — Git: repo already existed (remote origin kvarun314/pes-mtech-project; earlier notes saying "not a git repo" were wrong and are corrected). Branch `e23-leakfree-oct2026`, commits d1ada1b (Oct 9 audit fixes) and a1ec9fa (e23 work, 109 files, max 3.8MB), pushed by the user; verified `HEAD == origin/e23-leakfree-oct2026`. `.gitignore` extended: secrets (`kaggle_json.py` holds a Kaggle key: ROTATE), `.scratch`, pool/evid_train, review-text splits and training data, old adapters, traces, build files, tarballs. `main` untouched. Claude cannot push (classifier blocks); user runs `! git push`.

## 2026-10-19 (overnight, user asleep) — base LLaMA (no LoRA) scoring + analysis pipeline. PRE-DECLARED
- `e23/queue_base.sh` (caffeinate, resumable, markers `p1/stage_base.txt`, logs `p1/eval_base_<split>.log`) scores base LLaMA-3-8B Q4 without adapter, same zero-shot prompt, on dev, eval, natural_test (`eval_p1.py` arm `base`).
- Then `e23/base_vs_lora.py` runs once (output `e23/base_vs_lora.out`, `base_vs_lora_results.json`, figures `paper/figures/e23/`): base vs arm R vs Evidence; stacks [base+E], [R+E]=C0, [base+R+E] fit on balanced dev; C2 vs prior-adjusted Evidence paired CI on natural_test; confusion matrices and per-class recall figures. All variants are reported, none chosen on eval.

## 2026-10-20 — RESULTS base LLaMA (no LoRA) vs LoRA vs Evidence (S2a). Pipeline ran overnight, all splits DONE, `analysis_finished` in `e23/p1/stage_base.txt`. Output `e23/base_vs_lora.out`, `e23/base_vs_lora_results.json`, figures `paper/figures/e23/`.
- Base (no LoRA), same zero-shot prompt: dev 36.2 | eval 34.3 (macroF1 .269) | natural_test 16.8 (macroF1 .212). Predicts almost only 1* and 3* (2*/5* recall .01-.03). Arm R (LoRA): eval 54.0, natural_test 53.4. LoRA gain over base: +19.7 [+15.9,+23.3] eval, +36.7 [+34.2,+39.1] natural_test, p<1e-24. So LoRA is what makes LLaMA usable.
- Stacks fit on balanced dev, eval / natural_test exact: [base+E] 63.7 / 72.7 (vs Evidence +1.3 ns / +3.4 [+1.7,+5.0]); [R+E]=C0 63.9 / 73.4; [base+R+E] 63.7 / 73.6. [base+R+E] vs [R+E]: -0.2 ns / +0.1 ns.
- Finding: even base LLaMA probabilities, stacked, give nearly the same gain as the LoRA analyst (63.7 vs 63.9 on eval). Contribution to the stack comes from a second independent signal, not from the fine-tune; LoRA does not add to the stack once Evidence is present. Weakens the "LoRA Analyst" Phase 1 claim inside Phase 2; Phase 1 standalone gain over base stays large.
- C2 vs prior-adjusted Evidence on natural_test (TODO from Oct 18 closed): 79.0 vs 76.3, +2.7 [+1.4,+4.0], fixed 119 / regressed 65, p=8.4e-05. Real stacker gain at equal prior correction, but macroF1 .561 vs .517 and 2* recall .13 both.
- Honest summary: Phase 2 = modest, significant on natural_test, not significant on balanced eval. Angle decision stays Nov 1 (Gate 0 adjudicator, DistilBERT baselines, variance run pending).

## 2026-10-20 — Gate 0 for P2-2 (evidence-grounded adjudicator) on Electronics. PRE-DECLARED BEFORE ANY OUTCOME (script `e23/gate0_e23.py build|run|score`, output `e23/gate0_e23.out`)
- Rows: disagreement rows (argmax Analyst arm R != argmax Evidence) of dev (500) and natural_dev (500) ONLY. eval/natural_test not touched. Pair = (analyst star, evidence star); scored subset = rows whose truth is one of the two.
- LLM: base LLaMA-3-8B Q4 (no adapter), completion prompt: review + the two candidate stars + top-5 TF-IDF-LR n-grams per candidate (contribution = tfidf value x coef difference, refit evidence model identical to evidence.py). Prediction = higher first-digit logit of the two candidate digits; run in both candidate orders, probabilities averaged (cancels position bias).
- Stack baseline on same rows: C0 stacker (A+E log-probs, LR C=1); dev rows by 5-fold CV on dev, natural_dev rows by stacker fit on dev; choice restricted to the two candidates.
- PASS criterion: LLM pair accuracy >= stack pair accuracy + 3 pts AND paired-bootstrap 95% CI of the difference excludes 0 (pooled dev+natural_dev). Otherwise FAIL: adjudicator dropped, P2-1 router-only result kept. Single run, no prompt iteration after seeing scores.

## 2026-10-20 — RESULT Gate 0 (P2-2 evidence-grounded adjudicator): FAIL. Output `e23/gate0_e23.out`, prompts `e23/gate0_prompts.jsonl` (434 disagreement rows), answers `e23/gate0_answers.jsonl`, logs `gate0_run.log`.
- Pair accuracy on 371 rows whose truth is one of the two candidates (dev 193, natural_dev 178): LLM adjudicator 54.2% | stack 73.3% | Evidence always 64.2% | Analyst always 35.8%.
- LLM - stack: -19.1 pts, CI [-25.3,-12.4] (LLM-only right 46, stack-only right 117). Worse than just trusting Evidence (64.2). Same on both splits (55.4/71.0 dev, 52.8/75.8 natural_dev).
- Verdict per pre-declared criterion: FAIL. Adjudicator dropped; no prompt iteration (single run as declared). Replicates the Kaggle Gate 0 failure (52-57%), now with Evidence n-gram citations added. Phase 2 stays: Analyst + Evidence + LR Reflector (+ routing as P2-1 option). Oct 30 slot -> optional LoRA seed 2 (adapter variance).
- Paper angle consequence: "agentic LLM verification helps" is not supported; leaning angle B (leak audit + preprocessing ablation + honest modest stack result + negative result for LLM adjudication). Final decision Nov 1 after baselines + variance.

## 2026-10-20 — Variance run (S3). PRE-DECLARED BEFORE ANY OUTCOME (`e23/variance.py`, output `e23/variance.out`, per-seed `e23/variance_seed<k>.json`, resumable)
- 5 seeds (0-4). Per seed: Evidence refit (same config as evidence.py: title+text, 1-2gram, 400K, C=3) on a bootstrap resample (with replacement, 100K) of evid_train; stacker C0 (A+E log-probs, LR C=1) fit on a bootstrap resample of the 500 dev rows. Analyst probs (arm R) fixed (adapter variance NOT covered; second LoRA seed is the optional Oct 30 job).
- Reported per seed and mean +/- sd, on eval and natural_test: Evidence exact/macroF1, stack exact/macroF1, stack minus Evidence. Purpose: is +1.5 (eval) / +4.0 (natural_test) stable against evidence-training and stacker-fit randomness. No selection; eval/natural_test read as already frozen.

## 2026-10-20 — RESULT variance run (S3). Output `e23/variance.out`, `e23/variance_seed0-4.json`.
- Stack minus Evidence exact: eval +2.74 +/- 1.25 pts (seeds 1.6, 2.7, 4.1, 1.4, 3.9); natural_test +4.27 +/- 1.14 (4.7, 2.6, 4.95, 3.65, 5.45). macroF1: eval +0.027 +/- 0.013, natural_test +0.043 +/- 0.007. Positive in all 5 seeds on both splits.
- Caveat: bootstrap Evidence is weaker (eval 60.0 +/- 0.8, vs 62.4 for the single frozen fit; bootstrap duplicates ~37% of rows) so the stack gain is larger against a weaker Evidence. The frozen +1.5 (eval, ns) is the low end; honest range for the stack gain on eval is about +1.4 to +4.1 depending on Evidence/dev draw, sign stable. Adapter (LoRA) variance not covered.
- Reading: the stack gain is real in sign but small and sensitive to how strong the Evidence model is; report frozen values with this spread, not the best seed.

## 2026-10-21 — DistilBERT 2K baseline (S2b). PRE-DECLARED BEFORE ANY OUTCOME (`e23/encoder_2k.py`, output `e23/encoder_2k.out`, probs `e23/p1/probs_enc2k_<split>.jsonl`, log `e23/encoder_2k.log`)
- Purpose: size-matched baseline for the LoRA analyst. Not part of the proposed system; baseline row only (user 2026-10-20: not a method).
- Data: the exact 2,000 arm-R rows (stratified_sample 400/star, Random(42), same shuffle(42), 400 valid / 1,600 train as the LoRA run). distilbert-base-uncased (cached locally), MPS, max_len 256, batch 16, lr 5e-5, 3 epochs (~ LoRA's 3 epochs), linear decay, seed 0. Hyperparameters FIXED now, no tuning; valid loss/acc printed for monitoring only.
- Then scored once on dev, eval, natural_test. Report: exact / macroF1 / per-class recall vs arm R, base, Evidence; paired CI enc2k vs arm R; stack [enc2k+E] fit on balanced dev (same recipe as C0) vs [R+E]. All reported whatever the outcome.

## 2026-10-21 — RESULT DistilBERT 2K baseline (S2b). Output `e23/encoder_2k.out`, training log `e23/encoder_2k.log` (3 epochs, ~340 s on MPS, valid acc .490/.487/.542), probs `e23/p1/probs_enc2k_*.jsonl`, model `e23/p1/enc2k_model` (not committed). Report script had a path bug (evidence probs file name), fixed and rerun; no numbers changed by the fix.
- eval exact / macroF1: enc2k 52.8 / .522 | arm R 54.0 / .543 | Evidence 62.4 / .622. enc2k vs arm R -1.2 [-5.0,+2.2] ns. natural_test: enc2k 62.5 / .462 | R 53.4 / .488 | Evidence 69.3 / .541; enc2k vs R +9.0 [+6.3,+11.8] exact (prior effect: enc2k 5* recall .72 vs R .49, minority recalls lower; macroF1 lower than R).
- Reading: a 66M encoder on the same 2K rows is statistically level with the LLaMA-3-8B LoRA on balanced eval (-1.2 ns). Size-matched, LLaMA is no better; the 2K-row LoRA is not special. Answers "why LLaMA?": no accuracy advantage on balanced data at this data size.
- Stacks (fit on balanced dev): [enc2k+E] eval 61.6 (vs Evidence -0.8 ns; vs [R+E] -2.3 [-4.7,+0.1] p=.079); natural_test 70.2 (vs Evidence +0.9 ns; vs [R+E] -3.2 [-4.9,-1.6] p=.0002). The encoder adds nothing on top of Evidence; the LoRA analyst does (+1.5 eval ns, +4.0 natural_test sig). So LoRA's outputs are a more complementary signal to TF-IDF than a size-matched encoder's. But [base+E] ~ [R+E] (Oct 20), so the complementarity comes from LLM pretraining knowledge, not the fine-tune.
- Single seed for the encoder (seed 0); no variance estimate. Baseline row only.
