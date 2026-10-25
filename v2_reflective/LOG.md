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

## 2026-10-21 — LoRA seed 2 (adapter variance, S3b). PRE-DECLARED BEFORE ANY OUTCOME. Local, MLX, no Colab. Queue `e23/queue_seed2.sh` (resumable per stage, markers `p1/stage_seed2.txt`, logs `p1/train_R2.log`, `p1/eval_R2_<split>.log`); report `e23/seed2_report.py` -> `e23/seed2_report.out`.
- Arm R2 = identical to arm R (same 2,000 rows, `config_R.yaml` hyperparameters) except `seed: 43` (was 42), adapter `p1/adapter_R2`. Training randomness only; data not resampled.
- Scored once on dev, eval, natural_test. Report: R2 vs R exact/macroF1 and agreement; stack [R2+E] vs [R+E] (fit on balanced dev) with paired CIs; stack-minus-Evidence for both. Purpose: is the LoRA contribution (+1.5 eval ns / +4.0 natural_test) stable across adapter seeds. Both seeds reported, none chosen.

## 2026-10-21 — Frozen-table draft (S4/Oct 31 job done early): `e23/freeze_tables.py` -> `e23/results_frozen.md` + `paper/figures/e23/main_bars.png`. All numbers recomputed from stored probs and match earlier logs (e.g. eval Analyst 54.0, Evidence 62.4, C0 63.9; natural_test 53.4 / 69.3 / 73.4). Includes base, DistilBERT, base+E, enc2k+E stacks with bootstrap CIs. To be rerun on Nov 1 (adds R2 rows automatically once seed-2 probs exist). Seed-2 queue started 2026-10-21 ~13:30 (training, then 3 scoring stages, est. 3-4 h).

## 2026-10-21 — RESULT LoRA seed 2 (adapter variance). Output `e23/seed2_report.out`, probs `e23/p1/probs_R2_*.jsonl`, adapter `e23/p1/adapter_R2`, logs `p1/train_R2.log`, `p1/eval_R2_*.log`, markers `p1/stage_seed2.txt`. Frozen table rerun with R2 rows (`e23/results_frozen.md`).
- Balanced eval: R2 53.7 / macroF1 .516 vs R 54.0 / .543 (-0.3 [-3.3,+2.6] ns). Same exact accuracy; the two adapters agree on only 74% of rows.
- natural_test single-model exact is NOT stable: R2 74.8% (macroF1 .545) vs R 53.4% (.488), +21.3 [+19.1,+23.8]. Cause: same balanced training data, different seed -> different class bias. R2 predicts 5* on 64% of natural_test rows (R: 33%); 5* recall .88 vs .49, 2*/3*/4* recall lower (.35/.30/.47 vs .42/.50/.71). On balanced eval R2 5* recall .93 vs .56. So a single analyst's natural_test exact accuracy is mostly a seed-dependent prior artifact; do not report it as a model-quality number. macro-F1 / balanced eval are the stable measures.
- Stack is stable across adapter seeds: eval [R2+E] 64.6 vs [R+E] 63.9 (+0.7 ns), vs Evidence +2.2 [-0.3,+4.8] ns; natural_test [R2+E] 72.7 vs [R+E] 73.4 (-0.7 [-1.4,+0.0] p=.065), vs Evidence +3.4 [+1.6,+5.0]. macroF1 natural_test .576 vs .577.
- Conclusion: stack gain over Evidence is 1.5-2.2 (ns) on balanced eval and 3.4-4.0 (sig) on natural_test across both adapter seeds; a robust, modest, positive result. The Reflector absorbs the analyst's seed-specific bias (that is a real argument for the stacking layer). Analyst-alone numbers on natural_test must be shown with both seeds or omitted.

## 2026-10-21 — `v2_reflective/PAPER_OUTLINE.md` written (outline + facts index with sources, no prose). Backup `pes_20261021_2119.tar.gz` verified. Local commit 58d3f34 NOT pushed: ancestor commit 190816a still contains the 255MB enc2k_model (GitHub rejects); needs the user's amend command (see DAILY_STATUS).
Review agenda for Oct 22 written: progress/REVIEW_2026-10-22.md
## 2026-10-21 — Push fixed: squashed 190816a..3f56d72 into e8c7ae7 without enc2k_model (now gitignored); pushed, HEAD==origin.
## 2026-10-21 — Deadline: user restated 'November 3rd week' = third week of November, weekend Nov 21-22 2026; submit by Fri Nov 20 as planned. Reading now CONFIRMED (no longer 'unconfirmed'). Note: if it ever meant the week of Nov 3 (weekend Nov 7-8) the plan would not hold; user to say so.

## 2026-10-21 — Improvement plan written (`v2_reflective/IMPROVEMENT_PLAN.md`), before any run. Phases A metrics, B embedding/char agents, C LoRA-10K, D better Reflector fit, E combine+freeze, each with dev-only gates and kill criteria. Local only; user rule: Colab only if truly required (nothing planned needs it). LoRA speed measured from adapter timestamps (400 iters ~2h15m).

## 2026-10-21 — Phase A (metrics_extra) and Phase B (embedding + char n-gram agents) STARTED. PRE-DECLARED per IMPROVEMENT_PLAN.md.
- A1 `e23/metrics_extra.py` -> `metrics_extra.out`: reporting only, recomputed from stored probs. Polarity 3-class (1-2 neg / 3 neu / 4-5 pos), binary (excl 3*), within-1, MAE, expected-rating MAE for Analyst R, R2, base, enc2k, Evidence, stacks.
- B1 `e23/embed_agent.py`: all-MiniLM-L6-v2 embeddings (title + ". " + text, max_seq 256), chunked/resumable (`e23/emb/<split>_<chunk>.npy`), LR (C=1, 1000 iters) on evid_train 100K -> `evid2_<split>_probs.jsonl` for dev, natural_dev, eval, natural_test. Output `embed_agent.out`.
- B2 `e23/char_agent.py`: char_wb 2-5 gram TF-IDF (max 400K feats, sublinear) + LR C=3 on the same 100K -> `evid3_<split>_probs.jsonl`. Output `char_agent.out`.
- B3 `e23/stack_gate_B.py` (written after B1/B2 finish; gate declared now): dev-only 5-fold CV on dev and on natural_dev. Stacks: S0 [A+E] (current), S1 [A+E+E2], S2 [A+E+E3], S3 [A+E+E2+E3], all LR C=1 on log-probs. An agent set passes only if CV exact >= S0 + 1.0 on dev AND same direction (>= S0) on natural_dev AND macro-F1 not lower on dev. Only a passing set is read out (once) on eval + natural_test. No eval/natural_test numbers are looked at before the gate result is logged.

## 2026-10-21 — RESULT Phase A (metrics_extra). Output `e23/metrics_extra.out`. Reporting only.
- eval (balanced): 3-class polarity acc / macroF1 / binary acc: Analyst R .767/.719/.912 | Evidence .781/.737/.915 | Stack R+E .824/.775/.949 | Stack R2+E .826/.775/.953. Within-1: Evidence .931, stack .955-.960.
- natural_test: polarity acc Analyst R .884, R2 .899, Evidence .878, Stack R+E .907 (macroF1 .758 vs Evidence .714); binary acc .953 vs .926.
- Reading: on the coarser task the stack gain is larger and clearer than on 5-class exact (eval polarity +4.3 over Evidence, binary +3.4; natural_test polarity macroF1 +4.4). Useful framing for the paper: fine-grained star errors are mostly adjacent-star confusions; the stack is most helpful at the polarity level. Seed difference also shrinks on polarity (R .884 vs R2 .899) vs exact (53.4 vs 74.8).
- Paired CIs for these polarity gains NOT yet computed (descriptive only); compute before claiming significance.

## 2026-10-21 — RESULT Phase B gate (dev-only; `e23/stack_gate_B.out`, agents `embed_agent.out` / `char_agent.out`, probs `evid2_*` (MiniLM+LR) / `evid3_*` (char 2-5gram+LR)). Gate: FAIL for all three sets.
- Single agents (dev / natural_dev exact): TF-IDF E .586/.714 | MiniLM E2 .534/.612 | char E3 .568/.684. Both new agents are weaker than the word TF-IDF.
- 5-fold CV stacks, dev exact (macroF1): S0 A+E .640 (.640) | S1 +emb .627 | S2 +char .631 | S3 +both .619. natural_dev: S0 .804 (.558) | S1 .805 (.604) | S2 .779 | S3 .788.
- Reading: extra lexical/semantic views of the same text do not complement TF-IDF at this stacker fit size (500 rows, 15-20 features); they add noise. Embedding agent gains macroF1 on natural_dev only (.604 vs .558) - not a gate pass, noted. Agents dropped; no eval/natural_test read. 3 agent-set configs declared and all reported.
- Implication: the second signal that helps is the LLM (different knowledge source), not more classical views. Next lever is the LLM itself: Phase C (LoRA-10K), plus Phase D (more stacker fit data).

## 2026-10-21 — Phase C STARTED: LoRA-10K (arm R10). PRE-DECLARED. `e23/build_R10.py` -> `p1/data_R10/{train,valid}.jsonl`; `p1/config_R10.yaml`; queue `e23/queue_R10.sh` (markers `p1/stage_R10.txt`, logs `p1/train_R10.log`, `p1/eval_R10_<split>.log`).
- Data: 10,000 raw balanced Electronics rows (2,000/star) sampled from evid_train (disjoint from dev/natural_dev/eval/natural_test), Random(7), same prompt builder as arm R; 9,800 train / 200 valid. Hyperparameters identical to arm R (rank 4, alpha 64, lr 5e-5, batch 3 x accum 4, num_layers 3, max_seq 1024), iters 816 (= 1 epoch of 9,800 at 12 rows/iter), seed 42, save every 200.
- Time guard: speed checked on first iterations; if projected training > 9 h, kill and reduce rows to 6,000 (500 iters). If still > 9 h tell the user (Colab case).
- Queue scores dev ONLY, then stops. GATE (dev): R10 dev exact >= 58.0 (arm R 55.2; dev margin about +/-4.4) -> then score eval + natural_test (`queue_R10_rest.sh`) and add to the ladder. Otherwise log: more LoRA data does not rescue the analyst. No eval/natural_test looked at before the gate.
- 2026-10-21 22:12 speed check R10: ~4 it/min (iter 80 at 22:12), 816 iters projected ~3.4 h, ETA training done ~01:30, dev scored ~01:50. Under the 9 h guard: no row reduction, no Colab.

## 2026-10-22 — RESULT Phase C gate (LoRA-10K, arm R10): FAIL. Training 816 iters done (val loss .044), dev scored (`p1/probs_R10_dev.jsonl`, logs `p1/train_R10.log`, `p1/eval_R10_dev.log`, `p1/stage_R10.txt`).
- Dev exact / macroF1 / recall: R (2K) 55.2 / .554 | R2 55.4 / .532 | R10 (10K) 56.4 / .554 recall [.72,.59,.24,.51,.76]. Gate was >= 58.0: FAIL. eval/natural_test NOT scored (as declared).
- Reading: 5x more LoRA data moves dev exact by +1.2 (inside the +/-4.4 dev margin). The first-digit LoRA analyst is NOT data-starved at 2K; it plateaus around 55-56 on 5-class. So "LLaMA loses to TF-IDF because of little data" is refuted for this protocol; answer for the paper: scaling LoRA data 5x does not rescue it.
- Next lever (declared below): use LLaMA as a feature extractor (hidden-state probe) instead of the first-digit readout.

## 2026-10-22 — Phase F STARTED: LLaMA hidden-state probe agent. PRE-DECLARED. Script `e23/probe_agent.py` (chunked, resumable; `e23/hid/<split>_<chunk>.npy`; output `e23/probe_agent.out`; probs `evid4_<split>_probs.jsonl`), queue marker `e23/stage_F.txt`.
- Idea: a different, richer use of the same LLM: base LLaMA-3-8B Q4 (NO adapter), same zero-shot build_prompt as eval_p1, final-norm hidden state of the last prompt token (4096-d), then LR on labelled rows. Local MLX, no Colab. Est. ~0.5 s/row x ~14K rows ~ 2 h.
- Fit rows: 10,000 balanced rows from evid_train (2,000/star, Random(11); NOT the eval splits). Scored: dev, natural_dev, eval, natural_test. LR C chosen on dev from {0.01, 0.1, 1} (standardised features), dev only.
- GATE (dev-only, same as B): probe alone and stack [A+E+P] by 5-fold CV on dev and natural_dev. Pass: stack CV exact >= S0 + 1.0 on dev AND >= S0 on natural_dev AND dev macroF1 >= S0. Only then eval/natural_test read once. Probe alone is also reported vs Analyst (single-model comparison, descriptive).
- 2026-10-22 10:25 probe_agent.py bug: stratified_sample reads key 'rating' (fit set came out empty); fixed + assert len==10000; restarted (dev chunk kept, resumable).
- 2026-10-22 10:50 Phase F amendment (before any probe result): measured 2.0-2.2 s/row with the long task prompt (same as the base scoring runs: ~2.1 s/row), i.e. ~8 h for 14K rows. Probe now uses a compact prompt `Review: <text up to 1,200 chars>\nRating (1-5):` + space (no rubric/CoT text), expected ~3x faster; previously extracted dev chunk deleted so all features use one prompt. Everything else in the Phase F declaration unchanged. No gate/outcome seen yet.
- 2026-10-22 10:51 probe speed ~1.16 s/row (first 500-row chunk 581 s incl. load); ~14K rows => ETA ~15:30 same day. Local, no Colab. Next on completion: `probe_gate_F.py` (dev-only gate).

## 2026-10-22 — Polarity paired CIs (declared now, after Phase A descriptive numbers): `e23/polarity_ci.py` -> `e23/polarity_ci.out`. Metric: 3-class polarity accuracy (1-2 neg / 3 neu / 4-5 pos) and binary accuracy (excl 3*). Comparisons on eval + natural_test (read as already frozen): stack [R+E] vs Evidence, [R2+E] vs Evidence, [R+E] vs Analyst R. Paired bootstrap 3,000 + exact McNemar. 6 comparisons x 2 metrics declared, uncorrected p.

## 2026-10-22 — RESULT polarity paired CIs (`e23/polarity_ci.out`). Stack gain is significant on BOTH splits and BOTH adapter seeds at the coarser levels.
- eval, 3-class polarity acc: Evidence .781 | Analyst .767 | [R+E] .824 | [R2+E] .826. [R+E] vs Evidence +4.3 [+2.2,+6.4] p=1.2e-4; [R2+E] +4.5 [+2.4,+6.6] p=4.5e-5; [R+E] vs Analyst +5.7 [+3.3,+8.0].
- eval binary (excl 3*): Evidence .915 | [R+E] .949 | [R2+E] .953. +3.4 [+1.5,+5.2] p=3.6e-4; +3.8 [+2.1,+5.5].
- natural_test polarity: Evidence .878 | [R+E] .907 | [R2+E] .907; +2.9 [+1.7,+4.0] p=1.8e-6 (both seeds). binary .926 -> .953 (+2.6 [+1.7,+3.7], p=3.5e-7).
- HONESTY NOTE: polarity/binary was added as a reporting metric AFTER seeing that 5-class exact gains were small (this is a post-hoc metric choice). Defensible because: the original paper (Wang et al.) and the earlier Kaggle reproduction already reported polarity/binary; stack and metric definitions are fixed; all 12 comparisons are printed, none dropped; p-values uncorrected. State this in the paper. Even Bonferroni over the 12 comparisons (alpha .004) keeps all but none borderline: the smallest p is 4.5e-5 (eval) / 1.8e-6 (natural_test); largest p is 3.6e-4.
- Takeaway for the thesis: Phase 2 stacking gives a consistent, significant improvement in sentiment polarity (+3 to +4.5 points) across balanced and natural test sets and across two adapter seeds, while fine-grained 5-star exact gains are smaller. This is the first clearly publishable positive Phase 2 result.

## 2026-10-22 — Phase F: LLaMA hidden-state probe agent (dev-only gate)
- Extraction: `e23/probe_agent.py` (compact prompt, last-token hidden state of base LLaMA-3-8B Q4, no LoRA), 10K balanced fit rows from evid_train + dev/natural_dev/eval/natural_test. ~15.5K s local. Marker `F_extract_DONE`.
- Gate run: `e23/probe_gate_F.py` -> `e23/probe_gate_F.out`. LR on standardised states, C=0.01 chosen on dev (dev exact .600; C=0.1 .582; C=1 .548). Dev 5-fold CV x3.
- dev: S0 A+E .640/.640 (exact/macro-F1); SP A+E+P .651/.650; P alone .618; A+P .626.
- natural_dev: S0 .804/.558; SP .803/.567.
- Pre-declared rule: SP >= S0+.010 on dev AND >= S0 on natural_dev AND dev macro-F1 >= S0. Dev passes (+.011), natural_dev misses by .001 (.803 vs .804). **GATE FAIL by the rule.** Probe NOT carried to eval/natural_test. eval and natural_test untouched by this step.
- Note: lbfgs ConvergenceWarning at max_iter=500 for C=0.1/1 (not the chosen C). Probe alone (.618 dev) beats Evidence alone (.594 dev) and Analyst alone (.565), a finding for the paper's ablation, not a headline.

## 2026-10-22 — Phase D declared (before running): pooled-dev Reflector fit + ordinal decoding
- Script `e23/phase_D_gate.py` -> `phase_D_gate.out`. DEV-ONLY: dev (500 bal) + natural_dev (500 nat). eval/natural_test untouched.
- 5-fold CV x3 on pooled 1000 rows (stratified on split x label). Test on held-out dev part and held-out natural_dev part separately.
- S0 = A+E LR(C=1) fit on dev-train-fold only (current protocol). D1 = same LR fit on pooled-train-fold. D2 = D1 probs decoded by rounding expected rating. D3 = D1 + 1 extra feature log review length not used (no).
- PASS rule: D1 or D2 mean of (dev, natural_dev) exact >= S0 + 0.005 AND mean macro-F1 >= S0. Pick one config only; if both pass, take higher exact. FAIL -> keep S0, log, drop.
- Phase D gate result (`e23/phase_D_gate.out`): S0 mean exact .691 / F1 .622 (dev .640, natural_dev .741); D1 .719 / .623; D2 .722 / .643 (dev .642 / .640, natural_dev .802 / .647). GATE PASS. Config chosen by rule: D2 (higher exact) = A+E LR(C=1) fit on pooled dev+natural_dev (1000 rows), decode by rounded expected rating.
- Declared single readout: `e23/phase_D_readout.py` fits D2 on all 1000 pooled rows, scores eval + natural_test once, vs S0 (stack fit on dev only) with paired bootstrap 3000 + McNemar. Note: natural_dev exact gain is largely class-prior learning (5★ share); macro-F1 and polarity reported alongside. Disclose in paper as dev-gated, post-plan.
- Phase D readout (`e23/phase_D_readout.out`, run once): eval S0 63.9 / F1 .638 / polarity 82.4 vs D2 62.0 / .617 / 80.5 (exact -1.9 CI [-4.1,+0.3], p .12; polarity -1.9 CI [-3.0,-0.8]). natural_test S0 73.4 / .577 / 90.7 vs D2 75.0 / .570 / 91.2 (exact +1.6 CI [+0.1,+3.0], p .038; polarity +0.5 CI [-0.1,+1.1]; macro-F1 down; 1★ recall .67 -> .54, 5★ .81 -> .88).
- Verdict: D2 NOT ADOPTED. Dev gate passed but the gain was prior learning (more 5★ predictions): helps skewed natural_test exact a little, hurts balanced eval and minority-class recall. Not a model improvement. Keep S0 (dev-only balanced fit) as the headline stacker. Report D2 as a prior-shift trade-off finding in the paper. Lesson: natural_dev exact is prior-dominated, so a dev gate on it can pass a prior shift; use macro-F1/per-class recall as the gate next time.

## 2026-10-22 — Phase F2 declared (before running): multi-layer probe
- Why: F probe failed by .001 on natural_dev, but probe alone (.618 dev) beat Evidence (.594) and Analyst (.565). Mid-layer states usually carry more sentiment than the last layer.
- `e23/probe_agent2.py`: same base LLaMA, same compact prompt, same fit set (Random(11), 10K) and splits. Saves last-token states at layers 16, 20, 24 and mean-pool of layer 20 (float16) in `e23/hid2/`. Resumable, 500-row chunks, `stage_F2.txt` marker.
- Gate `e23/probe_gate_F2.py` (dev-only): pick layer-set + C on dev; same pass rule as F (dev +.010 over S0, natural_dev >= S0, dev macro-F1 >= S0). Plus macro-F1 on natural_dev must be >= S0 (lesson from Phase D). Multiple variants tried -> note multiplicity. eval/natural_test read once only if PASS.
- Local, ~4.3 h, no Colab.
- selftest: manual layer loop matches model.model output (maxdiff 0.0); vec 16384 = 3 last-token layers + mean-pool L20
- 2026-10-22 21:40 F2 extraction paused by user (laptop heat); killed cleanly, resumable from hid2 chunks. Restart: nohup caffeinate -i sh -c '.venv/bin/python v2_reflective/e23/probe_agent2.py >> v2_reflective/e23/probe_agent2.out 2>&1 && echo F2_extract_DONE >> v2_reflective/e23/stage_F2.txt' &
- Phase F2 extraction done (`F2_extract_DONE`, 2026-10-24; resumed after user pause). Gate `e23/probe_gate_F2.py` -> `probe_gate_F2.out`.
- Probe alone on dev (fit 10K): L16 .622, L20 .628, L24 .584, mean20 .628 (all at C=.003; larger C worse). Chosen on dev: L20, C=.003 (tie with mean20, first wins).
- Stack CV: dev S0 .640/.640, SP A+E+P2 .645/.644 (+.005, needs +.010); natural_dev S0 .804/.558, SP .813/.574. P2 alone dev .611, natural_dev .821/.577. GATE FAIL (dev gain +.005 < +.010). Not read on eval/natural_test.
- Takeaway: mid layers beat last layer (L24 .584 < L20 .628) but the probe is mostly redundant with Evidence+Analyst on balanced dev. Probe family (F, F2) closed: two variants tried, both FAIL; report in ablation with multiplicity note.

## 2026-10-24 — Phase G declared: Evidence learning curve (decides whether fetching more Electronics data is worth it)
- `e23/evid_curve.py` -> `evid_curve.out`. Hold out 10K balanced rows of evid_train (Random(5)) as validation; train TF-IDF+LR (same config as evidence.py, title from evidence_config.json) on balanced subsets 11.25K/22.5K/45K/90K of the rest. Report val exact + macro-F1. eval/natural_test untouched.
- Decision rule: if 45K->90K gain >= +1.0 pt on val, fetch more Electronics rows (beyond file row 2.5M, disjoint by id and text) to train a ~400K+ Evidence agent; else stop data scaling.
- Phase G curve result (`e23/evid_curve.out`, val = 10K held-out balanced): 11.25K .535 / 22.5K .549 / 45K .552 / 90K .568 (macro-F1 .533/.548/.550/.566). 45K->90K = +1.6 >= +1.0 -> rule says fetch more data.

## 2026-10-24 — Phase G2 declared: big Evidence agent (500K balanced)
- `e23/fetch_big.py` -> `evid_big.jsonl`, `fetch_big.log`. Stream the first 6M Electronics review rows (same URL, same MIN=20 chars). Skip any row whose id is in pool.jsonl or whose text is in evid_train/dev/eval/natural_test/natural_dev or already taken. Reservoir 80K per star, Random(2025). evid_big = evid_train (100K) + these 400K = 500K balanced.
- `e23/evidence_big.py` -> `evid6_<split>_probs.jsonl`, `evidence_big.out`: same TF-IDF+LR config as evidence.py (title from evidence_config.json, C=3) on 500K.
- Gate (dev-only, `e23/stack_gate_G.py`): S0 A+E vs SG A+E6, 5-fold CV x3 on dev and natural_dev. PASS if dev exact >= S0+.010 AND natural_dev exact >= S0 AND macro-F1 >= S0 on both. PASS -> E6 replaces E everywhere (one readout on eval/natural_test, all paired stats rerun); FAIL -> log, keep E.
- Phase G2 result: fetch_big DONE (6M rows streamed, 80K/star reservoir, 398,819 after exact-text dedup, 128,204 rows skipped as pool/split overlap). evidence_big.py: leak assert passed; 498,819 train rows; TF-IDF+LR fit 355 s; lbfgs ConvergenceWarning at max_iter=400 (model possibly under-converged). E6 alone: dev .602 (E .586 direct argmax), natural_dev .712 (E .714).
- Gate `stack_gate_G.out`: dev S0 .640/.640 vs SG A+E6 .632/.631; natural_dev S0 .804/.558 vs SG .800/.562. E6 alone CV dev .613 vs E .594. **GATE FAIL.** E6 not adopted; not read on eval/natural_test.
- Reading: 5x data makes the Evidence agent alone better (+1.9 dev CV), but the stack does not improve: the stronger Evidence agent overlaps more with the Analyst. Note the dev gates are underpowered: 500 rows gives about ±4.3 pt 95% margin on exact, so gains under ~2 pts cannot be resolved on dev.

## 2026-10-24 — Phase H declared (user approved "more training"): bigger dev (dev2) + bigger probe fit (P3)
- Why: dev (500) gives ±4.3 pt margin; every gate needed +1 pt. dev2 = 2,000 balanced rows gives about ±2.1.
- `e23/build_H.py`: dev2.jsonl = 400/star from evid_big_extra (Random(2026)); fit2.jsonl = 6,000/star from evid_train not in the Phase F fit set (Random(12)). Writes evid_big_extra_nodev2.jsonl (extra minus dev2).
- GPU queue `e23/queue_H.sh`: (1) Analyst R on dev2 via eval_p1.py; (2) `probe_agent3.py` (L20 last-token + mean-pool L20, stops at layer 20) on dev2, then fit2 (30K). Resumable chunks in `e23/hid3/`. Marker `stage_H.txt`.
- CPU `e23/evid_H.py`: E (same evidence.py config, 100K) probs on dev2; E6b = 500K config minus dev2 rows, max_iter 1000; probs on dev2/dev/natural_dev/eval/natural_test.
- P3 probe = LR on standardised L20(+mean20, choice made on dev2 CV) features of fit (10K, from hid2) + fit2 (30K); C in {.001,.003,.01} chosen on dev2.
- Gate `e23/gate_H.py` (dev2 + natural_dev only). Candidates: A+E6b, A+E+P3, A+E6b+P3, P3+E, P3+E6b, vs S0 A+E. Stacker LR C=1, 5-fold CV x3. Best candidate by dev2 exact. PASS if dev2 exact >= S0+.010 AND paired bootstrap 95% CI lower bound > 0 on dev2 CV preds (repeat 0) AND macro-F1 >= S0 on dev2 and natural_dev AND natural_dev exact >= S0. 5 candidates -> multiplicity disclosed. PASS -> single readout on eval/natural_test; then freeze. FAIL -> freeze current results.
- Phase H progress 2026-10-24 14:36: H_evid_DONE, H_analyst_dev2_DONE; probe_agent3 extraction at fit2 19,000/30,000.
- evid_H.out: E dev2 alone .554; determinism check vs evid_dev max|diff| 0.0; E6b (500K minus dev2, max_iter 1000, fit 1063 s) alone: dev2 .579, dev .600, natural_dev .712.
- **Protocol slip, disclosed:** evid_H.py also printed E6b-alone accuracy on eval (.625) and natural_test (.703); my script should have written those probs silently. Seen before the gate. Not used for any choice; the gate reads dev2 + natural_dev only and its rule was fixed beforehand. For reference E alone was eval .624 / natural_test .693, so E6b alone is level on eval.
- **Phase H gate PASS** (`e23/gate_H.out`). P3 chosen on dev2: L20 last-token, C=.001 (dev2 alone .622; fit 40K). dev2 CV: S0 A+E .600/.597; A+E6b .612; A+E+P3 .627; A+E6b+P3 .633; P3+E .639; **P3+E6b .639/.637** (best, tie with P3+E, first by order wins... max() picked P3+E6b). Gain +.039, paired CI rep0 [+.024, +.058]. natural_dev CV: S0 .804/.558, P3+E6b .818/.603. 5 candidates tried (disclose).
- Reading: frozen base-LLaMA layer-20 probe replaces the LoRA Analyst; adding the LoRA Analyst back does not help (A+E6b+P3 .633 < P3+E6b .639).
- Declared readout (before running) `e23/readout_H.py` -> `readout_H.out`: stackers LR C=1 fit on dev2 (2,000 balanced) for both S0 (A+E) and SH (P3+E6b); run once on eval and natural_test. Report exact, macro-F1, within-1, polarity, per-class recall; paired bootstrap 3000 + McNemar for SH vs S0, SH vs E6b alone, SH vs E alone, SH vs P3 alone. Also report S0 fit on dev (500), the original headline, for continuity.
- **Phase H readout (run once, `e23/readout_H.out`).** New headline config SH = P3 (frozen base-LLaMA L20 probe, 40K fit) + E6b (TF-IDF LR, ~497K) -> Reflector LR fit on dev2.
  - eval: SH 67.5 exact / macro-F1 .675 / within-1 96.7 / polarity 82.0. Old headline (S0 dev fit) 63.9 / .638. S0 refit on dev2 65.5 / .653.
  - natural_test: SH 75.7 / .612 / 96.7 / 91.5. Old headline 73.4 / .577. S0 dev2 fit 74.2 / .587.
  - SH vs old headline: eval +3.6 [+0.9,+6.3] p .011; natural_test +2.3 [+0.6,+4.0] p .0086.
  - SH vs S0 refit on dev2 (same fit data): eval +2.0 [-0.6,+4.7] p .14; natural_test +1.5 [-0.1,+3.0] p .081 (ns). Part of the gain comes from the bigger Reflector fit set.
  - SH vs best single agent: vs E6b eval +5.0 [+2.4,+7.4] p 1.7e-4, natural_test +5.5 [+3.8,+7.1] p 1.3e-10; vs P3 eval +6.3 [+3.8,+8.8] p 6.4e-7, natural_test +1.4 [-0.1,+2.9] p .074.
  - Polarity: SH vs E6b +3.1 [+1.0,+5.2] eval, +2.6 [+1.6,+3.6] natural_test. SH vs old headline polarity ns (-0.4 / +0.9).
- Verdict: SH adopted as the new main Phase 2 configuration (dev2-gated, read once). The LoRA Analyst becomes an ablation row. To do before freeze: variance over seeds for SH, regenerate frozen tables/figures, update outline + deck. Uncorrected p-values; 5-candidate selection on dev2 disclosed; evid_H eval/natural_test print slip disclosed above.

## 2026-10-24 — Phase H variance declared (before running)
- `e23/variance_H.py` -> `variance_H.out`, `variance_H_seed{0-4}.json` (resumable per seed). Per seed: bootstrap the 40K probe fit set and refit P3 (L20, C=.001); bootstrap dev2 for the Reflector (both SH and S0 refit on the same bootstrap). E6b held fixed (500K retrain x5 too slow; disclosed). Report mean ± sd over 5 seeds on eval and natural_test: SH, S0 (dev2 fit), E6b, P3 exact + macro-F1, and SH-minus-E6b, SH-minus-P3, SH-minus-S0. Post-selection robustness check, not a selection step.
- Phase H variance result (`e23/variance_H.out`, 5 seeds, E6b fixed):
  - eval SH: exact 65.8 ± 0.6  macroF1 0.657 ± 0.007
  - eval S0: exact 65.1 ± 0.8  macroF1 0.650 ± 0.007
  - eval E6b: exact 62.5 ± 0.0  macroF1 0.624 ± 0.000
  - eval P3: exact 60.7 ± 1.1  macroF1 0.605 ± 0.010
  - eval SH-minus-E6b: +3.34 ± 0.61 pts, per seed [3.9, 3.4, 3.5, 3.6, 2.3]
  - eval SH-minus-P3: +5.18 ± 0.86 pts, per seed [4.5, 5.9, 6.3, 4.4, 4.8]
  - eval SH-minus-S0: +0.70 ± 1.23 pts, per seed [2.1, 0.7, 1.6, 0.1, -1.0]
  - natural_test SH: exact 75.4 ± 0.8  macroF1 0.610 ± 0.005
  - natural_test S0: exact 74.4 ± 0.9  macroF1 0.591 ± 0.010
  - natural_test E6b: exact 70.2 ± 0.0  macroF1 0.550 ± 0.000
  - natural_test P3: exact 73.0 ± 0.7  macroF1 0.586 ± 0.012
  - natural_test SH-minus-E6b: +5.19 ± 0.82 pts, per seed [5.7, 4.8, 3.9, 6.0, 5.4]
  - natural_test SH-minus-P3: +2.43 ± 1.19 pts, per seed [3.0, 1.3, 1.0, 2.8, 3.8]
  - natural_test SH-minus-S0: +1.00 ± 0.61 pts, per seed [0.1, 0.9, 0.9, 1.7, 1.3]
- Reading: the multi-agent gain over each single agent is positive in 5/5 seeds on both splits (SH-E6b +3.3±0.6 eval, +5.2±0.8 natural_test; SH-P3 +5.2±0.9 eval, +2.4±1.2 natural_test). SH vs the old two-agent stack refit on the same dev2 bootstrap: +0.7±1.2 eval (4/5 positive), +1.0±0.6 natural_test (5/5 positive), small.

## 2026-10-24 — Frozen tables regenerated with Phase H rows
- `e23/freeze_tables.py` extended (no selection; fixed model list): adds P3 alone, E6b alone, Stack Analyst+E refit on dev2, and SH P3+E6b (dev2 fit, MAIN). Old rows kept for continuity. Probs-file loader now cached (`functools.lru_cache`): the old version re-read the file per row (O(n^2)), same numbers, 3 s instead of minutes.
- Output `e23/results_frozen.md`, `e23/freeze_tables.out`, figure `paper/figures/e23/main_bars.png`. Previous table saved to the session scratchpad before overwrite.
- Check: SH 67.5 / .675 eval, 75.7 / .612 natural_test, and S0 dev2-fit 65.5 / 74.2 match `readout_H.out` exactly.
- Backup `pes_20261024_1641` re-verified PASSED; it predates variance_H, so a new snapshot follows (see DAILY_STATUS backup ledger).

## 2026-10-24 (late) — Deck v6 and paper results section for Oct 25 presentation
- Deck: `build_deck_e23.py` now writes `../Agentic_LLaMA3_Presentation_v6_phaseH.pptx` (27 slides). v5 file left untouched on disk; previous builder copy in the session scratchpad. Changes: title date; slide 12 relabelled "first stack (Oct 13), superseded by SH"; old "Phase 2 focus" slide replaced by the experiment ledger (slide 15); new slide 16 SH main result (table, paired CIs, variance, honest-read caveat); new slide 17 frozen bar chart; roadmap checkpoints updated. Rendered via LibreOffice and checked visually.
- `freeze_tables.py` bar chart now colours the SH row red; numbers unchanged. Bootstrap CIs of rows after the inserted ones shift by about ±0.2 (rng order changed); point estimates identical.
- Paper: new shared file `paper/e23_results.tex` (section "Results — Amazon Reviews 2023 Electronics (Leak-Free, Frozen)": main table, paired table, variance table, bar figure, candidate ledger). It is `\input` in both `paper/main.tex` (before Run 1 results) and `paper/conference/ConferencePaper.tex` (before Experiments and Results), so the two papers stay in sync. Old Run 1 sections kept as they were (added alongside, not replaced). Prose not written: facts for the author are in `% FACT` comments. Both compile (main 20 pages, conference 11 pages), no overfull boxes from the new file. Pre-edit tex copies saved in the session scratchpad.
- Backup `pes_20261024_2301` VERIFIED (made before the deck/paper edits).
- Backup pes_20261024_2306 VERIFIED (includes deck builder + paper edits; the .pptx itself is outside the archive scope).

## 2026-10-24 (night) — Deck v6 update slide + speaker notes; presenter notes; experiments summary
- Deck v6 rebuilt (28 slides): new slide 10 "What changed since the last review" (tried / main system / result + caveat band); title-slide system line and agenda updated; speaker notes written into the notes pane of every slide (NOTES dict in `build_deck_e23.py`; historical slides get a one-line note). Rendered and checked.
- `Presenter_Notes.md`: new Part 0 for the Oct 25 v6 deck (slide map, scripts for slides 10/13/16/17/18, new expected questions, v6 numbers cheat sheet). Parts 1–6 kept, marked as written for the older v3 deck. Pre-edit copy in the session scratchpad.
- New `v2_reflective/EXPERIMENTS_SUMMARY.md`: whole-project ledger of what worked (W1–W10), what failed or was not adopted (F1–F16), baselines/robustness, historical results, integrity disclosures, and next steps.
- An email-style update draft was written and then deleted at the user's request (user wanted the update in the deck instead).

## 2026-10-24 (night) — First off-disk backup: Google Drive
- Google Drive for desktop installed by the user (Homebrew cask `google-drive`, v133.0.2), signed in as the project account; default streaming mode, no settings changed.
- New folder created: `My Drive/PES_backups/` (did not exist before; checked). Nothing else in Drive read, moved, changed or deleted.
- Copied with `cp -n` (never overwrite): `pes_20261024_2306.tar.gz` (1,811,570,828 bytes) + `.sha256`. SHA-256 of the Drive copy = local = recorded: `aca255f7ee81d11df865e96aaf4a415a48ce4322862b8e791552b0beec6857c5`.
- Cloud upload check (Drive web search): `.sha256` visible immediately; the 1.8 GB archive was still uploading at the time of this entry (see the backup ledger in progress/DAILY_STATUS.md for the confirmed state).
- Rules for Drive (user asked for great care): only write inside `PES_backups`; copy only (`cp -n`), no sync tools, no `--delete`, no moves; never delete anything in Drive without asking the user and naming each file.
- Confirmed: both files (archive + .sha256) visible in Drive web search, so the cloud upload is complete.

## 2026-10-24 (night) — Phase A1 DECLARED (before running): bigger probe fit (P3b), part of the Phase 2 v3 plan
- Why: P3 improved from a 10K to a 40K fit and is the best single agent on natural_test (74.3); the Phase G curve says more fit data still helps. Local only: the probe must use the same MLX Q4 model as the demo.
- Data: `e23/build_A1.py` -> `fit3.jsonl` = 80K balanced (16K/star, Random(2027)) from `evid_big_extra_nodev2` (asserted disjoint from dev, dev2, fit2, eval, natural_test, natural_dev).
- Extraction: `e23/probe_agent4.py` = probe_agent3 with the L20 last-token state only (the P3 feature; same prompt, same model, layers[:20]), chunks of 500 in `e23/hid4/`, resumable. Queue `e23/queue_A1.sh`, marker `stage_A1.txt`, log `probe_agent4.out`. ~5K rows/h, so about 16 h (2 nights); `caffeinate -i`.
- P3b = LR on standardised L20 last-token features of fit (10K) + fit2 (30K) + fit3 (80K) = 120K; C in {.0003, .001, .003} chosen on dev2 CV.
- Gate (dev2 + natural_dev only; script written before extraction ends): P3b+E6b vs SH (P3+E6b), stacker LR C=1, 5-fold CV x3. PASS if dev2 exact >= SH+.010 AND paired bootstrap CI lower bound > 0 (repeat 0) AND macro-F1 >= SH on dev2 and natural_dev AND natural_dev exact >= SH. One candidate. PASS -> single readout on eval/natural_test, SH replaced. FAIL -> logged, SH stays. If extraction is cut short for schedule reasons, the gate runs on the rows done by then (stated in the result).
- 2026-10-24 night: `Project_Journey.md` written next to `Presenter_Notes.md` (full Phase 1 to now history, failures, successes, lessons, why the v3 direction; linked from Presenter_Notes Part 0).
- A1 schedule change (user, Oct 25 00:05): laptop needed for the presentation, so extraction auto-pauses at 08:00 Oct 25 (~43K of 80K rows expected) and resumes afterwards by rerunning queue_A1.sh (resumable chunks). Gate rule unchanged; if the full 80K is not reached by the gate date, the gate uses the rows done (as declared).

## 2026-10-25 — C1 DECLARED (before running): SH variance with E6b retrained per seed
- `e23/variance_H2.py` -> `variance_H2.out`, `variance_H2_seed{0-4}.json` (resumable). Identical to variance_H.py (same seeds; same probe-fit and dev2 bootstraps, draw order kept) plus: E6b retrained each seed on a bootstrap of its ~497K training rows (evid_train + evid_big_extra_nodev2; same TF-IDF 1-2gram, 400K features, LR C=3, max_iter 1000). Reports the same table as variance_H. Robustness check only, not a selection step. CPU, `nice -n 10`, runs alongside A1; ~20-25 min per seed.
- Also written now, before A1 extraction ends: `e23/gate_A1.py` (implements the A1 gate exactly as declared; data-loading smoke test only, no gate numbers looked at).
- C1 STOPPED at 00:12 Oct 25 (seed 0 not finished, nothing written): laptop on battery and swap at 10.3 of 11.3 GB with A1 + C1 together. Rerun later when the GPU is idle and the Mac is plugged in. analysis_C ran (small).
- RESULT C2-C5 (`e23/analysis_C.out`, figures `paper/figures/e23/reliability_SH_{eval,natural_test}.png`):
  - C2 Holm over the 8 SH comparisons: still significant after correction: SH vs E6b (both splits), SH vs P3 (eval), SH vs old headline (eval adj .045, natural_test adj .043). Not significant: SH vs P3 on natural_test, SH vs S0 refit on dev2 (both splits).
  - C3 errors: off-by-one = 89.8% (eval) / 86.4% (natural_test) of SH errors; 2+ stars off 33 / 66 rows. Length buckets, agent-agreement split, per-class recall and example ids in the .out.
  - C4 calibration: ECE .047 (eval) / .062 (natural_test); SH slightly under-confident (mean confidence .650 vs accuracy .675; .704 vs .757).
  - C5 cost per review on the M1: P3 ~0.67 s (LLaMA forward to layer 20), TF-IDF ~0.06 ms, Reflector ~0.06 ms. So the probe is >99% of SH's cost, which motivates the Router.
- 2026-10-25 00:30: Phase 2 v3 spec written: `docs/superpowers/specs/2026-10-25-phase2-v3-design.md` (pending user review).

### 2026-10-25 08:00 — A1 planned pause
- The auto-pause timer stopped `probe_agent4.py` at 08:00 IST, before the presentation.
- Progress: fit3 45,000 / 80,000 rows (91 chunks in `hid4/`). Total time so far is about 8 h.
- `stage_A1.txt` shows `A1_extract_FAILED`. This is the expected result of the planned kill, not a crash.
- Resume after the presentation: `sh v2_reflective/e23/queue_A1.sh` from the repo root. Saved chunks are skipped. About 35K rows remain, about 6 h.

### 2026-10-25 — One deck; deck updated for the presentation
- There is now one current deck: `../Agentic_LLaMA3_Presentation.pptx` (30 slides), built by `v2_reflective/build_deck_e23.py`.
- All 14 older `.pptx` files were moved, unchanged, to `../old_decks/`. Nothing was deleted.
  - The builder's template now lives at `../old_decks/Agentic_LLaMA3_Presentation_FINAL_v3_aligned.pptx`.
  - `Agentic_LLaMA3_Presentation_v3.pptx` was removed from git tracking. It remains in git history and in `old_decks/`.
- New slide 19, robustness: Holm correction, error analysis, calibration and cost, all from `e23/analysis_C.out`.
- New slide 20, next steps: the Phase 2 v3 plan from `docs/superpowers/specs/2026-10-25-phase2-v3-design.md`. A1 status is shown as 45K / 80K.
- Consistency fixes:
  - Slide 4 Phase 2 card now says Electronics only.
  - Slide 7 is marked leaky and withdrawn.
  - Slides 8 and 9 are marked as the original design.
  - Slide 17 now reports Holm-adjusted p.
  - The roadmap shows the two-stage freeze (Nov 1 and Nov 5).
  - The agenda line is updated.
  - Section numbers now run 01–20 on every non-title slide.
- Updated `Presenter_Notes.md` Part 0: new slide map, scripts for slides 19–21, timing and cheat-sheet rows.
- Rounding note: `analysis_C.out` prints SH vs E6b on natural_test as +5.4 and SH vs S0 dev2 as +1.4. The exact values are 109/2000 = 5.45 and 29/2000 = 1.45. The deck and paper use the `readout_H.out` values, +5.5 and +1.5. It is the same number with different rounding.

### 2026-10-25 — Deck: v2_reflective slides and guide
- Added deck slides 22 and 23: the `v2_reflective/` workspace map and the 6-step experiment workflow. Historical slides are now 24–32, and the deck has 32 slides.
- New `V2_Reflective_Guide.md`, next to `Presenter_Notes.md`. It covers the folder map, why the folder has this name, the `e23/` pipeline groups, the key to probability file prefixes (evid … evid8), the talking script and expected questions.
- `Presenter_Notes.md` Part 0 updated: slide map, timing, and a link to the guide.
- The old decks stay in `../old_decks/` (kept on purpose, at the user's request).

### 2026-10-25 — Folder structure reorganized; rules made persistent
- User asked for a cleaner structure that persists across sessions. Nothing was deleted. Experiment code paths (`v2_reflective/e23/`) are unchanged, so A1 still resumes.
- New `README.md`: a START HERE map of every file. New short `CLAUDE.md`: working rules and where new files go. The old long CLAUDE.md is now `docs/archive/CLAUDE_2026-10-09.md`.
- Outer `PES_Project_Phase1/CLAUDE.md` imports the repo CLAUDE.md, so the rules load in every session.
- Moves:
  - Presenter_Notes, Project_Journey, V2_Reflective_Guide and the demo notebooks go to `presentation/`.
  - The 8 plan documents go to `v2_reflective/plans/`. Older LOG entries still name their old paths.
  - Reference PDFs go to `docs/reference/`.
  - CODE_EXPLAINED.md goes to `docs/archive/`.
  - build_presentation_v4.py goes to `v2_reflective/`; both deck builders were updated.
  - The v4 PDF goes to `../old_decks/`.
  - March 2026 outer files go to `../archive_early_2026/`.
- The wedding QR files in the outer folder are personal and were left untouched.
- The deck rebuilds fine after the moves (32 slides).

### 2026-10-25 — Tidy-up
- Wedding QR files (personal) moved out of the project to `~/Documents/Personal_Wedding_QR/`. They are not in git; the outer folder is not a repo.
- Found `v2_reflective/e23/dev2.jsonl`, which holds 2,000 reviews with their text, tracked in git since commit 1031426 (Phase H). This breaks the "no review text in git" rule.
  - Removed from tracking with `git rm --cached`. The file stays on disk, so the scripts are unaffected.
  - Added it to `.gitignore`.
  - It is still in the git history. Purging that history needs a rewrite and a force-push, and the user decides on that.
- Also gitignored: `fit3.jsonl`, `hid4/`, and the demo notebooks (which can contain review text).
- Now tracked: `probe_agent4.out`, `stage_A1.txt`, `variance_H2.out`, and `docs/archive/CODE_EXPLAINED_2026-10-11.md`.

### 2026-10-25 — Spec approved; implementation plan written; Instruct model downloaded
- The user approved the Phase 2 v3 spec, including the Phase 1 + Phase 2 demo for the December final presentation.
- Plan: `docs/superpowers/plans/2026-10-25-phase2-v3-plan.md`, 14 tasks:
  - A1 resume and gate
  - text, Evidence and Reflector modules, with the persisted SH agents verified against the frozen probabilities
  - shared LLaMA with a LoRA toggle
  - B1 Router gate
  - pack and Critic, Explainer, LangGraph v3
  - B2 runner, deletion test and report
  - C6 audit table
  - final demo with a headless check
  - freeze stage 1
  - A2 on Colab
- Downloaded Llama-3-8B-Instruct 4-bit MLX (`mlx-community/Meta-Llama-3-8B-Instruct-4bit`, 4.9 GB) to `.scratch/mlx-llama3-8b-instruct-q4`. The user approved this.
- Repo visibility: the GitHub repo is public. `dev2.jsonl` (review text) is still in its history. The history purge was blocked by the permission system and is waiting on the user (see the session reply).

### 2026-10-25 — History purge deferred
- User decision: do not purge `dev2.jsonl` from the git history. Keep the details for future reference.
- Saved in `docs/maintenance/git_history_dev2_purge.md`: the issue, when to revisit, the exact commands, and the consequences.
- `README.md` now lists `docs/maintenance/`. `CLAUDE.md` now forbids history rewrites and force-pushes.

### 2026-10-25 — Plan execution started (inline, user check-in after each task)
- Task 1 (A1 resume and gate, C1) is deferred until the user gives the go.
- Task 2 done. Created `src/agentic_sentiment/v3/paths.py` (all model and artifact paths) and `src/agentic_sentiment/v3/text.py`:
  - tokens, using the same pattern as sklearn TF-IDF
  - contiguous phrase containment
  - display span ("doesn work" → "doesn't work")
  - quoted phrases, straight and curly quotes
  - token drop for the deletion test
- `tests/v3/test_text.py`: 5 tests, red before the code and green after. Full suite: 134 passed (129 old + 5 new).

### 2026-10-25 — Deck brought in line with today's decisions
- Slide 20 now describes the final demo for the December presentation: Phase 1 and Phase 2 live, 3 minutes or less.
- Slide 21 roadmap: retitled to "Plan to Submission and the December Final Presentation", with a new row 6 (December final presentation and demo).
- Speaker notes and `presentation/Presenter_Notes.md` (slides 20–21) updated to match. Rebuilt and rendered; checked visually. The deck still has 32 slides.

### 2026-10-25 — Papers updated with the robustness analyses (C2–C5)
- `paper/e23_results.tex`, shared by both papers:
  - The paired table now shows Holm-adjusted p next to each raw p (the caption no longer says "uncorrected").
  - New Table `tab:e23errors`: errors off by one, per-class recall, agree/disagree accuracy, shortest and longest quartiles, ECE.
  - New Figure `fig:e23reliability`: the two reliability diagrams.
  - New `% FACT` lines, including agent complementarity (E6b strong on short reviews, the probe on long ones) and cost.
  - The disclosure FACT line now reflects that Holm was applied and C1 is pending.
- All numbers are copied from `e23/analysis_C.out`.
- Layout fix: section 10 has no author prose yet, so LaTeX pushed its heading below all of its floats. Added an invisible `\mbox{}\par` anchor (marked for removal once the prose is written). The heading now sits on page 10 with Table 4.
- Builds: `main.pdf` 21 pages (was 20), `ConferencePaper.pdf` 11 pages, 0 errors. The overfull boxes are all pre-existing, in the older Run 1 sections.

### 2026-10-25 — Paper integrity audit (plagiarism and AI content, target each under 5%)
- New `paper/integrity/audit.py`. It attributes every prose word with git blame: lines last changed in a commit with a `Co-Authored-By: Claude` trailer count as AI-drafted. It also finds 8-word verbatim overlaps with the reference PDFs. Output: `INTEGRITY_REPORT.md`.
- Result now:
  - **AI-drafted share 100.0%** (4,729 prose words). The paper text was created in AI-co-authored commits (`main.tex` 2026-03-31; `ConferencePaper.tex` 2026-10-02).
  - Verbatim overlaps are mostly with the user's own earlier proposal: the Abstract has 21 8-word overlaps and the Problem Statement 16.
  - Table/figure words (data): 2,766.
- New `paper/integrity/README.md` with the workflow:
  - the author rewrites the prose from the facts and commits it without the trailer
  - Claude provides tables, figures and facts only, never prose or "humanizing"
  - quote or rewrite reused text; disclose AI use
  - a checklist before each draft, and a suggested rewrite order
- `README.md` and `CLAUDE.md` now list `paper/integrity/`.

### 2026-10-25 — Phase 1 baseline framing (FACT notes for the author)
- The author felt "54.0 / 53.4 for Phase 1" reads badly. We agreed on honest framing instead of swapping in a higher old number.
- Added `% FACT` notes to `paper/e23_results.tex` covering:
  - Lead with the gain: +13.5 / +22.3 points, which is 29.3% / 47.9% fewer errors.
  - Phase 1 strengths on Electronics: within-one-star 93.7 / 94.4, binary 91.2 / 92.9.
  - Why natural_test is 53.4: a class-mix effect.
  - Reproduction on Kaggle with the paper protocol, leak-free: 69.1% exact.
- Numbers not to use, each for a stated reason: 70.7 / 60.1 (leaky), 74.8 (seed 2, unstable), 79.9 (prior-corrected).

### 2026-10-25 — Deck: Phase 1 → Phase 2 slide (honest baseline framing)
- New slide 18, "How Far Phase 2 Moves the Baseline", with four cards and one band:
  - Gain: +13.5 / +22.3 points, 29% / 48% fewer errors, both significant (readout_H: SH vs R p 2.1e-14 eval, 4.3e-71 natural).
  - Phase 1 strengths: within-one 93.7 / 94.4, binary 91.2 / 92.9.
  - Why natural is 53.4: class mix.
  - Reproduction: Kaggle, paper protocol, leak-free 69.1% (always-5 63.8%).
  - Band "Not used, on purpose": 70.7 / 60.1 (leaky), 74.8 (one seed), 79.9 (class-mix correction).
- The deck now has 33 slides. Later slides shift by one; historical slides are now 25–33.
- Slide references updated in the builder (slide 4 and slide 8 subtitles, speaker notes), in `Presenter_Notes.md` Part 0 (slide map, timing, a new script for slide 18, renumbered headings) and in `V2_Reflective_Guide.md`.
- Rendered and checked visually.
- Rounding note: readout_H prints the natural gain as +22.2. The exact value is 445/2000 = 22.25, and the deck and paper round half up to +22.3, the same convention as the 5.45 → 5.5 note above.

### 2026-10-25 — Neutral names in the deck and paper (internal IDs stay in LOG and code)
- User request: neutral names for the presentation and paper. Mapping:
  - SH → "main stack" / "Probe + Evidence stack"
  - P3 → "LLaMA probe"; P3b → "LLaMA probe (120K)"
  - E6b → "Evidence model (~497K)"; E → "Evidence model (100K)"
  - S0 / Analyst+E → "LoRA + Evidence stack"
  - Analyst arm R → "Phase 1 LoRA"
  - The phase letters (R10, F/F2, D, G2, H) were removed from slide text.
- Where it changed:
  - Deck builder: all visible text and speaker notes. A scan of the built deck finds no SH/P3/E6b/S0.
  - `freeze_tables.py`: model labels only. This changes the `results_frozen.md` labels and the `main_bars.png` bar labels; the main bar is coloured by "(main)". A diff of every table value before and after shows them identical.
  - `analysis_C.py`: reliability x-axis label "main stack confidence". After the rerun, `analysis_C.out` is identical except the live timing line: TF-IDF 0.06 → 0.04 ms, Reflector 0.06 → 0.05 ms. Run-to-run noise; the paper and deck say "well under 0.1 ms".
  - `paper/e23_results.tex`: tables, captions and FACT comments renamed, plus a NAMING comment block mapping neutral names to internal IDs. The `pdftotext` output of both PDFs contains no SH/P3/E6b. 0 LaTeX errors.
  - `presentation/Presenter_Notes.md` Part 0: renamed, plus a new "Names to say out loud" section.

### 2026-10-25 — Task 3: Evidence agent and Reflector code; rebuild of the saved agents started
- `src/agentic_sentiment/v3/evidence.py`: `EvidenceAgent` returns probabilities plus per-phrase contributions (coef[star] × tf-idf). Each phrase has `phrase`, `display` (original wording) and `weight`; top 5 for the star, top 3 against.
- `src/agentic_sentiment/v3/reflector.py`:
  - `Reflector.combine(probe or None, evidence)` returns the star, the probabilities and each agent's share. With no probe it uses an Evidence-only fallback.
  - `fit_reflector`
  - `choose_tau`: the smallest threshold whose routed 5-fold CV accuracy is within 0.5 points of the full stack.
- Tests: `tests/v3/test_evidence.py` (2) and `tests/v3/test_reflector.py` (3), red before the code and green after. `tests/v3` total: 10 passed.
- `v2_reflective/e23/persist_agents.py` started: it refits E6b, P3 and the Reflector with the frozen configs and asserts they reproduce the stored probabilities (max diff < 1e-6) and the main stack's 0.675 on eval.
  - Pre-flight: AC power, battery 100%, swap 2.1 GB, no other model job.
  - Output: `persist_agents.out`. Agents go to `v2_reflective/e23/agents/`, which is now gitignored along with `b2/`.
- Rebuild finished in 611 s (Evidence model fit 549 s). Verification:
  - E6b max |diff| vs `evid6b_eval_probs` = 0.0.
  - P3 max |diff| vs `evid7_eval_probs` = 0.0.
  - Reflector eval exact = 0.675, matching the frozen main stack.
- The saved agents reproduce the frozen results exactly. Files: `agents/e6b.joblib` (30.8 MB), `p3.joblib` (177 KB), `reflector.joblib` (2 KB). Task 3 complete.

### 2026-10-25 — Self-correction made a measured, three-level part of v3 (spec and plan updated)
- User requirement: "Self correction needs to be properly done."
- Spec (`docs/superpowers/specs/2026-10-25-phase2-v3-design.md`, Architecture section and experiments table) now defines three levels:
  - **SC1 escalation (rating).** The Evidence model gives a draft star. If it is unsure, the system escalates to the LLaMA probe and the Reflector revises the star. Measured as fixed / broken / net with a binomial test. It is part of the B1 gate (dev2 CV: fixed > broken, p < .05).
  - **SC2 reflective re-rating (rating).** Llama-3-8B-Instruct picks between the stack's top-2 stars on uncertain reviews (margin < 0.20), using the evidence phrases for both candidates. The order is swapped and inconsistent answers abstain. Gated on dev2 with CV stack predictions, plus a natural_dev net ≥ 0 check. This tests the original proposal's claim properly; a FAIL is reported as a negative result.
  - **SC3 explanation loop.** Critic feedback with up to 2 rewrites. Measured as faithfulness before vs after (McNemar).
- Plan (`docs/superpowers/plans/2026-10-25-phase2-v3-plan.md`):
  - Task 5 `gate_B1.py` now computes SC1 (dev2 CV and readout).
  - New Task 7b: `sc.py`, `test_sc.py` and `gate_SC2.py`, with full code.
  - Task 11 `freeze_tables.py` now writes one "Self-correction (three levels)" table.
- Out-of-scope wording narrowed: "LLM choosing the star for every review". SC2 retests only the narrower claim (revision on uncertain reviews).

### 2026-10-25 — Deck: build schedule slide; Next slide shows three-level self-correction
- New slide 22 "Phase 2 v3 Build Schedule", Oct 25 → Nov 5, 10 rows (when / work / compute / status), matching the plan.
- Slide 21 "Next" updated:
  - The Router card mentions SC1 escalation.
  - The accuracy card adds SC2 LLM re-rating, and the 120K probe status is corrected to "paused at 45K / 80K new".
  - The band now reads "Self-correction (3 levels)".
- The deck now has 34 slides; historical slides are 26–34.
- References updated in the builder (slide 4 and slide 8 notes, guide note), in `Presenter_Notes.md` Part 0 (map, timing, new slide 22 script) and in `V2_Reflective_Guide.md`.
- Rendered and checked.

### 2026-10-25 — Deck: full experiment ledger (3 slides); auto-sync to Google Drive for Slides
- User request: a table of every experiment tried, showing what we did, the result and why it was dropped.
- New slides:
  - 17 and 18, "Everything We Tried and Dropped": 19 experiments. These are F1–F16 from `EXPERIMENTS_SUMMARY.md`, plus TextBlob/VGST, the RAG Prover and the dissonance Critic from the audits.
  - 19, "What We Kept, and the Evidence for It": W1–W10.
  - Neutral names throughout.
- The deck now has 37 slides. Later slides shift by 3; historical slides are 29–37.
- Renumbered: builder references and speaker notes, `Presenter_Notes.md` Part 0 (map, timing, new script for slides 17–19, headings) and `V2_Reflective_Guide.md`. Rendered and checked.
- User request: "upload it to slides so the always updated one is there". `build_deck_e23.py` now copies the deck on every build to `My Drive/PES_backups/current_deck/Agentic_LLaMA3_Presentation.pptx`.
  - It overwrites that one file only, so the Drive link stays the same and Drive keeps revisions.
  - This is the single user-approved exception to the `cp -n` rule, recorded in `CLAUDE.md`.
  - First sync done at 13:49.

### 2026-10-25 — Deck: key-terms glossary (2 slides)
- User request: add terminology so the presenter and the audience can follow.
- New slides 10–11, placed right after the background:
  - "Key Terms: Models and Agents": 16 terms, from LLaMA-3-8B, LoRA, the hidden state and the LLaMA probe to TF-IDF, the Evidence model, stack, Reflector, main stack, Router, Explainer/Critic and self-correction.
  - "Key Terms: Data, Evaluation and Method": 16 terms, including balanced vs natural, dev/dev2, exact, within one, macro-F1, polarity, zero/one-shot, label leak, gate, bootstrap CI, McNemar, Holm, ECE, faithfulness and the deletion test.
  - All in plain language.
- The deck now has 39 slides. Slides from 10 on shift by 2; historical slides are 31–39.
- Renumbered: builder references (slides 4, 7, 8 and notes 5, 7, 8, 29) and all speaker-note keys, `Presenter_Notes.md` Part 0 (map, timing, new script for slides 10–11, headings) and `V2_Reflective_Guide.md`.
- Also fixed a stale reference in the slide 8 note, which still pointed to the main result as slide 17; it is now 22.
- Rendered and checked. Auto-synced to Drive `current_deck/`.

### 2026-10-25 — Deck confirmed in Google Slides
- Opened `PES_backups/current_deck/Agentic_LLaMA3_Presentation.pptx` from Drive in Google Slides: https://docs.google.com/presentation/d/1V08iho4p7gdllKPxRVroaPhj6oo8Km1_/edit
- Shows "39 total slides" and the speaker notes. The link is added to `README.md`.

### 2026-10-25 — Deck restructured: 22-slide main talk + appendix; presenter notes rebuilt to match
- User feedback: "ordering seems off and number of slides is way too much. Make it cleaner." Also: "Make sure presenter notes also is being reflected the same."
- **Main talk (22 slides), in story order:**
  - title, problem, gap
  - key terms ×2, at a glance
  - Electronics protocol, how one experiment flows
  - Phase 1 method, leak audit
  - where Phase 2 started (original design, retitled), hallucination study
  - tried/dropped ×2, kept
  - main result, Phase 1 → Phase 2, bar chart, robustness
  - next (v3), build schedule, roadmap
- **Appendix (18 slides) after a divider with a contents list:** datasets, Phase 1 pipeline, Kaggle baseline, dissonance Critic, first stack, preprocessing ablation, nine-lines summary, code workspace, and historical Run 1 ×9. Nothing deleted.
- **Builder refactor:**
  - Slides are ordered by key (`MAIN` / `APPX` lists), and speaker notes are keyed by slide name.
  - Every cross-reference is a `§key§` token resolved after ordering, so reordering no longer breaks references.
  - Eyebrows: main slides numbered 01–21; appendix slides "APPENDIX · …" and "HISTORICAL (SUPERSEDED) · …".
  - The "At a glance" slide was retitled from "What Changed Since the Last Review". The eyebrow "CODE · WORKFLOW" became "HOW WE RUN EXPERIMENTS".
  - Every build writes `presentation/deck_slide_map.md`: slide number, section, title and the speaker note.
- **Presenter notes:** `Presenter_Notes.md` Part 0 rewritten in deck order, without slide numbers (numbers live in the auto-generated map). It covers the structure, timing for the 22-minute main talk, the main-talk list, longer scripts per slide, and an appendix section. `V2_Reflective_Guide.md` references are now by slide name.
- Deck: 40 slides in total (22 main). Rendered and checked. Synced to Drive `current_deck/` (Google Slides link unchanged).

### 2026-10-25 — Deck restructured to the user's flow: 24 slides
- User specification: intro → Phase 1, the baseline paper (what it does) → its novelty and shortcomings → Phase 2 intro → in depth → how it aims to solve the problem → results of Phase 1 → results of Phase 2 → in depth → conclusions. No more than 25 slides.
- New order (24 slides):
  - **Intro:** title, problem, key terms ×2.
  - **Phase 1 paper:** "Baseline Paper (Wang et al., 2025): What It Does, Where It Stops" (retitled), data pipeline, LoRA.
  - **Shortcomings:** leak audit, hallucination study, preprocessing ablation.
  - **Phase 2:** original four-agent design; NEW "Phase 2 System: How a Review Is Rated and Explained" (five-box flow, decision core vs agentic layer); "How it solves the problem" (renamed from "Next"); protocol.
  - **Phase 1 results:** NEW "Phase 1 Results: The Baseline Paper's Method" (Electronics exact / within-one / binary / macro-F1 / MAE, plus the Kaggle paper-protocol reproduction).
  - **Phase 2 results:** dropped ×2, kept, main result, Phase 1 → Phase 2 (rewritten as gain / why it works / honest caveat, so it no longer repeats the Phase 1 slide), all-models chart.
  - **In depth:** robustness.
  - **Close:** build schedule, NEW "Conclusions".
- Removed from this deck but not deleted: they remain in `old_decks/`, and their code stays in the builder.
  - Run 1 historical ×9, Kaggle data slide, Kaggle baseline, dissonance Critic, first stack, nine-lines summary, code map, workflow, at-a-glance, roadmap, appendix divider.
  - The roadmap dates moved into the Conclusions band.
- `Presenter_Notes.md` Part 0 rewritten in the same eight-section order, with timing. `V2_Reflective_Guide.md` notes that its two slides are no longer in the deck. `CLAUDE.md` and memory record the flow and the 25-slide limit.
- Rendered all 24 slides (contact sheet) and checked. Synced to Drive / Google Slides.

### 2026-10-25 — Deck: baseline paper in simple points + why it falls short; key terms moved off the slides (23 slides)
- User: add a slide at the start with more on the chosen paper and what it does in simple points, then a slide on why it fails. Also: "key terms don't need to be a slide".
- New slide 3, "The Paper We Chose: What It Does":
  - Wang et al. (2025), ITM Web of Conferences 70, 04021.
  - Four cards: task, data (TextBlob, equal per star, VGST, neutral ×2), model (LLaMA-3-8B + LoRA, r=4 / α=64 / 3 epochs / lr 5e-5, 4-bit) and prompt (rubric, one-shot, step-by-step).
  - "Reported" band: 92.6% accuracy, 93.3% F1, 5-fold CV and a 40K test set, beats the classical baselines and XLNet.
- New slide 4, "Why It Falls Short":
  - Four cards: one pass with no checking; a fragile test (our first run's one-shot leak of ~8 pts, always-5 baseline at 63.8%, leak-free balanced 54.0%); TextBlob filtering backfires (−52% of 1★, −67% of 2★, −2.5 to −3.5 pts); template justifications (wrong in 53% of 150).
  - Band: what Phase 2 needs.
  - The speaker note says the leak came from our reproduction's example choice, not from the authors.
- The new pair replaces the combined "Baseline Paper: What It Does, Where It Stops" slide. It stays in the builder and in `old_decks/`.
- The key-terms slides were removed from the deck. Their 32 terms now sit in a table in `Presenter_Notes.md` Part 0, "Key terms (for you; not on the slides)".
- The deck now has 23 slides. Part 0 sections and timing were renumbered. Rendered and checked; synced to Google Slides.

### 2026-10-25 — Results visuals in the deck; labelled results store in the repo
- User: show more results in the deck, and store all saved results, labelled, in the repo.
- New `v2_reflective/e23/figures_deck.py` reads stored probabilities only, with the main-stack Reflector fit on dev2 exactly as in readout_H / analysis_C. It writes:
  - `cm_main_eval.png` and `cm_main_natural_test.png`
  - `recall_phase1_vs_main.png`
  - `agents_complement.png`
- Every printed number (confusion counts, per-star recall, length quartiles, agree/disagree) equals `analysis_C.out`. Output: `figures_deck.out`.
- New deck slides, the deck now has 25 (the limit):
  - "Where the Main Stack Is Right, and Where It Errs": confusion matrix, plus per-star recall Phase 1 vs Phase 2. It states honestly that Phase 1 finds more 4★ on natural test (.715 vs .639).
  - "Why Two Agents Beat One": length quartiles and disagreement accuracy.
  - Speaker notes and `Presenter_Notes.md` Part 0 updated (section 6 text, slide ranges, timing).
- New `v2_reflective/e23/collect_results.py` creates labelled copies of every saved result in `results/e23/<category>/` and writes `results/e23/INDEX.md` (file, what it is, result/status, source).
  - Categories: `01_main_results`, `02_gates`, `03_baselines`, `04_robustness`, `05_ablations_audits`, `06_phase1_kaggle_closeout`, `07_phase2_v3`, `08_extraction_logs`, `09_figures`.
  - 54 files copied; 4 pending (SC1, SC2, explanations report, A1 gate).
  - `halluc_p1.out` contains review excerpts, so only its summary lines are copied. A scan confirms no review text in `results/e23/`.
  - New `results/README.md` covers e23 (current) and run1/run2 (historical).
- `CLAUDE.md`: `collect_results.py` added to the after-each-step routine, plus a file-placement row. The `README.md` map now points to `results/e23/INDEX.md`.
- Side note: the original `v2_reflective/e23/halluc_p1.out` with review excerpts is tracked in git, the same class of issue as `dev2.jsonl`. The user decided against history rewrites; this is to be added to the maintenance note.

### 2026-10-25 — Presenter notes check
- Checked: all 25 slides have their own speaker note, and the `Presenter_Notes.md` Part 0 sections and slide ranges match `deck_slide_map.md`.
- Fixed:
  - Slide 12 note said "A1 is running"; it is paused at 45K / 80K.
  - Slide 12 note and Part 0 now mention the three self-correction levels.
  - "Old headline" wording replaced with "earlier LoRA + Evidence stack".
  - Sentence-start capitalisation of "main stack".
- Rebuilt; Google Slides synced.

### 2026-10-25 — Deck review: flow and explanation fixed (24 slides)
- User asked for a check of the deck and of the explanation flow. Problems found and fixed:
  1. **Problem slide pitched images and photos**, contradicting the later finding that photos don't help. It is now "No Second Opinion / Unverifiable Explanations / Hard Language", with a new "gap" line.
  2. **"Phase 2 System" and "Agents That Route…" duplicated each other**, and no slide mapped shortcomings to fixes. New slide "How Phase 2 Answers Each Shortcoming": 8 rows of shortcoming → Phase 2 answer → the number that checks it. "Agents That Route…" left the talk; it stays in the builder.
  3. **The protocol slide was outdated** (100K training reviews, 300K features) and broke the flow. Removed. Its facts moved to the Phase 1 Results subtitle: data, frozen disjoint splits, test sets read once.
  4. **Phase 2 results led with process.** The order is now main result → Phase 1→Phase 2 → all models (headline), then in depth: where it errs → why two agents → tried/dropped ×2 → kept → robustness.
  5. **Smaller fixes:**
     - The title slide says "Evidence model" plus "Self-checking agents", and the agenda line is updated.
     - The hallucination slide is retitled "Phase 1's Justifications Are a Template" (no "adapter" jargon).
     - The bar-chart subtitle no longer shows a file path.
     - The schedule says "self-correction 1/2/3" instead of SC1–SC3.
     - Conclusions gain a LIMITS card: one dataset; part of the gain is the fit set; agentic numbers pending.
- Speaker notes updated for problem, solves, Phase 1 results and conclusions. `Presenter_Notes.md` Part 0 is reorganised to the new order, with updated timing.
- The deck now has 24 slides. Rendered and checked; synced to Google Slides.

### 2026-10-25 — Presenter notes: Q&A from the pre-presentation questions
- `Presenter_Notes.md` "New questions to expect" gains:
  - "are these results good?", with the baseline ladder (20 / 34.3 / ~48 / 52.8 / 54.0 / 67.5), within-one 96.7, and the comparison with the paper's 92.6
  - "what is the natural test, and which other tests?": balanced eval, natural test, dev / dev2 / natural_dev, Kaggle paper-protocol
  - "what is the main stack / Probe + Evidence stack?"
  - "is the tried-and-dropped table final?"
  - a text block diagram
- The main-result section opens with the "two readers and a learned referee" line.
- Slide 14 (main result) speaker note now defines the main stack and gives the "is it good?" answer. Rebuilt; Google Slides synced.

### 2026-10-25 — A2 relabelled in the deck
- User: keep the DeBERTa run (A2) under wraps for now and label it something else.
- The deck and speaker notes now say "optional extra specialist (GPU run, kept only if it passes its check)". The schedule row is marked "Optional".
- The `Presenter_Notes.md` mention was neutralised the same way. A scan of the built deck finds 0 mentions of "DeBERTa".
- The plan and spec (`docs/superpowers/`) keep the technical name as internal documents. The experiment itself is unchanged: optional, dev-gated, Nov 1.

### 2026-10-25 — Task 4: shared LLaMA (Phase 1 LoRA on / off) and probe agent
- Pre-flight: AC power, swap 1.0 GB, no other model jobs.
- `src/agentic_sentiment/v3/llama.py`:
  - `SharedLlama` loads the LLaMA-3-8B Q4 MLX base once, with adapter R. `.lora(on)` sets the LoRA scale to the adapter value or to 0.
  - `ProbeAgent.hidden()` / `.probs()` give the L20 last-token state through the saved P3 classifier, always with the LoRA off.
  - `load_probe()`.
  - `phase1_probs()` gives the Phase 1 first-digit softmax with the LoRA on, using the same prompt as `eval_p1.py`.
- `tests/v3/test_llama.py`: 2 tests, red before the code and green after (14 s).
  - With the adapter loaded and the LoRA off, the live L20 state matches the stored `hid3` dev2 state (< 0.05, fp16).
  - `hidden()` forces the LoRA off even after `lora(True)`.
- Sanity check on 5 eval reviews, one model in memory, switching LoRA on and off per review:
  - Phase 1 probs vs stored `p1/probs_R_eval`: max diff 0.0.
  - LLaMA probe probs vs stored `evid7_eval`: max diff 6e-7.
  - The demo can therefore run Phase 1 and Phase 2 from one ~5 GB model load.
- Full suite passes (see the commit). Task 4 complete.

### 2026-10-25 — B1 Router + SC1 escalation self-correction DECLARED (before running)
- **Candidate:** skip the LLaMA probe when the max probability of the Evidence model (E6b) is ≥ tau.
  - Skipped reviews use an Evidence-only LR on log E6b probabilities, fit on dev2 (C=1). This gives the "draft star".
  - Escalated reviews use the full main stack: LR on [log P3, log E6b], fit on dev2. This gives the "revised star".
- **tau:** the smallest of 0.30, 0.35, …, 0.95 whose dev2 5-fold CV routed exact accuracy is within 0.5 points of the full stack. Same folds (StratifiedKFold, shuffle, seed 0).
- **SC1:** on escalated reviews (max E6b < tau), count fixed (draft wrong, revised right) and broken (draft right, revised wrong); exact binomial test. On dev2, draft and revised both come from the same 5-fold CV predictions.
- **Gate (all three must hold):**
  1. Such a tau exists.
  2. natural_dev routed exact (stack fit on dev2) ≥ natural_dev full-stack exact − 0.005.
  3. SC1 on dev2: fixed > broken with binomial p < .05.
- **If PASS:** one readout on eval and natural_test. Report routed vs full-stack exact, McNemar p, share of probe calls saved, and SC1 fixed / broken / net / p.
- **If FAIL:** reported as is; no test read.
- Script: `v2_reflective/e23/gate_B1.py` (code from the plan, Task 5). Probe probabilities: `evid7` (A1 not passed, so P3 stays).

### 2026-10-25 — B1 Router + SC1 RESULT: FAIL (natural_dev check missed by 0.3 pt); no test read
- tau chosen on dev2 CV = **0.60**: routed 0.637 vs full stack 0.642 (within 0.5 pt), Evidence-only on 52% of dev2 reviews.
  - The full curve is in `gate_B1.out`. At tau ≥ 0.70 routed equals the full stack (0.642), with 37% skipped at 0.70.
- **SC1 on dev2 (CV): PASS.** 961 reviews escalated, 326 changed; **fixed 169, broken 85, net +84, binomial p = 1.5e-7**.
- **natural_dev: FAIL.** Routed 0.776 vs full 0.784, required ≥ 0.779. Missed by 0.3 pt (about 2 reviews of 500); 64% skipped.
- **Gate: FAIL** (condition 2). Per the declaration, no readout on eval or natural_test.
- Reading: escalation clearly corrects draft stars on dev2. The chosen tau 0.60 is slightly too aggressive on natural-mix data.
- Any retry must be declared as a new candidate (B1-b) and disclosed as the second Router attempt.

### 2026-10-25 — B1-b Router (second attempt) + SC1 DECLARED (before running); user approved
- **Why a second attempt:** B1 failed only its natural_dev check (−0.3 pt) at tau 0.60. Disclosed as the second Router candidate.
- **What was already seen:** the dev2 CV curve (dev data), and natural_dev at tau 0.60 only. No test data.
- **The one change:** tau = the smallest of 0.30, 0.35, …, 0.95 whose dev2 5-fold CV routed exact is ≥ the full stack's (tolerance 0 instead of 0.5 pt). Same folds (seed 0).
- **Everything else is identical to B1:** draft and revised definitions, SC1 counts, the natural_dev check (routed ≥ full − 0.005), SC1 on dev2 (fixed > broken, p < .05), and on PASS a single readout on eval + natural_test.
- **Multiplicity:** two Router candidates were tried. Both are reported in the paper, and the readout p-values join the Holm family at freeze.
- **Script:** `gate_B1.py` called with `TOL=0 OUT=gate_B1b`. The code is otherwise unchanged; outputs are `gate_B1b.out` and `gate_B1b.json`.

### 2026-10-25 — B1-b Router + SC1 RESULT: PASS; single readout done
- **tau = 0.70** (smallest with dev2 routed ≥ full stack). On dev2 the probe is skipped on 37% of reviews with no accuracy loss.
- **SC1 on dev2 (CV):** 1,258 escalated, 371 changed; **fixed 194, broken 101, net +93, p = 6.7e-8**.
- **natural_dev:** routed 0.784 = full 0.784; 52% skipped.
- **Gate: PASS.**
- **Readout (read once):**

| Split | Full stack | Routed | LLaMA probe skipped | McNemar routed vs full | SC1 escalated | Fixed | Broken | Net | p |
|---|---|---|---|---|---|---|---|---|---|
| eval | 0.675 | 0.674 | 39% | p = 1.0 | 607 | 92 | 50 | +42 | 5.3e-4 |
| natural_test | 0.757 | 0.755 | 48% | p = 0.68 | 1,035 | 152 | 78 | +74 | 1.2e-6 |

- **Reading:** the Router skips the slow LLaMA call on 39–48% of test reviews with no significant accuracy change. On escalated reviews, revising the draft star fixes about 2 stars for each one it breaks, significant on both test sets. This is measured self-correction of the rating (SC1).
- **Disclosure:** this was the second Router attempt; B1 at tau 0.60 failed natural_dev.
- **Deck, notes and paper updated with the B1-b / SC1 results:**
  - New slide 23, "Self-Correction 1: Two Stars Fixed for Each One Broken": readout table, cost saving, and the attempt-1 / attempt-2 disclosure.
  - The "How Phase 2 Answers" rows now show the measured numbers. The Conclusions agentic card reads "first part measured". The schedule row is marked Done.
  - The deck has 25 slides.
  - `Presenter_Notes.md` Part 0 has a new SC1 section, plus updated sections, ranges and timing.
  - `paper/e23_results.tex` has FACT notes for the Router and SC1 (both attempts disclosed). Build OK.

### 2026-10-25 18:29 — A1 scheduled for tonight (user request)
- `v2_reflective/e23/night_A1.sh` is running in the background (nohup, pid 63966). Sequence:
  1. Wait until **22:30**.
  2. Require AC power (wait up to 2 h, otherwise skip and log) and swap under 6 GB.
  3. `caffeinate -is` the resume of `queue_A1.sh`: 35K reviews left (70 chunks), about 6.5 h, done around 05:00.
  4. The declared A1 gate (`gate_A1.py`, dev2 + natural_dev only, no test read) writes `gate_A1.out`.
- Status goes to `stage_A1.txt` and `night_A1.log`. The lid must stay open, because caffeinate does not prevent lid-close sleep.
- No other GPU job may run from 22:30 until it finishes. Tasks 6–7 run before then.
- If the gate passes, the readout on eval and natural_test happens in the next working session, after the result is logged.

### 2026-10-25 — Task 6: evidence pack and Critic
- `src/agentic_sentiment/v3/pack.py`: `build_pack(star, probs, shares, agreed, pos, neg)` returns the star, its confidence, each agent's share, whether the agents agreed, the top 5 phrases for the star and the top 3 against.
- `src/agentic_sentiment/v3/critic.py`:
  - Rule checks:
    1. The stated star equals the final star.
    2. Quoted phrases occur in the review and in the pack.
    3. At least 2 of the top 3 phrases are covered.
    4. Disagreement is mentioned when the agents disagreed.
  - Each failure produces specific feedback text.
  - `critique()` calls the LLM judge (check 5) only when the rules pass. An unparsable judge answer counts as a pass and is reported.
- `tests/v3/test_pack_critic.py`: 5 tests, red before the code and green after.
