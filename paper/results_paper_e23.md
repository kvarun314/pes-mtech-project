# Results draft — Amazon Reviews 2023 (Electronics), leak-free protocol

DRAFT for review (not yet merged into `main.tex`). Every number below traces to `v2_reflective/LOG.md` (entry dates in brackets) and the raw outputs named in brackets. Nothing is estimated. The earlier All_Beauty Run 1 / Run 2 numbers and the Kaggle 70.7% headline are NOT used as main results (see §7 and Appendix A).

## 1. Setup

- **Dataset.** Amazon Reviews 2023 (McAuley-Lab), Electronics only, for all reported evaluation. Frozen splits (`v2_reflective/e23/`, seed 2023, first 2.5M review rows streamed, reservoir sampled, disjoint by review id and exact text): `evid_train` 100,000 balanced (20K per star); `dev` 500 balanced; `eval` 1,000 balanced; `natural_test` 2,000 with the natural skew (1★ 189, 2★ 83, 3★ 150, 4★ 277, 5★ 1,301). [LOG 2026-10-11]
- **Phase 1 (Analyst).** LLaMA-3-8B, 4-bit base, LoRA trained with the paper's Table 1 hyperparameters (MLX QLoRA; 2,000 balanced raw Electronics rows; 400 updates ≈ 3 epochs at effective batch 12; zero-shot training-format prompt). Prediction = argmax of the first-digit token probabilities over 1–5. [LOG 2026-10-13]
- **Phase 2 agents.** Analyst (Phase 1 probabilities) + Evidence agent (TF-IDF 1–2-gram, 300K features, logistic regression, trained on the 100K balanced raw Electronics rows) → Reflector (logistic regression stacker on the log-probabilities of both agents, fit on the 500 dev rows; a second config adds log review length). Run once on `eval` and `natural_test`; two configs pre-declared. [`e23/phase2_e23.py`, `e23/phase2_e23.out`]
- **Statistics.** Paired bootstrap 95% CI on the exact-match difference (3,000 resamples) and exact binomial McNemar test on fixed vs regressed rows.
- **Transfer note.** The original paper's adapter was trained on Kaggle Fine Food reviews. Here Phase 1 is retrained on Electronics, so no Kaggle-trained component is used in the main results.

## 2. Main results

| Split | Analyst (Phase 1, arm R) | Evidence alone | Stack (Phase 2) |
|---|---|---|---|
| `eval`, 1,000 balanced | 54.0% exact | 62.4% | **63.9%** (within-1 95.5, MAE .411, macro-F1 .638) |
| `natural_test`, 2,000 natural | 53.4% exact | 69.3% | **73.4%** (within-1 96.2, MAE .320, macro-F1 .577) |

Paired comparisons for the stack (A+E config):

| Split | vs Analyst | vs Evidence alone |
|---|---|---|
| `eval` | +9.9 pts, CI [+6.7, +13.0], fixed 183 / regressed 84, McNemar p = 1.3e-9 | +1.5 pts, CI [−1.2, +4.1], fixed 97 / regressed 82, p = 0.30 (**not significant**) |
| `natural_test` | +20.0 pts, CI [+17.8, +22.2], fixed 490 / regressed 91, p = 4.6e-67 | +4.0 pts, CI [+2.4, +5.8], fixed 197 / regressed 116, p = 5.5e-6 |

The A+E+len config is statistically indistinguishable (63.5% / 73.2%). The two agents disagree on 44% of rows on both splits. On disagreement rows the Analyst is right 32% (`eval`) / 25% (`natural_test`), the Evidence agent 51% / 61%, and the stack 55% / 67%.

## 3. Honest reading

1. A 100K-row bag-of-words model beats the 2K-row LoRA-tuned LLaMA on exact five-class accuracy (62.4 vs 54.0 on `eval`). Most of the Phase 2 gain over Phase 1 therefore comes from the Evidence agent, not from LLaMA.
2. LLaMA's marginal contribution on top of the Evidence agent is small: +1.5 on balanced `eval` (not significant) and +4.0 on `natural_test` (significant).
3. The Reflector is fit on balanced dev rows. On the skewed `natural_test` part of the gain is therefore a class-prior effect (5★ recall .77 → .81); the macro-F1 of .577 is the fairer headline there.
4. The choice of Phase 1 arm (R, raw balanced, over P, the paper's preprocessing pipeline) was made on dev only (55.2 vs 52.2, margin about ±4.4), so it is a direction, not a significant result.

## 4. Preprocessing ablation (Evidence model; validation = 10K held-out raw balanced Electronics rows, ±1.0 pt at 95%) [LOG 2026-10-11, `e23/ablate_preproc.log`]

| Variant | Training rows | Accuracy |
|---|---|---|
| A raw | 90K | 51.2 |
| B TextBlob DQC | 68K | 47.7 |
| C random | 10K | 47.6 |
| D DQC + VGST | 10K | 44.5 |
| E DQC + random | 10K | 44.3 |
| F DQC then re-balance | 33K | 47.6 |
| G raw balanced | 33K | 50.1 |

TextBlob DQC hurts (−2.5 at equal size, −3.5 at full size): it deletes 52% of 1★ and 67% of 2★ reviews (negative reviews contain positive words) and drops mixed reviews. VGST adds nothing (D vs E +0.2, within noise) and selects very long reviews (156 vs 64 words). The Phase 2 Evidence agent therefore trains on raw balanced data; TextBlob and VGST are reported as ablations only, and test sets are never filtered. The same direction appears for the LLaMA adapter on dev (arm P 52.2% vs arm R 55.2%, not significant).

## 5. Phase 1 label leak (retraction of earlier headline numbers) [LOG 2026-10-11]

The earlier one-shot evaluation chose the example's rating to differ from the true rating, which leaks about 8 points (300 balanced rows, paired bootstrap): A (leaky protocol) 59.7%; B (random shot) 52.0%, −7.7 [−13.3, −2.0]; D (zero-shot) 58.3%, −1.3 [−7.0, +4.3]. All main results here use the zero-shot prompt. A post-hoc recalibrator that gave +5.0 on leaky probabilities gave +0.1 [−2.3, +2.5] on clean probabilities and is withdrawn: recalibration adds no information; a real gain needs an independent second signal, which is why the Evidence agent is used.

## 6. Hallucination study (Phase 1 CoT generations, 150 samples, Kaggle adapter) [LOG 2026-10-11]

0 unparseable; 41% wrong star; 16 severe misses (≥ 2 stars); 7 outputs run on into an invented "Review:/Example:" turn; the rationale is a canned template tied to the digit rather than evidence from the review (description never mismatches the digit, 0/150, because it is a fixed template per star). This study used the Kaggle adapter and is reported as an appendix-level diagnostic unless repeated on Electronics.

## 7. Items that must not be reported as main results

- All_Beauty Run 1 (16.1% → 38.7%) and Run 2 (Electronics, 50.95% → 46.40% before the dissonance fixes): obtained with the pre-fix 4-agent graph and a non-Electronics-trained Phase 1; keep only as a historical note.
- Visual Verifier: excluded (constant output with 3 vision models); no image signal is claimed.

## Appendix A — Kaggle reproduction (original paper protocol) [LOG 2026-10-11]

4,000 natural Kaggle rows (seed 2024), committed Kaggle adapter, zero-shot: 69.10% exact (always-5 baseline 63.78%), polarity 85.88% (always-positive 78.60%), binary 90.31%, within-1 93.45%, MAE .383, macro-F1 .549, weighted-F1 .709 (was 70.7% / 85.0% with the leaky one-shot). Kaggle stack (balanced 1,000 eval): LLaMA 59.2 | evidence 65.5 | stack 67.6 (+8.4 vs LLaMA, CI +5.3..+11.6; vs evidence +2.1, CI −0.3..+4.5, not significant).

## Open items before merging into `main.tex`

- Per-class recall table and confusion matrix figures for the Electronics stack (`e23/phase2_trace_{eval,natural_test}.json` has the predictions).
- Whether to repeat the hallucination study on the Electronics adapter.
- Which paper variant gets this: the thesis `paper/main.tex` (currently Run 1 narrative) or the conference paper `paper/conference/ConferencePaper.tex`.
