# Experiments summary: everything tried, what worked, what is next

Last updated: 2026-11-12 (SH3 adopted). Covers the whole project history.
- Detailed record: `LOG.md` (dated entries; each rule was declared before its run).
- Frozen numbers: `e23/results_frozen.md`.
- Outputs: `e23/*.out`.

Terms used below:
- **Exact**: 5-class star accuracy.
- **eval**: 1,000 balanced rows.
- **natural_test**: 2,000 rows with the natural star mix (65% are 5-star).
- **dev**: 500 balanced rows.
- **dev2**: 2,000 balanced rows.
- **natural_dev**: 500 rows with the natural star mix.
- **CI**: 95% bootstrap confidence interval.
- **Gate**: a pass/fail rule written before the run and checked on dev data only.

---

## 1. Current state (one paragraph)

Main system **SH3** (adopted 2026-11-10, results 2026-11-12) has three agents:
- **P3**: a logistic probe on the frozen base LLaMA-3-8B, layer-20 hidden state, fit on 40K rows.
- **E6b**: a TF-IDF + logistic regression Evidence model trained on about 497K rows.
- **DeBERTa-v3-base** (seed 0), fine-tuned on 100K balanced rows (Colab, D1).

An LR Reflector, fit on dev2, combines the three agents' log-probabilities. All data is Amazon Reviews 2023 Electronics.

Results:
- **Single readout:** eval 67.5% exact (macro-F1 .675); natural_test 78.0% (.634). MAE .353 / .253.
- **Previous main SH (P3 + E6b):** 67.5 / 75.7. SH3 vs SH on natural_test: +2.3, Holm-adjusted p .0043. On eval: no change.
- **5 seeds (SH, two agents):** eval 65.8 ± 0.6, natural_test 75.4 ± 0.8. DeBERTa seeds (D2): gain over SH +0.1 ± 0.3 eval, +2.2 ± 0.2 natural_test.
- **Gain over each single agent:** positive in 5/5 seeds on both splits.
- **Caveat:** the old LoRA + Evidence stack, refit on the same dev2, is only 2.0 / 1.5 points behind. That gap is not significant.

Numbers freeze on Nov 1. The paper is due Nov 21–22.

---

## 2. What worked (adopted)

| # | Change | Evidence | Where used |
|---|---|---|---|
| W1 | **Leak-free zero-shot prompt.** The old one-shot example was picked with rating ≠ truth, which leaks about 8 points. | B vs A −7.7 [−13.3, −2.0]; zero-shot D −1.3 (ns) | All results since Oct 11 |
| W2 | **Single dataset: Electronics 2023**, frozen disjoint splits (seed 2023) | User decision Oct 11 | All results |
| W3 | **Raw balanced training data**, no TextBlob DQC and no VGST | DQC −2.5 to −3.5 pts; VGST +0.2 (noise); LoRA arm R 55.2 vs arm P 52.2 on dev | Evidence model and LoRA |
| W4 | **Second independent signal**: TF-IDF Evidence model stacked with LLaMA | First stack 63.9 / 73.4 vs Analyst 54.0 / 53.4 | Phase 2 core idea |
| W5 | **LR Reflector** on agents' log-probabilities (replaces the dissonance Critic, which fired on 0.4% of rows) | Stack beats each agent; absorbs LoRA seed bias | Phase 2 |
| W6 | **Bigger Reflector fit set (dev2, 2,000 rows)**. Halves the dev noise (±4.3 down to ±2.1). | Old stack 63.9 → 65.5 eval when refit on dev2 | SH |
| W7 | **Frozen LLaMA hidden-state probe on 40K rows (P3)** replaces the LoRA Analyst | Phase H gate PASS: dev2 +3.9 [+2.4, +5.8] | SH |
| W8 | **Evidence on ~497K rows (E6b)** inside SH | Alone +1.9 dev CV; best stack partner for P3 | SH |
| W9 | **Polarity / binary reporting** | Stack vs Evidence polarity +2.9 to +4.5, significant on both splits and both seeds (metric added post hoc, disclosed) | Paper metrics |
| W11 | **Router with escalation (SC1, B1-b, tau .70)** | Two-agent SH: eval 92 fixed vs 50 broken (p 5e-4), natural_test 152 vs 78 (p 1e-6); LLaMA skipped on 39–48%, accuracy unchanged. **With SH3 the gate fails (natural_dev .798 vs .804; SC1 p .20), so the Router is off** | Two-agent SH only |
| W12 | **Explainer + Critic loop (SC3, B2)** on all 3,000 test reviews | SH3, max 4 rewrites: faithful 62.3 → 90.5% (eval), 63.65 → 91.6% (natural_test); 282 + 559 fixed, 0 broken; McNemar p 2.6e-85 / 1.1e-168; 2.93 / 2.96 LLM calls per review | Phase 2 v3 agentic layer |
| W13 | **Cited phrases carry the decision (deletion test)** | SH3: comprehensiveness .156 vs .077 / .209 vs .119; sufficiency .521 vs .301 / .616 vs .349 (cited vs random, Wilcoxon p < 1e-54) | Explanations |
| W14 | **Hand-off agent (selective prediction, S1)** | SH3: natural_test 92.8% exact at 56.1% coverage; eval 86.9% at 41.1% | Phase 2 v3 agentic layer |
| W15 | **Two more Critic rounds (I1, max 2 → 4 rewrites)**; default in the SH3 rerun | Faithful 88.9 → 90.6% (eval), 88.8 → 91.1% (natural_test); 17 + 46 more fixed, 0 broken; p 1.5e-5 / 2.8e-14; +0.34 LLM calls per review | Phase 2 v3 agentic layer (dev2 gate 19/62, PASS) |
| W10 | **Phase 2 audit fixes (Run 2, Oct 9)**: grounding floor removed, ties go to the Analyst, RAG prompt fixed, Visual Verifier excluded | Run 2 −91 net becomes +2 net on 96 rows | Historical pipeline |

## 3. What failed or was not adopted

| # | Idea | Result | Verdict | Lesson |
|---|---|---|---|---|
| F1 | LLM adjudicator, Kaggle (Gate 0, Oct 11) | 51.9–57.5% vs Phase 1 55.6–67.2% on the same pairs | FAIL | The LLM cannot judge between two candidate stars |
| F2 | LLM adjudicator with Evidence n-gram citations, Electronics (Gate 0, Oct 20) | 54.2% vs stack 73.3% on 434 disagreement rows | FAIL | Same; replicated |
| F3 | Recalibration of Phase 1 probabilities (+5.0 on leaky probs) | +0.1 [−2.3, +2.5] on clean probs | RETRACTED | Recalibration adds no information |
| F4 | Phase 1 local answer-only retrain on Kaggle | 55.3% vs 59.7% committed adapter | Worse | |
| F5 | Extra Reflector features (TextBlob, margin, entropy, length) | ≤ +1 pt, within noise | Dropped | |
| F6 | Natural-prior Reflectors C1–C3 | Raise natural exact but cost macro-F1 and balanced eval; prior-adjusted Evidence alone gets 76.3 | Not headline | Natural exact is dominated by the class prior |
| F7 | Disagreement routing (C4) | +0.8 ns on natural; fewer regressions | Optional, not adopted | |
| F8 | MiniLM embedding agent, char n-gram agent (Phase B) | Weaker than TF-IDF; stacks −0.9 to −2.1 on dev | FAIL | More views of the same text add noise |
| F9 | LoRA on 10K rows (Phase C, R10) | dev 56.4 < gate 58 | FAIL | LoRA is not data-starved at 2K rows |
| F10 | Hidden-state probe on 10K rows (Phase F) | dev +.011, natural_dev .803 vs .804 | FAIL by .001 | |
| F11 | Multi-layer probe (Phase F2) | dev +.005 | FAIL | Fixed later by more fit data (W7) |
| F12 | Pooled-dev Reflector + ordinal decoding (Phase D) | Gate passed, but gain = more 5-star predictions | NOT adopted | Gate on macro-F1, not natural exact |
| F13 | Evidence on 500K rows in the old stack (Phase G2) | Stack .632 vs .640 | FAIL in stack | Stronger Evidence overlaps the LoRA Analyst |
| F14 | Visual Verifier: BLIP, moondream2, Qwen2-VL | Constant output (always 3, then always 5) | Excluded permanently | A product photo does not show satisfaction |
| F15 | Claim-splitting for RAG grounding | Mean grounding .144 vs .340 whole-review | Reverted | |
| F16 | Adding LoRA back to SH (A + E6b + P3) | dev2 .633 vs .639 | Not adopted | |
| F17 | SC2: Instruct LLM re-rates uncertain reviews (Oct 25) | dev2 fixed 76 vs broken 89, p = .35; natural_dev net −6 | FAIL | Third LLM-judge failure; LLMs explain, not decide |
| F18 | A1: LLaMA probe on 120K rows (P3b, Oct 25 night) | dev2 CV +.003 [−.011, +.009] vs SH; natural_dev .827 vs .818 | FAIL | Probe saturated at 40K; SH unchanged |
| F19 | I2: learned Hand-off (8 features) vs confidence only (Oct 28) | dev2 AUROC .701 vs .706; accuracy at 50% coverage 78.2 vs 78.2 | FAIL | The Reflector's confidence already carries the extra signals |
| F20 | I3: fixed-form fallback explanation after 4 rewrites (Oct 28) | dev2 27 of 43 pass the Critic (gate 30) | FAIL | All 16 failures are the LLM judge rejecting a sentence that only quotes the review |
| F21 | Q2: later-time holdout (Oct 28) | Not run: every agent was fit on reviews from 1999-2023 | Dropped | Year bands (Q5) are the time check |
| F22 | Q3 / Q3b: Qwen2.5-7B-Instruct as a second judge (Oct 28) | dev2: accepts 0 of 38 Llama-rejected, rejects 60 of 60 Llama-accepted, under both prompts | FAIL | Constant verdict; flags the stated star and agent shares as unsupported |
| F23 | G1: gradient-boosted Reflector (Nov 2) | dev2 .6205 vs .642; natural_dev .756 vs .784 | FAIL | A nonlinear combiner overfits 2,000 fit rows; LR stays |
| F24 | N1: NB-SVM text agent, 497K rows (Nov 2) | Replace Evidence: dev2 +0.5 [−0.8, +1.8]; natural_dev .774 vs .784 | FAIL | Another view of the same words adds nothing, as in F8 |

## 4. Baselines and robustness (reported, not selection steps)

| Item | Result |
|---|---|
| Base LLaMA, no LoRA (zero-shot digit) | eval 34.3, natural 16.8; LoRA adds +19.7 / +36.7 |
| Base LLaMA + Evidence stack | 63.7 / 72.7, about equal to LoRA + Evidence (63.9 / 73.4) |
| DistilBERT, 2K rows | eval 52.8 vs LoRA 54.0 (ns); adds nothing to Evidence |
| Variance of first stack (5 seeds) | Stack − Evidence +2.7 ± 1.3 eval, +4.3 ± 1.1 natural; 5/5 positive |
| LoRA seed 2 | eval same (53.7 vs 54.0); natural single-model unstable (74.8 vs 53.4, prior artifact); stack stable |
| Hallucination study (150 generations) | LoRA rationales are a canned template per digit; probabilities are the usable signal |
| Kaggle paper-protocol re-score (appendix) | 69.1% exact leak-free (was 70.7% leaky) |
| Evidence learning curve (Phase G) | 11K .535 → 90K .568: still rising, which led to G2/E6b |
| SH variance (5 seeds, E6b fixed) | SH − E6b +3.3 ± 0.6 / +5.2 ± 0.8; SH − P3 +5.2 ± 0.9 / +2.4 ± 1.2; SH − old stack on dev2 +0.7 ± 1.2 / +1.0 ± 0.6 |
| Phase 1 vs Phase 2 explanations (X1, 150 eval) | SH3: grounded in the review 0% vs 75% (113 vs 0, p 1.9e-34); distinct outputs 100%; judge pass 17 vs 5 Phase 2-only (p .017) |
| Hard cases (H1) | SH3: negative-sounding positives +32.5 / +35.2 over Phase 1 (p .0023 / 5e-16); positive-sounding negatives +11.1 / +10.3 (ns); mixed reviews +10.0 (p .003) / +3.2 (ns); sarcasm cues too rare (n 6 / 1) |
| Star ranges (Q1, conformal, 90% target) | SH3: coverage 91.6 / 90.05%; mean range 1.77 / 1.44 stars vs Phase 1 2.40 / 2.08; single star 34% / 58% vs 5% / 11%; with ranges, a person is needed for 11% / 2% of reviews |
| Text changes (Q4, 300 eval) | SH3 star flips: flat text 9.3% vs Phase 1 29.7% (p 3e-11); typos 0.7% vs 2.0% (ns); **no title 15.7% vs 0%** (two-agent SH 8.0%; Evidence and DeBERTa both read the title) |
| Year and product popularity (Q5) | SH3 gain over Phase 1 positive in every band, all CIs > 0: eval +10.3 to +16.3, natural +21.5 to +26.5 |
| Ordinal metrics (O1) | SH3 quadratic weighted kappa .898 / .902 vs Phase 1 .824 / .779 (CIs of the difference exclude 0); MAE .353 / .253 vs .529 / .539 |
| DeBERTa-v3 third agent (D1, 100K, Colab) | Gate PASS: dev2 +2.0 [+0.6, +3.35], natural_dev +2.0. Readout: eval 67.5 → 67.5 (58 fixed / 58 broken), natural_test 75.7 → 78.0, +2.3 [+0.95, +3.75], p 8.6e-4. Alone 66.5 / 75.75. **Adopted 2026-11-10 (SH3)** |
| DeBERTa seed variance (D2, seeds 0, 1, 2; reporting only) | Gain over the main stack, mean ± SD: dev2 +2.4 ± 0.6 (stable), natural_dev +1.0 ± 0.9 (not stable), eval +0.1 ± 0.3 (not stable), natural_test +2.2 ± 0.2 (stable). Alone: eval 66.7 ± 1.0, natural_test 76.3 ± 1.0. Selects nothing. `e23/d2_seeds.out` |

## 5. Historical results (superseded, kept for the record)

- **Run 1** (All Beauty, 2,000 rows, Kaggle adapter, leaky prompt): Phase 1 16.1% vs Phase 2 38.7%. Superseded: different dataset, leaky prompt, transfer setting.
- **Run 2** (Electronics, 2,000 rows): Phase 2 regressed −4.55 pts because of the grounding-floor bug (fixed, W10).
- **Kaggle headline** 60.1% / 70.7%: withdrawn (leak, W1).

## 6. Integrity notes (disclose in the paper)

1. Five candidate agent sets were compared on dev2 for Phase H. The winner was chosen by a rule fixed beforehand.
2. E6b-alone accuracy on eval and natural_test was printed once before the Phase H gate. It was not used for any choice.
3. Polarity / binary metrics were added after the 5-class gains looked small (post hoc, but also reported by the original paper).
4. All p-values are uncorrected.
5. In the SH variance run, E6b was not retrained per seed (too slow).
6. The natural_test gain of stackers fit on balanced dev partly reflects a class-prior effect. Report macro-F1 beside exact.

## 7. What we will do next

| When | Task | Status |
|---|---|---|
| Oct 25 | Present the deck (`PES_Project_Phase1/Agentic_LLaMA3_Presentation.pptx`, 30 slides) | ready |
| Oct 25–31 | Off-disk backup to Google Drive (user installs Drive for desktop) | waiting on user |
| by Nov 1 | **Numbers freeze.** Rerun `e23/freeze_tables.py`; lock the paper angle | tables done Oct 24 |
| Nov 2–7 | Paper prose, one section per day. The author writes; facts are in `paper/e23_results.tex` (`% FACT` comments) and `PAPER_OUTLINE.md` | todo |
| Nov 8 | Draft 1 to supervisor + similarity check #1 | todo |
| Nov 9–14 | Supervisor fixes, thesis sections, archive, docs | todo |
| Nov 15 | Draft 2 + similarity check #2 | todo |
| Nov 16–20 | Buffer, final checks, submit by Nov 20 | todo |

**Optional experiments.** Only if time allows before Nov 1. Each needs a gate declared first, and none is needed for the paper.
- ~~Retrain E6b inside the variance seeds~~: done as C1 (Oct 26): SH 66.1 ± 0.6 / 75.4 ± 0.7, same as with E6b fixed. Disclosure 5 closed.
- ~~Probe on more rows~~: done as A1 (120K), failed (F18).
- Disagreement routing on top of SH. Earlier it gave +0.8, not significant.

**Not planned (closed lines):** LLM adjudicator, more LoRA data, vision agent, extra classical text views, natural-prior Reflectors.
