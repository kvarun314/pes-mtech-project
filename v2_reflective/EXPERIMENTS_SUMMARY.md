# Experiments summary: everything tried, what worked, what is next

Last updated: 2026-10-24. Covers the whole project history.
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

Main system **SH** has two agents:
- **P3**: a logistic probe on the frozen base LLaMA-3-8B, layer-20 hidden state, fit on 40K rows.
- **E6b**: a TF-IDF + logistic regression Evidence model trained on about 497K rows.

An LR Reflector, fit on dev2, combines the two agents' log-probabilities. All data is Amazon Reviews 2023 Electronics.

Results:
- **Single readout:** eval 67.5% exact (macro-F1 .675); natural_test 75.7% (.612).
- **5 seeds:** eval 65.8 ± 0.6, natural_test 75.4 ± 0.8.
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
| W11 | **Router with escalation (SC1, B1-b, tau .70)** | eval 92 fixed vs 50 broken (p 5e-4), natural_test 152 vs 78 (p 1e-6); LLaMA skipped on 39–48%, accuracy unchanged | Phase 2 v3 agentic layer |
| W12 | **Explainer + Critic loop (SC3, B2)** on all 3,000 test reviews | Faithful 60.5 → 88.9% (eval), 63.4 → 88.8% (natural_test); 793 fixed, 0 broken; McNemar p 6e-86 / 1e-153 | Phase 2 v3 agentic layer |
| W13 | **Cited phrases carry the decision (deletion test)** | Comprehensiveness .158 vs .089 / .212 vs .124; sufficiency .556 vs .308 / .641 vs .353 (cited vs random, p < 1e-46) | Explanations |
| W14 | **Hand-off agent (selective prediction, S1)** | natural_test 93.1% exact at 55.7% coverage; eval 85.0% at 42% | Phase 2 v3 agentic layer |
| W15 | **Two more Critic rounds (I1, max 2 → 4 rewrites)** | Faithful 88.9 → 90.6% (eval), 88.8 → 91.1% (natural_test); 17 + 46 more fixed, 0 broken; p 1.5e-5 / 2.8e-14; +0.34 LLM calls per review | Phase 2 v3 agentic layer (dev2 gate 19/62, PASS) |
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
| Phase 1 vs Phase 2 explanations (X1, 150 eval) | Grounded in the review 0% vs 79% (p 6e-36); distinct outputs 3% vs 100%; judge pass 85% vs 94% (p .011) |
| Hard cases (H1) | Negative-sounding positives: stack +27.5 / +28.0 over Phase 1 (p .007 / 9e-11); ironic-sounding negatives +6.9 (ns) / +14.7 (p .04); mixed reviews +7.1 (p .04) / −0.7 (ns); sarcasm cues too rare (n 6 / 1) |

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
