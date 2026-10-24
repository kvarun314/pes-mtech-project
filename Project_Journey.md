# Project Journey: From Phase 1 to Now, and Why We Are Heading Here

Companion to `Presenter_Notes.md`.
- **What this file is:** the full story of the project. What we built, what we measured, what failed, what worked, and why each result pushed us in the next direction.
- **How to use it:** read it once start to finish before presenting. Use sections 9 and 10 to answer "why this direction?". Section 11 handles the tricky questions.
- **Where the numbers come from:** every number is traceable to `v2_reflective/LOG.md` (dated entries) or `v2_reflective/e23/*.out`. A shorter ledger is in `v2_reflective/EXPERIMENTS_SUMMARY.md`.

Terms used throughout:
- **Exact:** 5-class star accuracy (predicted star equals the true star).
- **Balanced eval:** 1,000 reviews, 200 per star. Guessing gives 20%.
- **natural_test:** 2,000 reviews with the real-world star mix. About 65% are 5-star, so always guessing 5 gives about 65%. That is why macro-F1 is reported next to exact accuracy.
- **dev / dev2:** 500 and 2,000 balanced reviews, used only for choosing models. The test sets are read only after a choice is fixed.
- **Gate:** a pass/fail rule written in the log before an experiment runs, checked on dev data only.
- **Stack / Reflector:** a small logistic-regression model that combines the star probabilities from several agents into one final star.

---

## 1. The story in one paragraph

We set out to reproduce a paper that fine-tuned LLaMA-3-8B for review sentiment, then to beat it with an agentic, multimodal Phase 2: LLM agents that check each other and self-correct. Early results looked strong (+22.6 points). Careful auditing then showed that much of that gain, and later a +5.0 gain, came from bugs, a different dataset, and a label leak in the prompt.

After fixing those problems, we rebuilt the evaluation from scratch on one clean dataset (Amazon Reviews 2023 Electronics) and tested each agentic component on its own. Three components failed to help accuracy: the LLM agents that judge, verify images, or retrieve, and the self-correction loop. What did help was combining **independent signals**: a classical text model plus LLaMA. The best version reads LLaMA's internal hidden state instead of its generated answer.

That system, SH, reaches 67.5% on balanced eval and 75.7% on natural_test, up from 54.0% and 53.4% for the Phase 1 model. The next step puts the agentic part back where it can add measurable value. LLM agents will route work, explain decisions, and correct their own explanations, while the specialist agents keep deciding the star.

---

## 2. Starting point: the paper and the original plan

**The paper being reproduced** (Wang et al. 2025, `itmconf_dai2024_04021.pdf`):
- LLaMA-3-8B fine-tuned with LoRA on Amazon product reviews.
- Claims 92.6% accuracy.
- Uses TextBlob filtering (DQC), stratified sampling, VGST diversity sampling, neutral-class oversampling, and a one-shot + chain-of-thought prompt.

**Our original plan:**
- **Phase 1:** reproduce that baseline.
- **Phase 2:** add an agentic, multimodal layer, a LangGraph pipeline of four agents:
  - **Analyst:** the Phase 1 LLaMA model writes a rating.
  - **Visual Verifier:** looks at product images.
  - **RAG Prover:** retrieves similar reviews as grounding evidence.
  - **Critic:** measures disagreement between agents ("dissonance") and sends the case back for re-analysis when they disagree.

The research question was whether evidence from extra sources, plus agents checking each other, improves 5-star classification over a single model.

---

## 3. Stage 1: Run 1 looked like a big win (Oct 1–8)

**What we did:** ran Phase 1 and Phase 2 on 2,000 Amazon 2023 "All Beauty" reviews on Colab.

**Result:** Phase 1 16.1% exact, Phase 2 38.7% (+22.6 points). Self-correction fired on 46% of reviews; 491 errors fixed, 39 made worse.

**Why it was not the real answer:**
1. **Transfer setting.** The Phase 1 adapter was trained on a different dataset (Kaggle). Phase 1 collapsed under the dataset shift: it predicted 1 or 3 stars for almost everything, with 4-star recall at 0%. Phase 2 looked good mainly because Phase 1 was broken on this data, not because the agents were strong.
2. **Leaky prompt.** The prompt contained a label leak, found later in Stage 3.
3. **Wrong category.** All Beauty is not the category we later standardised on.

**What we kept:** the pipeline code and the idea. The numbers became historical slides only.

---

## 4. Stage 2: Run 2 got worse, and the audit found real bugs (Oct 9)

**What we did:** reran on 2,000 Electronics reviews.

**Result:** Phase 2 was **worse** than Phase 1: 46.4% vs 50.95% (−4.55). Every agent after the Analyst hurt.

**The audit:** we replayed the saved outputs locally (no GPU cost) and then ran real local inference. Four defects turned up:
1. **Over-firing Critic.** A low retrieval score alone forced re-analysis, even when the agents agreed. It fired on 79% of reviews, and the net effect was 110 fixed vs 201 broken.
2. **Bad tie-break.** The weaker RAG vote overruled the stronger Analyst about half the time.
3. **RAG prompt missing.** The RAG agent lacked the prompt framing the Analyst had. Fixing it raised RAG from 37.5% to 46.9% alone.
4. **Visual Verifier was dead.** It output "3 stars" for 96 of 96 reviews. We tried three vision approaches (BLIP captions, moondream2, Qwen2-VL). One always said 3, another always said 5.
   - The reason is fundamental: a product photo shows the product, not whether the owner was happy. Dissatisfaction ("broke after a week", "poor support") is invisible in a picture.

After the fixes, on 96 reviews, Phase 2 was +2 net over Phase 1. That is honest, but tiny.

**Also learned:** classical baselines (SVM, Naive Bayes) on a fresh Electronics sample were weaker than expected (48%). That suggested the paper's 65–70% "target band" may not apply to Electronics at all.

**Lesson:** "the agentic system improved by X" needs an audit before it is believed.

---

## 5. Stage 3: Kaggle reproduction, the leak, and the retraction (Oct 10–11)

We went back to Phase 1 on the paper's own Kaggle data to understand the baseline properly.

**What we found:**
- **The paper's 92.6% is very likely not 5-star exact accuracy.**
  - The paper's own definitions and baseline numbers match polarity accuracy (negative / neutral / positive) on a natural, skewed test set.
  - Its Table 3 even contains an impossible row: F1 higher than both precision and recall.
  - On the paper's protocol (4,000 natural reviews) our model got 70.7% exact and 85.0% polarity. The classical baselines matched the paper's numbers only when read as polarity.
- **Learning curve:** a simple TF-IDF + logistic regression model keeps improving with more data. At 128K reviews it reaches 90.4% polarity, near the paper's 92.6%. The task is data-hungry.
- **The Critic almost never fired:** 0.4% of reviews, because it needed two agents to disagree with the Analyst in the same direction. Structurally dead.
- **Gate 0 #1 (LLM judge) failed.** We trained LLaMA to pick between two candidate stars when the agents disagreed. It chose correctly 51.9–57.5% of the time, worse than just trusting Phase 1 (55.6–67.2%).
- **A +5.0 "win", then a retraction.** A recalibration model on top of Phase 1's probabilities gave +5.0 points, which looked like a real Phase 2 gain. Then we found **the label leak**:
  - The one-shot example in the prompt was always chosen so that its star was different from the true answer.
  - The model learned to avoid the example's star. That removes one wrong option for free and is worth about 8 points (59.7% leaky vs 52.0% with a random example).
  - Zero-shot, with no example, gave 58.3% with no leak.
  - With clean, leak-free probabilities the recalibration gain became **+0.1** (not significant). **We withdrew the +5.0**, along with the 70.7% / 60.1% headlines.
- **The idea that saved Phase 2.** Recalibration cannot add information; it only reshuffles what one model already knows. A real gain needs a **second, independent signal**. We stacked LLaMA with a TF-IDF model trained on 100K reviews. The stack beat LLaMA by +8.4 points, but the TF-IDF model alone was already close to the stack.

**Decision (user, Oct 11):** all main results move to **one clean dataset, Amazon Reviews 2023 Electronics**. Kaggle becomes reproduction and appendix only.

**Lessons:**
- Check prompts for leaks.
- Calibration is not new information.
- Combining independent signals is where gains come from.

---

## 6. Stage 4: Clean restart on Electronics (Oct 11–16)

**Frozen, disjoint splits** (seed 2023), disjoint by review ID and by exact text:
- 100K balanced training reviews for the Evidence model
- dev (500)
- eval (1,000 balanced)
- natural_test (2,000)

**Preprocessing ablation:** the paper's pipeline turned out to be harmful.
- **TextBlob filtering** lowered accuracy by 2.5–3.5 points. It deletes 52% of 1-star and 67% of 2-star reviews, because angry reviews still contain positive words ("I wanted to love it").
- **VGST** added nothing (+0.2, noise) and prefers very long reviews.
- **Decision:** train on raw, balanced data. Report the paper's steps as ablations. Never filter test sets.

**Phase 1 retrained on Electronics:** paper hyperparameters, 2,000 reviews, local MLX.
- Paper pipeline (arm P): 52.2% on dev.
- Raw balanced (arm R): 55.2%. Chose R.

**First Phase 2 stack:** Analyst (LoRA LLaMA) + Evidence (TF-IDF, 100K) + Reflector, read once.
| | Analyst | Evidence | Stack |
|---|---|---|---|
| eval | 54.0 | 62.4 | **63.9** |
| natural_test | 53.4 | 69.3 | **73.4** |

**Honest reading:**
- The 100K-review TF-IDF model beats the 2K-review LLaMA alone.
- The stack's gain over Evidence was small on balanced eval (+1.5, not significant) and larger on natural_test (+4.0, significant). Part of that natural_test gain comes from class balance.

---

## 7. Stage 5: Hardening the claim (Oct 17–21)

Before trusting the stack, we stress-tested it.
| Test | Result | What it told us |
|---|---|---|
| Hallucination study (150 generations) | LoRA explanations are a fixed template per star; no review-specific reasoning | Use LLaMA's probabilities, not its text |
| Natural-prior Reflectors | Big natural_test exact gains, but a prior-adjusted Evidence model alone got 76.3; they hurt balanced eval and minority stars | Natural-test exact accuracy rewards predicting 5 stars. Report macro-F1 |
| Base LLaMA, no fine-tuning | 34.3% alone, but stacked with Evidence 63.7 ≈ LoRA stack 63.9 | The fine-tune matters alone but adds little inside the stack |
| Gate 0 #2: LLM judge with evidence phrases | 54.2% vs stack 73.3% on 434 disagreements | LLMs cannot pick the star. Second failure, now on Electronics |
| Variance, 5 seeds | Stack minus Evidence +2.7 ± 1.3 eval, +4.3 ± 1.1 natural; positive 5/5 | The gain is real in sign, modest in size |
| DistilBERT, same 2K reviews | 52.8 vs LoRA 54.0 (not significant) | A model 100× smaller matches the LoRA at this data size |
| LoRA seed 2 | Balanced eval same; natural_test alone 74.8 vs 53.4 | Single-model natural numbers are unstable; the stack absorbs that instability |
| Polarity paired CIs | Stack beats Evidence by +2.9 to +4.5 on polarity, both splits and both seeds, significant | Clearest positive result so far (metric chosen after seeing results; disclosed) |
| Extra text agents (embeddings, character n-grams) | Weaker than TF-IDF; stacks got worse | More views of the same text add noise. The helpful second signal is the LLM |

---

## 8. Stage 6: The improvement push (Oct 21–24)

Every candidate had a gate written before it ran. Failures never touched the test sets.
| Phase | Idea | Result | Verdict |
|---|---|---|---|
| C | LoRA on 10K reviews (5× more) | dev 56.4 vs gate 58 | **Fail.** LoRA is not data-starved |
| F | Logistic probe on LLaMA's layer-20 hidden state, 10K fit | dev +1.1, natural_dev missed by 0.1 | **Fail** (by a hair) |
| D | Reflector fit on pooled dev + natural_dev | Gate passed, but the gain was just predicting more 5-stars | **Not adopted.** Lesson: gate on macro-F1 |
| F2 | Multi-layer probe | dev +0.5 | **Fail** |
| G / G2 | Evidence model on 500K reviews | Better alone (+1.9) but the stack did not improve | **Fail** in stack |
| **H** | Bigger dev set (dev2, 2,000) + probe fit on 40K | dev2 +3.9, paired CI [+2.4, +5.8] | **PASS** |

**The new main system, SH:**
- **Agent 1 (P3):** a probe on frozen base LLaMA's layer-20 hidden state, fit on 40K reviews.
- **Agent 2 (E6b):** TF-IDF + logistic regression trained on about 497K reviews.
- **Reflector:** fit on dev2.

**Results:**
| | eval | natural_test |
|---|---|---|
| Phase 1 LoRA Analyst | 54.0 | 53.4 |
| Old headline stack | 63.9 | 73.4 |
| Old stack refit on dev2 | 65.5 | 74.2 |
| **SH** | **67.5** (macro-F1 .675) | **75.7** (.612) |
| SH, 5 seeds | 65.8 ± 0.6 | 75.4 ± 0.8 |

- SH beats each single agent in 5/5 seeds.
- **Caveat:** part of the gain over the old headline comes from the bigger Reflector fit set. Against the old stack refit on dev2, SH leads by only +2.0 / +1.5, not significant.

---

## 9. What all of this adds up to (the six lessons)

1. **Audit before believing.** Three headline gains (+22.6, +5.0, and the 70.7% headline) came from transfer collapse, bugs, or a leak.
2. **LLM agents are poor judges of the star.** The LLM judge failed twice, the vision agent produced constant output, and the retrieval vote was weak.
3. **Gains come from independent signals.** A classical text model and LLaMA know different things. Combining them helps; more views of the same text do not.
4. **LLaMA's hidden state beats its generated answer.** A simple probe on the frozen model outperforms the fine-tuned answer and stacks better. Fine-tuning was not needed for Phase 2.
5. **Data size matters more than model tricks.** More data for the Evidence model, the probe and the Reflector helped. More LoRA data did not.
6. **Measure honestly on skewed data.** Natural-test exact accuracy rewards predicting 5 stars, so macro-F1 and per-class recall must be reported beside it.

---

## 10. Why we are heading in this direction now

**The honest gap.** SH is accurate, but it is closer to "specialists + learned combiner" (stacking) than to agentic AI in the strict sense: LLMs that decide, use tools, and correct themselves. An examiner could fairly call it an ensemble.

**The fix** is not to force LLM agents back into choosing the star; the evidence says they fail there. Instead, give them jobs they can do well, and measure the benefit.

**How the original design maps onto the new one:**
| Original agent | What the experiments showed | New role |
|---|---|---|
| Analyst (LoRA writes a rating) | Probe on the hidden state is better | **Analyst = probe on frozen LLaMA** |
| RAG Prover (retrieval vote) | Weak | **Evidence agent** (TF-IDF on ~497K reviews); also supplies phrases for explanations |
| Visual Verifier | Constant output across 3 models | **Dropped**, reported as a negative result |
| Critic (dissonance decides re-analysis) | Fired 0.4%; the LLM judge failed | Split: the **Router** decides when to call expensive agents; the learned **Reflector** decides the star |
| Reflection loop | Did not improve ratings | **Kept, aimed at explanations.** Explainer writes, Critic checks faithfulness, Explainer rewrites (max 2) |

**The plan** (full detail in the spec once written):
- **Workstream A, accuracy:**
  - **A1:** probe fit grown from 40K to 120K reviews. **Running now, locally**, about 16 h.
  - **A2:** a DeBERTa-v3 specialist fine-tuned on about 200K reviews. **On Colab from Nov 1**, about 2 GPU hours. It still runs on the Mac for the demo.
  - Each has a pre-declared gate.
- **Workstream B, agentic layer** on the existing LangGraph code: Analyst, Evidence, Router, Reflector, Explainer, Critic.
  - **Measured by:**
    - LLM calls saved
    - faithfulness before vs after self-correction
    - a deletion test: do the cited phrases actually drive the prediction?
  - Explanations are run on all 3,000 test reviews.
- **Workstream C, paper rigor:**
  - variance rerun with the Evidence model retrained per seed
  - multiple-comparison correction
  - error analysis
  - calibration
  - cost and latency
  - an agent-audit table
- **Workstream D, demo:** one local notebook that runs in about 2 minutes. Live reviews go through the full graph, showing highlighted evidence phrases, agent votes, and the rationale before and after correction. Frozen result charts follow.

**Why this is the right direction:**
- It keeps the accuracy we earned.
- It turns the agentic claim from a design idea into measured behaviour.
- It answers "is this just an ensemble?" directly.
- Everything stays demoable on a laptop.
- Colab is used once, for the single experiment that genuinely needs a GPU.

**Timeline:** stage-1 freeze **Nov 1** (local work), stage-2 freeze **Nov 5** (DeBERTa in or out), prose Nov 2–8, draft 1 Nov 8, submission by Nov 20.

---

## 11. Explaining the tricky points (short answers)

- **"Your first result was +22.6 and now it's +13.5. What happened?"** The first number compared against a Phase 1 model that collapsed under dataset shift, used a leaky prompt, and ran on a different category. The current numbers are on clean, frozen splits, read once, and audited.
- **"Is the LoRA fine-tune wasted?"** It is the Phase 1 baseline and raises LLaMA from 34% to 54% alone. Inside Phase 2 the frozen model's hidden state is the better signal, so LoRA becomes an ablation row.
- **"Why not just use TF-IDF?"** TF-IDF alone gets 62.5 / 70.2. Adding the LLaMA probe gives +5.0 / +5.5, in every seed.
- **"Why did you drop images?"** Three vision models gave constant output. A photo doesn't show whether the owner was satisfied. That's a task limit, reported as a finding.
- **"You tried many things. Isn't the winner luck?"** Every rule was written before running. Failures never touched the test sets. The multiple attempts are disclosed, and the 5-seed check shows the winner is stable.
- **"Is it agentic?"** Today, two specialists plus a learned arbiter. The next stage adds LLM agents that route, explain and self-correct, each with a measured benefit. An LLM choosing the star was tested twice and failed; that is a contribution in itself.

---

## 12. Numbers timeline (balanced 5-class exact unless noted)

| Date | System | Result | Status |
|---|---|---|---|
| Oct 1–8 | Run 1, All Beauty: Phase 1 → Phase 2 | 16.1 → 38.7 | Historical (transfer collapse, leak) |
| Oct 9 | Run 2, Electronics: Phase 1 → Phase 2 | 50.95 → 46.4 | Bugs found and fixed |
| Oct 10 | Kaggle Phase 1, leaky prompt | 60.1 (balanced), 70.7 (natural) | Withdrawn |
| Oct 11 | Kaggle Phase 1, leak-free zero-shot | 59.2 | Reproduction / appendix |
| Oct 11 | Recalibration "Phase 2" | +5.0, then +0.1 | Retracted |
| Oct 16 | Electronics Phase 1 (LoRA, arm R) | 54.0 / 53.4 natural | Phase 1 baseline |
| Oct 16 | First stack (Analyst + Evidence) | 63.9 / 73.4 | Superseded |
| Oct 24 | **SH (probe + 500K Evidence + Reflector on dev2)** | **67.5 / 75.7** | **Current main** |
| Oct 24 → | A1 bigger probe | running | Gate pending |
| Nov 1 → | A2 DeBERTa (Colab) | planned | Gate pending |
