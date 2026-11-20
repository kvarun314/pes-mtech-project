# Presenter Notes Part 0 for the dense 25-slide deck (archived 2026-11-20)

Deck copy: `old_decks/Agentic_LLaMA3_Presentation_dense_2026-11-20.pptx`; builder: `docs/archive/build_deck_e23_2026-11-20.py`.

## Part 0 — Current deck (25 slides). READ THIS FIRST

Deck: `PES_Project_Phase1/Agentic_LLaMA3_Presentation.pptx`, the only current deck, built by `v2_reflective/build_deck_e23.py`. Google Slides (always current): https://docs.google.com/presentation/d/1V08iho4p7gdllKPxRVroaPhj6oo8Km1_/edit
**Flow (as agreed 2026-10-25):** Intro → Phase 1, the reference paper (what it does) → its shortcomings → Phase 2 intro → in depth → how it solves the problem → results of Phase 1 → results of Phase 2 → in depth → conclusions. At most 25 slides.
**Slide numbers and short scripts:** `deck_slide_map.md`, next to this file. It is regenerated on every build from the speaker notes, so it always matches the deck. This part gives the longer scripts in the same order.
Slides no longer in the deck (Run 1 historical, Kaggle baseline, dissonance Critic, first stack, nine-lines summary, code map, workflow) are still in `PES_Project_Phase1/old_decks/` and stay defined in the builder. Parts 1–6 below are for the older v3 deck; use them only for the glossary and metric explanations. Full history: `Project_Journey.md`. Code: `V2_Reflective_Guide.md`.

**Timing (about 22 minutes):**
- Intro: 2 min
- Phase 1 paper and why it falls short: 4 min
- Evidence for the shortcomings: 3 min
- Phase 2 intro, system, and how it answers each shortcoming: 4 min
- Phase 1 results (with the data line): 1 min
- Phase 2 results (main result, gain, Hand-off agent): 3 min
- In depth (errors, why it works, tried/dropped, kept, robustness, self-correction 1 and 3): 6 min
- Schedule and conclusions: 2 min

### 1. Intro: title and problem (slides 1–2)
Problem: one LLM pass gives a star with no second opinion, no evidence, and nothing checking its explanation. Sarcasm, mixed reviews and neighbouring stars trip it up. One minute; the rest of the talk shows how Phase 2 fixes each point.

### Key terms (for you; not on the slides)
Use these when explaining or when someone asks what a word means.

| Term | Meaning |
|---|---|
| LLaMA-3-8B | Meta's open large language model (8 billion parameters); the base model in both phases |
| LoRA | Small add-on weights trained on top of a frozen model, so fine-tuning is cheap (Phase 1) |
| Phase 1 LoRA | LLaMA-3-8B + LoRA trained to output the star: the reference paper's method |
| Frozen model | Used as-is, never retrained |
| Hidden state (layer 20) | LLaMA's internal numeric summary of the review, halfway through the network |
| LLaMA probe | Small classifier that reads the layer-20 hidden state and outputs 1–5★ probabilities |
| TF-IDF | Scores each word / word pair by how distinctive it is in a review |
| Evidence model | TF-IDF + logistic regression on ~497K reviews; its top phrases become the explanation evidence |
| Logistic regression (LR) | Simple, fast classifier that outputs a probability for each star |
| Agent | A component with one job (rate, combine, route, explain, check) that passes results to the next |
| Stack / stacking | Several models predict; a small learned model combines their predictions |
| DeBERTa-v3 | A smaller language model (encoder) fine-tuned on our 100K training reviews; the third agent |
| Reflector | The learned combiner: weighs the probe, the Evidence model and DeBERTa, outputs the final star |
| Main stack | LLaMA probe + Evidence model + DeBERTa + Reflector: the system behind the headline results |
| Router | Decides whether the slow LLaMA probe is needed for a review (worked with two agents; off with three) |
| Hand-off agent | Answers automatically when the stack is confident; sends unsure reviews to a person |
| Explainer / Critic | Explainer (instruction-tuned LLM) writes the rationale; Critic checks it and asks for rewrites |
| Self-correction | System revises its own output: escalation, LLM re-rating (rating) and the Critic loop (explanation) |
| Balanced eval | 1,000 test reviews, 200 per star: every star counts equally |
| Natural test | 2,000 test reviews with the real star mix (65% are 5★) |
| dev / dev2 | Separate development reviews (500 / 2,000) used for every choice; test sets are read once |
| Exact accuracy | Share of reviews whose star is predicted exactly |
| Within one star | Share predicted exactly or one star off |
| Macro-F1 | Average quality over the five stars, so rare stars count as much as 5★ |
| Polarity / binary | Negative vs neutral vs positive / negative vs positive only |
| Zero-shot / one-shot | Prompt with no example / with one worked example |
| Label leak | Example picked using the true answer, which inflated old scores by ~8 points |
| Gate | Pass/fail rule written in the lab log before an experiment runs, checked on dev data only |
| 95% CI (bootstrap) | Range the true value likely falls in, from resampling the test reviews |
| McNemar / paired test | Tests whether two systems differ on the same reviews |
| Holm correction | Adjusts p-values when many comparisons are made, to avoid lucky wins |
| ECE (calibration) | Gap between the model's confidence and its real accuracy; lower is better |
| Faithfulness | Explanation states the right star and only uses evidence that is really in the review |
| Deletion test | Remove the cited phrases vs random words: if confidence drops more, the explanation points at what mattered |

### 2. Phase 1, the reference paper: what it does and why it falls short (slides 3–6)
- **Framing (say it this way):** "We chose this paper as our reference. Phase 1 is our reproduction of it and our exploration of where it holds and where it does not. Phase 2 is what we built from what we learned." Call it the reference paper, not "the baseline".
- **The paper we chose:** Wang et al. (2025). Walk the four cards:
  - **Task:** one model rates a review 1–5★ and justifies it.
  - **Data:** TextBlob cleaning, equal reviews per star, varied wording (VGST), extra neutral reviews.
  - **Model:** LLaMA-3-8B with a small LoRA add-on, using the exact Table 1 settings.
  - **Prompt:** a rubric, one worked example, and "step by step".
  - Then the reported 92.6%. Say: "We rebuilt exactly this as Phase 1."
- **Why it falls short**, four plain reasons:
  - One pass, with nothing to catch a mistake.
  - The score depends heavily on the set-up: a leaked example in our first run, and a skewed test where always guessing 5★ gets 63.8%. Leak-free and balanced, it's 54%.
  - The data cleaning removes most negative reviews.
  - The justifications are a template.
  - Be fair: the leak came from our first reproduction's example choice; it isn't a claim about the authors.
- **Pipeline and LoRA slides:** how the method works in detail. Keep each to a minute.

### 3. Evidence for the shortcomings (slides 7–9)
- **Leak:** the one-shot example was chosen using the true star, worth about 8 points. Every later number is leak-free.
- **Hallucination study:** the LoRA's explanations are one canned sentence per star, wrong 53% of the time. The probabilities are usable; the explanations are not.
- **Preprocessing:** TextBlob filtering deletes most negative reviews and costs 2.5–3.5 points. VGST adds nothing.

### 4. Phase 2: intro, system, and how it answers each shortcoming (slides 10–12)
- **Where Phase 2 started:** the original four-agent design (Analyst, Visual Verifier, RAG Prover, dissonance Critic). The audits showed the Visual Verifier gave constant output, RAG was weak, and the Critic barely acted.
- **System slide:** walk the five boxes. Evidence model + DeBERTa → Router → LLaMA probe → Reflector → Explainer + Critic.
  - The decision core (main stack) has three agents: the LLaMA probe, the Evidence model and DeBERTa. It is built and frozen.
  - The agentic layer (Router, Hand-off agent, Explainer, Critic) is built in LangGraph and measured on all 3,000 test reviews.
  - The Router worked for the two-agent stack. With three agents it failed its check, so it is off and every review goes to all three agents.
  - Self-correction: the explanation loop (faithful 62–64% → 91–92% with max 4 rewrites, 0 broken). Escalation worked with two agents (92 vs 50 and 152 vs 78 fixed vs broken). An Instruct LLM re-rating unsure reviews was tested and failed: 76 fixed vs 89 broken on dev2.
- **"How Phase 2 answers each shortcoming"** (the table): read it as problem → fix → the number that checks it. Spend time on three rows:
  - Three agents instead of one pass.
  - LLMs explain and check instead of judging, because the judge failed twice.
  - Template justifications replaced by evidence-based explanations that a Critic checks.
  - The last row is new: unsure reviews go to the Hand-off agent.

### 5. Results of Phase 1 (slide 13)
Start with the data line in the subtitle. It answers any leakage question:
- One dataset (Amazon Reviews 2023 Electronics).
- Splits drawn once and disjoint by review id and exact text.
- Every choice made on dev; test sets read once.

"This is the reference paper's method on our data, leak-free."
- Exact star 54.0 / 53.4, but within one star about 94% and positive vs negative 91–93%. It gets the direction right.
- On the paper's own Kaggle data with its protocol and no leak: 69.1% (always guessing 5★ gives 63.8%).
- The paper's 92.6% used K=5 cross-validation and a 40K test set.

### 6. Results of Phase 2: the headline (slides 14–16)
**Main result:** "The main stack is three readers and a learned referee: the LLaMA probe reads meaning, the Evidence model reads key words, DeBERTa reads the wording." **Read the table top to bottom:** LoRA + Evidence stack refit on dev2 65.5 / 74.2; probe alone 61.2 / 74.3; Evidence alone 62.5 / 70.2; DeBERTa alone 66.5 / 75.8; previous two-agent main 67.5 / 75.7; **main stack 67.5 / 78.0**.
**Paired tests (Holm over 12):** versus the previous main +0.0 eval (not significant) and +2.3 natural (adjusted p .004). Versus DeBERTa alone +1.0 eval (ns), +2.3 natural (adj. p .003). Versus Evidence alone +5.0 / +7.8; versus the probe alone +6.3 / +3.7.
**Key message:** "Combining different kinds of signal beats any one alone. DeBERTa helps on the real star mix, not on balanced eval; the gain held over 3 DeBERTa seeds (+2.2 ± 0.2 natural, +0.1 ± 0.3 eval). The frozen LLaMA model already contains the useful information; fine-tuning it was not needed."

**Phase 1 → Phase 2:** lead with the gain, +13.5 / +24.6 points (29% / 53% fewer errors), significant after Holm. Then why: the agents complement each other; the Evidence model is strong on short reviews, the probe on long ones, and DeBERTa adds on the natural mix. Say the caveat yourself: the LoRA + Evidence stack refit on the same dev reviews is only 2.0 behind on eval (not significant). Never quote 70.7 / 60.1 (leaky), 74.8 (one seed) or 79.9 (class-mix correction).

**Hand-off agent (slide 16):**
- **Say:** "Exact accuracy has plateaued at about 67 and 78. The agentic answer is for the system to know when it is unsure. The Hand-off agent answers automatically only when the stack is confident and sends the rest to a person."
- The threshold was fixed on dev2 before the test sets were read.
- **Numbers:** natural test, answers 56.1% of reviews and is right 92.8% [91.3, 94.3]; balanced eval, 86.9% on 41.1%. The handed-off reviews are only 54–59% right, so they are the hard ones. Phase 1 with the same rule: 67.9% / 66.9%.
- **Star ranges (Q1):** when the system is unsure, it can answer with a range of stars instead of handing the review on. The range is calibrated (conformal prediction) to contain the true star 90% of the time: 91.6% on eval and 90.1% on natural test. Ranges are 1.8 stars wide on eval and 1.4 on natural test; Phase 1 needs 2.4 and 2.1 and still misses 90% on natural test (88.0%). The range is a single star for 34% / 58% of reviews (Phase 1: 5% / 11%). With three options (answer, range, person) only 11% (eval) and 2% (natural) still need a person. Be open: the range option on its own covers 86–91%, because it gets the harder reviews; the 90% holds over all reviews.
- The "All models" bar chart is out of the deck (it is in `paper/figures/e23/main_bars.png` if asked).

### 7. In depth (slides 17–23)
**Where the main stack is right and where it errs:**
- Left, the confusion matrix: mistakes hug the diagonal, and 92% of errors are one star off (88% on natural).
- Right, the share of each star found, Phase 1 vs Phase 2.
- Say the exception yourself: Phase 1 finds slightly more 4★ reviews (natural .715 vs .664; eval .635 vs .630) because it over-predicts 4★. Phase 2 is far better on 5★ (.855 vs .490), the most common star.

**Why several agents beat one:**
- Left: the Evidence model wins on short reviews (70.0 vs 54.6), the LLaMA probe on long ones (64.3 vs 55.8), and the stack is best or near-best in every length band.
- Right: when the probe and the Evidence model disagree, the stack is still right 58% (eval) / 62% (natural) of the time, against 36–50% for either alone. DeBERTa is the third view the Reflector uses there.
- **Hard cases (band):** reviews that sound negative but are rated 4–5★: stack 67.5 vs Phase 1 35.0 (eval, p .002), 78.2 vs 43.0 (natural, p 5e-16). Positive-sounding 1–2★ reviews: +11.1 (eval) and +10.3 (natural), both not significant. Mixed reviews with "but / however": +10.0 (eval, p .003), +3.2 on natural (not significant). Explicit sarcasm cue words: only 6 and 1 reviews, too few to measure; say so if asked about sarcasm.

**Tried / dropped / kept:** **Don't read every row.** Pick three: the LLM judge (failed twice, which is why the LLM doesn't decide the star), the Visual Verifier (constant output) and the recalibration (the gain was the leak). The last two rows: the probe on 120K reviews added nothing (+0.3 on dev2), and the Instruct-LLM re-rating, the third LLM-judge failure (76 fixed vs 89 broken). On "What we kept", walk the eleven kept changes quickly; the new one is DeBERTa (+2.3 natural, Holm p .004; stable over 3 seeds).

**Robustness:**
**Say:** "I checked whether the result survives four questions."
- **Multiple comparisons:** after Holm correction over 12 comparisons, 8 stay significant: all six on natural test (including vs the previous main and vs DeBERTa alone), and vs Evidence alone and vs probe alone on eval. On eval the main stack is not significantly better than DeBERTa alone, the previous main or the LoRA + Evidence stacks.
- **Errors:** 92% (eval) and 88% (natural) of errors are one star off. 2★ and 3★ are hardest (recall .51–.58).
- **Calibration:** ECE .031 and .052. On eval confidence .679 vs accuracy .675; on natural slightly under-confident. The Hand-off agent relies on this.
- **Cost:** the LLaMA probe takes about 0.67 s per review; DeBERTa 86 ms; TF-IDF and the Reflector under 0.1 ms each.
- **Text changes (Q4, 300 eval reviews):** lowercase text without punctuation changes the main stack's star for 9.3% of reviews and Phase 1's for 29.7% (p 3e-11); typos change 0.7% vs 2.0% (not significant). **Be open:** with the title removed, the star changes for 15.7% of reviews (the two-agent stack: 8.0%), exact 68.0 → 63.7, because the Evidence model and DeBERTa both read the title; Phase 1 never reads it.
- **Years and products (Q5):** the gain over Phase 1 is positive in every review-year band (2001–2023) and every product-popularity band, all confidence intervals above 0: eval +10.3 to +16.3, natural +21.5 to +26.5. This is not a future-data test; all agents saw reviews from every year.

**Self-correction 1, measured on the two-agent stack (slide 22):**
- **Say:** "When the quick Evidence-model draft is unsure, the system calls the LLaMA probe and the Reflector revises the star."
- **Numbers (two-agent stack):**
  - Balanced eval: 607 reviews escalated, 92 fixed vs 50 broken, p = 0.0005.
  - Natural test: 1,035 escalated, 152 fixed vs 78 broken, p = 0.000001.
  - LLaMA was skipped on 39% / 48% of reviews, with no significant accuracy change (67.4 vs 67.5; 75.5 vs 75.7).
- **Three agents (band):** the same check was rerun on dev data for the three-agent stack and failed: 50 fixed vs 37 broken (p .20), and natural dev 0.6 points lower than using all agents. By the rule written beforehand, the Router is off: every review uses all three agents.
- **Be open about it:** "For two agents this was our second threshold. The first (0.60) missed its natural-dev check by 0.3 points and was never tested on the test sets."

**Self-correction 3, measured (slide 23):**
- **Say:** "The Explainer writes a short explanation from the evidence only. The Critic checks five things: the star, that quotes really occur in the review, the key phrases, the disagreement, and no invented claims. If one fails, it sends specific feedback and the Explainer rewrites, at most four times."
- **Numbers (three-agent stack):** faithful first drafts 62.3% (eval) / 63.7% (natural); after the loop 90.5% / 91.6%. 282 + 559 fixed, 0 broken; McNemar p < 1e-84. About 3.0 LLM calls per review. The previous two-agent stack reached 90.6% / 91.1%. A fixed-form fallback for the rest (I3) failed its dev2 check.
- **Deletion test (right):** removing the cited phrases lowers the stack's confidence by .156 vs .077 for random words (eval) and .209 vs .119 (natural); keeping only the cited phrases holds .521 vs .301 and .616 vs .349. p < 1e-54 everywhere.
- **Phase 1 comparison (left, small text):** same 150 reviews, same judge: Phase 1's justification never quotes the review (0% vs 75%), is one of 5 stock sentences (3% distinct vs 100%), and Phase 2 is judged free of unsupported claims 93% of the time (17 vs 5 cases where only one passes, p .017).
- **Worked example (bottom strip, from the live demo):** a 2★ backpack review, rated 2★. Walk it left to right: the evidence pack (for: "disappointing", "Poor", "lacks"; against: "great", "easy"); draft 1 claims "Poor quality", which the review does not say, so the Critic asks to remove it; draft 2 drops it but quotes too few key phrases, so the Critic asks for at least 2; draft 3 quotes the review and passes. Say: "This is what the 62% to 91% means for one review." The demo (section 4) shows the same example live.
- **Be open:** about 9% of final explanations still fail a check (mostly the LLM judge); they are labelled, not hidden. The judge is the same Llama-3-8B-Instruct model that writes the explanation; the deletion test and the Phase 1 comparison are the independent checks. A human check of 50 explanations (Nov 20, blind to the Critic): author 80% faithful vs Critic 92%, agreement 76%. When the agents agreed the author passed all 34; when they disagreed only 6 of 16 (Critic 15 of 16). So the Critic is too lenient on contested reviews: it misses statements about the system such as "the reviewer gave N stars".

### 8. Build schedule and conclusions (slides 24–25)
**Say:** "Every row had its rule written first and is logged. The bigger probe failed its check (120K reviews gave +0.3 on dev2), so the 40K probe stays. Explanations and the Critic loop ran on all 3,000 test reviews, then the deletion test. The extra specialist became DeBERTa: it passed its dev2 check, was stable over 3 seeds, and was adopted. Every Phase 2 result was then rerun with three agents on Nov 11–12; the Router failed its check with three agents and is off."

**Conclusions:** four cards.
- Phase 1 was reproduced honestly, and two problems were found in the original protocol.
- Phase 2's three-agent stack beats the single fine-tuned model, significantly (+13.5 / +24.6). DeBERTa adds +2.3 on the natural mix only.
- The agentic layer is measured on 3,000 reviews: the Critic loop makes 91–92% of explanations faithful with none broken; the Hand-off agent is right 93% of the time on the 56% it answers; the Router helped with two agents but not with three. LLM re-rating was tested and dropped (third LLM-judge failure).
- **Limits, said plainly:** one dataset (but the gain holds in every year band); the main stack needs the review title (16% of stars change without it); part of the gain comes from the larger Reflector fit set; explicit sarcasm is too rare in the data to measure.

End with the dates: human check of 50 explanations done (Critic too lenient when agents disagree), paper by Nov 20, final presentation in December with a live demo.

### New questions to expect
- **"Are these results good or bad?"** Good, and they hold up. Exact five-star rating is hard, because 4★ vs 5★ is often a judgement call.
  - On the same balanced eval: random 20%, base LLaMA 34.3%, SVM / Naive Bayes ~47–48%, DistilBERT 52.8%, Phase 1 (the paper's method) 54.0%, **main stack 67.5%**.
  - Within one star 97.3%; 92% of errors are one star off; macro-F1 .675. Natural test 78.0% (within one 97.4%).
  - The paper's 92.6% came from skewed Kaggle data with a different protocol. On that kind of data our leak-free reproduction gets 69.1%, while always guessing 5★ gets 63.8%.
- **"What is the natural test, and what other tests did you use?"**
  - **Balanced eval:** 1,000 reviews, 200 per star. Skill on every star equally; random gets 20%.
  - **Natural test:** 2,000 reviews with the real mix (65% 5★). What real use looks like; always guessing 5★ gets ~65%, so macro-F1 is reported too.
  - **Development sets:** dev (500) and dev2 (2,000, balanced) plus natural_dev (500), used for every decision. Test sets are read once.
  - **Kaggle paper-protocol test:** 4,000 reviews, used only to reproduce the paper.
  - No review appears in two sets (checked by id and by exact text).
- **"What is the main stack?"** Three readers and a referee (adopted Nov 10).
  - The **LLaMA probe**: frozen LLaMA-3-8B whose layer-20 state goes to a small classifier; reads meaning.
  - The **Evidence model**: TF-IDF on ~497K reviews; reads key words and supplies the phrases for the explanation.
  - **DeBERTa-v3**: a smaller language model fine-tuned on our 100K training reviews (Colab); reads the wording.
  - A **Reflector** (a learned referee) combines them into the final star.
  - The readers make different mistakes: alone 61–67%, together 67.5% (eval) and 78.0% (natural).
  - The earlier "Probe + Evidence stack" (67.5 / 75.7) is now called the previous main stack.
- **"Did you try letting the LLM fix the star?"** Three times. Gate 0 on Kaggle and on Electronics, then on Oct 25 with Llama-3-8B-Instruct, evidence for both candidate stars and an order-swap consistency check. It fixed 76 and broke 89 on dev2, so it was dropped before any test read. LLMs are better at explaining and checking than at deciding.
- **"Is the tried-and-dropped table final?"** For everything run so far, yes. A pending experiment that fails adds a row; one that passes moves to "What We Kept". This is updated at the Nov 1 / Nov 5 freezes.

**Block diagram (draw it if asked):**
```
Review ─┬─► LLaMA probe (meaning)  ─► P(1..5★) ─┐
        ├─► Evidence model (words) ─► P(1..5★) ─┼─► Reflector ─► final star ─► Hand-off (answer / range / person)
        └─► DeBERTa (wording)      ─► P(1..5★) ─┘
Phase 2 v3 around it: Router (two-agent stack only; off with three) ·
LLM re-rating of uncertain reviews (gated) · Explainer ⇄ Critic loop (max 4 rewrites) → checked explanation
```
- **"Why did the gain over the old system shrink in the fair comparison?"** The old Reflector was fit on only 500 rows. Given 2,000 rows it improves too. The main stack is still ahead in 4 of 5 seeds on eval and 5 of 5 on natural, but the margin is small (+0.7 ± 1.2 and +1.0 ± 0.6).
- **"You tried many things; isn't the winner luck?"** Every rule was written before running, and failures never touched the test sets. Five candidates were compared on dev2, and I disclose that. The test sets were read once.
- **"Then what is the role of LoRA?"** Phase 1 is still our reproduction and exploration of the reference paper. Inside Phase 2 it is an ablation row: adding it to the main stack did not help (dev2 .633 versus .639).
- **"Why is natural-test accuracy higher than balanced?"** About 65% of natural reviews are 5-star, which are easier. That is why macro-F1 (.634) is reported next to it.
- **"Why does DeBERTa help on natural test but not on balanced eval?"** Its gain is on the common 4★ / 5★ reviews (natural test is 65% 5★). On balanced eval it fixes and breaks the same number (58 / 58). Three DeBERTa seeds give the same picture: +2.2 ± 0.2 natural, +0.1 ± 0.3 eval.
- **"Why is the Router off?"** With three agents, skipping the probe lowered natural-dev accuracy by 0.6 points and its corrections were not significant (p .20). The rule written beforehand said: if the check fails, the Router is off.
- **"Is this agentic?"** Today: three independent agents plus a learned Reflector that arbitrates between them. The LLM-as-judge version (Gate 0) was tested and failed, and that is reported. Phase 2 v3 (the "How it solves the problem" slide) adds agents for routing, hand-off, explanation and self-correction: escalation (worked with two agents, off with three), the Hand-off agent (93% right on the 56% it answers) and the Critic loop on explanations (62–64% → 91–92% faithful, 0 broken); LLM re-rating of the star was tested a third time and failed, so LLMs explain and check rather than decide.
- **"One slip?"** Yes, disclosed: Evidence-alone test accuracy was printed once before the gate ran. It was not used for any choice.

### Names to say out loud
Say **"main stack"** (three agents), **"previous main stack"** (probe + Evidence), **"LLaMA probe"**, **"Evidence model"**, **"DeBERTa"**, **"earlier LoRA + Evidence stack"** and **"Phase 1 LoRA"**. The internal IDs (SH3, SH, P3, E6b, deb3, S0) live only in `LOG.md` and the code.

### Numbers cheat sheet
| | eval (1,000 balanced) | natural_test (2,000) |
|---|---|---|
| main stack (three agents) exact / macro-F1 | 67.5 / .675 | 78.0 / .634 |
| previous main (probe + Evidence) | 67.5 / .675 | 75.7 / .612 |
| DeBERTa alone | 66.5 / .667 | 75.8 / .614 |
| DeBERTa gain over 3 seeds | +0.1 ± 0.3 | +2.2 ± 0.2 |
| previous main 5-seed mean | 65.8 ± 0.6 | 75.4 ± 0.8 |
| Earlier LoRA + Evidence stack (dev fit) | 63.9 / .638 | 73.4 / .577 |
| LoRA + Evidence stack refit on dev2 | 65.5 / .653 | 74.2 / .587 |
| LLaMA probe alone | 61.2 | 74.3 |
| Evidence model alone | 62.5 | 70.2 |
| LoRA Analyst alone | 54.0 | 53.4 |
| Base LLaMA zero-shot | 34.3 | 16.8 |
| main stack within-1 / polarity | 96.7 / 82.0 | 96.7 / 91.5 |
| main stack errors one star off | 92% | 88% |
| main stack ECE | .047 | .062 |

Sources: `v2_reflective/e23/results_frozen.md`, `readout_H.out`, `variance_H.out`; full history in `v2_reflective/EXPERIMENTS_SUMMARY.md`.

---


(Parts 1–6, older v3 deck) Deck: `Agentic_LLaMA3_Presentation_FINAL_v3.pptx` (22 slides). The original `..._FINAL_FORMATTED_v2.pptx` is untouched.
Speaking time: about 18–22 minutes plus questions. Slides 1–9 are the Phase 1 half (about 9 minutes), slides 10–17 are the earlier Phase 2 work (about 6 minutes), slides 18–21 are the audit and what comes next (about 5 minutes). Slide 22 is a backup slide.

This file has six parts:
1. The story in one minute (read this first)
2. Plain-English glossary: every technique, what it is, why we used it
3. Slide-by-slide script: what to say, what to explain, caveats, likely questions
4. Question bank with answers
5. Caveats checklist
6. Numbers cheat sheet

Wording note: everything below is written so you can read it out or paraphrase it. "Say" blocks are close to spoken language. "Explain" blocks are background you can use if someone asks.

---

