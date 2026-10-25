# Presenter Notes — Agentic Self-Correction in LLaMA-3

## Part 0 — Oct 25 presentation. READ THIS FIRST

Deck: `../Agentic_LLaMA3_Presentation.pptx`, the only current deck (30 slides; built by `v2_reflective/build_deck_e23.py`). Short speaker notes are also in the PowerPoint notes pane of every slide. Full project history (what failed, what worked, why this direction): `Project_Journey.md`, next to this file.
Older decks were moved to `../old_decks/` (unchanged, not used). Parts 1–6 below were written for the older v3 deck (22 slides). Their slide numbers no longer match. Use the glossary and metric explanations, but take numbers and slide order from this part.

Suggested timing (about 22 minutes): slides 1–9 background (6 min), slide 10 update (2 min), slides 11–15 protocol and audits (4 min), slides 16–18 experiments and main result (5 min), slide 19 robustness (2 min), slide 20 next steps (2 min), slide 21 roadmap (1 min). Slides 22–30 are historical backups; show them only if asked.

### Slide map
| Slide | Content | Status |
|---|---|---|
| 1 | Title | updated (Oct 25, new agenda line) |
| 2–9 | Problem, gap, data, pipeline, LoRA, Kaggle baseline, original Phase 2 design, dissonance Critic | background; slides 4, 7, 8, 9 subtitles now say what is historical |
| 10 | What changed since the last review | NEW |
| 11 | Leak audit | unchanged |
| 12 | Electronics protocol | unchanged |
| 13 | First stack, Oct 13 (superseded) | relabelled |
| 14 | Preprocessing ablation | unchanged |
| 15 | Hallucination study | unchanged |
| 16 | Experiment ledger (nine lines of work) | NEW |
| 17 | Main result: SH | NEW |
| 18 | Frozen bar chart, all models | NEW |
| 19 | Robustness: Holm, errors, calibration, cost | NEW (Oct 25) |
| 20 | Next: Phase 2 v3 (Router, Explainer, Critic loop, A1, A2) | NEW (Oct 25) |
| 21 | Roadmap | updated (two-stage freeze) |
| 22–30 | Historical Run 1 slides | unchanged, backup only |

### Slide 10 — What changed since the last review
**Say:** "Since the last review I finished every planned experiment. Each one had a pass/fail rule written down before it ran, checked only on development data. Five ideas failed. One passed, and it is now the main system."
**Explain the main system (SH) in one breath:** "Two agents. The first reads the hidden state of the base LLaMA-3-8B model at layer 20 and puts a small logistic classifier on top. LLaMA is frozen, with no fine-tuning. The second is a TF-IDF keyword model trained on about half a million reviews. A Reflector, which is a logistic regression, combines the two agents' probabilities. It is fit on 2,000 balanced development reviews."
**Numbers:** balanced eval 67.5% exact (was 63.9%); natural test 75.7% (was 73.4%).
**Caveat (say it, don't wait to be asked):** "Part of the gain comes from fitting the Reflector on 2,000 rows instead of 500. The old system given those same 2,000 rows is only 2.0 and 1.5 points behind, and that gap is not significant."

### Slide 13 — First stack (superseded)
**Say:** "This was the system three weeks ago: the LoRA Analyst plus the Evidence model. It got 63.9 and 73.4. I keep it in the deck because it is the baseline the new system is compared against."

### Slide 16 — Experiment ledger
Go row by row, about 10 seconds each.
- **Gate 0, LLM adjudicator:** when the two agents disagree, ask LLaMA to choose. It was right 54.2% of the time versus 73.3% for the stack. Dropped.
- **Base LLaMA without LoRA:** weak alone (34.3%), yet stacked with Evidence it gives 63.7, almost the same as LoRA plus Evidence (63.9). So the fine-tune adds little inside the stack.
- **DistilBERT on the same 2K rows:** 52.8% versus LoRA 54.0%, not significant. A model 100 times smaller matches it.
- **Variance and LoRA seed 2:** the stack gain over Evidence was positive in 5 of 5 seeds.
- **LoRA on 10K rows:** dev 56.4% against a 58% gate. More LoRA data does not help much.
- **Hidden-state probe on 10K rows (F, F2):** missed the gate by a hair.
- **Pooled-dev Reflector (D):** passed its gate, but the gain only came from predicting more 5-star reviews (a class-prior shift), so not adopted.
- **Evidence on 500K rows (G2):** better on its own, but the stack did not improve.
- **Phase H:** 2,000-row dev set plus the probe fit on 40K rows. Passed: +3.9 points on dev2, CI [+2.4, +5.8].

### Slide 17 — Main result
**Read the table top to bottom:** old headline 63.9 / 73.4; old stack refit on dev2 65.5 / 74.2; probe alone 61.2 / 74.3; Evidence alone 62.5 / 70.2; **SH 67.5 / 75.7**; 5-seed mean 65.8 ± 0.6 / 75.4 ± 0.8.
**Paired comparisons:** versus the old headline +3.6 [+0.9, +6.3] eval and +2.3 [+0.6, +4.0] natural, both significant. Versus Evidence alone +5.0 and +5.5. Versus the probe alone +6.3 eval; +1.4 natural (not significant).
**Key message:** "Combining two different kinds of signal beats either one alone, every seed. The frozen LLaMA model already contains the useful information; fine-tuning it was not needed for the stack."

### Slide 18 — Bar chart
**Say:** "Every model, with 95% confidence intervals. SH is in red. Notice base LLaMA: as a zero-shot rater it gets 34%, but a probe on its hidden state gets 61 and 74."

### Slide 19 — Robustness
**Say:** "I checked whether the result survives four questions."
- **Multiple comparisons:** after Holm correction over 8 comparisons, 5 stay significant. SH beats Evidence alone on both splits, the probe on eval, and the old headline on both splits (adjusted p .045 and .043). The probe on natural test and the old stack refit on dev2 are not significant.
- **Errors:** 90% (eval) and 86% (natural) of errors are one star off. 2★ and 3★ are hardest. When the two agents agree (64–72% of rows) SH is right 74–84% of the time; when they disagree, 56%.
- **Calibration:** ECE .047 and .062, slightly under-confident. The confidence can be trusted, which is what the Router needs.
- **Cost:** the LLaMA probe takes about 0.67 s per review; TF-IDF and the Reflector take under 0.1 ms each.

### Slide 20 — Next: Phase 2 v3
**Say:** "An LLM choosing the star failed twice, so the specialists keep deciding the star. The agents now do three things: a Router skips the slow LLaMA call when the Evidence agent is confident; an Explainer writes a short rationale from the evidence; and a Critic checks it and sends it back with specific feedback until it is faithful, at most twice."
**Measured:** on all 3,000 test reviews: faithfulness before and after the Critic loop, rewrite rate, a deletion test (remove the cited phrases versus random words), and latency with and without the Router.
**Accuracy candidates:** A1, the probe on 120K rows instead of 40K, is running (45K of 80K new rows extracted). A2, a DeBERTa-v3 specialist, runs on Colab on Nov 1. Each needs to pass a dev-only gate first.
**Demo:** one local notebook, about 2 minutes, no network.

### Slide 21 — Roadmap
Freeze stage 1 on Nov 1 (local work), stage 2 on Nov 5 (Colab result in or out). I write the prose from Nov 2; draft 1 to the supervisor Nov 8; submit by Nov 20.

### New questions to expect
- **"Why did the gain over the old system shrink in the fair comparison?"** The old Reflector was fit on only 500 rows. Given 2,000 rows it improves too. SH is still ahead in 4 of 5 seeds on eval and 5 of 5 on natural, but the margin is small (+0.7 ± 1.2 and +1.0 ± 0.6).
- **"You tried many things; isn't the winner luck?"** Every rule was written before running, and failures never touched the test sets. Five candidates were compared on dev2, and I disclose that. The test sets were read once.
- **"Then what is the role of LoRA?"** Phase 1 is still the paper's baseline reproduction. Inside Phase 2 it is an ablation row: adding it to SH did not help (dev2 .633 versus .639).
- **"Why is natural-test accuracy higher than balanced?"** About 65% of natural reviews are 5-star, which are easier. That is why macro-F1 (.612) is reported next to it.
- **"Is this agentic?"** Today: two independent agents plus a learned Reflector that arbitrates between them. The LLM-as-judge version (Gate 0) was tested and failed, and that is reported. Phase 2 v3 (slide 20) adds LLM agents for routing, explanation and self-correction of explanations, each measured.
- **"One slip?"** Yes, disclosed: Evidence-alone test accuracy was printed once before the gate ran. It was not used for any choice.

### Numbers cheat sheet
| | eval (1,000 balanced) | natural_test (2,000) |
|---|---|---|
| SH exact / macro-F1 | 67.5 / .675 | 75.7 / .612 |
| SH 5-seed mean | 65.8 ± 0.6 | 75.4 ± 0.8 |
| Old headline (LoRA + E, dev fit) | 63.9 / .638 | 73.4 / .577 |
| LoRA + E refit on dev2 | 65.5 / .653 | 74.2 / .587 |
| P3 probe alone | 61.2 | 74.3 |
| Evidence E6b alone | 62.5 | 70.2 |
| LoRA Analyst alone | 54.0 | 53.4 |
| Base LLaMA zero-shot | 34.3 | 16.8 |
| SH within-1 / polarity | 96.7 / 82.0 | 96.7 / 91.5 |
| SH errors one star off | 90% | 86% |
| SH ECE | .047 | .062 |

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

## Part 1 — The story in one minute

A fine-tuned LLaMA-3 model reads a product review and predicts a 1–5 star rating in a single pass. That works well for obvious reviews and badly for the in-between ones: it confuses neighbouring stars, it cannot check its own work, and it cannot use anything beyond the text.

Phase 1 is our reproduction of a published paper (Wang et al., 2025) that fine-tunes LLaMA-3-8B with LoRA on Amazon reviews and reports 92.6% accuracy. We rebuilt its pipeline, trained its adapter with its exact hyperparameters, and tested it. We reproduce the method. We do not reproduce its headline number as stated, and we show why: the paper's numbers look like polarity accuracy on a skewed test set, and its baselines land where ours do when scored the same way.

Phase 2 is the add-on that shows why a single pass is not enough. It wraps the same model in several agents (an analyst, a visual verifier, a retrieval agent, and a critic) and lets disagreement between them trigger a second look. We ran it, audited it honestly, found and fixed four real defects, and learned that the remaining errors are swaps between neighbouring star ratings. The next version (reflective orchestration) is aimed exactly at those errors. It is a design, and we do not yet claim results for it.

The honest one-line summary for the end: **Phase 1 reproduces the method but not the headline, Phase 2 is hardened and slightly net-positive, and the next step targets the specific confusions that remain.**

---

## Part 2 — Plain-English glossary (what each piece is, and why we used it)

### Star rating and polarity
- **5-class exact accuracy:** the predicted star (1–5) equals the true star. Strict.
- **Polarity:** group stars into three buckets, 1–2 = negative, 3 = neutral, 4–5 = positive, and ask whether the bucket matches. This forgives the most common mistake, a swap between neighbouring stars on the same side (4 vs 5, 1 vs 2). The paper itself defines these buckets (§2.2).
- **Within ±1:** the prediction is at most one star away. Shows how "close" errors are.
- **MAE:** mean absolute error in stars. 0.5 means the average miss is half a star.
- **Precision / recall / F1:** for one star class, precision = of the reviews we called that star, how many really were; recall = of the reviews that really were that star, how many we caught; F1 combines them. **Macro-F1** averages F1 over the five stars equally (so rare stars count as much as common ones). **Weighted-F1** weights by how common each star is.
- **Trivial baseline:** what a model gets by always guessing the most common answer. On our paper-style test set, 63.8% of reviews are 5-star, so "always 5" scores 63.8%, and "always positive" scores 78.6% on polarity. Any score only means something relative to these.

### Data quality and sampling (Phase 1 pipeline, slide 5)
- **Kaggle Amazon Product Reviews:** about 500,000 to 568,000 reviews with a 1–5 rating and text. Heavily skewed toward 5 stars (about 64%). Training on raw data would teach the model to say "5" for everything.
- **TextBlob DQC (data quality control):** TextBlob is a small library that gives each text a polarity score from -1 (negative) to +1 (positive). The paper uses it to throw away reviews whose rating contradicts their text. Rule we implemented: drop a review if polarity is negative but the rating is above 3, or polarity is positive but the rating is below 3. Keep it if polarity is exactly zero or TextBlob fails. **Why:** star ratings are noisy (someone clicks 5 stars but writes a complaint). Removing the clearest contradictions gives cleaner labels. **Limit:** TextBlob is crude (a dictionary-based scorer), so it is a rough filter, not truth. It also removes sarcasm cases, which is exactly the hard data, so it makes the training set a little easier than reality.
- **Stratified sampling:** after filtering, take an equal number of reviews per star (1 to 5), up to a cap of 10,000 in our pipeline. **Why:** removes the 5-star skew so the model has to learn all five classes. **Consequence:** the model is trained balanced but real data is skewed. That mismatch is why a simple class-balance correction raises scores on the skewed test set (see slide 8).
- **VGST (Variant Greedy Search Technique):** the paper's way of choosing which reviews to keep when you cannot train on everything. The idea is diversity. Instead of random sampling, keep the reviews that add the most new vocabulary. The paper's description: randomly pick a batch of candidates, find the one that adds the most new word-pieces (tokens) to what has already been selected, add it to a "wishlist", repeat until the wishlist is the target length, then choose the best from it. Parameters in our implementation: batch size 32, wishlist length 10. If no candidate adds anything new, fall back to random. **Why the paper used it:** reviews are repetitive ("great product, works well"), so random sampling wastes training rows on near-duplicates; VGST spends them on varied language. **Why "variant":** plain greedy search scans the entire dataset every step, which is slow; looking only at small random batches is a cheap approximation. **A known side effect:** because it counts new token ids, it favours long reviews. In the demo notebook, VGST-selected reviews average about 139 words against about 63 for a random sample, and cover about twice the vocabulary. So part of the "diversity" is just length. **Our honest position:** we implemented it as described and ran the pipeline, but we did not run an experiment comparing training with and without VGST, so we cannot claim it improves accuracy. We only claim we followed the paper's method.
- **Oversampling the neutral class:** the paper deliberately increases 3-star reviews because neutral is hardest to tell apart. We duplicate them (roughly doubled). **Why:** more examples of the hardest class. **Limit:** duplicates add no new information; they just weight the class more.
- **Cap at max_samples:** shuffle and keep the first N rows. The paper's Table 1 says 2,000, its text also mentions 4,000 and 5,000, and our pipeline default is 4,000. We say "capped (paper Table 1: 2,000)" to stay safe.

### The model and training (slide 6)
- **LLaMA-3-8B:** an open-source language model with 8 billion parameters. It generates text, so we frame classification as writing the answer "Sentiment (1-5): 4".
- **Fine-tuning:** continue training the model on our labelled reviews so it gets better at this one task.
- **LoRA (Low-Rank Adaptation):** training all 8 billion weights is expensive. LoRA freezes the original weights W and learns a small correction added on top: h = Wx + BAx, where B and A are two thin matrices. If a weight matrix is d × d, a full update has d² numbers; LoRA uses two matrices of size d × r and r × d with r tiny, so only 2·r·d numbers. A starts as random noise and B starts at zero, so at the start the model behaves exactly like the original. **Our setting (the paper's Table 1):** rank r = 4, alpha = 64 (the correction is scaled by alpha/r = 16), applied only to the query and value projections (q_proj, v_proj) in the last three transformer layers (29–31). That works out to roughly 160,000 trainable numbers, about 0.002% of the model, and the adapter file is only about 640 KB.
- **Epochs / batch / gradient accumulation:** 3 passes over the data, 3 reviews per step, accumulating 4 steps before updating (effective batch 12). **Learning rate 5e-5**, cosine schedule (the rate decays smoothly to zero), fp16 (half-precision numbers to save memory).
- **Quantisation (4-bit):** storing the frozen base weights in 4 bits instead of 16 so an 8-billion-parameter model fits on a modest GPU. Slight accuracy cost.

### Prompting
- **System instruction + rubric:** the model is told to evaluate the sentiment and is given a plain-language description of what 1, 2, 3, 4, 5 stars mean (paper §2.2).
- **One-shot example:** one worked example (a review with its answer) placed in the prompt so the model sees the format. **Caveat in our code:** the example is chosen using the true rating of the row being tested (we pick one with a different rating), a mild label leak inherited from the earlier notebook.
- **Zero-shot chain of thought (CoT):** adding the phrase "Let's take it one step at a time" nudges the model to reason briefly before answering.
- **Greedy decoding:** always pick the single most likely next word (temperature 0), so results are repeatable.

### Phase 2 pieces (slides 10–11)
- **Analyst:** the Phase 1 model produces a draft rating.
- **Visual Verifier:** downloads the product image, a captioning model (BLIP) writes a short description, and the LLM re-rates using that caption. **Later finding:** its answers were nearly constant, so it is excluded from the final vote (slide 18).
- **RAG Prover (retrieval-augmented generation):** looks up three similar past reviews (or product spec text) in a vector database (LanceDB, embeddings from all-MiniLM-L6-v2) and rates using that evidence.
- **Critic and dissonance:** collects the votes and measures how much they disagree: D = standard deviation of the votes divided by 2. If D ≥ 0.4, the Critic writes targeted feedback and the Analyst reruns, at most two times.
- **Vector database:** stores text as numbers (embeddings) so similar texts can be found by distance. We have one with about 7,300 product-spec snippets for 489 Electronics products.

---

## Part 2b — The Phase 1 data pipeline, step by step (what the code does and what to say)

Source: `src/agentic_sentiment/phase1/data/{dataset,sampling,preprocessing}.py`, defaults in `phase1/config/settings.py`. The whole point of the pipeline is to turn a huge, skewed, noisy review file into a small, clean, balanced training set. Run order: load, TextBlob filter, stratified sample, VGST, neutral oversample, cap, split, prompt, tokenise and mask.

**Code defaults to quote if asked:** read the first 250,000 CSV rows; stratified pool capped at 10,000; final cap `max_samples` = 2,000 in the code default (the paper's Table 1 says 2,000, its text also mentions 4,000 and 5,000); 80/20 train/validation split; max sequence length 1,024 tokens.

### Step 0 — Load
Reads the first 250,000 rows of the Kaggle CSV, finds the text and rating columns by name, forces the rating to an integer between 1 and 5 (an unreadable rating becomes 3), and stores each row as `{text, rating}`.
- *Why:* one clean format, and the row cap keeps memory within what Colab allows.
- *Caveat:* it takes the first N rows in file order, not a random draw, so it inherits any ordering in the file.

### Step 1 — TextBlob data-quality filter (DQC)
TextBlob scores each review's polarity from -1 to +1 by counting words from a built-in sentiment dictionary ("great" is positive, "terrible" is negative).
- *Rule:* drop the row if polarity is below 0 and the rating is above 3 (negative words, high stars), or if polarity is above 0 and the rating is below 3 (positive words, low stars). Polarity exactly 0 is kept with any rating, and rows where TextBlob fails are kept.
- *Why:* when words and stars contradict, the label is probably a mis-click or a mislabel. Removing the clearest cases gives cleaner labels.
- *Caveat:* TextBlob is a word counter. It cannot read sarcasm, negation ("not good") or mixed reviews ("lovely but it broke"). So the filter also deletes genuinely hard, mixed reviews. The result is a training set that is easier than real data, which is one reason the paper's headline number looks optimistic.
- *Likely question:* "Is TextBlob's polarity the same thing as your polarity metric?" No. The TextBlob polarity is a dictionary score of the text. The polarity metric (negative, neutral, positive buckets of the star rating) is an evaluation view. They only share a name.

### Step 2 — Stratified sampling
Groups rows by star, shuffles each group, and takes the same number from each. With a cap of 10,000 that is 2,000 rows per star.
- *Why:* about 64% of Amazon reviews are 5-star. A model trained on that learns to say "5" and still looks accurate. Equal classes force it to learn all five.
- *Caveat:* the model is now trained on balanced data but will be used on skewed data. That is a prior mismatch. It is why a simple class-balance correction changes our scores on the skewed test set (slide 8).

### Step 3 — VGST (Variant Greedy Search Technique)
Chooses a diverse subset instead of a random one.
1. Keep a running set of every token id already in the chosen sample.
2. Build a wishlist of 10 candidates. For each, draw a random batch of 32 reviews and keep the one that adds the most new token ids.
3. From the wishlist add the review with the most new tokens to the sample, and update the running set.
4. Repeat until the target size is reached. Target size is the larger of `max_samples` and 1% of the post-filter pool, capped at the current row count. If the target is not smaller than the pool, VGST is skipped.
- *Why "variant":* a true greedy search would scan the whole dataset at every step, which is far too slow. Sampling small random batches approximates it cheaply.
- *Why diversity:* reviews are repetitive ("great product, works well"). Random sampling spends training rows on near-duplicates. VGST spends them on varied language.
- *Caveat 1:* "most new token ids" favours long reviews, because a long text simply contains more distinct tokens. In the demo, VGST picks average about 139 words against about 63 for random picks. Part of the "diversity" is length.
- *Caveat 2:* the running set only grows, so late picks must be increasingly unusual, which drifts toward rare words and odd text.
- *Caveat 3:* it measures vocabulary variety, not difficulty or meaning. We did not run training with and without VGST, so we cannot say it improves accuracy.

### Step 4 — Oversample the neutral class
Adds extra 3-star rows, drawn with replacement, so the 3-star count doubles (ratio 2.0).
- *Why:* the paper says neutral is subtle and deliberately boosts it.
- *Caveat:* the extras are exact duplicates. They add weight, not information, and can encourage memorising those reviews. They also nudge the model toward answering 3.

### Step 5 — Cap
If more than `max_samples` rows remain, shuffle and keep that many.
- *Caveat:* a random cut after oversampling, so the final star mix is close to, but not exactly, equal. The exact rows used to train the committed adapter were not recorded.

### Step 6 — Split and build the prompt
Shuffle, split 80/20 into train and validation, and build a one-shot pool from the training part only (one random example per star where possible). Each row becomes a prompt and an answer.
- *Prompt:* the task instruction ("1-2 negative, 3 neutral, 4-5 positive"), the line "Let's take it one step at a time.", one worked example, "Review to classify:" plus the review, and a request to end with a line starting "Sentiment (1-5):".
- *Answer:* the rating and its description, for example "4. Users have a positive experience overall, with room for improvement (rating 4)."
- *Why the full sentence and not a bare digit:* so training matches the format used at inference.

### Step 7 — Tokenise and mask the prompt
Prompt and answer are tokenised together, cut at 1,024 tokens. Every prompt token gets label -100, which means "ignore in the loss". So the model is graded only on the answer line, not on copying the prompt. Each batch is padded to its own longest example, which changes speed only, not the maths.
- *Caveat:* the one-shot example comes from the training pool, so a training row can be shown its own review as the example.

### One-sentence summary for the room
"About 250 thousand raw reviews become about 2,000 that are label-clean by TextBlob, equal across stars, varied in vocabulary and boosted in neutral, and the model is then tested on raw, skewed reviews, which is a harder and more realistic job than the one it practised."

### Is Phase 2 preprocessed the same way?
No. Phase 2 does not train anything, so it skips TextBlob, VGST and oversampling. Its test rows are class-balanced, need product metadata and at least 20 characters of text, and are not filtered for sarcasm or mixed sentiment. Its Reflector is fit on 500 separate labelled dev rows and tested on 1,000 held-out rows. State this because it is a fair question: "why is the training data filtered but the test data not?" Answer: filtering the test set would remove exactly the hard cases we want to measure.

---

## Part 2c — Every metric explained, with one worked example

### A toy example (10 reviews) you can put on the board
| Review | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| True stars | 5 | 5 | 4 | 4 | 3 | 3 | 2 | 2 | 1 | 1 |
| Predicted | 5 | 4 | 4 | 5 | 3 | 4 | 1 | 2 | 1 | 2 |

Walk through every metric on this one table:

**1. 5-class exact accuracy = 5 / 10 = 50%.** The prediction equals the true star in reviews 1, 3, 5, 8, 9. This is the strictest metric: predicting 4 when the truth is 5 is simply wrong. It answers "did the model pick the exact star?"

**2. Within ±1 star = 10 / 10 = 100%.** Every miss was one star off. This answers "when it is wrong, is it nearly right?" For us it is about 92%, which shows nearly all of the errors are neighbouring-star swaps.

**3. MAE (mean absolute error) = 0.5 stars.** Five misses of one star each, divided by 10. Lower is better. It treats a 1-versus-5 blunder (4 stars off) as eight times worse than a 4-versus-5 slip, which exact accuracy cannot see.

**4. Polarity (3-class) = 9 / 10 = 90%.** Fold stars into three buckets: 1-2 negative, 3 neutral, 4-5 positive. Only review 6 (true 3, predicted 4) crosses a bucket. Swaps inside a bucket (5 versus 4, 2 versus 1) are forgiven. This is the paper's own bucket definition (§2.2). It answers "did we get the overall feeling right?"

**5. Binary (3-star dropped) = 8 / 8 = 100%.** Remove neutral reviews, then ask positive versus negative only. This is the easiest framing and gives the highest numbers. Reviews 5 and 6 are removed here.

**6. Precision, recall and F1 for one star.** Take 4-star:
- Reviews predicted 4: numbers 2, 3, 6. Only review 3 really is 4, so *precision* = 1/3 (when it says 4, it is right a third of the time).
- Reviews that really are 4: numbers 3 and 4. Only review 3 was caught, so *recall* = 1/2 (it finds half of the real 4-star reviews).
- *F1* = 2 x (1/3 x 1/2) / (1/3 + 1/2) = 0.40. It is the harmonic mean, so it is low unless both precision and recall are decent.
- **Macro-F1** averages F1 over the five stars with equal weight, so a weak rare class drags it down. **Weighted-F1** weights each star by how many reviews it has, so it follows the common stars.

**7. Trivial baseline.** The score from always guessing the most common answer. In the toy table "always 5" would score 20%. On our paper-style test set 63.8% of reviews are 5-star, so always-5 scores 63.8% exact, and always-positive scores 78.6% polarity. A number only means something next to its baseline.

### How the same model gives very different numbers
Same predictions, different ruler. This is the central message of slide 8.

| Metric | What counts as correct | Our LLaMA on the paper-style 4,000-row set | On our balanced 1,000-row set |
|---|---|---|---|
| 5-class exact | exact star | 70.7% | 60.1% (Phase 2: 65.2%) |
| Within ±1 | off by at most 1 | 91.7% | 92.0% (Phase 2: 93.6%) |
| Polarity (3-class) | right bucket | 85.0% | 79.2% (Phase 2: 81.1%) |
| Binary | positive versus negative, 3-star dropped | not the paper-style headline | 90.8% (Phase 2: 93.8%) |
| Trivial "always positive" | | 78.6% | 40% (4 and 5 stars are 2 of the 5 equal classes) |

Two reasons the numbers move:
1. **Coarser metric = higher number.** Exact, then polarity, then binary, are progressively more forgiving. The paper reports 92.6%, which our results suggest is a polarity-style number on a skewed test set, not a 5-class exact number.
2. **Test set mix.** On a skewed set (64% 5-star) an easy majority class lifts every score. On a balanced set each star counts equally, so the hard middle stars show up. Balanced numbers are lower and more honest about weakness.

**What to say:** "Our model's 85.0% polarity on the paper-style set is about the same as an SVM's 84.5%, so we did not reproduce the claim that LLaMA clearly beats classical models. On the harder balanced test the 5-class exact accuracy is 60%."

### Which metric answers which question
- *Is it basically right?* Polarity or binary.
- *Is it exactly right?* 5-class exact.
- *When wrong, how wrong?* Within ±1 and MAE.
- *Is it fair to every star?* Per-star recall, and macro-F1.
- *Is the score impressive?* Compare with the trivial baseline.

### Likely questions on metrics
- **"Why report so many?"** Because one number hides things. Exact accuracy alone makes a model that is nearly always one star off look bad. Polarity alone hides all star confusion.
- **"Which is the real accuracy?"** For a 5-star task, 5-class exact on a balanced set. The others are supporting views.
- **"Is 65% good?"** Against 20% chance on balanced classes, yes. Against 80% polarity-style claims in the literature, it is a different, harder measurement. Say which ruler you are using each time.
- **"How do you know Phase 2's gain is real?"** Paired bootstrap on the same 1,000 held-out rows: +5.0 points, 95% confidence interval +2.5 to +7.4. It fixed 105 rows and broke 55 (McNemar p = 0.0001).

---

## Part 3 — Slide-by-slide script

### Slide 1 — Title
**Say:** "This project asks whether an LLM sentiment classifier can check itself the way a careful reviewer would. I'll show a faithful reproduction of a published baseline, what an agentic wrapper does to it, what we found when we audited our own system, and where we go next."
**Explain:** The subtitle bar at the bottom is the roadmap of the talk: problem, baseline, agentic system, results, what we learned, reflective v2.
**Caveat:** none yet.

### Slide 2 — The problem: why single-pass LLMs fail
**Say:** "A single pass has no second look. Three things go wrong: it can't check claims against evidence, it ignores context like images and product details, and it trips on sarcasm and mixed reviews."
**Explain:** Example for each. *Claims:* "this watch isn't waterproof" could be a defect or a user mistake, and nothing in the prompt can settle it. *Context:* a one-line review next to a photo of a shattered screen. *Language:* "Amazing, it broke in two days" is negative but contains a positive word.
**Transition:** "So the gap is a system that adds evidence and a review loop."
**Likely question:** "Is that really a problem with 5-star ratings?" Answer: partly; the real measurable failure turns out to be neighbouring-star confusion (slides 19), which is what we target.

### Slide 3 — Research question and gap
**Say:** "The baseline is Wang et al. 2025: LLaMA-3-8B with LoRA on Amazon reviews, reporting 92.6% accuracy. It stops at a single pass, text only, no retrieval. We call those three gaps: gullibility (no evidence), critic (no auditing), and modality (text only)."
**Explain:** The left panel is what the paper establishes: LoRA fine-tuning, TextBlob quality filtering, VGST sampling, one-shot plus chain-of-thought prompting. The right panel is what it leaves open.
**Caveat:** "I'll show later that the 92.6% is hard to compare directly with our numbers."

### Slide 4 — Data and experimental protocol
**Say:** "Phase 1 trains and tests on the Kaggle reviews, around half a million rows, heavily skewed to 5-star. Phase 2 needs product metadata and images, which Kaggle does not have, so it uses Amazon Reviews 2023: All Beauty in Run 1 and Electronics in Run 2."
**Explain:** The challenge on the Kaggle side is bias and noisy labels; the fix is the five-step pipeline on the next slide.
**Caveat (important):** The Phase 1 adapter never saw Amazon 2023 data. So Phase 1 versus Phase 2 on that data is a transfer test, not a same-distribution test. Say this out loud.

### Slide 5 — Phase 1 data preprocessing pipeline
**Say:** "Five steps turn the raw corpus into a clean, balanced, diverse training pool."
**Walk the steps:**
1. *TextBlob DQC:* drop reviews whose polarity contradicts the rating. Cleaner labels. (Limit: it's a crude word-list scorer.)
2. *Stratified sampling:* equal reviews per star. Removes the 5-star skew.
3. *VGST:* greedy diversity selection. Keep the reviews that add the most new vocabulary, so the training set isn't full of repeats. (Explain batch 32, wishlist 10 if asked.)
4. *Oversample neutral:* double 3-star reviews because neutral is the hardest class.
5. *Cap:* shuffle and keep the first N rows.
**Caveat:** "We implemented each step as the paper describes it. We did not run an ablation on any single step, so I'm claiming faithfulness, not that each step improves accuracy."
**Likely question:** "Why does balancing matter?" Because trained on skewed data, the model learns to say 5 for everything and still looks accurate (64% of the data is 5-star).

### Slide 6 — Phase 1: LLaMA-3-8B + LoRA fine-tuning
**Say:** "We freeze the 8-billion-parameter model and learn a tiny correction, called LoRA. The formula h = Wx + BAx says the output equals the original layer plus a small low-rank update."
**Explain with numbers:** With rank 4 applied to the query and value projections in the last three layers, about 160 thousand numbers are trained, roughly 0.002% of the model, and the saved adapter is about 640 KB. At the start B is zero, so the model begins identical to the base model.
**Hyperparameters (right table):** learning rate 5e-5 with a cosine schedule, 3 epochs, batch 3 with gradient accumulation 4, rank 4, alpha 64, last 3 layers (29–31) on q_proj and v_proj, sequence length 1,024, 4-bit base weights.
**Caveat:** "An earlier version of this slide listed a stronger configuration (rank 16, 8 layers). The audit showed that configuration was only ever a plan; the adapter we actually use matches the paper's Table 1, so the slide now says that." Also: "which rows the adapter trained on, and the exact count, were not recorded."

### Slide 7 — Phase 1 baseline results (balanced evaluation)
**Say:** "On a class-balanced set of 1,000 reviews, 200 per star, the model gets 60.1% exact, 92% within one star, and a mean error of half a star. Guessing would score 20%."
**Explain:** Exact accuracy is a strict five-way test. Coarser views are much higher: 79.2% on three classes and 90.8% on binary (neutral dropped). The model reads direction well; what it struggles with is telling neighbouring stars apart.
**Caveats:**
- The paper's 92.6% used a different protocol (K=5 cross-validation, 128K training rows per fold, a separate 40K test set), so it is not a like-for-like comparison.
- An earlier figure of 66.6% came from a skewed test set where always guessing 5-star already scores about 63%, so we don't use it.
**Likely question:** "Why is this lower than the paper?" See slide 8.

### Slide 8 — Reproducing the paper's claims
**Say:** "Here are four models scored on the same 4,000 random held-out reviews, with the natural 64% five-star mix, using the paper's polarity buckets."
**Walk the table:**
- Always guessing the majority: 63.8% five-class, 78.6% polarity. Everything is judged against this.
- Decision Tree 58.5 / 75.3, SVM 70.0 / 84.5, Naive Bayes 67.4 / 83.1 (the paper reports 78.6, 87.2, 87.6).
- XLNet: not re-run; the paper's 91.6% is shown for reference.
- Our LLaMA: 70.7% five-class, 85.0% polarity, against the paper's 92.6%.
- Last row: with a class-balance correction, 79.9% and 92.0%.
**The key insight:** the paper's classical baselines land close to our polarity numbers, not our five-class ones. That is the evidence that the paper's headline is a polarity-style number on a skewed test set. The remaining gap comes from training scale (the paper used up to 128K training rows per fold; our classical runs used 16K).
**Explain the last row:** the adapter was trained on balanced data but the test set is 64% five-star, so a small correction shifts predictions toward 5-star. It is a class-balance fix, not better understanding of language. On the balanced set the same correction lowers accuracy (60.2% to 57.9%), which is how we know. Don't headline it.
**Caveats (read from the slide):** training scale differs; community 4-bit base versus fp16 training weights; one-shot example chosen using the true rating; training rows not recorded; the correction row's definition.
**Honest claim to make:** "LLaMA is level with SVM here. The paper's claim that it beats the classical models by four or five points does not reproduce at our scale."
**Likely question:** "So did you reproduce the paper?" Answer: the method, yes, down to the hyperparameters. The headline number, no, and here is the evidence for why.

### Slide 9 — Data scale: why the paper's numbers are higher
**Say:** "We trained simple classifiers on 1,000 up to 128,000 reviews and scored them on the same 4,000 test rows."
**Explain:** Each four-times increase in data adds about three to four points, and the curve is still rising at 128K. At the paper's scale a plain TF-IDF logistic regression reaches 90.4% polarity, close to the paper's 92.6%. Our LLaMA, trained on only 2–4K rows, beats the classical models at the same size.
**Caveat:** classical models trained on the natural distribution, so part of their gain is learning the 64% five-star prior; our adapter was trained balanced.
**What it means:** more training data is a real lever, which is why we plan a larger run. LLMs usually gain less per extra row than classical models, so expect a modest gain. We have not measured it for LLaMA.

### Slide 10 — Phase 2: the multi-agent pipeline
**Say:** "Phase 2 wraps the same model in three additional agents. The Analyst is the Phase 1 model giving a draft rating. The Visual Verifier looks at the product photo through a caption. The RAG Prover retrieves similar reviews or product specs. The Critic compares the votes."
**Explain:** If the votes disagree enough (a score called dissonance reaches 0.4), the Critic writes specific feedback and the Analyst reruns, up to two more times.
**Caveat:** "The Visual Verifier still runs but its vote is no longer used in the final decision; I'll explain why on the audit slide."
**Run 1 fact:** 922 correction events across 669 high-disagreement samples.

### Slide 11 — How disagreement triggers reflection
**Say:** "The Critic turns three votes into a number. D is the standard deviation of the votes divided by two. If all agree, release. Low disagreement, majority vote. D at or above 0.4, the correction loop, at most twice."
**Explain Run 1 numbers:** 922 events, 669 samples (33.5% of 2,000), mean dissonance 0.211, maximum 0.471.
**Why a threshold:** correcting every sample would triple cost and risk changing correct answers.

### Slide 12 — The comparison: does the agentic system improve? (Run 1)
**Say:** "On 2,000 All Beauty reviews, accuracy went from 16.1% to 38.7%, macro-F1 from 0.095 to 0.273, and the average miss from 1.52 stars to 0.96, at 3.7 times the latency."
**Caveat (must say):** This is a transfer setting: a Kaggle-trained model applied to a new category. The Phase 1 baseline collapsed (it heavily under-predicted), so the gain reads as "context and agents rescue a collapsed baseline", not a general twenty-point improvement.
**Also true:** 491 errors recovered and 39 regressed.
**Likely question:** "Why was the baseline so low here?" Different product category and vocabulary; the adapter had only ever seen Kaggle reviews.

### Slide 13 — Where the improvement appears (confusion matrices)
**Say:** "On the left, the baseline collapses onto a few ratings; four-star recall is only 6.6%. On the right, Phase 2 puts more weight on the diagonal."
**Explain how to read a confusion matrix:** rows are the true star, columns are the predicted star; a perfect model has numbers only on the diagonal.
**Note:** four-star remains hard in both. This foreshadows slide 19.

### Slide 14 — Which ratings improve, and which remain difficult
**Say:** "The biggest gain is on 3-star, from 45.8% to 97.9%. One-star and five-star also improve. Two-star and four-star barely move."
**Explain:** per-class F1 on the left, gain per class on the right. n values: 147, 105, 190, 377, 1,181.
**Caveat:** the 3-star jump is partly the system defaulting to 3 when unsure, so it is not purely better reading.

### Slide 15 — Agent dissonance: where does reflection matter?
**Say:** "Dissonance is a proxy for how hard a sample is. Gains show up at every level, not just where correction fired."
**Numbers:** +23.9 points when agents agree, +36.0 at moderate disagreement, +11.7 where correction triggered.
**Caveat (read the footer box):** the overall gain is not mainly from the correction loop; adding context (metadata and the caption) is the dominant observed mechanism, and no component-level ablation was run. Say this plainly.

### Slide 16 — Why Phase 2 improves: three observed mechanisms
**Say:** "Three mechanisms, stated as observations and not proven causes."
1. *Context breaks rating collapse:* 90.2% of Phase 1 errors were under-predictions, and 288 of the 491 recoveries were a 3-star guess corrected to 5-star.
2. *Multiple evidence sources reduce Analyst bias:* the Analyst alone was right on 64.9% of the samples Phase 2 got right; the RAG context lined up on 91.2%.
3. *Self-correction targets ambiguity:* 922 events across 669 samples; +16.6 points on corrected samples versus +27.7 on uncorrected; 12.6 recoveries per regression.
**Caveat:** the Visual Verifier's earlier 97.7% alignment figure is no longer shown because the later audit found its output nearly constant.

### Slide 17 — Evaluation framework and latency trade-off
**Say:** "Phase 2 costs 2.83 seconds per sample against 0.77, about 3.7 times, because it makes three to five model calls."
**Explain:** the right panel lists the metrics: accuracy, F1 and MAE for both phases; dissonance, per-agent votes and error recovery for Phase 2 only.
**Honest point:** latency is a cost of Phase 2. The next design reduces it by only reflecting on uncertain rows.

### Slide 18 — What the audit found and fixed
**Say:** "We then audited our own system on a second dataset (Electronics) and found it had regressed: 46.4% versus the baseline's 50.95%. We traced four real defects."
1. *Over-triggered correction:* a rule meant to flag poor grounding forced reflection whenever retrieval looked weak, even when the agents agreed. It fired on 78.6% of rows. Now only real vote disagreement triggers it.
2. *Tie-break favoured the weak vote:* when two agents disagreed, the weaker retrieval vote overruled a correct Analyst about half the time (retrieval alone 37.5% versus Analyst 54.2%). Ties now go to the Analyst.
3. *Prompt missing framing:* the retrieval agent lacked the one-shot and chain-of-thought framing; adding it raised its standalone accuracy from 37.5% to 46.9%.
4. *Visual Verifier constant:* it returned 3 stars on all 96 real test rows. A caption describes what the product is, not whether the buyer is happy. We tried three different vision models; all failed the same way, so its vote is excluded (still logged).
**Result:** before the fixes, net minus 91 on rows that triggered correction. After the fixes, on a 96-row real check: 52.08% to 54.17%, nine fixed, seven regressed, net plus two.
**Caveat:** 96 rows is small; treat it as directional.

### Slide 19 — Where the remaining errors live
**Say:** "Exact accuracy is 60.1%, but three-class is 79.2%, binary 90.8%, and within one star 92%. The errors are swaps between neighbours."
**Walk the per-star table (balanced eval):**
- 2-star is weakest: recall 0.34, and 46% of 2-star reviews get called 1-star.
- 3-star drifts to 2-star (22%), 4-star to 5-star (28%).
- 1-star and 5-star are fine (F1 0.70 and 0.72).
- Macro-F1 0.588. Same pattern on the 4,000-row set (2-star recall 0.36, macro-F1 0.548).
**Why it matters:** in the first Phase 2 design the Critic override reached only 0.4% of rows, so reflection barely touched these cases. That is the headroom.
**Also true:** the model is not overconfident; on 4,000 rows no prediction reached 0.9 probability, so the issue is not confident hallucination.

### Slide 20 — Reflective orchestration v2
**Say:** "The redesign makes self-reflection the core mechanism and aims it where the errors are."
1. *Analyst:* rating plus the full probability over 1–5.
2. *Uncertainty gate:* if the top two ratings are close, reflect; if confident, release.
3. *Reflector:* critiques the draft against the star rubric and weighs evidence for the two neighbouring ratings.
4. *Adjudicator:* a binary decision between the two candidates (for example 3 versus 4). We are training a small specialised model for this.
5. *Loop control:* at most two rounds.
Plus *reflection memory:* lessons from corrected mistakes stored and retrieved as guidance.
**Why it should work:** the correct star is in the model's top two on 88–89% of rows, so the headroom is almost entirely a choice between two neighbours.
**The honest bar:** inside the gate the baseline already picks the right one of two about 60% of the time. The adjudicator must beat that clearly (about 68–70%) or it adds nothing.
**Status (say this):** design locked; training of the adjudicator is in progress; no results claimed yet.

### Slide 21 — Conclusion and future work
**Say:** "Phase 1 is a faithful, paper-aligned baseline; the method reproduces but the headline does not, and we know why. Phase 2 is hardened and slightly net-positive after the audit. The next step is reflection aimed at neighbouring-star confusion."
**Future work:** reflective orchestration v2; more Phase 1 training data; ordinal metrics alongside exact accuracy; latency optimisation.
**Close:** "The main lesson is that how you measure matters as much as the model: the same predictions score 60%, 70%, 85% or 92% depending on the test set and the metric."

### Slide 22 — Appendix: run card (backup)
Use only if asked how a number was produced. It lists the model, inference setup, test set and caveats. Full detail in `v2_reflective/PHASE1_RUN_CARD.md`.

---

## Part 4 — Question bank

**Why isn't your accuracy 92.6% like the paper?**
Different protocol. The paper's accuracy appears to be polarity-style on a skewed test set, trained on much more data. Under the same polarity scoring on the same rows we get 85.0%, and the classical baselines also land below the paper's values. The remaining gap is mostly training scale.

**Is the 79.9% / 92.0% result just cheating?**
It is a class-balance correction fit by cross-validation. It is legitimate for a skewed deployment distribution, and we disclose it. It is not better language understanding; on the balanced set it lowers accuracy.

**What is VGST and did it help?**
A greedy method to choose diverse training reviews: from small random batches it picks the review that adds the most new vocabulary, repeated to fill a wishlist. We implemented it as the paper describes. We did not test it against random sampling, so we do not claim it helps accuracy.

**Why does TextBlob filtering make sense when it is so crude?**
It is a cheap consistency check: it removes the clearest cases where the star rating contradicts the words. It also removes sarcasm, so the training data is somewhat easier than real data.

**Why LoRA instead of full fine-tuning?**
Full fine-tuning needs to update eight billion numbers. LoRA trains about 160 thousand, fits on a small GPU, and saves a 640 KB file. The paper uses it.

**Why only the last three layers and only q and v?**
That is the paper's Table 1 configuration. We kept it to reproduce the method. A stronger configuration exists as a plan but was never trained.

**Does self-correction actually help?**
In Run 1 the overall gain came mostly from added context, and no ablation isolated the correction loop. After the audit fixes the net effect is slightly positive on a 96-row check. That is why the next design changes what triggers reflection.

**Why did you drop the Visual Verifier?**
It returned a constant rating regardless of the image, with three different vision approaches. A product photo mostly shows the product looking normal; dissatisfaction usually comes from things a photo cannot show. It still runs and is logged.

**What are the "hallucinations" you mention?**
We have not yet measured generated-reasoning hallucination. What we measured: the model is not overconfident (no prediction reached 0.9 probability on 4,000 rows), and its errors are neighbour swaps. Case studies of invented details in the reasoning text are planned from a 200-row generation sample.

**Why train on balanced data but test on skewed?**
The paper's protocol does both. Balanced training prevents the model from always answering 5; skewed testing reflects real data. The side effect is the gap that the class-balance correction recovers.

**Would more training data help?**
Probably a little. Classical models gain about 3 to 4 points per four-times more data. LLMs usually gain less. We have not measured it for LLaMA. A local 2,000-row retrain is queued, but it is a reproduction check of the committed adapter, not an improvement test; a real data-scale test needs a larger run.

**What does "answer-only loss" mean?**
During fine-tuning the model can be scored on every token or only on the answer. The original Phase 1 training already scored only the answer (prompt tokens are masked). Our early local experiments on the Mac did not, so they were weaker. The new local runs mask the prompt like the original, which makes them a fair reproduction rather than a new improvement.

**What is the adjudicator?**
A small fine-tuned model that answers one question: "is this review N stars or N+1?" It is trained on neighbouring pairs, and the plan is to call it only when the main model is torn between two ratings.

**What would make Phase 2 fail?**
If the adjudicator cannot beat the main model's roughly 60% pick rate between its top two choices. We test that first, and we report the result either way.

**Latency?**
Phase 2 is slower (3.7 times in Run 1). The next design only reflects on uncertain rows, roughly two thirds of rows at most.

**Why is Run 2 lower than Run 1?**
Different dataset, different baseline behaviour, and the Run 2 pipeline had the defects described on slide 18. Run 1's baseline had collapsed, which exaggerated the gain.

**What are the limits of your evidence?**
Small test sets for some checks (96 and 300 rows), a community 4-bit base, a one-shot example chosen with the true rating, unrecorded training rows, and no ablations of VGST or the quality filter.

---

## Part 5 — Caveats checklist (say each at least once)
1. Phase 1 applied to Amazon 2023 is a transfer test.
2. 60.1% is on a balanced set; the paper's 92.6% used a different protocol and looks polarity-style.
3. Run 1's 16.1% to 38.7% started from a collapsed baseline and is not a general claim.
4. The correction loop's contribution was not isolated by an ablation.
5. The post-fix result is 96 rows, so directional.
6. The class-balance correction is a prior fix, not better understanding.
7. The 4-bit community base differs from the fp16 weights used in training.
8. The one-shot example is chosen using the true rating (mild label leak).
9. VGST and the TextBlob filter were implemented as in the paper but not ablated.
10. v2 (reflective orchestration) is a design; no results claimed yet.

## Part 6 — Numbers cheat sheet

| Item | Value |
|---|---|
| Paper (Wang et al.) accuracy / precision / F1 | 92.6% / 95.3% / 93.3% |
| Our LLaMA, paper-style test (4,000 rows) | 70.7% exact, 85.0% polarity, 91.7% within-1, MAE 0.422 |
| Trivial baselines on that test | 63.8% (always 5), 78.6% (always positive) |
| Our LLaMA, balanced eval (1,000 rows) | 60.1% exact, 92.0% within-1, MAE 0.505, 79.2% 3-class, 90.8% binary |
| Classical baselines, paper-style test | DT 58.5 / 75.3, SVM 70.0 / 84.5, NB 67.4 / 83.1 (exact / polarity) |
| Logistic regression at 128K rows | 78.7% exact, 90.4% polarity |
| Class-balance correction on LLaMA | 79.9% exact, 92.0% polarity (balanced eval drops 60.2% to 57.9%) |
| Per-star recall, balanced eval | 1★ 0.86, 2★ 0.34, 3★ 0.59, 4★ 0.47, 5★ 0.75 |
| Macro-F1 | 0.588 (balanced), 0.548 (paper-style test) |
| Correct star in model's top two | 88.4% (dev) / 89.0% (eval) |
| Run 1 (All Beauty, 2,000 rows) | accuracy 16.1% to 38.7%, macro-F1 0.095 to 0.273, MAE 1.52 to 0.96, latency 0.77 s to 2.83 s |
| Run 1 recovery | 491 fixed, 39 regressed; 922 correction events over 669 samples (33.5%) |
| Run 2 (Electronics) before fixes | text-only 50.95%, full graph 46.40% |
| After fixes (96 real rows) | 52.08% to 54.17%, 9 fixed, 7 regressed |
| LoRA | r 4, alpha 64, layers 29–31, q and v, about 160K parameters, about 640 KB |
| Training | lr 5e-5, 3 epochs, batch 3 x 4, fp16, cosine |
| Dissonance | D = std(votes) / 2; threshold 0.4; at most 2 corrections |
| Vector DB | 7,301 spec snippets, 489 Electronics products |
