
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Current repository state (read this first)

The original Phase 1/2 milestones below (Run 1) were built with **no local
build/lint/test tooling** — no `requirements.txt`, no local venv, no `.py`
source, everything living as notebook cells run on Colab (this machine is
macOS with no CUDA GPU — see §1 of the mission spec below for why). That's
no longer the whole story: `src/agentic_sentiment/` is a real, locally
testable Python package (pytest, `.venv/`, `pyproject.toml`) added for the
full Phase 2 rebuild (see the table row below) — GPU-only paths still run
on Colab via the four notebooks in `colab/`, but parsing, dissonance math,
the LangGraph routing, checkpoint/resume, and classical baselines all have
real local tests. Run them with:
```bash
.venv/bin/pip install -e ".[dev]" && .venv/bin/pytest
```
The only thing besides that package that builds locally is the paper(s).

**Build the thesis-style paper:**
```bash
cd paper && pdflatex main.tex && pdflatex main.tex   # run twice to resolve the TOC/refs
```
Only `paper/main.tex` and `paper/main.pdf` are tracked in git; `main.aux`,
`.log`, `.out`, `.toc` are local build byproducts (untracked, regenerate
freely).

**Build the EAI/LNCS conference paper** (shorter, Run 1 results, lives at
`paper/conference/ConferencePaper.tex`):
```bash
cd paper/conference && pdflatex ConferencePaper.tex && pdflatex ConferencePaper.tex
```
Figures are under `paper/conference/figures/`. Same untracked-byproduct rule
applies to its `.aux`/`.log`/`.out`.

**All four milestones from §0 below are substantially complete**, not
greenfield:

| Milestone | File | Status |
|---|---|---|
| Phase 1 baseline | `colab/llama_sentiment_baseline_train.ipynb` | Implemented: TextBlob DQC → stratified sampling → VGST → neutral oversampling → LLaMA-3-8B+LoRA (r=16, α=128, 5 epochs) → one-shot+CoT eval |
| Phase 1 vs 2 scaffold + full agentic run | `colab/phase2_agentic_full_comparison.ipynb` | **Completed and executed** — Run 1, 2,000 samples from Amazon Reviews 2023 `raw_review_All_Beauty`, seed 42. Results checkpointed at `results/run1/phase2_checkpoint.jsonl` (2,000 rows) + 9 PNG charts in `results/run1/`. |
| Context ablation (bonus, §3.6a) | `colab/phase1_vs_phase2_amazon2023_presentation.ipynb` | Implemented: text-only vs. single enriched prompt (metadata + BLIP caption), no agent loop |
| Small LangGraph demo | `colab/phase2_agentic_vs_phase1_amazon2023.ipynb` | Implemented as a real `langgraph.graph.StateGraph`, but as a smaller-scale demo with **toy RAG grounding** (metadata string, not real vector search) |
| Final paper | `paper/main.tex` / `main.pdf` | Drafted with real Run 1 numbers already filled in (not a template) |
| Full Phase 2 rebuild (LangGraph + spec RAG + Norm dissonance) | `src/agentic_sentiment/` + `colab/01_train_phase1.ipynb`..`04_report.ipynb` | Code complete, unit-tested locally with a stub LLM (56 tests passing); **Run 2 numbers pending** — GPU steps (LoRA retrain, XLNet, live Phase 2 eval on Electronics) run on the user's Colab, not here. Design: `docs/superpowers/specs/2026-10-02-full-phase2-design.md`; plan: `docs/superpowers/plans/2026-10-02-full-phase2-plan.md`. |

**Important architecture wrinkle — read before touching Phase 2 code:** the
notebook that actually produced the paper's numbers,
`colab/phase2_agentic_full_comparison.ipynb`, does **not** use the
`langgraph` library despite being described as "LangGraph-style" in its
markdown — its 4-agent pipeline (Analyst, Visual Verifier, RAG Prover,
Critic) is plain Python functions operating on an `AgentState` dataclass,
with a manual `if dissonance > threshold` retry loop (`DISSONANCE_THRESHOLD
= 0.4`, `MAX_CORRECTION_STEPS = 2`). The dissonance score is
`stdev(votes)/2` over the three agent ratings, not the paper mission spec's
`Norm(H − (Ev + Gf))` formula. Its RAG Prover retrieves the top-k (k=3)
similar *reviews* via LanceDB + `all-MiniLM-L6-v2` embeddings for empirical
grounding — it does **not** check claims against a product spec store (the
"USB-C vs Micro-USB" style factual mismatch check in §5.3 of the mission
spec below). The *other* Phase 2 notebook,
`phase2_agentic_vs_phase1_amazon2023.ipynb`, is the one with a real
`StateGraph` but only toy grounding, run at smaller scale. Don't assume
either notebook alone matches the full mission spec — if asked to modify
"the Phase 2 pipeline," ask which notebook, since they've diverged.

**Real headline results (Run 1, 2,000 samples, logged in
`paper/main.tex` §"Results — Run 1"):** Phase 1 accuracy 16.1% / macro-F1
9.5% / MAE 1.52 vs. Phase 2 accuracy 38.7% / macro-F1 27.3% / MAE 0.96
(+22.6pp accuracy). Self-correction triggered on 46.1% of samples; net
error recovery +452 samples (491 fixed, 39 regressed). There is currently
no separate `RESULTS.md` — the run log lives directly in `paper/main.tex`'s
results sections instead; if you add new runs, either start `RESULTS.md` as
the mission spec (§7) expects, or keep extending `main.tex` consistently
with what's already there — don't do both in parallel.

**Gaps vs. what the mission spec below assumes exists:** no
`requirements.txt`, no `output/final/` (the trained LoRA adapter lives only
in Colab / as an `output_final.zip` uploaded per-session, never committed),
no `RESULTS.md`, no `CODE_EXPLANATION.md` / `PROJECT_CHANGES_SUMMARY.md`.
Treat every reference to those paths below as aspirational unless you
create them.

---

## Original project mission (verbatim init prompt)

The rest of this file is the original project brief this repo was built
from. It's still the source of truth for *why* things are structured this
way (dataset choices, prompt wording, target metric bands, the Phase
1/Phase 2 dataset split rationale in §3.6) — treat the "Current repository
state" section above as a status update layered on top of it, not a
replacement.

### Agentic Self-Correction in LLaMA-3: Multimodal E-Commerce Sentiment Analysis

Paste this whole document as your first message to Claude Code in the project
repo (or save it as `CLAUDE.md` at the repo root so Claude Code loads it
automatically). The attached paper
(`Agentic_Self-Correction_in_LLaMA3___E-Commerce_Sentiment_Analysis.pdf`)
should also be placed in the repo — reference it as the spec of record.

---

## 0. Mission

Build this project in four sequential milestones, each gated on the
previous one working end-to-end with real results (no placeholder/mock
numbers, no "TODO: run this later"):

1. **Phase 1 — Baseline**: LLaMA-3-8B + LoRA sentiment classifier,
   reproducing the paper's methodology (not necessarily its exact numbers).
   → `colab/llama_sentiment_baseline_train.ipynb`
2. **Phase 1 vs Phase 2 comparison scaffold**: load a multimodal Amazon
   Reviews 2023 slice, run the Phase 1 adapter single-pass as a first
   column, leave a clean insertion point for Phase 2.
   → `colab/phase2_agentic_vs_phase1_amazon2023.ipynb` (same file as #3 —
   built in two passes, not two separate notebooks; see §3.6a)
3. **Phase 2 — Proposed agentic system**: the LangGraph multi-agent
   Verifier-in-the-Loop architecture (Analyst → Visual Verifier + RAG
   Prover → Critic with dissonance scoring and reflection loop), wired into
   the same notebook as #2 as its second column, executed for real.
   → `colab/phase2_agentic_vs_phase1_amazon2023.ipynb` (completed)
4. **Final paper**: once #1–#3 have produced real results, write up a
   paper reporting them, using the attached project paper as the
   structural/stylistic template. See §6.

**Execution environment policy: try local first.** Attempt everything on
this machine. Only fall back to Google Colab if local hardware genuinely
can't run a step (see §1). If you fall back, produce a Colab-ready notebook
(self-contained install cells, no local-path assumptions) and tell me
exactly what to click/run there — don't just assume Colab and skip local
entirely.

Do not stop at "the code looks right." Actually execute every notebook
cell-by-cell, capture real metrics, and iterate on bugs/OOM/data issues
until the target results in §3 and §5 are achieved — milestone #4 depends
on real numbers existing, not projected ones — or you've exhausted
reasonable options and reported why.

---

## 1. Environment check (do this first)

Before writing any training code:

- Detect GPU: `nvidia-smi` (or equivalent). Report VRAM available.
- Report free disk space and RAM.
- Check whether `bitsandbytes` (4-bit quantization) will actually work on
  this OS/driver combo — it's Linux+CUDA only. If this machine is macOS or
  lacks a CUDA GPU, say so explicitly and propose the Colab fallback
  *before* burning time on a doomed local fp16 load.
- Decision rule: LLaMA-3-8B needs ~16GB VRAM in fp16, ~8–10GB in 4-bit.
  If local VRAM < 10GB or bitsandbytes won't load, stop and recommend the
  Colab path (T4 16GB) rather than attempting a broken local run.

Set up a virtual environment (`venv` or `conda`) and pin dependencies in a
`requirements.txt`: `transformers`, `peft`, `bitsandbytes`, `accelerate`,
`torch`, `textblob`, `sacrebleu`, `rouge-score`, `datasets>=3`, `pandas`,
`scikit-learn`, `langgraph`, `langchain`, `lancedb`, `pillow`,
`transformers` (for BLIP), plus anything else you pull in.

---

## 2. Repo layout — inventory before you build anything

**This is not a greenfield repo.** It already contains (at minimum):

```
paper/                      # source paper(s) — includes the PDF this prompt is based on
colab/                      # existing notebooks — the paper's own §10/§17 text
                             #   references colab/llama_sentiment_baseline_train.ipynb,
                             #   colab/phase1_vs_phase2_amazon2023_presentation.ipynb,
                             #   colab/phase2_agentic_vs_phase1_amazon2023.ipynb
                             #   by name — check if these already exist here
results/                    # some prior run outputs
assets/                     # supporting files (images, diagrams, etc.)
README.md                   # existing project description
*.pptx                      # presentation decks already built from this project
itmconf_dai2024_04021.pdf   # CONFIRMED: this is Wang et al. 2025, "Sentiment Analysis of
                             #   Product Reviews Using Fine-Tuned LLaMa-3 Model" (ITM Web of
                             #   Conferences 70, 04021) — the exact baseline paper cited in
                             #   §3.1 of the main project paper. It's the source of the 92.6%
                             #   accuracy / 95.3% precision / 93.3% F1 claim and Table 1's
                             #   hyperparameters. Treat it as the primary methodology
                             #   reference for Phase 1, alongside the main project paper.
```

A few details from Wang et al. that aren't in the main project paper's own
description of §3.1 and are worth Claude Code knowing before it builds Phase 1:

- Wang et al.'s reported 92.6% run used **K=5 cross-validation** with 128K
  train / 32K val per fold, plus a **separate held-out 40K test set** — a
  materially bigger and differently-structured eval than the ~4000-sample
  single train/test split our notebook target (§3.5) uses. This is very
  likely a large part of why the paper's own reference re-run only reached
  ~66% rather than 92.6% — it's not just an implementation gap, it's a
  smaller/different eval protocol. Don't chase the 92.6% number by trying
  to replicate the K=5/40K setup unless I explicitly ask for that — it's a
  much bigger training job.
- Wang et al. also benchmarked **LLaVA1.5-7B** (marginally higher
  BLEU-4/ROUGE-1, ~4x slower) and **Qwen2-7B** (similar accuracy to LLaMA3-8B)
  as alternative backbones and picked LLaMA3-8B as the best
  accuracy/speed tradeoff — useful context if I later ask you to swap
  backbones, but out of scope for now.
- Wang et al.'s classical baselines (Table 3) used specific configs worth
  reusing verbatim if you ever build the optional §17.3 comparison table:
  Decision Tree + CountVectorizer, SVM (sigmoid kernel, gamma=1.0) +
  TfidfVectorizer, Multinomial NB (alpha=0.2) + CountVectorizer, and
  XLNet-Large-Cased.

**Step 0.5, before writing any new code:** open and actually read `README.md`,
list everything in `colab/`, `results/`, and `paper/`, and open any existing
notebooks in `colab/` to see how far this project already got. Specifically
figure out:

- Do `colab/llama_sentiment_baseline_train.ipynb` and the two
  `phase1_vs_phase2...` notebooks already exist? If so, do they run as-is
  locally, or were they written Colab-specific (Drive mounts, `!pip install`
  cells, hardcoded Colab paths)?
- Does `results/` already contain a completed Phase 1 baseline run? If a
  real eval matching §3.5's target band already exists, don't retrain from
  scratch — validate the existing numbers, log them to `RESULTS.md`, and
  move on to Phase 2 unless I say otherwise.
- Does `README.md` describe conventions (naming, folder structure, how to
  run things) that this prompt should defer to instead of the layout below?
- Paper §17.1 also references `CODE_EXPLANATION.md` and
  `PROJECT_CHANGES_SUMMARY.md` as existing repo-root docs from prior work
  on this project — check if they're present and read them too; they may
  contain implementation notes (e.g. exact deviations already made from
  the paper spec) that should inform your inventory report.

**Report this inventory back to me before proceeding** — one paragraph:
what already exists, what state it's in, what's actually missing. Then
adapt the plan in §3–§5 to fill the gaps rather than re-scaffolding
everything that's already there.

**No `.py` files.** Everything — preprocessing, prompt builders, eval
metrics, and the Phase 2 LangGraph agent nodes — lives directly in notebook
cells, not in importable modules. If the same function is needed in more
than one notebook (e.g. the eval metrics used by both Phase 1 and the
Phase 1-vs-2 comparison, or the preprocessing pipeline needed again in
Phase 2's data-loading cells), copy the function's cell into each notebook
that needs it rather than factoring it into a `.py` file and importing it.
Keep each such cell clearly commented so it's obvious it's a duplicate of
"the same DQC/VGST/eval logic as notebook X" — that's the acceptable cost
of staying notebook-only. (If `colab/` already has `.py` helper files from
before, leave them as-is and note it in your inventory report rather than
either deleting them or building new notebooks against them — flag it and
I'll decide.)

```
├── colab/                             # existing + new notebooks live here (matches paper's naming)
│   ├── llama_sentiment_baseline_train.ipynb        # Phase 1 (may already exist — check first)
│   ├── phase1_vs_phase2_amazon2023_presentation.ipynb
│   └── phase2_agentic_vs_phase1_amazon2023.ipynb   # Phase 2 (preprocessing, prompts, eval,
│                                                    #   and all LangGraph agent-node code
│                                                    #   defined in-notebook, cell by cell)
├── output/final/                      # trained LoRA adapter + tokenizer
├── results/                           # existing — add new run outputs here
├── paper/                             # existing
├── requirements.txt
└── RESULTS.md                         # running log of every eval you produce (create if absent)
```

---

## 3. Phase 1 — Baseline (`colab/llama_sentiment_baseline_train.ipynb`)

Implement exactly what the paper's §5–§10 describe. Specifics:

### 3.1 Dataset

- Kaggle `arhamrumi/amazon-product-reviews` (~500k rows: review text,
  rating 1–5, summary, category). If Kaggle API credentials aren't
  configured, tell me what you need rather than silently substituting a
  different dataset.
- Support an optional `max_csv_rows` cap for faster iteration (e.g. 250k).

### 3.2 Preprocessing pipeline (as notebook cells)

Implement in this exact order — each stage is described in the paper §6:

1. **TextBlob DQC**: drop rows where polarity contradicts the rating
   (`pol<0 and rating>3` → drop; `pol>0 and rating<3` → drop; `pol==0` or
   TextBlob failure → keep).
2. **Stratified sampling**: equal count per rating (1–5) up to
   `stratified_max_total` (default 10,000).
3. **VGST (Variant Greedy Search Technique)**: only runs if the stratified
   pool exceeds `max_samples`. Greedy diversity selection: batches of
   `batch_size=32` candidates, `wishlist_len=10` per round, pick the
   candidate adding the most new token IDs vs. the running `current_tokens`
   set; fall back to random fill if no candidate adds new tokens.
4. **Oversample neutral**: rating-3 rows duplicated with `multiplier=2.0`
   (i.e. doubled) via sampling with replacement, then concatenate + shuffle.
5. **Cap `max_samples`**: shuffle, take first `max_samples` (default 4000).

Before running this on the full dataset, sanity-check each stage in its own
cell against a small synthetic in-notebook DataFrame (a handful of hand-built
rows covering each drop/keep case) and print the before/after counts — so a
bug here doesn't silently corrupt 3 hours of training later. This is a
throwaway verification cell, not a formal test file.

### 3.3 Model & training

- Base: `LLaMA-3-8B`, loaded in 4-bit if VRAM-constrained.
- LoRA on `q/k/v/o` projections. Use the paper's Table 1 values as the
  *floor* (`r=4, alpha=64, epochs=3, lr=5e-5`) but the notebook default
  described in §9/§17.1 as the *target config*: `r=16, alpha=128`, last 8
  layers, `epochs=5`, `max_samples=4000`, cosine LR scheduler, AdamW,
  batch size 3 / grad accumulation 4 (effective batch 12), cutoff length
  1024, best-checkpoint selection on `eval_loss`, gradient checkpointing on.
- Prompting: one-shot example + zero-shot CoT directive ("Let's take it one
  step at a time") + the 5-point rating rubric below (paper §8's exact
  wording — use it verbatim so prompts are consistent across Phase 1 and
  Phase 2's Analyst/Critic, which reuse the same adapter and should see the
  same task framing):

  > Evaluate the sentiment expressed in user reviews and classify each one
  > according to its sentiment rating.
  >
  > **Rating 1**: Comments show a high level of dissatisfaction and
  > negativity; serious flaws, unpleasant experiences; strong negative
  > feedback.
  > **Rating 2**: Still negative but slightly softened; dissatisfaction
  > plus some areas to improve or unmet expectations.
  > **Rating 3**: Generally neutral; both positive and negative aspects;
  > mediocre experience; balanced attitude.
  > **Rating 4**: Positive overall; praise with a few shortcomings;
  > generally satisfactory, room for improvement.
  > **Rating 5**: High satisfaction; praise and testimonials; little or no
  > mention of deficiencies; willing to reuse or recommend.
  >
- Train/val split 80/20.
- Save the adapter + tokenizer to `output/final/`.
- FYI on expected wall-clock (paper §11, so you can sanity-check you're not
  stuck/hung): ~10–15 min/epoch on a T4 with ~2000 samples (so ~30–45 min
  for 3 epochs); scale roughly linearly for `max_samples=4000` and
  5 epochs. Local timing will vary with your actual GPU.

### 3.4 Evaluation (§8/§10 methodology)

- Held-out eval pool: disjoint from train (exclude by index/hash), shuffled
  with a **fixed seed** (use 42 for reproducibility), capped at
  `MAX_EVAL_SAMPLES` (default 4000). No TextBlob filtering on eval data.
- Inference: one-shot + CoT, greedy decoding (`do_sample=False`,
  `max_new_tokens=128`).
- Rating extraction: parse the **last** `Sentiment (1-5):` occurrence in
  the output (CoT reasoning may mention other digits earlier — don't grab
  those).
- Report: accuracy, weighted precision/recall/F1, macro F1, BLEU-4 (via
  `sacrebleu`, correct reference pairing — one reference list per
  hypothesis, referencing the full `Sentiment (1-5): {r}. {description}`
  string), BLEU-1, chrF++, ROUGE-1 (0–100 scale), and samples/sec
  throughput.

### 3.5 Target / exit criteria

The paper's own reference run (documented in §10.1) after these exact
pipeline fixes landed **~66% accuracy / ~68% weighted F1 / ~50% macro F1**
on a 4000-sample eval — the paper authors' original claim of 92.6% assumes
their full undisclosed setup and is **not** the target to chase. Treat
"consistent with the ~65–70% accuracy / ~65–70% weighted-F1 band, with
BLEU-4 and ROUGE-1 in a sane range (BLEU-4 well above 0, not near-perfect
either)" as the success bar for this phase. If your run is wildly below
that band (e.g. near-random ~20% or BLEU-4 ≈ 0), that's a bug signal (check
prompt formatting, rating parser, or reference-pairing for BLEU) — debug
and rerun rather than reporting a broken number. Log every run's config +
metrics to `RESULTS.md`.

---

## 3.6 Important: Phase 1 and Phase 2 use two different, disjoint datasets

Don't conflate these — this is intentional in the paper's design, not
something to "fix":

|         | Phase 1 (train + its own eval, §3)                                   | Phase 2 + comparison (§4, §5)                                                                                                   |
| ------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Dataset | Kaggle`arhamrumi/amazon-product-reviews`                            | Amazon Reviews 2023 (UCSD McAuley Lab)                                                                                            |
| Fields  | review text + star rating only                                        | review text + rating + product metadata/specs + product images                                                                    |
| Why     | Phase 1 is text-only, so the smaller/simpler Kaggle set is sufficient | Phase 2 needs images + specs for the Visual Verifier and RAG Prover, which Kaggle doesn't have (paper §4.2 says this explicitly) |

Practical consequence: when §4/§5 evaluate the Phase 1 LoRA adapter on
Amazon Reviews 2023 rows, that adapter has **never seen this dataset during
training** — it was trained entirely on Kaggle. So the "Phase 1" column in
the comparison table is a generalization/transfer test, not a same-
distribution held-out comparison. That's expected and matches the paper's
own design (§10.4's `phase2_agentic_vs_phase1_amazon2023.ipynb` does exactly
this). Report it as such in `RESULTS.md` — don't retrain Phase 1 on Amazon
Reviews 2023 to make the comparison "fairer" unless I explicitly ask for
that, and don't silently merge/dedupe the two datasets into one pool.

---

## 3.6a Important: two similarly-named notebooks, different jobs — don't mix them up

The paper names three notebooks (§10.3, §10.4, §17). Two of them sound alike
but are **not** the same thing:

- **`colab/phase2_agentic_vs_phase1_amazon2023.ipynb`** — this is the real
  Phase 1 vs Phase 2 comparison and **is what §4 and §5 below build**. It
  contains, in one notebook: the Amazon Reviews 2023 data loading, a
  Phase-1-style single-pass text-only generation column, the full LangGraph
  agentic pipeline (Analyst → Visual Verifier + RAG Prover → Critic with
  reflection), and the side-by-side scoring table. Build it in two passes —
  Phase 1 column first (§4), then add the LangGraph pipeline as the second
  column (§5) — but it's one file throughout, not two.
- **`colab/phase1_vs_phase2_amazon2023_presentation.ipynb`** — a
  **separate, simpler ablation**, per paper §10.3: same LoRA adapter,
  Phase-1-style text-only vs. a *single* enriched prompt (metadata + an
  optional BLIP caption stuffed into the prompt text), with **no LangGraph
  and no Critic at all**. It answers "does just showing the model more
  context help, without any agentic verification loop." This is **not**
  required for the three milestones — treat it as optional/bonus, and only
  build or run it if I ask, or if it already exists in `colab/` and you're
  just validating/documenting what's there during your inventory (§2).

If you find that the existing repo already uses these two filenames
differently than described here, defer to what's actually in the notebooks
and tell me — don't silently rename or restructure existing work to match
this prompt.

---

## 4. Phase 1 vs Phase 2 comparison scaffold (`colab/phase2_agentic_vs_phase1_amazon2023.ipynb`, pass 1 of 2)

Build this **before** Phase 2's agent nodes exist, as the first half of the
one notebook this and §5 share (see §3.6a — don't create it as, or confuse
it with, `phase1_vs_phase2_amazon2023_presentation.ipynb`):

- Loads the Phase 1 adapter from `output/final/` (trained on Kaggle — see
  §3.6).
- Loads a filtered multimodal slice of Amazon Reviews 2023 (see §5.1 for
  filter criteria — same filters Phase 2 will need, so build it once here).
- Runs the pure Phase-1-style single-pass generation (text-only prompt, no
  metadata/image) as the "Phase 1" column of the results table.
- Leaves a clean, obviously-marked insertion point (e.g. a
  `run_phase2(row)` stub raising `NotImplementedError`) for Phase 2's
  Analyst→Critic pipeline to be dropped in during the next milestone.
- Scoring: exact-match accuracy and MAE against the review's star rating as
  ground truth, plus a checkpointed CSV (`results.csv`, resumable via a
  `sample_idx` column, flush every `CHECKPOINT_EVERY=25` rows) so a crash
  mid-run doesn't lose progress.

Don't run this notebook to conclusions yet — it's scaffolding. Confirm the
Phase 1 column produces sane numbers, then move to §5.

---

## 5. Phase 2 — Proposed agentic system (`colab/phase2_agentic_vs_phase1_amazon2023.ipynb`, pass 2 of 2 — same file as §4)

This is the paper's §12–§13 architecture, added as the second column of the
notebook you started in §4. Implement the **real** thing, not
the toy stand-ins — i.e. don't hardcode a fake spec-matching string for the
RAG Prover if a real LanceDB store is achievable locally; use LLaVA (or the
closest locally-runnable VLM) for the Analyst rather than skipping straight
to a plain LLaMA-3 call, unless resource constraints force a documented
substitution (say so explicitly in `RESULTS.md` if you substitute — e.g.
"used BLIP captioning + the Phase-1 LoRA adapter as Critic instead of
LLaVA-o1 due to local VRAM limits").

### 5.1 Data (Amazon Reviews 2023, UCSD McAuley Lab)

Reminder: this is a **different dataset from Phase 1's Kaggle training
data** — see §3.6. Nothing in Phase 2 should read from or merge with the
Kaggle CSV.

- Load via `datasets>=3`: reviews with
  `load_dataset("json", data_files=...)` on the Hub JSONL shards
  (`raw/review_categories/{Category}.jsonl`), metadata with
  `load_dataset("parquet", data_files=...)` over `raw_meta_{Category}/*.parquet`.
  **Do not** pass `trust_remote_code` — the Hub scripts are disabled for
  this dataset. Default category: `All_Beauty` (smallest, good for
  iteration) — confirm with me before switching categories.
- Build an ASIN → metadata index (cap `META_INDEX_MAX_ROWS`, e.g. 500k).
- Filter to rows with: review text ≥ `MIN_REVIEW_CHARS` (default 20),
  metadata with at least a title or details field, and at least one
  fetchable product image URL.
- Target slice size: start small (~100–200 rows) to validate the pipeline
  end-to-end fast, then scale to the paper's ~1000-row target once correct.
- Rating parser nuance specific to this notebook (paper §10.4): decoded
  continuations here are often just the text *after* `Sentiment (1-5):`
  (not the full labeled string as in §3.4's Phase 1 eval), and may start
  with patterns like `5.` or `5 ...` rather than repeating the label. Write
  the parser to handle both the leading-digit-only case and the full-line
  `Sentiment (1-5): ...` case — don't reuse §3.4's parser unmodified and
  assume it covers this.

### 5.2 Knowledge Grounding (LanceDB)

- Stand up a real LanceDB table of product specs/metadata for the chosen
  category, embedded with CLIP (ViT-L/14) so text specs and product images
  share a vector space.
- This is what the RAG Prover queries — it should be an actual vector
  search, not a hardcoded string match.

### 5.3 LangGraph nodes (defined in-notebook, one cell per node function)

- **Analyst**: decomposes the review into atomic claims, produces a
  preliminary sentiment + hypothesis vector `H`.
- **Visual Verifier**: patch/image-level check against the product image
  (BLIP captioning is an acceptable, documented substitute for full
  patch-level LLaVA auditing if that's what's achievable locally) →
  evidence vector `Ev`. Per paper §13.3, prompt this node as a strictly
  isolated "Hardware Auditor" — it should see the product image (and at
  most the specific claim being checked), but **not** the review's
  emotional language or star rating, so it can't just echo the user's tone
  back as "confirmation" (the paper calls this failure mode "semantic
  leakage" — the model 'sees what it reads' instead of what's actually in
  the image).
- **RAG Prover**: queries LanceDB for the claim's relevant spec, flags
  user-expectation mismatches (e.g. claim references a feature the product
  spec doesn't have) → factual grounding score `Gf`. Worked example from
  the paper (§13.4) to model this after: a 1-star review complains "the
  USB-C cable doesn't fit the port"; the spec store says the device only
  has Micro-USB; the Prover should flag this as a **user error**, not
  a product defect — i.e. it's checking claims against specs, not
  rubber-stamping the user's framing.
- **Critic**: computes dissonance `D = Norm(H − (Ev + Gf))`.
  If `D < threshold` (paper suggests 0.3), release the verified sentiment.
  If `D ≥ threshold`, send specific feedback back to the Analyst (reflection
  edge) and re-run, up to a max reflection count (avoid infinite loops —
  cap at e.g. 2 reflections and release best-effort after that). Feedback
  should be specific and actionable, not just "try again" — paper §13.5's
  example: *"Re-evaluate: User is complaining about a feature this product
  never claimed to have."*
- Orchestrate with LangGraph as a stateful graph, not a manually chained
  sequence of function calls — the reflection loop is a real graph edge
  back to the Analyst node, conditioned on the Critic's output.
- Generation config note (paper §10.4): when the Critic reuses the Phase 1
  LoRA adapter for its second pass, set the tokenizer's
  `truncation_side="left"` so that if the combined prompt (review + claims
  + evidence + spec results) runs long, truncation trims from the front
    and preserves the trailing `Sentiment (1-5):` cue the model was trained
    to complete. Also pass an explicit `GenerationConfig(do_sample=False, ...)`
    rather than relying on model defaults, to avoid warnings/errors from
    merged sampling parameters.

### 5.4 Execution

- Wire the `run_phase2(row)` stub from §4's notebook into the LangGraph
  pipeline you just built.
- Run on the filtered slice, streaming results to a checkpointed CSV exactly
  as in §4 (resumable, `CHECKPOINT_EVERY`).
- Track LLM/VLM calls per row (paper's reference implementation used 3:
  Analyst + Critic + the separate Phase 1 comparison call) so cost/latency
  is visible.
- Produce simple plots (paper §10.4): accuracy/MAE bar chart comparing
  Phase 1 vs Phase 2, plus a short line chart of per-sample predicted
  ratings vs. ground truth for a prefix of the slice (e.g. first ~50 rows)
  to eyeball where the two diverge.

### 5.5 Target / exit criteria

No single paper-reported number exists for the fully faithful Phase 2 —
that's expected, this is the novel contribution. Success here means:

- The graph actually runs end-to-end without manual intervention on the
  full target slice (≥100 rows, ideally ~1000).
- Report exact-match accuracy and MAE vs. star rating for Phase 2 **and**
  the Phase 1 column from the same rows, side by side, plus the additional
  Phase-2-only metrics from paper §14 you can actually compute (faithfulness
  = claims supported by LanceDB / total claims; dissonance-score
  distribution; average reflection edges per input). Skip §14's Spatial
  IoU metric — it needs bounding-box localization, which isn't meaningful
  if the Visual Verifier is using BLIP captioning rather than true
  patch-level detection; note this omission explicitly in `RESULTS.md`
  rather than fabricating an IoU number.
- Phase 2 doesn't need to beat Phase 1 on every metric to be a successful
  build — report whatever the real numbers say. Do not adjust thresholds or
  cherry-pick a subset to force a win.

---

## 6. Final paper — write up the real results

Once Phase 1 and Phase 2 have both produced real, logged results (§3.5,
§5.5), write a paper reporting them. The attached project paper
(`Agentic_Self-Correction_in_LLaMA3___E-Commerce_Sentiment_Analysis.pdf`)
is the structural and stylistic template — match its section structure and
academic tone, but this is a **results paper**, not a proposal, so the
content differs in specific ways:

### 6.1 What carries over as-is

Sections 1–2 (Abstract framing, Problem Statement), §3 (Research Gap,
including the Wang et al. 2025 baseline and the three gaps: gullibility,
critic, modality isolation), and §4 (Datasets) describe motivation and
setup that doesn't change based on your results — reuse this content,
rewritten in your own words rather than copied verbatim (this is the
user's own paper, so verbatim reuse of their own prior text is fine
too if they prefer; ask if unsure).

### 6.2 What must be rewritten from real numbers, not projected ones

- **§10-equivalent (Baseline Results)**: replace the original paper's
  92.6%/95.3%/93.3% framing with what Phase 1 **actually achieved**
  (§3.5's logged numbers). Explicitly discuss the gap vs. Wang et al.'s
  92.6% and the likely explanation (§2's note on their K=5/128K-32K/40K
  eval protocol vs. this project's ~4000-sample single split) — don't
  present the gap as unexplained or as a failure without that context.
- **§12–14-equivalent (Proposed System + Evaluation)**: replace the
  architecture description's future/proposed tense ("the framework
  moves...", "the Critic will...") with what was actually built and run.
  Report Phase 2's real accuracy/MAE alongside Phase 1's on the same
  Amazon Reviews 2023 slice (§5.5's numbers), the faithfulness and
  dissonance-distribution metrics actually computed, and state plainly
  which paper §14 metrics were skipped and why (e.g. Spatial IoU, per
  §5.5) rather than omitting them silently.
- **§15/Conclusion-equivalent**: rewrite as an honest summary of what was
  found — including if Phase 2 didn't beat Phase 1 on every metric, or if
  the reflection loop rarely triggered, etc. Report what happened, not
  what the original proposal hoped would happen.
- **§17-equivalent (Implementation Status)**: update from "proposed/not
  yet implemented" to what's actually complete, matching your inventory
  report and the final state of both notebooks.

### 6.3 New content this results paper needs that the original didn't

- A brief methodology note that Phase 1 (Kaggle) and Phase 2 (Amazon
  Reviews 2023) use disjoint datasets (§3.6), and what that means for how
  the Phase 1-vs-2 comparison should be read (generalization test, not
  same-distribution held-out comparison).
- Include the actual plots produced in §5.4 (accuracy/MAE bar chart,
  per-sample line chart) as real figures, not the original's architecture
  diagrams alone.
- Add Wang et al. 2025 (the `itmconf_dai2024_04021.pdf` paper) as an
  explicit numbered reference — the original paper cites it in-text as
  "Wang et al., 2025" in §3.1 but doesn't appear to include it in its own
  §16 reference list.

### 6.4 Ground rules for this milestone specifically

- **No fabricated or extrapolated numbers.** Every metric in the paper
  must trace back to a line in `RESULTS.md`. If a result is missing
  (a run didn't finish, a metric wasn't computed), say so in the paper
  rather than estimating it.
- Draft as Markdown first in `paper/` (e.g. `paper/results_paper.md`) so
  it's easy for me to review and edit inline. Don't jump straight to a
  polished Word/PDF export — ask me whether I want that conversion once
  the Markdown draft is approved, and what filename/format I'd like.
- Keep citations consistent with the original paper's style (numbered,
  matching its §16 format) rather than switching citation conventions.

---

## 7. Ground rules

- **Real execution only.** Every metric in `RESULTS.md` must come from an
  actual completed run you watched finish, not an estimate or a partial
  run extrapolated forward. If something is still running when you report
  interim status, say so explicitly.
- **Checkpoint everything long-running** (training, the ~1000-row Phase 2
  eval) so a crash doesn't cost hours.
- **Ask before big downloads/installs** you're unsure about — e.g. before
  pulling the full 570M-review Amazon Reviews 2023 dataset, or before a
  multi-GB model download, confirm the plan.
- **If you hit a hard local blocker** (OOM even at 4-bit, bitsandbytes
  won't load, no CUDA), stop, explain exactly what failed, and produce the
  Colab-ready version of that notebook (self-contained `!pip install`
  cells, no local file-path assumptions, uses Drive or upload for data) so
  I can run it there. Don't silently degrade to a smaller model or skip a
  component to force a local pass.
- Keep `RESULTS.md` updated after every notebook run: date, config, dataset
  slice, metrics, and any deviations from the paper's spec and why.
