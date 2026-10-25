# Code Explained

A guided tour of the code behind the Phase 1 reproduction and the Phase 2 agentic pipeline: what each file does, how data flows through it, and where the known quirks are. Paths are relative to `pes-mtech-project/`. Function names are given instead of line numbers because line numbers drift.

Contents
1. The 60-second map
2. Phase 1: data pipeline, training, evaluation
3. Phase 2: the agent graph
4. The evaluation harness
5. Supporting modules (RAG store, Amazon 2023 loader, baselines, run directory)
6. The local (no-GPU) path used for the recent experiments
7. The v2_reflective folder (new work)
8. Known quirks and gotchas (read before trusting any number)

---

## 1. The 60-second map

```
src/agentic_sentiment/
  phase1/                      PHASE 1: reproduce the paper (LLaMA-3-8B + LoRA)
    config/settings.py           all hyperparameters as dataclasses (paper Table 1)
    data/preprocessing.py        prompt text, rating descriptions, one-shot, training example format
    data/sampling.py             TextBlob filter, stratified sampling, VGST, neutral oversampling
    data/dataset.py              load CSV -> run the pipeline -> tokenize -> mask prompt tokens
    models/lora.py               build the LoRA config, freeze the base model, load the 4-bit model
  train/
    phase1_train.py              wires config + data + HF Trainer; produces the adapter
    run_dir.py                   checkpoint folder management (built for Colab + Drive)
  agents/                      PHASE 2: the multi-agent pipeline
    graph.py                     LangGraph graph: analyst, visual, rag, critic nodes + prompts
    critic.py                    dissonance formulas and threshold
    state.py                     the shared state passed between nodes
    parsing.py                   turns raw model text into a 1-5 rating
  rag/spec_store.py            LanceDB vector store of product specs; query + grounding score
  data/amazon2023.py           Amazon Reviews 2023 loader and spec-record builder
  eval/
    run_eval.py                  runs one ablation over many rows, metrics, error recovery
    one_shot.py                  builds the one-shot example pool
    checkpoint.py                append-only JSONL results with resume
    results_log.py               formats Markdown result blocks for docs/RESULTS.md
  baselines/classical.py       Decision Tree, SVM, Naive Bayes (paper Table 3 configs)
  baselines/xlnet.py           XLNet baseline training
  colab_sync.py                pushes Colab outputs back to the git repo

colab/                         the four Colab notebooks that run the GPU parts
output/final/                  the trained Phase 1 LoRA adapter (committed)
results/run1, results/run2     saved per-sample results and charts
docs/                          PROGRESS.md (build log), RESULTS.md, CHECKPOINT_HISTORY.md
v2_reflective/                 new work (Part 7); the older code above is untouched
.scratch/                      throwaway scripts and local artifacts (not part of the package)
```

Rule of thumb: Phase 1 code is about *building and training* the classifier. Phase 2 code is about *running* the classifier inside several cooperating prompts.

---

## 2. Phase 1: data pipeline, training, evaluation

### 2.1 Configuration: `phase1/config/settings.py`
Three dataclasses hold every setting.

- `ModelConfig`: base model name (`meta-llama/Meta-Llama-3-8B`), LoRA rank 4, alpha 64, dropout 0, `trainable_layers = 3`. The comments record why: these match the paper's Table 1 and the committed adapter. An older docstring claimed a stronger r=16, 8-layer config produced the early results; git history showed that was never true, so the defaults were corrected.
- `TrainingConfig`: learning rate 5e-5, 3 epochs, batch size 3 with gradient accumulation 4 (effective 12), cosine schedule, fp16, max sequence length 1024, validation ratio 0.2, `max_samples = 2000`, NEFTune noise 0.02, best-checkpoint selection on `eval_loss`.
- `DataConfig`: which preprocessing steps are on, VGST batch size 32 and wishlist length 10, VGST target ratio 0.01, neutral oversample ratio 2.0, the system prompt, one-shot and chain-of-thought switches.

Many fields (Galore, BAdam, LLaMA Pro, and so on) mirror paper Table 1 entries that are set to zero or off; they exist so the table can be checked field by field, and most are not used by the trainer.

### 2.2 Prompts and training examples: `phase1/data/preprocessing.py`
- `SENTIMENT_INSTRUCTION`: the task sentence given to the model, including "1-2 negative, 3 neutral, 4-5 positive".
- `RATING_DESCRIPTIONS`: one sentence per star, used both in one-shot examples and as the training answer text.
- `build_prompt(review, cfg, one_shot_example)`: assembles the prompt in this order: instruction, the chain-of-thought phrase ("Let's take it one step at a time."), an optional worked example, the review, a reminder to end with `Sentiment (1-5):`, and the string `Sentiment (1-5):` itself, so the next token the model writes is the rating.
- `prepare_conversation_format(text, rating, cfg, one_shot)`: returns the prompt and the **answer**, which is `"{rating}. {description}"` (for example `"4. Users have a positive experience overall, with room for improvement (rating 4)."`), not a bare digit. This matters because the evaluation parser and the one-shot example expect the same shape, so training matches inference.
- `create_one_shot_pool` and `get_one_shot_from_pool`: pick worked examples from the training rows (one per rating where possible).

### 2.3 The sampling pipeline: `phase1/data/sampling.py`
`apply_paper_preprocessing(rows, data_cfg, tokenizer, max_samples, seed)` runs the paper's steps in order. Each step is its own function.

1. **`filter_by_textblob_polarity`** (data quality control). TextBlob scores each review from -1 to +1. A row is discarded if polarity < 0 and rating > 3, or polarity > 0 and rating < 3. Polarity exactly 0, or a TextBlob failure, keeps the row. For large inputs (2,000 or more rows) polarities are computed in a process pool; the decision per row is identical either way, only faster.
2. **`stratified_sample`**: groups rows by rating 1 to 5 and takes the same number from each (from `max_total // 5`, or the smallest class if nothing is set). This removes the roughly 64% five-star skew.
3. **`vgst_sample`** (Variant Greedy Search Technique). The loop:
   - keep a running set `current_tokens` of token ids already covered by the chosen reviews;
   - to fill a wishlist: draw `batch_size` (32) random candidate rows, tokenize each, count how many token ids it would *add* to `current_tokens`, keep the candidate that adds the most; repeat `wishlist_len` (10) times;
   - from the wishlist, pick the row with the most new tokens, append it to the sample, and merge its tokens into `current_tokens`;
   - repeat until `target_size` rows are chosen.
   The target size is `min(len(rows), max(max_samples, 1% of the post-filter pool))`, and VGST only runs if that target is smaller than the pool. Intuition: reviews repeat the same phrases, so picking by "new vocabulary added" keeps the training set varied. Note it samples *with replacement* from random batches, so a row can in principle be chosen twice.
4. **`oversample_neutral`**: duplicates rating-3 rows (extra copies = `(multiplier - 1) x count`, sampled with replacement), concatenates, shuffles.
5. Final cap at `max_samples` after a shuffle.

### 2.4 Turning rows into tokens: `phase1/data/dataset.py`
`build_sft_dataset(data_path, tokenizer, data_cfg, training_cfg)`:
1. `load_raw_data` reads the CSV or JSON and returns dicts with `text` and `rating` (it infers column names).
2. Runs `apply_paper_preprocessing`.
3. Shuffles, splits 80/20 into train and validation.
4. Builds a one-shot pool from the **training** rows only, and attaches a one-shot example to each prompt.
5. Tokenizes `prompt + " " + answer`, then builds `labels`: the prompt part is set to `-100` (ignored by the loss) and only the answer tokens keep their ids. **So the original Phase 1 training does compute loss on the answer only.** The Colab notebook has the same line (`labels = [-100] * prompt_len + input_ids[prompt_len:]`).
6. Padding is left to the data collator so each batch is padded only to its longest example.

If no data file is given it falls back to a 30-row dummy set, so a missing dataset can silently train on dummy data. That is why a guard was added in the notebooks.

### 2.5 The model and LoRA: `phase1/models/lora.py`
- `get_lora_config(cfg)`: builds a PEFT `LoraConfig`. Only the last `trainable_layers` of 32 get LoRA (layers 29, 30, 31 for 3). **Quirk:** if `lora_target_modules` is not set it defaults to `q_proj, k_proj, v_proj, o_proj`, but the committed adapter was trained with only `q_proj` and `v_proj` (see Part 8).
- `freeze_base_model`: freezes every non-LoRA parameter ("Freeze technique").
- `detect_use_4bit`, `detect_fast_training_config`, `load_model_and_tokenizer`: pick 4-bit loading (bitsandbytes) on smaller GPUs and turn gradient checkpointing on or off depending on GPU memory.

### 2.6 Training: `train/phase1_train.py`
`run_training(data_path, run_dir)`:
1. Builds the three configs, merges any auto-detected overrides.
2. Writes the config to the run directory.
3. Loads the model (4-bit if needed), builds the datasets.
4. `_build_trainer` creates a HuggingFace `Trainer` with the Table 1 settings. The data collator pads per batch; `neftune_noise_alpha` adds small noise to embeddings during training.
5. Trains, resuming from the latest checkpoint if the run id already has one.
6. Evaluates, saves the best adapter and `metrics.json`.

`train/run_dir.py` (`RunDir`) manages a run folder: config, history rows, latest and best checkpoint tracking, and a callback that copies checkpoints to Google Drive during a Colab run.

### 2.7 Evaluation parsing: `agents/parsing.py`
`parse_rating(text)` turns raw model output into 1 to 5. It tries, in order: `Sentiment (1-5): N`; a leading digit; then any digit that is not part of the "(1-5)" scale label or an "N out of 5" pattern. If nothing matches it returns `None` and callers default to 3. This default is one reason 3-star can look better than it is.

---

## 3. Phase 2: the agent graph

### 3.1 State: `agents/state.py`
`AgentState` is a typed dictionary passed through every node. It carries the review (`review_text`, `asin`, `meta_title`, `meta_description`, `image_caption`), switches (`use_metadata`, `use_image`, `use_rag`, `use_cot`, `dissonance_method`, `max_correction_iters`), the `one_shot_example`, and the results each node adds: `analyst_rating`, `visual_rating`, `rag_rating`, `rag_grounding`, `critique`, `dissonance`, `correction_iters`, `final_rating`, `self_corrected`, `rationale`.

### 3.2 The graph: `agents/graph.py`
`build_graph(llm_fn, spec_store=None, max_correction_iters=2)` returns a compiled LangGraph `StateGraph`. `llm_fn` is any function that takes a prompt string and returns text, so the same graph runs against a real model, a local server, or a stub in tests.

Nodes, in order: `analyst` then `visual` then `rag` then `critic`, with a conditional edge from `critic` back to `analyst` or to `END`.

- **`analyst_node`**: builds `ANALYST_PROMPT` (instruction, chain-of-thought phrase, optional one-shot example, optional product metadata, the review, and any critic feedback from a previous round), calls the model, parses the rating (default 3).
- **`visual_node`**: if an image caption exists, asks the model to rate sentiment based only on the caption and product title. Otherwise returns `None`. The caption comes from BLIP in the notebooks.
- **`rag_node`**: assembles context from the product description plus the top matching spec snippets from the spec store (filtered to this product's `asin`), asks the model to rate with that context, and separately computes a **grounding score**: how well the review text matches the product's specs.
- **`critic_node`**: the decision point.
  1. Convert votes to a 0 to 1 scale with `normalize_rating`.
  2. Compute dissonance `d` (see 3.3).
  3. If `d >= 0.4` and fewer than `max_correction_iters` rounds have run: ask the model for a one-sentence critique (including a note if grounding is below 0.5), store it, increment `correction_iters`, and the router sends control back to `analyst` (because `final_rating` is still unset).
  4. Otherwise decide the final rating. **The Analyst's rating is the default.** It is overridden only if two or more *other* votes unanimously agree on a different rating. Because the Visual vote is now excluded (see Part 8) and RAG is the only other vote, this condition can never be met, so `final_rating` effectively equals `analyst_rating`. That is the "structurally dead override" found in the audit.
- **`route_after_critic`**: returns `"analyst"` if `final_rating` is not set, else `END`.

### 3.3 Dissonance: `agents/critic.py`
- `DISSONANCE_THRESHOLD = 0.4`.
- `normalize_rating(r) = (r - 1) / 4`.
- **Default `dissonance_norm(h, ev, gf)`**: `raw = h - (ev + gf) / 2`, then `abs` and clamp to [0, 1]. Here `h` is the Analyst's normalized rating, `ev` the Visual one, and `gf` the RAG sentiment (the code passes the RAG *rating*, not the grounding score, on purpose, so all three are on the same axis).
- **`dissonance_stdev(votes)`**: `statistics.stdev(votes) / 2`, the formula Run 1's notebook used. Run 1 numbers (mean dissonance 0.211, maximum 0.471) come from this one. The deck describes it for that reason; the rebuilt package uses the `norm` formula by default and keeps `stdev` for the ablation that reproduces Run 1.

Important behaviour change recorded in the code comments: an earlier version also forced `d = max(d, 1 - rag_grounding)`. Grounding is near-binary and low on about 71% of real rows, so correction fired on 78.6% of rows regardless of agreement and cost about 91 net accuracy on Run 2. That line was removed.

---

## 4. The evaluation harness

### `eval/run_eval.py`
- `ABLATIONS`: dictionary of named configurations (`text_only`, `plus_metadata`, `plus_image`, `plus_rag_reviews`, `plus_rag_specs`, `full_graph`) that switch the graph's inputs on cumulatively.
- `run_ablation(name, rows, llm_fn, checkpoint_path, ...)`: for each row not already in the checkpoint, builds the state (one-shot example chosen with a random generator seeded by `review_id`, so a resumed run gives the same example), invokes the graph, times it, and appends a result record. Resumable.
- `compute_metrics(records, key)`: accuracy, macro-F1, MAE and the 5x5 confusion matrix.
- `error_recovery(baseline, improved)`: matches rows by `review_id` and counts errors **fixed** (wrong before, right now), **regressed** (right before, wrong now), and the net.
- `average_grounding`, `average_correction_iters`, `average_latency`: the Phase 2 extras.

### `eval/one_shot.py`
`build_one_shot_pool(rows)` picks one row per rating from the **evaluation rows themselves** (sorted first, so the order of the input does not change the pool). `pick_one_shot_text(row, pool, rng)` prefers an example whose rating differs from the row's *true* rating. That is the mild label leak mentioned in the notes: the example choice depends on the answer.

### `eval/checkpoint.py` and `eval/results_log.py`
Append-only JSONL storage that tolerates a truncated last line (`read_records`), and a formatter that writes the Markdown result blocks kept in `docs/RESULTS.md`.

---

## 5. Supporting modules

### `rag/spec_store.py` (`SpecStore`)
- `build(records)`: embeds every spec record's text and writes a LanceDB table `specs` with columns `asin`, `field`, `text`, `vector`.
- `query(claim, asin, k)`: vector search, filtered to one product (`prefilter`), returns the top k snippets.
- `grounding_score(claims, asin, threshold=0.5)`: for each claim, find the closest spec snippet for that product and count a hit if cosine similarity is at least 0.5; the score is hits divided by claims. In the pipeline the "claim" is the whole review, which is why scores are almost always exactly 0 or 1.

The store used in the experiments lives at `.scratch/lancedb_eval` (7,301 snippets, 489 products).

### `data/amazon2023.py`
- `load_balanced_slice(...)`: keeps reviews that have metadata and enough text, then samples up to `n_per_rating` per star with a fixed seed. Builds the `review_id`, `meta_title`, `meta_description`, `meta_details`, and first `image_url`.
- `build_spec_records(meta_by_asin)`: one record per product field (title, description, each detail such as Brand or Item Weight) for `SpecStore.build`.

### `baselines/classical.py`
Three scikit-learn pipelines using the paper's Table 3 settings: Decision Tree with `CountVectorizer`; SVM (sigmoid kernel, gamma 1.0) with `TfidfVectorizer`; Multinomial Naive Bayes (alpha 0.2) with `CountVectorizer`. `evaluate` returns accuracy and *macro* precision, recall and F1.

### `baselines/xlnet.py`
Trains the XLNet baseline. A code comment records an earlier failure mode: with a bad setting it collapsed to predicting one majority class for the whole run.

### `colab_sync.py`
`push_repo_changes` commits result files and pushes them from a Colab session so results are not lost with the session; it redacts the token from any printed output.

---

## 6. The local (no-GPU) path used for the recent experiments

The training code above needs CUDA and bitsandbytes, which this Mac does not have. The recent local work therefore uses two other tools.

1. **llama.cpp (inference).** The committed adapter was converted to GGUF with `convert_lora_to_gguf.py`; the base model is a community 4-bit file (`Meta-Llama-3-8B.Q4_K_M.gguf`). Served with:
   `llama-server -m <base.gguf> --lora <adapter.gguf> -c 8192 --parallel 1 --port 8090`.
   `.scratch/probe_rating_probs.py` calls it per review with temperature 0 and `n_probs`, then reads the probabilities of the digits 1 to 5 at the answer position. Llama-3 emits a space and then the digit, so the digit position is the second step. The predicted rating is the argmax. This matched the stored generated ratings on 999 of 1,000 rows.
2. **MLX (small training experiments).** `mlx_lm lora` trains LoRA adapters on Apple silicon using a 4-bit MLX copy of the model (`.scratch/mlx-llama3-8b-q4`). Earlier runs (A, B, C) used `{"text": prompt + answer}` data, which puts loss on every token, including the prompt. Run C crashed with a Metal error near iteration 1,200. The newer runs use `{"prompt", "completion"}` data with `mask_prompt: true`, which scores only the answer, like the Colab trainer.

Other `.scratch` scripts you may see referenced: `calibrate.py` (per-rating bias fit), `knn_rag_vote.py` (nearest-neighbour retrieval vote, fitted weight was zero), `test_retrieved_fewshot.py`, `mlx_eval_scale.py`, `build_mlx_*_data.py`.

---

## 7. The `v2_reflective/` folder (new work)

| File | What it does |
|---|---|
| `LOG.md` | Dated log of every step, result and decision (the audit trail) |
| `PHASE1_RUN_CARD.md` | Every setting behind the Phase 1 numbers |
| `PHASE2_PLAN.md` | The reflective-orchestration plan and the gate that decides whether to build it |
| `build_paper_protocol_test.py` | Makes the 4,000 random natural-distribution test rows (seed 2024) |
| `paper_protocol_rows.jsonl`, `paper_protocol_probs.jsonl` | The test rows and the model's digit probabilities for each |
| `score.py` | Scores a probabilities file: exact, within-1, MAE, polarity, binary, F1, with trivial baselines |
| `classical_paper_protocol.py` | Runs the three classical baselines on the same 4,000 rows |
| `build_pair_data.py` | Builds the "is it A stars or B stars?" training data for the adjudicator |
| `pair_config.yaml`, `pair_data/`, `pair_adapter/`, `pair_train.log` | The adjudicator's training config, data, saved adapter, log |
| `mlx_model/` | Symlinked weights plus a plain-concatenation chat template so `mask_prompt` works |
| `gate0.py` | The first check: can the adjudicator beat Phase 1 on its own top-two choices? Also simulates the resulting accuracy on dev rows |
| `build_p1_answeronly_data.py`, `p1_config.yaml`, `eval_p1_answeronly.py` | The 2,000-row Phase 1 retrain on the local MLX path and its evaluation |
| `queue.sh` | Waits for the adjudicator training, then runs gate0, the Phase 1 retrain and its eval |
| `build_deck.py` | Rebuilds the presentation from the previous deck, applying all the edits |

Key idea of `gate0.py`: for each development row, take the model's two most likely ratings `a` and `b`; ask the adjudicator "is it `a` or `b`?" in both display orders and average, so position bias cancels; compare its pick with the model's own top choice, restricted to rows where the true rating really is one of the two. It then simulates "inside the uncertainty gate, let the adjudicator decide" and prints the exact accuracy for several gate thresholds.

---

## 8. Known quirks and gotchas

1. **LoRA target modules differ between code and adapter.** `phase1/models/lora.py` defaults to `q_proj, k_proj, v_proj, o_proj`, but the committed adapter (`output/final/adapter_config.json`) uses only `q_proj` and `v_proj`. Retraining with the package defaults would not reproduce the committed adapter's structure. To match, set `lora_target_modules = ["q_proj", "v_proj"]`.
2. **Training rows were never recorded.** We know the settings (from `adapter_config.json` and `training_args.bin`) but not which rows or how many (Table 1 says 2,000; the pipeline default path can produce 4,000). Test-set overlap with training is therefore excluded only statistically.
3. **One-shot label leak.** The example is chosen using the row's true rating (`pick_one_shot_text`), and the pool is built from the evaluation rows themselves.
4. **The Critic's override is dead code in practice.** It needs two unanimous other votes, and only one other vote can exist. The Analyst's rating always wins. The fired-on-0.4%-of-rows number follows from this.
5. **Visual Verifier is logged but not used.** It returned a constant 3 on 96 of 96 real rows. It still runs, so it still adds latency.
6. **Two dissonance formulas.** Run 1 used `stdev(votes)/2`; the rebuilt package defaults to `Norm(H - (Ev+Gf)/2)`. Same threshold (0.4), different scales. Do not compare dissonance values across the two.
7. **Parser default.** `parse_rating` returns `None` on failure and callers default to 3, which can flatter or distort the 3-star class.
8. **`max_samples` is ambiguous in the paper** (2,000, 4,000 and 5,000 all appear). The config uses 2,000; the notebooks default to 4,000.
9. **Quantisation mismatch.** The adapter was trained on fp16 or 4-bit weights but is served locally on a community Q4_K_M file. Expected effect: about a point, unmeasured.
10. **Defaults that hide failure.** A missing data file silently becomes a 30-row dummy dataset in `build_sft_dataset`; this is guarded against in the notebooks but not in the function itself.
11. **VGST samples with replacement**, uses token-id sets from the model's own tokenizer, and counts *new* token ids, so it favours long reviews. In the demo notebook its picks average about 139 words versus 63 for random, and cover about twice the vocabulary; "diversity" here means token coverage, partly driven by length, not meaning.
12. **TextBlob filter bias.** It is dictionary-based and removes exactly the sarcastic and mixed reviews that are hardest, so the training set is easier than real data.

---

## How to run things (reference)

```bash
# tests (CPU, no GPU needed)
.venv/bin/pip install -e ".[dev]" && .venv/bin/pytest

# score a probabilities file under the paper's protocol
.venv/bin/python v2_reflective/score.py v2_reflective/paper_protocol_probs.jsonl

# classical baselines on the same 4,000 rows
.venv/bin/python v2_reflective/classical_paper_protocol.py

# rebuild the presentation
<venv with python-pptx>/bin/python v2_reflective/build_deck.py
```
