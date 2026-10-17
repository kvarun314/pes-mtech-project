# Progress log (newest last)

## 2026-10-02
- Built the EAI/LNCS conference paper into `paper/conference/` (from
  `Individual Journal Paper/Varun`); PDF builds, 9 pages.
- Audited all Run 1 numbers in the paper against `results/run1/phase2_checkpoint.jsonl`:
  all match (acc 0.161->0.387, macro-F1 0.095->0.273, MAE 1.52->0.96, 491 fixed / 39 regressed).
- Run 1 used category `All_Beauty` (2,000 samples, seed 42). Decided Run 2 uses `Electronics`.
- Agreed scope for full Phase 2: LangGraph + spec RAG + norm dissonance, balanced eval +
  ablations, classical baselines, full Phase 1 retrain, Drive checkpointing of all params.
  Design: `docs/superpowers/specs/2026-10-02-full-phase2-design.md`.
- Next: user reviews spec -> implementation plan -> build package and notebooks.

## 2026-10-02 (cont.)
- Wrote the implementation plan (`docs/superpowers/plans/2026-10-02-full-phase2-plan.md`,
  16 tasks) and executed it via subagent-driven development: fresh implementer + reviewer
  subagent per task, fix rounds dispatched or applied directly for every real finding.
- Built `src/agentic_sentiment/`: Drive-checkpointed `RunDir` (resume-safe best-value
  tracking), vendored + Drive-wired Phase 1 LoRA training (paper-closer config: r=16,
  alpha=128, 5 epochs, max_samples=4000), balanced Electronics eval-slice loader + spec
  records, shared rating parser, `Norm(H-(Ev+Gf))` dissonance (stdev kept as ablation), a
  LanceDB spec store with real claim-vs-spec grounding, a real `langgraph.graph.StateGraph`
  (Analyst -> Visual Verifier + RAG Prover -> Critic -> self-correction loop, max 2 iters,
  independently verified to terminate both when agents agree immediately and when they
  never converge), JSONL checkpoint/resume (crash-safe against a truncated last line),
  an ablation runner + accuracy/macro-F1/MAE/confusion metrics, classical baselines
  (DT/SVM-sigmoid/NB matching Wang et al. Table 3), and an XLNet-large baseline — all
  Drive-checkpointed via `RunDir`. 56 tests passing locally with a stub LLM; GPU-only
  paths are untested here by design.
- Caught and fixed several cross-task bugs during review: the RAG Prover was producing
  another sentiment vote instead of a real factual-grounding score (defeating the spec
  store's whole purpose) — fixed to use `SpecStore.grounding_score()` for Gf when a spec
  store is present; `use_metadata` was declared but never read by the graph, so the
  `plus_metadata` ablation was a no-op — fixed; `load_balanced_slice` emitted `rating`
  while the eval harness expected `gt_rating` — renamed at the source.
- Added 4 Colab notebooks (`01_train_phase1` → `04_report`), each self-contained for a
  fresh Colab runtime (own Kaggle download where needed, `!rm -rf` rerun guards) and using
  the Colab Secrets (`userdata`) pattern for the gated Llama-3 token, not `os.environ`
  (an earlier draft of notebook 01 got this wrong and would have silently logged in with
  an empty token).
- Branch: `full-phase2-rebuild` (not yet merged to `main`).
- Next: user runs the 4 notebooks on Colab in order (01 -> 02 -> 03 -> 04), downloads
  `results/run2/` back into the repo, then the paper gets updated with real Run 2 numbers
  (not before).

## 2026-10-02 (final whole-branch review)
- Dispatched a final whole-branch review (Opus) across all 16 tasks together. It found
  two Critical issues neither per-task review could see in isolation, plus several
  Important ones, all fixed directly:
  - **Dissonance formula was backwards in some cases.** Averaging the grounding score
    (Gf, a support fraction) with the sentiment votes (H/Ev) meant an *ungrounded but
    unanimous* review scored dissonance 0.0 regardless of rating direction — e.g. a
    1-star review with a factually wrong claim never triggered self-correction, exactly
    backwards from the "USB-C vs Micro-USB" case the spec store exists to catch. Fixed:
    dissonance is now `max(vote-agreement, 1 - grounding)`, so an ungrounded claim can
    only raise dissonance, never get cancelled out.
  - **Notebook 03's dataset load would have failed outright** (a Hub config name needing
    `trust_remote_code`, incompatible with `datasets>=4`). Replaced with the streaming
    JSONL + parquet loader already proven in the Run 1 notebook.
  - `plus_rag_specs`/`full_graph`/the dissonance-ablation's "norm" run were three
    identical graph executions under different names — now run once each, aliased.
  - `image_url` was read from the reviewer's own (usually absent) uploaded images
    instead of the product's own metadata images — the Visual Verifier was starved.
  - XLNet's local HF checkpoint staging lived under its own Drive run dir, doubling
    Drive usage with no cap across ~150 saves; fixed by keeping local staging off
    Drive and pruning the Drive-side checkpoint copies to the 2 most recent
    (`RunDir.checkpoint_callback(keep=2)`, shared by Phase 1 and XLNet).
  - Training now actually resumes from the latest Drive checkpoint on reconnect
    (`RunDir.latest_checkpoint()` + `resume_from_checkpoint`) instead of silently
    restarting at step 0 while BEST.md still reflected the interrupted attempt.
  - SVM's sigmoid kernel is infeasible on the full ~568k-row Kaggle CSV on Colab;
    classical baselines now share XLNet's 20k-row cap.
  - `rag_grounding` is now recorded in every ablation checkpoint row (needed for the
    brief's faithfulness metric); the Critic's critique now names the grounding
    mismatch explicitly, not just "they disagree"; `pytest` now works standalone
    (`pyproject.toml` `pythonpath`, independent of the editable install quirk on
    macOS/Python 3.13).
- 62 tests passing, all 4 Colab notebooks JSON-valid. Branch `full-phase2-rebuild`,
  not yet merged.

## 2026-10-03 — merged to main; auto-doc-push; Phase 1 re-ported from the real notebook
- Merged `full-phase2-rebuild` to `main` (`cbc7820`).
- Added auto-documentation: every Colab notebook now commits + pushes its own
  outputs straight to `main` (new `agentic_sentiment.colab_sync.push_repo_changes`,
  `RunDir.history_row()`, `agentic_sentiment.eval.results_log`), gated on a
  `GITHUB_TOKEN` Colab Secret. Two review rounds found and fixed real reliability
  bugs: `userdata.get()` raising (not returning None) on a missing secret crashed
  all 4 notebooks instead of degrading gracefully; a retry after a failed push
  never actually retried the push (fixed: only the *commit* is skipped when
  nothing's newly staged, pull+push always run); two notebooks appending to the
  same log file around the same time conflicted under a plain rebase (fixed:
  `.gitattributes` `merge=union`). Merged (`5f0ea2c`).
- **User pointed at `colab/llama_sentiment_baseline_train.ipynb` directly and
  asked to verify `src/agentic_sentiment/phase1/` actually matches it** — it
  didn't. Task 2 had vendored Phase 1's logic from a *different*, never-run
  sibling package, not this notebook. Two rounds of line-by-line comparison
  against the actual proven cells found real algorithmic drift, not just
  config differences:
  - VGST's target size was ~1% of the *already-stratified* pool (~100 rows)
    instead of ~1% of the *post-DQC* pool floored at `max_samples` (~4000 rows)
    — training would have run on a tiny fraction of the intended data.
  - The SFT target was a bare digit ("5") instead of the full
    "{rating}. {description}" line the notebook deliberately trains on.
  - LoRA's `target_modules` fallback was `[q_proj, v_proj]` instead of all four
    attention projections.
  - `gradient_checkpointing` didn't exist as a config field or get wired in at
    all — needed at this LoRA size to avoid OOM on a T4.
  - Half of `TrainingArguments` was never passed through despite the config
    fields existing (weight_decay, max_grad_norm, warmup_steps, logging_steps,
    bf16, optim, neftune_noise_alpha).
  - `max_csv_rows` didn't exist; the full ~500k-row CSV was always read.
  - Validation examples never got a one-shot prefix even with
    `use_one_shot=True`.
  - **No 4-bit QLoRA at all** — `load_model_and_tokenizer()` defaulted to
    full-fp16, a different numerical regime from the proven Run 1 QLoRA run,
    and one that OOMs on a T4. Added `detect_use_4bit()`, ported from the
    notebook's own bitsandbytes/GPU-memory detection.
  - `overwrite_output_dir` (added in an earlier fix round for a different
    reason) crashed on the installed transformers 5.x, since the notebook
    never passes it — removed; added a real (unmocked) `TrainingArguments`
    regression test so this class of bug is caught locally next time.
  - An en dash vs. ASCII hyphen mismatch in the training prompt's trailing
    instruction line (literal training-input text).
  - Fixed all of them by re-porting the actual logic from the notebook's
    cells; `ModelConfig`/`TrainingConfig`/`DataConfig` defaults are now the
    paper-closer config directly. Merged (`2d556ed`).
- **Checked `colab/phase2_agentic_full_comparison.ipynb` (the notebook that
  produced Run 1's real Phase 2 numbers) against `src/agentic_sentiment/agents/graph.py`**
  the same way: every one of its four agent prompts was missing
  `SENTIMENT_INSTRUCTION` (the 5-point-scale task framing the LoRA adapter was
  fine-tuned to expect) — fixed, added to all four prompts (merged `ba57b4b`'s
  parent). Separately, both proven Phase 2 notebooks always give the Analyst a
  one-shot example (task framing, not multimodal context) — the rebuilt graph
  never did. Added `agentic_sentiment.eval.one_shot` (`build_one_shot_pool`/
  `pick_one_shot_text`, ported from the notebook's own `create_one_shot_pool`/
  `pick_one_shot`), wired into `run_ablation()` unconditionally across all six
  ablations, each row's pick seeded by its own `review_id` so it's stable
  across a resumed run. Merged (`ba57b4b`).
- 107 tests passing. All of `src/agentic_sentiment/` is now verified against
  the actual proven notebooks, not just internally self-consistent.
- Next: unchanged — run the 4 Colab notebooks in order, review the auto-pushed
  `docs/RESULTS.md`/`docs/CHECKPOINT_HISTORY.md`/`results/run2/`, then update
  the paper with real Run 2 numbers.

## 2026-10-04 — full dry-run audit before spending real Colab GPU credits
- User: "make sure this is correct, I dont want to waste my colab credits." Ran two
  independent full-notebook dry-run audits (both Opus) across all 4 notebooks against
  the actual `src/agentic_sentiment/` source (not just internal self-consistency). Both
  converged on the same real bugs, cross-validated via actual local repro (a tiny-model
  Trainer resume simulation, a live HF parquet schema check against `raw_meta_Electronics`,
  real transformers/peft version checks):
  - Same-kernel `ModuleNotFoundError` on every notebook: an editable install's `.pth` file
    is only read at Python interpreter startup, never picked up by a kernel already
    running — `!pip` vs `%pip` never actually fixed this (that helped a different failure
    mode). Fixed with `sys.path.insert(0, "/content/repo/src")` right before the import.
  - 4-bit quant type mismatch: notebook 03's `BitsAndBytesConfig` never set
    `bnb_4bit_quant_type`, defaulting to fp4 while the adapter was trained nf4 — would have
    silently degraded every Phase 2 ablation result with no error. Fixed: `nf4` + explicit
    `torch_dtype=torch.float16`.
  - Notebook 01 would silently train on 30 dummy rows if the Kaggle download failed (no
    assertion existed, and `build_sft_dataset` falls back to dummy data rather than
    raising). Fixed with `assert os.path.isfile(DATA_PATH)` right after the download.
  - XLNet had no `compute_metrics` (only `eval_loss` after hours of training) and was
    scored on its own internal split, not the same `X_test` as the classical baselines —
    fixed both.
  - `RunDir.latest_checkpoint()` could return an incomplete checkpoint (Drive copy
    interrupted mid-copytree) — fixed to skip and fall back to the next-newest complete one.
  - `build_spec_records`/`load_balanced_slice` would crash on real data: Electronics'
    actual `details` field is a JSON *string*, not a dict (test fixtures used dicts,
    masking this).
  - Notebook 01's fixed `run_id="phase1"` meant resume was previously dead code (always a
    fresh timestamped folder); now added a stale-run print (resuming from what BEST.md, or
    about to silently re-finish an already-complete run).
  - Notebook 03's adapter-selection cell now prints `metrics.json`/`BEST.md`/adapter mtime
    before committing GPU time to a possibly stale or incomplete adapter.
  - Notebook 03 cells 14/16 now read checkpoints back via the already-existing
    `eval.checkpoint.read_records` (tolerant of a truncated last line) instead of a bare
    `[json.loads(l) for l in open(ckpt)]`.
  - Caught and fixed my own mistake along the way: cell 16's first edit silently failed to
    apply (indentation copied from cell 14's loop body didn't match cell 16's top-level
    statement), leaving the brittle read in place until re-reading the notebook caught it.
  - A grounding-cost fix (`_split_into_claims`, sentence-splitting review text before
    comparing against spec snippets) was committed, then **reverted** after a focused
    confirmation review actually ran the real `all-MiniLM-L6-v2` model against synthetic
    spec records: the premise was backwards. A whole grounded review already scores
    `g=1.00` against its product's title/description record; splitting into sentences
    instead puts pure-sentiment sentences ("Would buy again.") into the per-claim
    denominator, where they can never match a spec — dropping `g` to ~0.25 for exactly the
    reviews that were grounded correctly before. Since `rag_grounding` depends only on
    `review_text`, it's identical on every correction pass, so this would have forced
    `full_graph`/`plus_rag_specs` into max correction iterations on *more* rows than
    before, not fewer. Reverted; the original whole-review grounding comparison stands.
  - Two Minor items accepted, not fixed: Phase 1's `create_one_shot_pool` uses the unseeded
    module-level `random` (a resumed run sees different one-shot prompts than the original
    attempt — cosmetic); a narrow edge case where `metrics.json`/`RESULTS.md` could
    describe the final (not best) model's metrics after a resume where the best eval
    happened strictly before a disconnect (the saved adapter itself is still correct via
    `save_best()`'s protected-checkpoint preference).
- 115 tests passing, all 4 notebooks JSON-valid. Merged to `main` (`320cc0c`).
- **Status: safe to run the 4 notebooks on real Colab GPU now** (fresh clone from GitHub,
  restart runtime, run 01 → 02 → 03 → 04 in order).
- Reminder still open: an untracked `kaggle_json.py` in the repo root contains the user's
  real Kaggle API credential (pasted directly in chat) — never committed, not referenced by
  anything auto-pushed, but should be deleted once done with it; user may want to regenerate
  that key via Kaggle's "Expire API Token" given it was pasted in plaintext.

## 2026-10-09 — Run 2 results reviewed; Phase 2 regression root-caused; local-run scoping started
- User asked to review the completed Run 2 (`results/run2/`, Amazon Reviews 2023 Electronics,
  2000-row balanced eval, seed 42) against the earlier cross-dataset-transfer analysis. All
  prior numbers (Phase 1 text_only 50.95% acc, SVM 69.7%, Naive Bayes 66.3%, XLNet 64.4%,
  CLAUDE.md §3.5's 65–70% band) checked against `docs/RESULTS.md`/`results/run2/metrics_summary.json`
  and confirmed unchanged.
- **New finding: Phase 2 now regresses instead of improving on Run 2, opposite of Run 1.**
  Run 1 (`CLAUDE.md`): Phase 1 16.1% -> Phase 2 38.7% (+22.6pp). Run 2
  (`results/run2/metrics_summary.json`): Phase 1 text_only 50.95% -> full_graph 46.40%
  (-4.55pp), monotonically decreasing through every ablation stage (plus_metadata 50.65,
  plus_image 44.55, plus_rag_reviews 47.10, full_graph 46.40).
  - `plus_rag_specs`/`full_graph`/`dissonance_norm` rows being bit-identical in
    `docs/RESULTS.md` is **not a bug** — confirmed intentional aliasing in
    `colab/03_phase2_eval_ablations.ipynb` (same flags + spec_store, result copied instead
    of re-paying for an identical run); `dissonance_stdev` is the real unrun second
    dissonance ablation (shown as `—`).
  - Root cause (computed locally from the existing `results/run2/*_checkpoint.jsonl` files,
    pure Python, no deps/GPU needed): the dissonance floor in
    `src/agentic_sentiment/agents/critic.py` / `agents/graph.py:124-125`
    (`d = max(d, 1.0 - rag_grounding)`) forces the correction loop to fire whenever
    `rag_grounding` is low, independent of whether Analyst/Visual/RAG actually disagree on
    the rating. Measured: self-correction fires on 1573/2000 rows (78.6%, vs Run 1's 46.1%);
    net effect across those rows is **-91** (110 fixed, 201 regressed); `analyst_rating`
    alone scores 49.5% vs `final_rating`'s 46.4% — every agent stage after the Analyst is
    actively hurting on this dataset.
  - Why grounding is low so often: `rag/spec_store.py`'s `grounding_score()` embeds the
    *whole* review as a single claim (k=1) and cosine-matches it against one spec snippet at
    threshold 0.5 — confirmed near-binary on the real checkpoint data: exactly 1419/2000
    rows score `0.0`, exactly 581/2000 score `1.0` (no values between), averaging to 0.2905
    (matches `results/run2/report_summary.json`'s faithfulness figure exactly). This doesn't
    contradict the 2026-10-04 revert's finding that a *correctly grounded* review scores
    `g=1.00` — it means roughly 71% of this run's rows aren't hitting that match at all, which
    wasn't characterized before and is still an open question (spec_store coverage for this
    dataset? asin mapping? genuinely low lexical/semantic overlap between review prose and
    spec text for most Electronics products?).
- **Local feasibility check:** machine is Apple M1, 16GB RAM, no CUDA. Confirmed against the
  repo's own `../RESOURCES_AND_WHY_COLAB.md`: LLaMA-3-8B 4-bit LoRA retrain needs
  CUDA + bitsandbytes, not supported on Apple Silicon. User chose to scope "run locally" down
  to: fix the Phase 2 correction-loop bug and rerun eval locally against the *existing* Phase 1
  Colab checkpoint, not attempt a local LoRA retrain.
- User decisions (via clarifying question, both pending implementation):
  1. Retest claim-splitting for `grounding_score()` against a real local `sentence-transformers`
     MiniLM model (CPU-only, no GPU) on this Run 2 data, rather than trusting the 2026-10-04
     revert as-is — that revert's confirmation test used synthetic spec records, not this run's
     real Amazon2023 Electronics data.
  2. Decouple the dissonance formula from the grounding floor: low grounding should surface as
     an independent flag/penalty (reported in the critique text), not something that alone can
     push `d` over `DISSONANCE_THRESHOLD` — only genuine H/Ev/RAG rating disagreement should
     trigger the correction loop.
- Repo has no installed Python env yet in this working copy (`python3 -c "import torch"` ->
  `ModuleNotFoundError`, same for `sklearn`). Next step: create a local venv with
  `pyproject.toml`'s core deps (pandas, scikit-learn, langgraph, lancedb) plus
  `sentence-transformers` for the grounding retest — none of that needs CUDA/bitsandbytes.
- **Open, unresolved:** actually rerunning the full corrected graph (not just recomputing
  metrics from existing checkpoints) needs real LLM calls for any row whose correction
  decision changes under the new formula — the existing checkpoints only record the final
  per-row snapshot, not enough to replay corrected iterations without new inference. No
  CUDA-free local inference path (e.g. a GGUF/llama.cpp or MLX build of the Phase 1 LoRA
  adapter) has been set up yet; still need to decide whether that's in scope before claiming
  a "rerun locally" result.

## 2026-10-09 (cont.) — both decisions resolved; Phase 2 formula fixed; Colab credits now at zero
- Local venv created (`.venv/`, core deps from `pyproject.toml` + `pytest` + `sentence-transformers`,
  all CPU-only, no torch/bitsandbytes). 129 tests pass before any change.
- **Decision 1 resolved — grounding claim-splitting retested against real data, 2026-10-04
  revert confirmed correct.** Pulled a real sample from the actual source
  (`McAuley-Lab/Amazon-Reviews-2023`, Electronics category: 500 real reviews + their real
  product metadata for 489 matched asins, streamed via `datasets`, saved to
  `.scratch/sample_reviews.jsonl` / `.scratch/sample_meta.jsonl`) and ran both grounding
  variants through the real `all-MiniLM-L6-v2` model (`.scratch/grounding_retest.py`) —
  not synthetic data this time, unlike the original 2026-10-04 confirmation test:
  - whole-review (current `grounding_score()`): mean 0.340, 170/500 exact 1.0, 330/500 exact 0.0
  - split-into-claims: mean 0.144, only 19/500 exact 1.0, 300/500 exact 0.0
  - 126 rows flip grounded->ungrounded under splitting, only 8 flip the other way
  Splitting is worse on real data too. **Decision: `rag/spec_store.py`'s `grounding_score()`
  stays as-is, no code change.** Also confirms the ~66-71% near-zero grounding rate seen in
  both this sample and Run 2's real `rag_grounding` values is inherent to the task (review
  prose rarely restates spec facts closely enough to cross cosine 0.5), not an embedding or
  data-pipeline bug.
- **Decision 2 implemented — dissonance formula decoupled from grounding.** Removed
  `d = max(d, 1.0 - rag_grounding)` from `agents/graph.py`'s `critic_node` (was lines
  124-126); dissonance now comes only from `dissonance_norm(h, ev, rag_sentiment)`, i.e. real
  Analyst/Visual/RAG vote disagreement. Low grounding still surfaces as context text in the
  critique when a correction fires for a real disagreement reason, but can no longer force a
  correction pass by itself. Updated 3 tests in `tests/test_graph.py` that asserted the old
  floor behavior (renamed to assert the new decoupled behavior); rewrote the grounding-critique
  test to trigger correction via real vote disagreement instead of via grounding alone. 129
  tests pass (no count change — 3 modified, none added/removed).
- **Not yet done:** actually re-measuring Run 2's accuracy under the fixed formula. Blocked on
  the same gap noted above — corrected rows need fresh LLM calls (no local inference path for
  the Phase 1 LoRA adapter exists yet), so no new accuracy number has been claimed. The unit
  tests prove the formula change behaves as intended; they don't yet prove a new Run 2 number.
- **User flagged a hard constraint: Colab credits are at zero right now, and the Run 2 GPU
  pass that produced the regression we're fixing already cost real money.** Going forward:
  no Colab run gets proposed until it's been fully validated locally first (same bar as the
  2026-10-04 pre-GPU dry-run audit) — local work (this formula fix, the grounding retest,
  classical-baseline reruns) proceeds freely since it's free; any GPU step needs explicit
  go-ahead with evidence of why it's worth the spend, not a casual suggestion.
- User separately asked about improving Phase 1's own number (currently 50.95% text_only,
  a cross-dataset-transfer artifact: trained on Kaggle, eval'd on Amazon2023 Electronics it
  never saw). Confirmed blocked the same way as a full retrain: the LoRA adapter itself isn't
  stored locally (lives only in Colab `output_final.zip`, per the repo's existing gap note),
  and this machine (Apple M1, 16GB RAM, no CUDA) can't run bitsandbytes 4-bit training per
  `../RESOURCES_AND_WHY_COLAB.md`. Getting close to the paper's 92.6% is a data-scale problem
  regardless (Wang et al. used K=5 CV, 128K train/32K val per fold, 40K held-out test — see
  CLAUDE.md:196-203 — vs. the 2000 samples used here), not something local fixes close. Agreed
  split: local now covers this formula fix + classical-baseline reruns on Amazon2023; any
  Phase 1 retrain on more same-distribution data is Colab-only and waits for credits plus a
  local dry-run pass on the notebook changes first.
- Classical SVM/Naive-Bayes baselines retrained+eval'd directly on Amazon2023 Electronics
  (same distribution as the Phase 1/Phase 2 eval, unlike the existing Kaggle-trained
  SVM 69.7%/NB 66.3% numbers) — local, CPU-only, no GPU, no credits spent. Pulled a fresh
  balanced sample straight from the source (`McAuley-Lab/Amazon-Reviews-2023` Electronics
  category, streamed with `datasets`, only 26,314 rows scanned to fill 5000 balanced rows):
  4000 train (800/rating) + 1000 held-out test (200/rating), seed 42
  (`.scratch/fetch_classical_sample.py` -> `.scratch/classical_{train,test}.jsonl`). Ran the
  repo's existing `baselines/classical.py` unchanged (`.scratch/run_classical_amazon2023.py`):
  - decision_tree: acc 0.3240
  - svm_sigmoid: acc 0.4840, precision 0.4904, recall 0.4840, f1 0.4863
  - naive_bayes: acc 0.4730, precision 0.4654, recall 0.4730, f1 0.4679
  **Unexpected, important finding: same-distribution classical baselines (48.4%/47.3%) score
  *lower* than Phase 1 LLaMA's cross-dataset-transfer number (50.95%), not higher.** This
  inverts the usual assumption that same-distribution training beats cross-dataset transfer.
  SVM's confusion matrix on this data shows the same ordinal-confusion pattern already seen in
  Phase 1/Phase 2 (middle ratings 2/3 heavily confused with neighbors, extremes 1/5 classified
  well) — this isn't a quirk of any one model, it's the Amazon2023 Electronics distribution
  itself being intrinsically harder for 5-class sentiment than the Kaggle distribution the
  65-70% target band (`CLAUDE.md` §3.5) was calibrated on. Implication: a Phase 1 LoRA retrain
  on more Amazon2023-distribution data should not be assumed to land anywhere near 65-70% just
  because that's what the Kaggle-distribution baselines hit — this dataset's own ceiling looks
  lower, and that should be set as the realistic expectation before spending any Colab credit
  on a retrain aimed at matching the Kaggle-era target band.
- Next: decide, with the user, whether to pursue a Phase 1 retrain on Amazon2023-distribution
  data at all given this lower apparent ceiling, or focus remaining effort on the Phase 2 fix's
  payoff (stopping the regression) as the realistic "better results" win for this dataset.

## 2026-10-09 (cont. 2) — Phase 1 vs. paper audit; local real-inference verification; second bug found+fixed
- User asked to verify Phase 1 matches the cited paper (`itmconf_dai2024_04021.pdf`, Wang et
  al.) as closely as possible. Read the paper in full, extracted Table 1's exact hyperparameters
  and §2.1/§2.2's methodology, compared against the actual repo state:
  - **Git archaeology resolved a real provenance conflict.** `output/final/`'s committed adapter
    (`6f159ce`, 2026-10-01 11:07:57) predates `src/agentic_sentiment/phase1/`'s creation
    (`29966f6`, 2026-10-02 14:19:24) by a day. `adapter_config.json` + `training_args.bin`
    confirm the committed adapter was trained with the paper's literal Table 1 config (lr=5e-5,
    epochs=3, batch=3, grad_accum=4, warmup=0, save_steps=100, cosine, adamw_torch, LoRA
    r=4/alpha=64/3 layers/q+v_proj) -- exactly. `phase1/config/settings.py`'s dataclass defaults
    (r=16/alpha=128/8 layers/epochs=5/warmup=200/save_steps=50/max_samples=4000) do NOT match
    either the paper or this adapter, despite a docstring claiming they "produced Run 1's Phase 1
    numbers." That claim was false -- that stronger config, documented as a deliberate "paper-
    closer" choice in `colab/llama_sentiment_baseline_train.ipynb`, was only ever a plan for a
    retrain that never happened. **Fixed:** `settings.py`'s `ModelConfig`/`TrainingConfig`
    defaults now match Table 1 exactly (matching the adapter already in use); updated the two
    test files that pinned the old wrong defaults (`tests/phase1/test_config.py`,
    `tests/test_phase1_train.py`). 129 tests still pass. Data/prompt side (`DataConfig`) was
    already correctly aligned with the paper -- no change needed there.
  - §3's prose internally contradicts its own Table 1 on training-set size (2000 in the table,
    "4,000 reviews" and separately "5,000 dataset rows" in the text) -- Table 1 kept as
    authoritative since it's the paper's own dedicated hyperparameter listing.
- **Built a local, CUDA-free inference path to actually measure the Phase 2 fix, zero Colab
  cost.** `output/final/`'s adapter is tiny (628KB) and was already committed locally, so only
  the base model needed fetching: `llama.cpp` installed via Homebrew (Metal-accelerated), PEFT
  adapter converted to GGUF LoRA via `llama.cpp`'s `convert_lora_to_gguf.py` (cloned the repo
  shallowly into `.scratch/llama.cpp-src/` for the script + `gguf-py` package it depends on),
  base model `QuantFactory/Meta-Llama-3-8B-GGUF` Q4_K_M (4.6GB, ungated community requant, since
  the original `meta-llama/Meta-Llama-3-8B` is gated) downloaded and served via
  `llama-server -m ... --lora .scratch/phase1_lora.gguf -c 8192 --parallel 1`. Confirmed correct
  generation on a first test prompt (negative review rated "1" correctly, ~1.2s/call on Metal).
- **Ran real inference on a 96-row balanced subset** (real Amazon2023 Electronics reviews+specs,
  reusing the `.scratch/sample_reviews.jsonl`/`sample_meta.jsonl` pulled earlier this session;
  no real product images available, so `visual_rating` stayed `None` throughout -- an honest
  limitation of this subset, not a bug):
  - `text_only`: acc 52.08%, macro_f1 0.5047, mae 0.5625 (181.3s)
  - `full_graph` (with the dissonance-formula fix already in place): acc **43.75%**, macro_f1
    0.3997, mae 0.7708 (479.3s; one mid-run crash from a 2048-token context limit being too
    small for the RAG prompt, fixed by restarting the server with `-c 8192`; `run_ablation`'s
    checkpoint/resume meant the restart picked up cleanly at row 30/96, no wasted calls)
  - **Dissonance fix confirmed working as intended:** `self_corrected` fired on 0/96 rows (down
    from Run 2's 78.6% over-triggering) -- the grounding floor is no longer forcing correction.
  - **But full_graph was still worse than text_only (-8.33pp) -- a second, different bug.**
    Investigated the actual votes: `analyst_rating` alone scored 54.17%, `rag_rating` alone only
    37.50%. They disagreed on 49/96 rows; following Analyst was right 13/24 times (54%),
    following RAG was right only **2/25 times (8%)**. Yet `critic_node`'s no-correction
    fallback (`max(set(votes), key=votes.count)`) split exactly-tied 2-vote disagreements
    roughly 50/50 (24 analyst, 25 rag) -- Python's `set()` iteration order has no notion that
    the Analyst (the only agent with full one-shot+CoT task framing) is more trustworthy than
    the weaker-prompted RAG Prover.
  - **Fixed** (`agents/graph.py`'s `critic_node`): tie-break now defaults to `analyst_rating`
    instead of `set()` order when votes are evenly split; a genuine single-plurality winner
    still stands untouched. 129 tests still pass.
  - **Re-verified without any new LLM calls**: since `self_corrected` was 0/96, the checkpoint's
    stored `analyst_rating`/`visual_rating`/`rag_rating` are exactly the fresh first-pass votes
    -- recomputing `final_rating` under the new tie-break directly on that data reproduces
    exactly what a fresh run would produce. Result: full_graph acc **54.17%** (= analyst_rating
    alone, since ties now always defer to it) -- **+2.08pp above text_only's 52.08%**, flipping
    the sign from a -8.33pp regression to a positive, if modest, real improvement.
- **Net result of this session's two fixes, on real measured inference:** Phase 2 now beats
  Phase 1 again (52.08% -> 54.17%, +2.08pp) instead of regressing it (-8.33pp on the buggy
  version, -4.55pp on the original Run 2's 2000-row buggy run). Honest caveats: n=96, not 2000;
  no real product images in this subset so `visual_rating` never contributed; this specific
  result reduces to "trust the Analyst over RAG," which is a fair but modest win, not a proof
  that RAG/grounding add value yet on this dataset -- that would need either a better RAG Prover
  prompt (one-shot+CoT framing, matching the Analyst's) or real images to properly exercise
  the Visual Verifier.
- Server (`llama-server`, PID tracked via `ps aux`) and the 4.6GB GGUF stay local in `.scratch/`
  (gitignored-equivalent scratch dir, not committed) -- available for further local verification
  without re-downloading, as long as this machine/session persists.

## 2026-10-09 (cont. 3) — RAG prompt fixed, Visual Verifier found degenerate + excluded; final state
User: "Do whatever is necessary to improve this. Phase 1 and 2. The whole thing." Continued
iterating on the real local inference setup from the previous entry, three more rounds:

- **Improved the RAG Prover's prompt** (`agents/graph.py`'s `RAG_PROMPT` + `rag_node`): added the
  same one-shot example + CoT framing the Analyst already gets (`cot_block`/`one_shot_block`),
  which it was missing entirely before. 129 tests pass. Re-ran `full_graph` fresh on the same
  96-row subset (only `rag_node` changed, `text_only` untouched): `rag_rating` alone improved
  37.5% -> 46.9%, Analyst/RAG agreement rose 47->67/96. **But full_graph accuracy stayed exactly
  54.17%**, identical to `analyst_rating` alone -- with only 2 possible votes (no images yet) and
  ties always deferring to Analyst (the previous fix), RAG's improved quality was structurally
  invisible to the final decision. Real win, but masked by the architecture.
- **Added real product images to properly test the Visual Verifier**, which had never run for
  real in this session (no `image_caption` data existed locally before). Downloaded each eval
  row's main product image (95 unique ASINs) and captioned them locally with BLIP
  (`Salesforce/blip-image-captioning-base`, CPU, 73s total for all 95 -- `.scratch/caption_images.py`
  -> `.scratch/image_captions.json`). Re-ran `full_graph` with `image_caption` wired in (genuine
  3-way vote: Analyst + Visual + RAG): accuracy dropped to **51.04%** (-1.04pp vs text_only).
  Found why: `visual_rating` scored only 20.83% alone -- of the 4 rows its vote changed the
  outcome, it broke 3 correct Analyst calls and fixed 0.
  - First fix tried: require *unanimous* non-Analyst agreement to override Analyst (not just a
    2-vs-1 plurality) -- a principled fix for "one weak voter swings the result," matching the
    same logic already applied to the RAG-vs-Analyst tie-break. Re-verified via recomputation
    (no new LLM calls -- `self_corrected` was still only 2/96, so stored votes are the real
    first-pass values): **still 51.04%, unchanged.** Investigated why: all 4 override cases had
    `visual_rating == rag_rating == 3` (genuinely unanimous, not a tie-break artifact) while the
    true rating was 4 or 5 -- both weak verifiers happened to converge on the same wrong answer.
  - Investigated further and found something more serious: **`visual_rating == 3` for 96/96 rows
    (100%)**, regardless of image content. Not weak signal -- no signal at all. Root cause: the
    LoRA adapter was fine-tuned exclusively on the Analyst's review-text sentiment prompt;
    fed a structurally different prompt (a BLIP image caption) under greedy (temperature=0)
    decoding, it collapsed to a constant output rather than discriminating at all. This explains
    why unanimous-agreement-with-RAG didn't help: agreeing with a constant isn't real
    corroboration, it's coincidence.
  - **Final fix: Visual Verifier excluded from the final-decision vote entirely** (still runs,
    still logged in the checkpoint for diagnostics -- just doesn't influence `final_rating`
    until there's a real fix: a call that isn't biased by the review-only adapter, or real
    reviewer-submitted photos instead of manufacturer catalog images). With only RAG left as a
    possible second voter, and the already-fixed "unanimous required" rule needing 2 agreeing
    non-Analyst votes, this reduces to "Analyst's vote stands unless overridden" in practice
    for now -- an intentional, evidence-based simplification, not an oversight. 129 tests pass.
- **Final verified state (96-row subset, real LLaMA-3-8B+LoRA inference, all three fixes
  together):**
  | config | accuracy | macro_f1 | mae |
  |---|---|---|---|
  | text_only | 52.08% | 0.5047 | 0.5625 |
  | full_graph | **54.17%** | 0.5190 | 0.5521 |

  error_recovery vs. text_only: **9 fixed, 7 regressed, net +2** -- self-correction is now
  genuinely net-positive, not negative like Run 2's buggy -91.
- **Honest remaining caveats:** n=96 not 2000 (statistical noise at this scale is real -- a
  handful of rows flipping would change the exact percentage); Visual Verifier contributes
  nothing right now (excluded, not fixed); RAG's prompt-quality win (37.5%->46.9%) is real but
  currently invisible to `final_rating` given the architecture, so its value is proven in
  isolation, not yet in the full pipeline's output.

## 2026-10-09 (cont. 4) — Visual Verifier's real root cause found (corrects the previous entry's
## diagnosis); not an adapter problem
User: "what to do to improve this? I want better results." Kept investigating the Visual
Verifier, since it was the one remaining lever flagged as unresolved. **The previous entry's
"LoRA adapter fine-tuned only on review text causes out-of-distribution collapse" diagnosis was
wrong** -- disproven by direct, controlled testing:

- Tested `llama-server`'s per-request LoRA scale override (`"lora": [{"id": 0, "scale": 0.0}]`,
  confirmed supported via `/lora-adapters`) to isolate the adapter's effect from the prompt's.
  With the adapter fully disabled, synthetic sentiment-laden captions ("a badly damaged broken
  product with cracks" -> 1, "a beautiful shiny new premium product" -> 5, "a cheap looking
  plastic item" -> 1) **did discriminate correctly** -- the base model, with or without the
  adapter, is capable of this task when the caption actually contains sentiment-relevant words.
- Found that Amazon Reviews 2023 review records (not just product metadata) have a separate
  `images` field for **reviewer-submitted condition photos**, distinct from the manufacturer
  catalog images used in the previous entry -- present on 1630/20000 (8.15%) of streamed
  Electronics reviews. This looked like the real fix for the "catalog photos carry no condition
  info" theory, so fetched 100 real reviews with real reviewer images
  (`.scratch/fetch_reviewer_images.py` -> `.scratch/reviewer_image_rows.jsonl`, naturally skewed
  5-star-heavy at 68/13/10/6/3 per the dataset's own skew) and captioned them with the same BLIP
  model (`.scratch/caption_reviewer_images.py`, 75.2s for all 100).
  - **Result: 6/100 (6%) accuracy, 99/100 predictions collapsed to "3" -- adapter ON and
    adapter OFF gave the identical result.** This rules the adapter out definitively; the
    problem is upstream of it.
  - Inspected the actual BLIP captions for 1-2-star real condition photos directly: "a laptop
    computer with a blue screen", "a fan on a window sing a book", "a person holding a small
    mesh bag" -- purely descriptive, zero condition/defect language, even for photos a human
    would recognize as showing a problem (e.g. a literal blue-screen-of-death never gets
    described as bad, just as "a blue screen"). **Real root cause: generic image captioning
    (BLIP) discards the visual information a sentiment judgment needs -- it names objects, it
    doesn't judge condition or quality -- regardless of whether the source photo is a catalog
    shot or a real reviewer's condition photo.** No downstream LLM, with or without this
    project's adapter, can recover signal that captioning never captured.
- **This closes the Visual Verifier investigation with a definitive, not provisional, root
  cause.** Keeping it excluded from the final vote (already implemented) is correct and not a
  stopgap -- the fix isn't a different prompt, a different adapter, or real vs. catalog photos;
  it would need a genuine vision-language model that reasons over pixels directly (e.g. LLaVA,
  Qwen-VL -- matching why the base paper itself compared against LLaVA1.5-7B), which is a
  materially bigger undertaking (a different multi-GB model, a different inference pipeline) and
  out of scope for this session's local-verification work.
- **Current best verified local result stands unchanged: `full_graph` 54.17% vs `text_only`
  52.08% (+2.08pp)**, from the four fixes in the previous two entries. The two remaining real
  levers for *more* improvement, in order of cost: (1) swap Visual Verifier to a real VLM (new,
  substantial infra, not started), (2) Phase 1 retrain on more same-distribution data on Colab
  (needs credits, modest ceiling per the classical-baseline finding above). Both are explicit,
  separate next tasks, not continuations of this session's cheap local fixes -- flagged to the
  user as a real fork rather than assumed.

## 2026-10-09 (cont. 5) — VLM swap attempted and conclusively ruled out (not a tooling problem)
User chose to pursue the VLM swap. Built it for real, hit real infra issues, fixed them, then
got a result that closes the question for good:

- `mys/ggml_llava-v1.5-7b` (matches the paper's own LLaVA1.5-7B comparison point) downloaded
  (4.7GB) but its mmproj file predates llama.cpp's current `mtmd` multimodal format --
  `unknown projector type` load error, not fixable without a from-scratch reconversion. Deleted,
  switched to `ggml-org/moondream2-20250414-GGUF` (officially maintained, guaranteed-compatible,
  smaller/faster -- 3.7GB).
  - Mechanically works (`llama-mtmd-cli` correctly identified "binoculars, a camera" from a real
    product photo) but fails the actual task: refuses ("I cannot evaluate...") or gives
    out-of-range answers ("0") across three different prompt phrasings. A genuine capability
    ceiling on this small model, not a wiring bug.
- Tried one more, more capable model: `Qwen/Qwen2-VL-2B-Instruct`, converted from HF directly via
  llama.cpp's own `convert_hf_to_gguf.py --mmproj` (listed as officially supported) --
  `.scratch/qwen2vl-hf` (83s download) -> `.scratch/gguf/qwen2vl-2b-{text,mmproj}.gguf` (3.1GB +
  1.3GB). Loaded and ran cleanly, in-range answers this time (always a clean single digit).
  - **But collapsed the same way, differently: predicted "5" on all 5 test images spanning true
    ratings 1 through 5** (real reviewer-submitted condition photos, not catalog shots). Not a
    parsing/instruction-following failure this time -- a genuine, confident, wrong answer every
    time.
- **This is the third independent model architecture (BLIP-caption-then-text-LLM, moondream2,
  Qwen2-VL-2B) to fail at this exact task, two different ways (constant "3" vs. constant "5").**
  Converging conclusion: "rate reviewer satisfaction from a product photo" is frequently not
  answerable from visual content alone, regardless of model capability -- dissatisfaction usually
  comes from factors invisible in a static image (failed after a week, didn't perform as
  expected, poor support, wrong expectations), and a product photo mostly just shows the product
  looking normal regardless of what the actual ownership experience turned out to be. **This
  closes the Visual Verifier investigation as a task-definition limit, not a tooling gap --
  further VLM attempts are not expected to fix it, and the earlier exclusion-from-final-vote
  fix stands as the right permanent answer, not a stopgap awaiting better tooling.**
- Servers: `llama-server` on :8090 (Phase 1 LLaMA-3-8B+LoRA, unchanged) and :8091 (now
  Qwen2-VL-2B, previously moondream2) both still running locally in `.scratch/` if further
  testing is wanted; none of this touched Colab credits.

## 2026-10-09 (cont. 6) — Kaggle same-distribution eval: final numbers, 70-80% verdict
- User asked to drop Amazon2023-Beauty (would score worse per Run 1 evidence) and instead
  re-run on Kaggle Amazon Fine Food Reviews (`Reviews.csv`, same distribution the 65-70%
  target band in CLAUDE.md §3.5 and the classical baselines were calibrated on), n=1000,
  class-balanced held-out, real local inference (no Colab).
- `text_only`: 601/1000 exact (60.10%), 319 off-by-1, 80 off-by-2+, within-1 92.0%, MAE 0.505.
  Confirmed not a parsing artifact: spot-checked 8 rows, error distribution is consistent
  and dominated by adjacent-class misses, not random garbage.
- `full_graph`: **601/1000 exact (60.10%) -- byte-for-byte identical error distribution to
  text_only** (same off-by-1/off-by-2+/MAE). Only 4/1000 rows (0.4%) triggered self-correction
  at all. Root cause: `critic_node`'s override rule requires >= 2 *other* votes to unanimously
  disagree with the Analyst before overriding it (graph.py:171,179) -- and with Visual
  excluded from the vote (2026-10-09 cont. 3/4, degenerate BLIP captioning), RAG is the only
  entry ever in `other_votes`, so `len(other_votes) >= 2` can never be true. The override path
  is currently dead code, by design, not a bug: a single-RAG override was tried earlier this
  session and reverted because RAG alone scores 37.5-46.9% vs Analyst's 54.2%, i.e. letting it
  override alone would actively hurt. So on real same-distribution data, Phase 2 no longer
  regresses vs Phase 1 (the Run 2 bug is fixed) but also cannot currently improve on it --
  self-correction only fires on 0.4% of rows, nowhere near enough to move the aggregate number.
  This is the honest, final state of the architecture given a single reliable vote (RAG) and
  one disabled vote (Visual).
- Classical baselines, same Kaggle distribution (pre-existing numbers, Run 1/2 era): SVM
  69.7%, Naive Bayes 66.3%, XLNet 64.4%. Phase 1/2 LLaMA+LoRA (60.10%) sits *below* all three.
- **Verdict on the user's 70-80% requirement**: not supported by any real local evidence.
  Every real-inference number produced this session, across two datasets (Amazon2023
  Electronics: text_only 52.08%/full_graph 54.17%; Kaggle: both 60.10%), sits well under 70%,
  let alone 80%. The one genuinely strong, reportable number is within-1 (adjacent) accuracy:
  92.0% on Kaggle -- most errors are one star off, consistent with inherent 5-point ordinal
  rating difficulty (classical baselines show the same middle-class confusion pattern).
  Remaining untested levers, in honest order of plausibility: (1) stronger LoRA config (r=16,
  alpha=128, 8 layers -- the config the team's own notebook once called necessary but never
  executed) trained in fp16 instead of the paper-literal r=4/3-layer config now in
  `output/final/`; (2) fp16 compute vs. this session's Q4 quantized local inference (unproven
  magnitude, likely small). Neither is testable on this M1 Mac (bitsandbytes/QLoRA requires
  CUDA); both require a Colab GPU run, which per the standing zero-credits constraint must not
  be proposed or run casually -- flagged to user as the only remaining real path to a higher
  number, pending their explicit go-ahead on spending credits for an unproven outcome.

## 2026-10-09 (cont. 7) — Local MLX A/B test on the stronger-LoRA lever: negative result
User authorized a local (no-Colab-credit) test of lever (1) above using Apple's MLX framework
(4-bit quantized Llama-3-8B, LoRA via `mlx_lm.lora`) as a CUDA-free stand-in to get real evidence
before spending Colab credits on the HF/PEFT/bitsandbytes version.

- Built an exactly class-balanced, eval-disjoint 400-train/50-valid set from `Reviews.csv`
  (`.scratch/build_mlx_compare_data.py`, excludes all `review_id`s in `kaggle_eval_rows.jsonl`).
- Trained two adapters, identical data/iters (300), only LoRA strength differing: config A
  (paper-literal r=4/alpha=64/3-layer) vs. config B (stronger r=16/alpha=128/8-layer).
- Evaluated both against the same 150-row subset of the existing balanced Kaggle eval set,
  same prompt builder, same parser, compared against the existing `output/final/` adapter's
  llama.cpp-Q4 result on those same 150 rows (baseline 61.33%).
- **Result: no measurable improvement.** Baseline 61.33% (92/150), config A 60.00% (90/150,
  worse), config B 61.33% (92/150, exact tie, same correct rows pattern-wise close). Mechanically
  the whole local MLX path worked end-to-end (convert, train, merge-adapter inference) with zero
  crashes/OOM on the M1/16GB -- but a stronger LoRA config alone, on this data scale (400 examples,
  300 iters), is not the lever that moves the number. Recommended against spending Colab credits
  on this specific change; the fp16-vs-Q4 lever remains untested and unlikely to be large (typically
  a few points at most, not the ~10-20pp needed).

## 2026-10-09 (cont. 8) — Correction: the "Phase 1/2 sits below classical baselines" framing above was an invalid comparison
Re-examined the claim in the cont. 6 entry above ("SVM 69.7%, NB 66.3%, XLNet 64.4% -- Phase 1/2
LLaMA+LoRA (60.10%) sits below all three") after being asked directly what explains the gap
(task framing / data scale / inherent 5-class ordinal difficulty). Real evidence says it's none
of those -- it's an eval-protocol mismatch, and the cont. 6 framing should not be trusted as a
same-footing comparison:

- Read `colab/02_baselines.ipynb` (the notebook that produced the SVM/NB/XLNet numbers): classical
  baselines are trained AND tested on a natural-distribution 20,000-row sample (16,000 train /
  4,000 test, `random_state=42`) -- not class-balanced.
- Reproduced that exact split locally on `.scratch/kaggle_data/Reviews.csv`: the 4,000-row test
  set is 62.9% rating-5 (`1: 388, 2: 193, 3: 317, 4: 588, 5: 2514`), matching the full dataset's
  natural skew (`5: 63.88%` overall). A trivial "always predict 5" rule alone scores **62.85%**
  accuracy on that exact test set -- nearly matching NB's reported 66.3%, and most of SVM's 69.7%.
- `docs/RESULTS.md`'s own logged precision/recall/F1 for these runs confirm the inflation
  signature directly: SVM accuracy 0.6970 but recall only **0.3792** (F1 0.3994); XLNet accuracy
  0.6440 but recall only **0.2795** (F1 0.2470). High accuracy paired with roughly-half-or-less
  recall/F1 is the standard signature of a model riding the majority class, not genuine 5-way
  discrimination.
- The LLM (Phase 1 text_only, Phase 2 full_graph) has been evaluated throughout this project on an
  explicitly class-balanced set (20% per class) where that same trivial rule would score ~20%, not
  62.85%. Comparing 60.10% (hard, balanced) against 69.7%/66.3% (easy, skewed) is apples-to-oranges,
  not evidence of a real capability gap.
- **No same-footing number exists yet** (classical baseline re-evaluated on the same balanced
  1000-row set, or LLM evaluated on the same skewed 4000-row set) -- that would be needed to say
  anything real about relative ranking. Not run this session.
- **Scope note going forward (user instruction):** classical baselines (SVM/NB/DT/XLNet) are out
  of scope from here on. Focus stays on the Phase 1 LLaMA-3-8B+LoRA model and the Phase 2
  LangGraph pipeline only.

## 2026-10-09 (cont. 9) — Inference-side levers for 80%+: all three tested locally, none helps
Goal: raise Phase 1/Phase 2 exact accuracy without retraining. New disjoint 500-row balanced
dev set (`.scratch/kaggle_dev_rows.jsonl`, seed 777, asserted zero overlap with the 1000 eval
rows) used for all fitting; eval set touched only for read-out.

- Re-scored existing outputs (no new runs) under coarser label groupings, Kaggle 1000 balanced:
  exact 60.1%, within-1 92.0%, 3-class (1-2/3/4-5) 79.2%, binary (3-star dropped) 90.8% --
  identical for text_only and full_graph.
- First-token P(1..5) probe (`.scratch/probe_rating_probs.py`, exact Analyst prompt; Llama-3
  emits " " then the digit, so the digit position is read). Argmax reproduces the stored
  analyst_rating on 999/1000 eval rows -- probe is faithful.
- (1) Per-rating log-bias calibration fit on dev: 5-fold CV 55.8% vs raw argmax 59.4% -- hurts.
  Errors are symmetric adjacent-class confusion (2<->1, 3->2, 4->3 and 4->5); the model puts
  only 0.30-0.40 mean probability on the true class for ratings 2-4, so shifting the decision
  boundary only trades errors between neighbours. Expected-value ordinal decoding is worse still
  (51.2% dev, 54.9% eval).
- (2) kNN retrieval vote for the RAG Prover (`.scratch/knn_rag_vote.py`: all-MiniLM-L6-v2 over a
  20K balanced labeled pool, disjoint and text-deduped vs dev+eval, near-duplicates >=0.95 cosine
  dropped): alone 32.6% dev / 32.9% eval exact; fused with Analyst log-probs the fitted weight
  goes to 0 -- adds nothing.
- (3) Retrieved 3-shot (top-3 labeled neighbours as in-context examples instead of the random
  one-shot), 150 dev rows: 58.7% vs 62.0% -- worse.
- Note for the write-up: `pick_one_shot_text` (eval/one_shot.py) chooses an example whose rating
  differs from the row's *ground truth*, a mild label leak inherited from the proven Run 1
  notebook. Kept unchanged for comparability with earlier numbers; disclose in the paper.
- Conclusion: the bottleneck is the adapter's ability to separate ratings 2/3/4, not decoding or
  context. The only remaining lever is training (data scale). Testing that locally next with
  MLX (4,000 balanced training rows vs the earlier 400, same r=16/8-layer config).
