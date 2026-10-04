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
