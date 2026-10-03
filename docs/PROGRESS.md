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
