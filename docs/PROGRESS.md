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
