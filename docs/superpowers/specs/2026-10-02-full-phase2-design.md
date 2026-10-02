# Full Phase 2 (agentic) + Phase 1 retrain: design

Date: 2026-10-02. Status: awaiting user review.

## Goal
Replace the Run 1 prototype (plain-Python agents, review-only RAG, vote-stdev dissonance,
2,000 skewed All_Beauty samples) with a full implementation matching the mission spec in
`CLAUDE.md`, trained and evaluated on Colab, with every run checkpointed and reproducible.

## Decisions (user-confirmed)
- **Compute:** code is written and tested locally (stub LLM, tiny-model CPU smoke test);
  full training/eval runs on the user's Colab; results come back into `results/run2/`.
- **Scope:** LangGraph + spec-store RAG + new dissonance; larger balanced eval + ablations;
  classical baselines; full Phase 1 retrain.
- **Dataset category:** Amazon Reviews 2023 **Electronics** (Run 1 used `All_Beauty`).
  Electronics has real technical specs (ports, battery, compatibility), which the spec-store
  RAG Prover needs. Eval set is balanced per star rating (not skewed to 5-star).
- **Persistence:** every training stage writes checkpoints, best adapter, and all parameters
  to a Google Drive run directory so best params can never be lost to a Colab disconnect.
- **Documentation:** `docs/PROGRESS.md` is a running log, updated after every step.

## Architecture
Package `src/agentic_sentiment/` (testable locally) + four thin Colab notebooks.

| Module | Role |
|---|---|
| `data/phase1.py` | Kaggle prep: TextBlob DQC, stratified, VGST, neutral oversample, cap, split |
| `data/amazon2023.py` | Electronics loader; balanced eval slice; spec-store source records |
| `train/phase1.py` | LoRA r=16, alpha=128, 5 ep. Steps-based save, best-on-val, resume |
| `rag/spec_store.py` | LanceDB store of product specs/descriptions; claim-vs-spec check -> Gf |
| `agents/graph.py` | LangGraph StateGraph: Analyst -> (Visual Verifier, RAG Prover) -> Critic -> loop (max 2) |
| `agents/critic.py` | Dissonance `D = Norm(H - (Ev + Gf))`; vote-stdev kept as ablation |
| `eval/` | Per-sample JSONL checkpoint + resume; metrics; charts; ablations |
| `baselines/` | DT+CountVec, SVM(sigmoid)+TF-IDF, MultinomialNB, XLNet-large (checkpointed) |

Notebooks: `01_train_phase1`, `02_baselines`, `03_phase2_eval_ablations`, `04_report`.
Each is idempotent: mount Drive, read/write `runs/<id>/`, skip finished work.

## Run directory (Drive)
`runs/<timestamp>/`: `config.json` (all hyperparameters, seed, git commit), `checkpoints/`,
`best_adapter/`, `trainer_state.json`, `metrics.json`, `BEST.md` (best step, val loss).

## Ablations
text-only, +metadata, +image, +RAG(reviews), +RAG(specs), full graph; dissonance
vote-stdev vs Norm formula; self-correction on/off.

## Error handling
Bad parse / missing image / missing spec degrade gracefully and are logged per sample;
the run never aborts on one bad row.

## Testing
pytest with a stub LLM (parser, dissonance, graph routing, resume logic); 2-step tiny-model
CPU training smoke test proving checkpoint + resume. Real numbers only from Colab runs.

## Paper
After `results/run2/` is returned, regenerate tables/figures and update
`paper/conference/ConferencePaper.tex` and `paper/main.tex`. No numbers before then.
