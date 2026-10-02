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
