# Results log

Per CLAUDE.md §7: every real completed run's config, dataset slice, and
metrics, appended automatically by the Colab notebooks right after each run
finishes (`agentic_sentiment.eval.results_log.format_results_entry()`). No
projected or estimated numbers belong here — only runs that actually
completed.

Run 1 (All_Beauty, 2,000 samples) predates this file; its numbers live in
`paper/main.tex`'s "Results — Run 1" section. Everything below is Run 2
(Electronics, balanced) and later.
## 2026-10-04 — Phase 1 LoRA retrain

- **dataset:** Kaggle arhamrumi/amazon-product-reviews
- **run_id:** phase1

| name | eval_loss |
|---|---|
| phase1 | 0.0443 |
