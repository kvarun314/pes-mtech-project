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
## 2026-10-04 — Classical + XLNet baselines (Wang et al. Table 3 configs)

- **dataset:** Kaggle arhamrumi/amazon-product-reviews
- **n_samples:** 20000

| name | accuracy | precision | recall | f1 | eval_loss | eval_accuracy | eval_precision | eval_recall | eval_f1 |
|---|---|---|---|---|---|---|---|---|---|
| decision_tree | 0.5840 | 0.3481 | 0.3383 | 0.3425 | — | — | — | — | — |
| svm_sigmoid | 0.6970 | 0.5030 | 0.3792 | 0.3994 | — | — | — | — | — |
| naive_bayes | 0.6625 | 0.4635 | 0.4342 | 0.4469 | — | — | — | — | — |
| xlnet | — | — | — | — | 1.1576 | 0.6278 | 0.1256 | 0.2000 | 0.1543 |
## 2026-10-04 — Classical + XLNet baselines (Wang et al. Table 3 configs)

- **dataset:** Kaggle arhamrumi/amazon-product-reviews
- **n_samples:** 20000

| name | accuracy | precision | recall | f1 | eval_loss | eval_accuracy | eval_precision | eval_recall | eval_f1 |
|---|---|---|---|---|---|---|---|---|---|
| decision_tree | 0.5840 | 0.3481 | 0.3383 | 0.3425 | — | — | — | — | — |
| svm_sigmoid | 0.6970 | 0.5030 | 0.3792 | 0.3994 | — | — | — | — | — |
| naive_bayes | 0.6625 | 0.4635 | 0.4342 | 0.4469 | — | — | — | — | — |
| xlnet | — | — | — | — | 1.0069 | 0.6440 | 0.2539 | 0.2795 | 0.2465 |
