# Paper plan — written 2026-10-17 (read this first in a new session)

See also `PROJECT_ROADMAP.md` for the whole project (thesis, deck, notebook, reproducibility, archive, submission).

Rule for every step below ([[document-everything-checkpoints]] in memory): resumable scripts (append, skip done ids), a log file, output teed to a `.out` file next to the script, and a dated `LOG.md` entry per experiment. No number goes in the paper unless it traces to `LOG.md` or a `.out` file.

## Where we are (all documented in LOG.md)
- Dataset: Amazon Reviews 2023, Electronics only. Frozen splits in `e23/`: evid_train 100K balanced, dev 500 balanced, eval 1,000 balanced, natural_test 2,000 natural. Kaggle = Appendix A only.
- Phase 1: LLaMA-3-8B MLX QLoRA trained on 2,000 raw balanced Electronics rows (arm R, `e23/p1/adapter_R`), zero-shot prompt. Arm P (paper preprocessing) kept as a dev-only ablation.
- Phase 2: Analyst (arm R probs) + Evidence (TF-IDF LR, 100K raw balanced) -> stacker fit on 500 balanced dev rows. Output: `e23/phase2_e23.out`.
- Results: eval 54.0 / 62.4 / 63.9 (analyst / evidence / stack); natural_test 53.4 / 69.3 / 73.4. Stack vs evidence: +1.5 (ns) on eval, +4.0 (sig) on natural_test.
- Draft text: `paper/results_paper_e23.md` (not merged into main.tex; main.tex is still the Run 1 narrative).
- Leak audit (one-shot label leak worth ~8 pts), withdrawn calibrator gain, and preprocessing ablation (TextBlob DQC hurts) are finished findings.
- Hallucination study on the Electronics adapter: `e23/halluc_p1.py gen` was running (150 rows). When `halluc_DONE` is in `e23/p1/stage2.txt`, run `halluc_p1.py score | tee e23/halluc_p1.out` and log it. If the process died, rerun `gen` (it resumes).

## The weakness a reviewer will hit
1. TF-IDF LR on 100K rows beats the fine-tuned LLaMA (62.4 vs 54.0 on eval).
2. Stack beats evidence-alone by only +1.5 on balanced eval (not significant).
3. natural_test gain is confounded: stacker was fit on balanced dev, so part of the +4.0 is a class-prior effect.
4. The "Reflector" is a logistic regression over two probability vectors; "agentic" is a stretch.

## Work plan (in order; each step ends with an `.out` file + LOG.md entry)
- **S1. Natural-prior dev set.** Build a natural-distribution dev set from the Electronics pool (disjoint from every existing split by id and text; same sampling seed convention). Refit the stacker on it (and on prior-reweighted balanced dev as a second config), re-read natural_test once. Removes confound 3. Script `e23/stack_natural_dev.py`. Needs: Analyst and Evidence probs on the new dev rows (Analyst scoring via `eval_p1.py`-style resumable run).
- **S2. Baselines that answer "why LLaMA?".** (a) base LLaMA-3-8B without LoRA, same zero-shot prompt, on eval + natural_test; (b) a small fine-tuned encoder (DeBERTa-v3-base or similar) on the same 100K-or-2K rows. Report each against the stack. Decide the encoder size to fit the M1 16GB machine (MLX or CPU/MPS); log time and config.
- **S3. Variance for the +1.5.** Resample (bootstrap or repeated seeds for the evidence training subset and the stacker fit) and report the spread of stack-minus-evidence on eval and natural_test.
- **S4. Figures/tables for the paper.** Per-class recall table and confusion matrices for analyst / evidence / stack from `e23/phase2_trace_{eval,natural_test}.json`; a bar chart of the main table with CIs. Script `e23/make_figures.py`, outputs under `paper/figures/e23/`.
- **S5. Decide the angle with the user, then merge.** Two honest framings: (A) "agentic verification helps" — only defensible if S1/S2/S3 still show a clear stack gain; (B) "leak audit + preprocessing ablation + honest negative/small result for LLM agents vs classical evidence". Choose the target file (thesis `paper/main.tex` or `paper/conference/ConferencePaper.tex`), then rewrite from `paper/results_paper_e23.md`. Run 1 / Run 2 numbers are historical only.
- **S6 (optional, lowest priority).** Rebuild deck (`build_deck.py`) and demo notebook (`build_demo_notebook.py`) from the e23 numbers; both are Kaggle-centric today.

## Decisions still open (ask the user)
- Paper angle A vs B (S5) and target paper file.
- Encoder baseline choice for S2 given the 16GB machine.
- Whether the Visual Verifier / images stay out entirely (current state: excluded, constant output with 3 vision models).

## Things not to redo
- Do not touch eval/natural_test for model selection; all choices on dev only. Each stacker config is read out once.
- Do not filter test sets. Evidence model trains on raw balanced data.
- Do not switch datasets. Do not report the 70.7% Kaggle number or the +5.0 calibrator gain as results.
