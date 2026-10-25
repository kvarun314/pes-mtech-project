# pes-mtech-project — START HERE

**PES MTech project:** agentic sentiment analysis with LLaMA-3-8B on Amazon Reviews 2023 Electronics (1–5 stars).
- **Phase 1:** reproduce the baseline paper (Wang et al. 2025, LLaMA-3-8B + LoRA).
- **Phase 2 (current focus):** a multi-agent system with leak-free, pre-registered evaluation.
- **Current main system (SH):** a frozen LLaMA-3-8B layer-20 probe plus a TF-IDF Evidence agent, combined by an LR Reflector.
- **SH results:** eval 67.5%, natural_test 75.7% exact.

## Where everything is

| I want… | Go to |
|---|---|
| **The presentation** | `../Agentic_LLaMA3_Presentation.pptx`, the **only** current deck. Build it with `/opt/homebrew/bin/python3 v2_reflective/build_deck_e23.py` |
| The deck in Google Slides (always current; synced by the builder) | https://docs.google.com/presentation/d/1V08iho4p7gdllKPxRVroaPhj6oo8Km1_/edit (Drive `PES_backups/current_deck/`). View and present there; make changes through the builder, because edits in Slides are overwritten on the next build |
| Old decks (kept, never deleted) | `../old_decks/` |
| What to say on each slide | `presentation/Presenter_Notes.md` (Part 0 = current deck) |
| Full story: Phase 1 to now, failures, why | `presentation/Project_Journey.md` |
| Explaining the `v2_reflective/` code | `presentation/V2_Reflective_Guide.md` |
| Demo notebook | `presentation/Presentation_Demo.ipynb` (being rebuilt for Phase 2 v3) |
| **Lab notebook**: every experiment, rule declared first | `v2_reflective/LOG.md` |
| One-page summary of what worked and what failed | `v2_reflective/EXPERIMENTS_SUMMARY.md` |
| Frozen result tables | `v2_reflective/e23/results_frozen.md` (regenerate: `python v2_reflective/e23/freeze_tables.py`) |
| Status updates, weekly notes | `v2_reflective/progress/` (`DAILY_STATUS.md`, `week_NN.md`, `UPDATE_<date>.md`) |
| Current Phase 2 design (v3) | `docs/superpowers/specs/2026-10-25-phase2-v3-design.md` |
| Older plans (paper plan, roadmap, contingency) | `v2_reflective/plans/` |
| Papers | `paper/main.tex` (thesis), `paper/conference/ConferencePaper.tex`; shared results section `paper/e23_results.tex` |
| Baseline paper and original proposal (PDF) | `docs/reference/` |
| Plagiarism / AI-content audit (run before every draft) | `paper/integrity/` (`README.md`, `audit.py` → `INTEGRITY_REPORT.md`) |
| Deferred maintenance notes (e.g. the git-history decision) | `docs/maintenance/` |
| Backups | `v2_reflective/backup_artifacts.sh` and `verify_backup.sh`; off-disk copy in Google Drive `My Drive/PES_backups/` |

## Folder layout

```
pes-mtech-project/
├── README.md            this map
├── CLAUDE.md            working rules (loaded automatically by Claude Code)
├── presentation/        everything for talking: presenter notes, journey, v2_reflective guide, demo notebook
├── v2_reflective/       Phase 2 experiment workspace (all current results)
│   ├── LOG.md, EXPERIMENTS_SUMMARY.md
│   ├── plans/           older plan documents
│   ├── progress/        status updates
│   ├── e23/             Electronics 2023: data, agents, gates, readouts, freeze  ← all current numbers
│   ├── build_deck_e23.py, build_presentation_v4.py, build_demo_notebook.py
│   └── (top-level *.py) Kaggle close-out scripts, Oct 9–12 (historical; paths inside are repo-root relative)
├── paper/               LaTeX papers and figures (paper/figures/e23/)
├── docs/
│   ├── superpowers/     design specs and plans
│   ├── reference/       baseline paper, original proposal
│   ├── archive/         superseded docs (old CLAUDE.md, CODE_EXPLAINED)
│   ├── maintenance/     deferred decisions with exact steps (git_history_dev2_purge.md)
│   └── PROGRESS.md, RESULTS.md, CHECKPOINT_HISTORY.md   (Phase 1 / Run 1–2 era)
├── src/agentic_sentiment/, tests/   LangGraph package (Phase 1 / Run 1–2; Phase 2 v3 agents plug into agents/graph.py)
├── colab/               Colab notebooks (Phase 1 training, Run 1–2)
└── results/run1, run2   Run 1 / Run 2 outputs (historical)
```

Outside the repo (`PES_Project_Phase1/`): the deck, `old_decks/`, `results_archive/`, `archive_early_2026/` (March 2026 material), `Individual Journal Paper/`, `llama_sentiment_baseline/`.

## Run the checks

```bash
.venv/bin/pytest                                         # package tests
python v2_reflective/e23/freeze_tables.py                # recompute every frozen table (~3 s)
cd paper && pdflatex main.tex && pdflatex main.tex       # thesis PDF
```
