# CLAUDE.md — working rules for this project

The map of every file is in `README.md`. Read it first. The previous long version of this file (Run 1/2 history and the original mission brief) is in `docs/archive/CLAUDE_2026-10-09.md`.

## Start of every session

1. Read `README.md`.
2. Read the last entries of `v2_reflective/LOG.md` and `v2_reflective/progress/DAILY_STATUS.md`.
3. Read any `v2_reflective/e23/stage_*.txt` for paused or running jobs. Check `pgrep -fl python` before starting anything heavy.
4. The current plan is `docs/superpowers/specs/2026-10-25-phase2-v3-design.md`. Phase 2 is the only focus.

## Where new files go (do not create files anywhere else)

| New thing | Location |
|---|---|
| Experiment script, its `.out` / `.log`, probabilities | `v2_reflective/e23/` (paths in scripts are repo-root relative: `v2_reflective/e23/...`) |
| Labelled copy of every result (index) | `results/e23/` via `v2_reflective/e23/collect_results.py` (add new outputs to its list) |
| Experiment declaration and result | a dated entry in `v2_reflective/LOG.md`, written **before** the run (gate) and after it (result) |
| Change in what worked / failed | `v2_reflective/EXPERIMENTS_SUMMARY.md` |
| Status update, weekly note | `v2_reflective/progress/` |
| Design spec / implementation plan | `docs/superpowers/specs/`, `docs/superpowers/plans/` |
| Presenter notes, talk scripts, demo notebook | `presentation/` |
| Paper tables/figures | `paper/e23_results.tex`, `paper/figures/e23/` |
| Temporary files | the session scratchpad, never the repo |
| Superseded docs | `docs/archive/` (move, never delete) |
| Paper integrity audit (AI share, overlap) | `paper/integrity/`; rerun `audit.py` after any paper change |
| Deferred decisions / maintenance notes | `docs/maintenance/` |

Do not add new top-level files to the repo or to `PES_Project_Phase1/`. If something does not fit the table, ask the user.

## Deck

- One deck only: `presentation/Agentic_LLaMA3_Presentation.pptx` (moved there 2026-11-20, user request; gitignored). Rebuild it in place with `/opt/homebrew/bin/python3 v2_reflective/build_deck_clean.py`. Never create v7/v8 copies. The previous builder is `docs/archive/build_deck_e23_2026-11-20.py`.
- Old decks stay in `../old_decks/` and are never deleted.
- **Deck structure: at most 25 slides (user, 2026-10-25). Order (user, 2026-11-20): reference paper in a nutshell → its stack → our implementation and what we found → its novelty; then our approach → stack → why it works → approaches tried → what we kept → results → why it is better → conclusion. Apple-inspired layout (user, 2026-11-20): one idea per slide, large titles, grey rounded tiles, deep-navy stage for big moments. Since 2026-11-20 a clean deck (user: "more clean, more straightforward and simple, showcasing the analysis, same colour palette"; "don't cram info into slides, make it flow naturally").** One message per slide, big numbers, native charts; details go in the speaker notes. Flow: intro → Phase 1, the reference paper (what it does, what we found, its result) → Phase 2 (principle, system) → results → conclusions. Order is the order of the slides in `build_deck_clean.py`; every build writes `presentation/deck_slide_map.md`. Keep `Presenter_Notes.md` Part 0 in the same order.
- After a deck change, update `presentation/Presenter_Notes.md` Part 0 (slide map and scripts), then render the changed slides with `soffice` and check them visually.

## Experiment discipline

- Every experiment gets a gate in `LOG.md` before it runs, checked on dev / dev2 / natural_dev only. A failure never touches eval or natural_test.
- Every reported number traces to a `.out` file. `freeze_tables.py` regenerates the tables. The deck and the papers copy from it.
- Long jobs are chunked and resumable (`queue_*.sh` + `stage_*.txt`).
- One heavy job at a time. Run only on charger, never overnight on battery, and never during a presentation. Check swap before launching.
- Single dataset: Amazon Reviews 2023 Electronics. Run locally; flag explicitly when Colab is needed (credits from November; every Colab run must be worth it).

## After each logged step

1. Write the `LOG.md` entry.
2. Refresh the labelled results: `.venv/bin/python v2_reflective/e23/collect_results.py`.
3. Back up: `backup_artifacts.sh`, then `verify_backup.sh` with an absolute path. The script then copies the backup to Drive.
4. Commit and push to branch `e23-leakfree-oct2026`. No Co-Authored-By trailer (user, 2026-11-20: commits are the user's own).

## Never

- Commit review text, data `.jsonl`, `.pptx`, `.tar.gz`, or any file over 100 MB. `kaggle_json.py` is a secret and is gitignored.
- Leave paper drafts unmarked. Since 2026-11-14 (user's decision) Claude writes full draft content for the papers and the user rewrites it in their own words before submission (plagiarism and AI-content each under 5%). Every draft file says at the top that it is Claude-drafted, and every number in it traces to a `.out` file.
- Do anything in Google Drive except copy (`cp -n`) into `My Drive/PES_backups/`. No sync tools, moves or deletes.
  - One exception, approved by the user on 2026-10-25: the deck builder (`build_deck_clean.py`) overwrites `My Drive/PES_backups/current_deck/Agentic_LLaMA3_Presentation.pptx` on every build, so the copy opened in Google Slides is always current.
- Delete old material. Archive it instead.
- Rewrite git history or force-push. The user decided against it on 2026-10-25; see `docs/maintenance/git_history_dev2_purge.md`.
