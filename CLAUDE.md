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

- One deck only: `../Agentic_LLaMA3_Presentation.pptx`. Rebuild it in place with `/opt/homebrew/bin/python3 v2_reflective/build_deck_e23.py`. Never create v7/v8 copies.
- Old decks stay in `../old_decks/` and are never deleted.
- **Deck structure (user, 2026-10-25): at most 25 slides.** Flow: intro → Phase 1, the baseline paper (what it does) → its shortcomings → Phase 2 intro → in depth → how it solves the problem → results of Phase 1 → results of Phase 2 → in depth → conclusions. Order is the `MAIN` list in `build_deck_e23.py`; references use `§key§` tokens; every build writes `presentation/deck_slide_map.md`. Keep `Presenter_Notes.md` Part 0 in the same order.
- After a deck change, update `presentation/Presenter_Notes.md` Part 0 (slide map and scripts), then render the changed slides with `soffice` and check them visually.

## Experiment discipline

- Every experiment gets a gate in `LOG.md` before it runs, checked on dev / dev2 / natural_dev only. A failure never touches eval or natural_test.
- Every reported number traces to a `.out` file. `freeze_tables.py` regenerates the tables. The deck and the papers copy from it.
- Long jobs are chunked and resumable (`queue_*.sh` + `stage_*.txt`).
- One heavy job at a time. Run only on charger, never overnight on battery, and never during a presentation. Check swap before launching.
- Single dataset: Amazon Reviews 2023 Electronics. Run locally; flag explicitly when Colab is needed (credits from November; every Colab run must be worth it).

## After each logged step

1. Write the `LOG.md` entry.
2. Back up: `backup_artifacts.sh`, then `verify_backup.sh` with an absolute path. The script then copies the backup to Drive.
3. Commit and push to branch `e23-leakfree-oct2026`. End the message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Never

- Commit review text, data `.jsonl`, `.pptx`, `.tar.gz`, or any file over 100 MB. `kaggle_json.py` is a secret and is gitignored.
- Write paper prose. The user writes it (plagiarism and AI-content each under 5%). Claude supplies facts, tables, figures and `% FACT` comments.
- Do anything in Google Drive except copy (`cp -n`) into `My Drive/PES_backups/`. No sync tools, moves or deletes.
  - One exception, approved by the user on 2026-10-25: `build_deck_e23.py` overwrites `My Drive/PES_backups/current_deck/Agentic_LLaMA3_Presentation.pptx` on every build, so the copy opened in Google Slides is always current.
- Delete old material. Archive it instead.
- Rewrite git history or force-push. The user decided against it on 2026-10-25; see `docs/maintenance/git_history_dev2_purge.md`.
