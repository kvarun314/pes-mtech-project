# Deferred: purge `dev2.jsonl` from git history (NOT done — kept for future reference)

Status: **deferred by user decision, 2026-10-25** ("don't purge history"). Nothing has been rewritten.

## The issue
- `v2_reflective/e23/dev2.jsonl` holds 2,000 balanced Amazon Reviews 2023 Electronics reviews, including their text. It was committed in `1031426` (Phase H, 2026-10-24) on branch `e23-leakfree-oct2026`.
- The GitHub repo `kvarun314/pes-mtech-project` is **public**. As of 2026-10-25 the API returned HTTP 200 to an unauthenticated request.
- 2026-10-25: removed from tracking (`git rm --cached`) and added to `.gitignore` (commit `1c1e9ea`). The file is still on disk, and it is still present in the history of commits `1031426` through `1c1e9ea^`.
- `main` never contained it.
- Other history checked on 2026-10-25: no other committed file contains a `review_text` field. The Run 1 checkpoint `results/run1/phase2_checkpoint.jsonl` (on `main`) has a short `snippet` field. That is Run 1 record data and is left as is.

- Also found 2026-10-25: `v2_reflective/e23/halluc_p1.out` (tracked) contains short review excerpts (~20 quotes from the hallucination study). If purging, add `--path v2_reflective/e23/halluc_p1.out` (or replace the file with its summary lines first). The labelled copy in `results/e23/` is summary-only.

## When to revisit
- Before the paper or thesis links to the repo (code availability statement), or before sharing it with examiners.
- Alternative with no history rewrite: make the repo private (GitHub settings), or publish a fresh clean repo for the paper.

## How to purge, if decided later (run by the user; it rewrites history)
```bash
cd /Users/varunk/Documents/PES_Project_Phase1/pes-mtech-project
cp v2_reflective/e23/dev2.jsonl ../results_archive/                                  # keep the data locally
git bundle create ../results_archive/pes_git_before_dev2_purge.bundle --all            # full history backup
.venv/bin/pip install -q git-filter-repo
.venv/bin/git-filter-repo --force --invert-paths --path v2_reflective/e23/dev2.jsonl --refs e23-leakfree-oct2026
git remote get-url origin || git remote add origin git@github.com:kvarun314/pes-mtech-project.git
git push --force origin e23-leakfree-oct2026
```
Consequences:
- Every commit ID from `1031426` onward changes, so update any IDs quoted in `LOG.md` or the docs.
- GitHub can keep the old commits reachable by their ID until garbage collection. Full removal needs a request to GitHub Support.
