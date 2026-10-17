# Project roadmap — everything, not only the paper (written 2026-10-17)

Companion to `PAPER_PLAN.md` (detailed paper steps S1-S6). Same rule on every item: resumable runs, log file, `.out` file, dated `LOG.md` entry, nothing reported unless it traces to a logged output. Deadlines, venue and viva date are NOT known to Claude: fill the "Due" column with the user.

## Workstreams

| # | Workstream | Deliverable | State | Depends on | Due |
|---|---|---|---|---|---|
| W1 | Experiments (Electronics 2023) | Frozen splits, Phase 1 adapter R, evidence model, stack, ablations, leak audit | Core done; S1-S3 pending (natural-prior stacker, base-LLaMA + encoder baselines, variance of +1.5); hallucination study on adapter R finishing | - | ? |
| W2 | Research paper (journal/conference) | Final PDF in target venue template | Draft text only: `paper/results_paper_e23.md` | W1 S1-S3, angle decision | ? |
| W3 | Thesis/report (`paper/main.tex`) | Full report with Electronics results | Still the Run 1 / All_Beauty narrative; needs rewrite of Abstract, Datasets, Results, Analysis, Conclusion | W1, W2 text | ? |
| W4 | Conference paper (`paper/conference/ConferencePaper.tex`, EAI/LNCS) | Short version | Run 1 numbers; decide if still needed | W2 | ? |
| W5 | Presentation/viva deck | `Agentic_LLaMA3_Presentation_v5_electronics.pptx`, built by `v2_reflective/build_deck_e23.py` | v5 built 2026-10-17 with current results + plan; REBUILD from the script after the Nov 1 freeze; `Presenter_Notes.md` still stale | W1 final numbers | ? |
| W6 | Demo notebook | `build_demo_notebook.py` rebuilt on e23 splits and probs | Kaggle-centric; stale | W1 | ? |
| W7 | Code and reproducibility | One command per experiment; README with exact run order; pinned `requirements`/`pyproject`; seeds; tests green | `src/agentic_sentiment` package + 129 tests exist; the e23 scripts live outside it in `v2_reflective/e23/` | W1 | ? |
| W8 | Artifact archive | Adapters, splits (ids only if license requires), probs, traces, `.out` files, LOG.md in one tagged bundle | Files exist in place; no bundle, no checksum list, no git (not a git repo) | W1 | ? |
| W9 | Docs cleanup | Update stale `CLAUDE.md`, `README.md`, `PROJECT_CHANGES_SUMMARY.md`, `IMPLEMENTATION_STATUS.md`, `CODE_EXPLANATION.md` to the Electronics leak-free state; mark Run 1/Run 2 as historical | Stale | W1 | ? |
| W10 | Submission logistics | Venue, template, page limit, plagiarism/similarity check, supervisor review, co-author list, ethics/data statement, code/data availability statement | Unknown | W2 | ? |

## Order of work (proposed)
1. W1: finish hallucination study, then S1 (natural-prior stacker), S2 (baselines), S3 (variance). These decide the paper's claim.
2. Angle decision with the user (PAPER_PLAN S5), then W2 and W4 text.
3. W1 figures/tables (S4), then W3 thesis rewrite.
4. W7 + W8 + W9 in parallel with writing: make the repo reproducible and freeze an archive before the final numbers go in the paper.
5. W5 + W6 (deck, notebook) last, from frozen numbers only, so they cannot drift.
6. W10 submission checklist with supervisor feedback loops.

## Risks to track
- The agentic stack's gain over the TF-IDF evidence model is small and partly confounded (see PAPER_PLAN). The claim may have to be reframed.
- Single seed for the Phase 1 adapter; no variance estimate for LoRA training yet.
- Phase 1 arm choice (R vs P) is within dev noise.
- Not a git repo: no history, easy to overwrite results. Consider `git init` and committing after each logged experiment (needs user approval).
- Machine is M1 16GB with no CUDA; anything heavy runs via MLX or goes to Colab, which has zero credits as of 2026-10-09.

## Open questions for the user
- Target venue(s) and every deadline (paper, thesis, viva, demo).
- Which of W3/W4/W2 is actually required, and whether co-authors or the supervisor impose a structure.
- OK to `git init` this folder?
- Is the Visual Verifier / multimodal part still in scope for the thesis, given it is excluded in the results?

## DEADLINE: 3rd weekend of November 2026 = Sat Nov 21 / Sun Nov 22 (user said "November 3rd weekend" on 2026-10-17; Claude's reading, confirm). Work to finish by Fri Nov 20. User also needs to SHOW STEADY PROGRESS until then.

Schedule (5 weeks). Every week ends with a visible, dated checkpoint: a `LOG.md` entry plus a one-page `progress/week_NN.md` (what ran, numbers, what changed, next week) that can be shown to the supervisor. Final days are buffer only.

| Week | Dates | Work | Visible checkpoint |
|---|---|---|---|
| 0 | Oct 17-18 | Hallucination study scored and logged; plan, roadmap, results draft written | DONE (`progress/week_00.md`) |
| 1 | Oct 19-25 | S1 natural-prior dev set + stacker refit, natural_test re-read; S2 start: base-LLaMA no-LoRA scoring (resumable, runs in background); `make_figures.py` from existing traces | `progress/week_01.md`: confound resolved or quantified; first figures |
| 2 | Oct 26-Nov 1 | S2 encoder baseline (small encoder sized for M1 16GB); S3 variance of stack-minus-evidence; **angle decision with user (S5) by Nov 1; numbers FREEZE Nov 1**; git init if approved | `progress/week_02.md`: "why LLaMA?" table + variance; claim fixed |
| 3 | Nov 2-8 | User writes paper prose from frozen facts/outline (W2); Claude supplies tables, figures, sourced outline, reference checks; repo reproducibility pass (W7); supervisor draft 1 by Nov 8 | Draft 1 + similarity check #1 |
| 4 | Nov 9-15 | Supervisor feedback fixes; thesis/report sections (W3) as required; docs cleanup (W9); archive bundle with checksums (W8); deck (W5) from frozen numbers | Draft 2 + archive + deck, similarity check #2 |
| 5 | Nov 16-20 | BUFFER: proofreading, final similarity/AI-disclosure check, final PDFs, submission logistics (W10), viva rehearsal | Submitted by Nov 20; Nov 21-22 spare |

Rules for staying on schedule:
- Tighter than the first plan: drop W4 (conference paper) and W6 (demo notebook) unless the user says they are required. Never cut S1-S3 or the Nov 1 freeze.
- After the freeze any new result must be justified and logged; deck/paper must be regenerated from the freeze, not edited by hand.
- If the encoder baseline is too slow on M1, use the smaller encoder and say so in the paper.

## Writing-integrity policy (user requirement 2026-10-17: plagiarism and AI-generated content each under 5%)
- Claude cannot guarantee a detector score. AI-text detectors are unreliable and similarity tools (Turnitin/iThenticate, the institution's checker) are the real gate. Plan: run the institution's similarity checker on every full draft (Week 4 draft 1, Week 5 thesis, Week 6 final) and log the percentages in `progress/`.
- Division of labour: Claude produces facts, tables, numbers, figures, and an outline with a source for every claim. The user writes the paper's prose in their own words from that material. `paper/results_paper_e23.md` is a notes/scaffold file, NOT submission text; do not paste it into the template.
- Do not reuse prose from Wang et al., the old `main.tex`, the sample EAI papers, or any other source. Quote sparingly with quotation marks and a citation; paraphrase only after understanding, then cite.
- Check the target venue's and institution's policy on AI-assisted writing and disclose tool use if required (also for code and figures).
- Every number and reference must be verifiable: numbers trace to `LOG.md`/`.out`; references are checked against the actual papers (no unverified citations).

## FOCUS (user, 2026-10-17): Phase 2 is the paper's contribution. Phase 1 is the Analyst component and its audit.
Consequence: the paper must show what the Phase 2 agent layer adds, and the current Phase 2 is thin (a logistic-regression stacker over two probability vectors). Evidence so far: stack vs evidence-alone +1.5 (ns) on balanced eval, +4.0 (sig, confounded) on natural_test. Earlier LLM pair adjudicator FAILED Gate 0 on Kaggle dev (picked the right star of the top-2 pair 52-57% vs Phase 1's 56-67%; LOG 2026-10-11). So Phase 2 work in Weeks 1-2 is about building and testing a Phase 2 that can honestly justify the word "agentic", with kill criteria.

### Phase 2 candidates (each tested on dev only; kill if it does not beat the current stack in dev CV)
- **P2-1 Disagreement-triggered reflection.** Analyst and Evidence disagree on 44% of rows (analyst right 25-32%, evidence 51-61%, stack 55-67% there). Route only those rows to a reflection step; agree-rows are passed through (stack already 71-78% right there). Report cost saved (LLM calls only on 44% of rows) and the accuracy on routed rows.
- **P2-2 Evidence-grounded adjudicator (Gate 0 on Electronics).** Rerun Gate 0 on e23 dev, but give the LLM what the old adjudicator lacked: the two candidate stars plus the Evidence agent's top contributing n-grams (from the TF-IDF-LR coefficients) as cited evidence. Pass: pairwise accuracy clearly above the stack's on the same disagreement rows (dev CV). Fail: log it and keep P2-1 as a router-only result.
- **P2-3 Better Reflector features.** Add margin, entropy and disagreement indicator, ordinal/prior-adjusted output, and an abstain/escalate option; compare to the current LR stacker. Cheap, likely small.
- **P2-4 Honest ablation ladder for the paper:** Analyst | Evidence | + stacker | + routing | + adjudicator (only if P2-2 passes), each with paired CIs, all on eval and natural_test read once.
- Natural-prior stacker fit (S1) stays: it is needed so the Phase 2 gain on natural_test is not a class-prior artifact.
- Visual Verifier / images stay excluded (constant output across 3 vision models); say so as a negative result.

### Revised Week 1-2 (replaces the Phase-1-centric S2 priority)
- Week 1 (Oct 19-25): S1 natural-prior dev + stacker refit; P2-3 features; P2-1 routing analysis from existing traces (no new scoring); `progress/week_01.md`.
- Week 2 (Oct 26-Nov 1): P2-2 Gate 0 on e23 dev (resumable local LLM runs); only if it passes, build the adjudicator and read out once; baselines needed to defend the claim (base LLaMA no-LoRA, small encoder) in the background; **freeze Nov 1**.
- Every Phase 2 candidate gets a LOG.md entry with config, what was touched, and a pass/fail against its stated criterion before anything is written for the paper.

## Compute policy (user, 2026-10-17): run everything locally; Claude must flag when Colab credits are needed
Local = Apple M1 16GB, MLX (QLoRA training + scoring), llama.cpp, sklearn. Current plan needs NO Colab: natural-dev scoring, stacker variants, base-LLaMA-no-LoRA scoring, evidence-grounded adjudicator Gate 0 (local Q4 LLM) all run on MLX/CPU. Possible Colab triggers (will ask first, with GPU-hour estimate): a transformer encoder baseline on the full 100K rows (use a small encoder on 2K rows locally instead), multi-seed LoRA retrains if each local run exceeds a few hours, any 4-bit bitsandbytes training.
