# Paper integrity: plagiarism and AI content (each under 5%)

## Current state (2026-10-25)
- `audit.py` reports **100% of the current prose as AI-drafted** (4,729 words across `main.tex`, `e23_results.tex` and `ConferencePaper.tex`).
  - Every commit that created that text carries a `Co-Authored-By: Claude` trailer: `main.tex` on 2026-03-31, `ConferencePaper.tex` on 2026-10-02.
- **Verbatim overlap:** mostly with your own earlier proposal (`docs/reference/Agentic Self-Correction in LLaMA3 – E-Commerce Sentiment Analysis.pdf` and its revised copy). That shows up mainly in the Abstract, the Problem Statement and the agent-specification sections.
  - Similarity checkers flag reuse of your own earlier submitted work too.
- Tables, figures and numbers (2,766 words) are data and are counted separately.

## What makes the target achievable
1. **You write every sentence of prose yourself, from the facts, not from the old text.**
   - Facts to work from: the `% FACT` comments in `paper/e23_results.tex`, `presentation/Project_Journey.md`, `v2_reflective/EXPERIMENTS_SUMMARY.md` and the tables.
   - Write each section fresh. Do not edit or paraphrase the existing AI-drafted paragraphs line by line; paraphrased AI text is still AI-shaped, and similarity tools still match it.
2. **Commit your own prose without the Claude trailer.** `audit.py` then moves those words to the author column. The git history is also your evidence of authorship if a checker ever flags something.
3. **Claude's role stays limited:** tables, figures, numbers and `% FACT` comments only (rule in `CLAUDE.md`). No prose, no rewording of your text, and no "humanizing". Those tools are detectable, and using them to hide AI text is academic misconduct.
4. **Quote, or don't reuse.** Any sentence taken from a source (including your own proposal) is either quoted and cited, or rewritten from scratch.
5. **Disclose AI use honestly**, following PES policy. For example, a statement that AI tools were used for code, experiment scripts, figures and tables, and that the text was written by the author. Ask your supervisor for the required wording.

## Checks before each draft (Nov 8, Nov 15, final)
- [ ] `.venv/bin/python paper/integrity/audit.py`. The AI-drafted share is under 5% and every section shows `ok`.
- [ ] No verbatim 8-word overlaps outside quotations and the bibliography.
- [ ] University similarity check (Turnitin or equivalent) on the PDF. Keep the report in `paper/integrity/` (gitignored if it contains other people's text).
- [ ] AI-use disclosure present.

Note: AI detectors are unreliable in both directions. The only real guarantee is that the prose is yours. This audit tracks that objectively through git; it does not try to guess.

## Order to rewrite (biggest first)
1. Abstract, Introduction / Problem Statement
2. Datasets, Methods (Phase 1 pipeline, Phase 2 system as it is now: SH plus the v3 agents)
3. Results interpretation: the `% AUTHOR` markers in `e23_results.tex`
4. Discussion, Limitations, Conclusion
5. Sections that are now historical (Run 1 analysis, the old agent specifications). Decide with your supervisor whether to cut, shorten or move them to an appendix; cutting is the fastest way to remove AI text.
