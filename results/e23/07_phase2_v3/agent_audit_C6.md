# Agent audit (C6): original Phase 2 design → evidence → role in Phase 2 v3

Every number below traces to the named file in `v2_reflective/e23/` (or to the dated `LOG.md` entry).

| Original agent | What we found | Evidence | Role in Phase 2 v3 |
|---|---|---|---|
| Analyst (LLaMA-3-8B + LoRA, one-shot CoT) | One-shot leak (~8 pts); LoRA learns the label, its rationale is a repeated template; a probe on the frozen model is stronger | Leak audit B vs A −7.7 [−13.3, −2.0]; `halluc_p1.out`; `x1_explain_compare.out`; P3 61.2 vs LoRA 54.0 eval | Replaced by the L20 probe agent (P3); LoRA kept as the Phase 1 baseline and an ablation row |
| Visual Verifier (BLIP / moondream2 / Qwen2-VL) | Constant output (always 3, then always 5) | 96/96 rows rated 3; Qwen2-VL 5/5 rated 5 | Excluded permanently |
| RAG Prover (LanceDB k-NN reviews) | Weak vote (37.5%); bag-of-words over many reviews is the useful second signal | Run 2 audit; TF-IDF 62.5 eval | Replaced by the Evidence agent (E6b), which also supplies the phrases for explanations |
| Critic (dissonance D ≥ 0.4) | Override reached 0.4% of rows; the LLM adjudicator lost Gate 0 twice, and LLM re-rating (SC2) failed a third time | Gate 0: 54.2% vs 73.3%; `gate_SC2.out` (dev2 fixed 76 / broken 89) | Star decision moved to the learned Reflector; Critic re-tasked to check explanations |
| — (new) Reflector | Learned combination of the two agents beats either agent in all 5 seeds | `variance_H2.out` (SH − E6b +4.4 / +6.7; SH − P3 +5.4 / +2.4) | Decides the star: 67.5% eval / 75.7% natural_test |
| — (new) Router (SC1) | Probe costs 0.67 s/review, Evidence < 0.1 ms; escalation fixes about two stars per one broken | `gate_B1b.out`: tau 0.70, probe skipped on 39.3% / 48.3%, accuracy unchanged; SC1 fixed 92 / broken 50 (eval), 152 / 78 (natural_test) | Skips the probe when the Evidence agent is confident; escalates otherwise |
| — (new) Hand-off agent | Confidence ranks errors well; uncertain reviews go to a human | `s1_selective.out`: natural_test 93.1% at 55.7% coverage; eval 85.0% at 42% | Answers automatically above the dev2 threshold, hands off below it |
| — (new) Explainer | First drafts are faithful 60.5% / 63.4% of the time | `b2_report.out` | Writes a 1–2 sentence rationale from the evidence pack only |
| — (new) Critic for explanations (SC3) | Loop raises faithful explanations to 88.9% / 88.8% with max 2 rewrites and 90.6% / 91.1% with max 4 (I1); 856 fixed, 0 broken; cited phrases pass the deletion test | `b2_report.out` (McNemar p 6e-86 / 1e-153; comprehensiveness and sufficiency cited > random, p < 1e-46) | 4 rule checks + LLM judge, max 4 rewrites (`i1_more_rounds_<split>.out`); a fixed-form fallback (I3) failed its dev2 gate |
