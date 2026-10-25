"""Labelled results store: copies every saved result (outputs, summaries, figures; never review text or data) into results/e23/<category>/
with a descriptive file name, and writes results/e23/INDEX.md. Source of truth stays in v2_reflective/e23/ (scripts write there);
rerun after every logged run / freeze:  python v2_reflective/e23/collect_results.py"""
import os, shutil, glob, datetime
E, V, F, OUT = "v2_reflective/e23/", "v2_reflective/", "paper/figures/e23/", "results/e23/"
# (category, label, source, what it is, verdict / status)
R = [
 ("01_main_results", "frozen_results_tables.md", E + "results_frozen.md", "All frozen result tables (exact, CI, macro-F1, within-1, MAE) for every model", "frozen 2026-10-24"),
 ("01_main_results", "main_stack_readout_paired_tests.out", E + "readout_H.out", "Main stack (LLaMA probe + Evidence model) single readout on eval + natural_test, paired tests", "MAIN RESULT: 67.5 / 75.7"),
 ("01_main_results", "freeze_tables_run.out", E + "freeze_tables.out", "Console output of the freeze script", "—"),
 ("01_main_results", "first_stack_readout_oct13.out", E + "phase2_e23.out", "First LoRA + Evidence stack readout (Oct 13)", "superseded: 63.9 / 73.4"),
 ("02_gates", "gate_llm_judge_electronics.out", E + "gate0_e23.out", "LLM judge between agents' stars (Gate 0, Electronics)", "FAIL: 54.2% vs 73.3%"),
 ("02_gates", "gate_extra_text_views_minilm_char.out", E + "stack_gate_B.out", "MiniLM embedding and char n-gram agents added to the stack", "FAIL"),
 ("02_gates", "gate_llama_probe_10k.out", E + "probe_gate_F.out", "LLaMA hidden-state probe, 10K fit", "FAIL by .001"),
 ("02_gates", "gate_llama_probe_multilayer.out", E + "probe_gate_F2.out", "LLaMA probe, several layers", "FAIL"),
 ("02_gates", "gate_pooled_dev_reflector.out", E + "phase_D_gate.out", "Reflector fit on pooled dev + natural_dev, ordinal decoding", "passed, not adopted (class-mix gain)"),
 ("02_gates", "readout_pooled_dev_reflector.out", E + "phase_D_readout.out", "Readout of the pooled-dev Reflector", "not adopted"),
 ("02_gates", "gate_evidence_500k_in_old_stack.out", E + "stack_gate_G.out", "Evidence model on 500K reviews inside the LoRA stack", "FAIL in stack"),
 ("02_gates", "gate_main_stack_phase_H.out", E + "gate_H.out", "Bigger dev2 + LLaMA probe on 40K (main stack selection)", "PASS: dev2 +3.9 [+2.4, +5.8]"),
 ("03_baselines", "base_llama_vs_lora.out", E + "base_vs_lora.out", "Base LLaMA (no LoRA) vs LoRA, alone and stacked", "LoRA adds little in a stack"),
 ("03_baselines", "base_llama_vs_lora.json", E + "base_vs_lora_results.json", "Numbers for the above", "—"),
 ("03_baselines", "distilbert_2k_baseline.out", E + "encoder_2k.out", "DistilBERT on the same 2K rows", "52.8 vs LoRA 54.0 (ns)"),
 ("03_baselines", "prior_corrected_baseline.out", E + "prior_baseline.out", "Natural-prior corrected stackers and Evidence model", "class-mix effect, not headline"),
 ("03_baselines", "stack_week1_natural_prior.out", E + "stack_week1.out", "Week-1 stack variants (natural prior, routing)", "not adopted"),
 ("03_baselines", "stack_week1_natural_prior.json", E + "stack_week1_results.json", "Numbers for the above", "—"),
 ("04_robustness", "variance_first_stack_5seeds.out", E + "variance.out", "First stack, 5 seeds", "gain over Evidence 5/5 positive"),
 ("04_robustness", "variance_main_stack_5seeds.out", E + "variance_H.out", "Main stack, 5 seeds (Evidence model fixed)", "65.8 ± 0.6 / 75.4 ± 0.8"),
 ("04_robustness", "variance_main_stack_e6b_retrained_PARTIAL.out", E + "variance_H2.out", "Main stack, Evidence model retrained per seed (C1)", "stopped 2026-10-24 (laptop strain); rerun pending"),
 ("04_robustness", "lora_seed2_report.out", E + "seed2_report.out", "Second LoRA seed", "eval stable; natural single-model unstable"),
 ("04_robustness", "polarity_paired_ci.out", E + "polarity_ci.out", "Polarity (neg/neu/pos) paired CIs", "stack beats each agent on both sets"),
 ("04_robustness", "polarity_binary_ordinal_metrics.out", E + "metrics_extra.out", "Polarity, binary, within-1, MAE for all models", "reporting"),
 ("04_robustness", "holm_errors_calibration_cost.out", E + "analysis_C.out", "Holm correction, error analysis, calibration (ECE), cost per review", "5 of 8 comparisons significant after Holm"),
 ("04_robustness", "deck_figures_numbers.out", E + "figures_deck.out", "Numbers behind the deck figures (identical to analysis_C)", "—"),
 ("05_ablations_audits", "preprocessing_textblob_vgst_ablation.log", E + "ablate_preproc.log", "TextBlob filtering and VGST ablation (Evidence model)", "TextBlob −2.5 to −3.5; VGST +0.2"),
 ("05_ablations_audits", "hallucination_study_lora_150.out", E + "halluc_p1.out", "What the Phase 1 adapter generates (150 reviews)", "template; wrong star 53%"),
 ("05_ablations_audits", "evidence_learning_curve.out", E + "evid_curve.out", "Evidence model accuracy vs training size", "still rising at 90K"),
 ("05_ablations_audits", "evidence_500k_training.out", E + "evidence_big.out", "Evidence model trained on 500K", "better alone (+1.9 dev)"),
 ("05_ablations_audits", "evidence_main_stack_e6b_training.out", E + "evid_H.out", "Evidence model (~497K, dev2 excluded) used in the main stack", "—"),
 ("05_ablations_audits", "evidence_natural_dev.out", E + "evid_natural_dev.out", "Evidence probabilities on natural_dev", "—"),
 ("05_ablations_audits", "char_ngram_agent.out", E + "char_agent.out", "Character n-gram agent", "weaker than TF-IDF"),
 ("05_ablations_audits", "minilm_embedding_agent.out", E + "embed_agent.out", "MiniLM embedding agent", "weaker than TF-IDF"),
 ("06_phase1_kaggle_closeout", "phase1_answer_only_retrain.json", V + "p1_answeronly_eval.json", "Phase 1 answer-only retrain (Kaggle)", "worse: 55.3 vs 59.7"),
 ("06_phase1_kaggle_closeout", "phase2_reflective_v2_analysis.out", V + "phase2_analysis.out", "Reflective orchestration v2 analysis (leaky probs)", "retracted"),
 ("06_phase1_kaggle_closeout", "phase2_reflective_v2_analysis_zero_shot.out", V + "phase2_analysis_zs.out", "Same, leak-free zero-shot probs", "recalibration gain +0.1 (ns)"),
 ("06_phase1_kaggle_closeout", "gate0_llm_judge_kaggle.log", V + "gate0.log", "LLM pair adjudicator, Gate 0 (Kaggle)", "FAIL"),
 ("07_phase2_v3", "persisted_agents_verification.out", E + "persist_agents.out", "Saved agents reproduce frozen probabilities and 0.675", "verified: diff 0.0"),
 ("07_phase2_v3", "router_sc1_attempt1_gate.out", E + "gate_B1.out", "Router + SC1, attempt 1 (tau 0.60)", "FAIL: natural_dev -0.3 pt; no test read"),
 ("07_phase2_v3", "router_sc1_attempt1_gate.json", E + "gate_B1.json", "Numbers for the above", "—"),
 ("07_phase2_v3", "router_sc1_attempt2_gate_and_readout.out", E + "gate_B1b.out", "Router + SC1, attempt 2 (tau 0.70), gate + single readout", "PASS: skip 39-48%, accuracy unchanged; SC1 fixed 92/50 eval, 152/78 natural"),
 ("07_phase2_v3", "router_sc1_attempt2_gate_and_readout.json", E + "gate_B1b.json", "Numbers for the above", "—"),
 ("07_phase2_v3", "llm_rerating_sc2_gate.out", E + "gate_SC2.out", "LLM re-rating of uncertain reviews (SC2)", "pending"),
 ("07_phase2_v3", "explanations_report.out", E + "b2_report.out", "Explanation faithfulness, Critic loop (SC3), deletion test", "pending"),
 ("07_phase2_v3", "probe_120k_gate_A1.out", E + "gate_A1.out", "LLaMA probe on 120K reviews (A1)", "pending (extraction 45K / 80K)"),
 ("08_extraction_logs", "probe_extraction_10k.out", E + "probe_agent.out", "Hidden-state extraction log (10K)", "log"),
 ("08_extraction_logs", "probe_extraction_multilayer.out", E + "probe_agent2.out", "Hidden-state extraction log (multi-layer)", "log"),
 ("08_extraction_logs", "probe_extraction_40k.out", E + "probe_agent3.out", "Hidden-state extraction log (Phase H, 40K)", "log"),
 ("08_extraction_logs", "probe_extraction_120k_A1.out", E + "probe_agent4.out", "Hidden-state extraction log (A1, paused)", "log"),
]
figs = [("09_figures", os.path.basename(p), p, d, "figure") for p, d in [
 (F + "main_bars.png", "All models, exact accuracy with 95% CI (main stack in red)"), (F + "cm_main_eval.png", "Main-stack confusion matrix, balanced eval"),
 (F + "cm_main_natural_test.png", "Main-stack confusion matrix, natural test"), (F + "recall_phase1_vs_main.png", "Per-star recall, Phase 1 LoRA vs main stack"),
 (F + "agents_complement.png", "Accuracy by review length; agents' disagreement"), (F + "reliability_SH_eval.png", "Reliability diagram (calibration), balanced eval"),
 (F + "reliability_SH_natural_test.png", "Reliability diagram (calibration), natural test"), (F + "cm_stack_eval.png", "First stack confusion matrix, eval (superseded)"),
 (F + "cm_stack_natural_test.png", "First stack confusion matrix, natural test (superseded)"), (F + "recall_eval.png", "First stack per-star recall, eval (superseded)"),
 (F + "recall_natural_test.png", "First stack per-star recall, natural test (superseded)")]]
SUMMARY_ONLY = {E + "halluc_p1.out"}
rows, missing = [], []
for cat, label, src, what, verdict in R + figs:
    if not os.path.exists(src): missing.append((cat, label, src, what, verdict)); continue
    os.makedirs(OUT + cat, exist_ok=True)
    if src in SUMMARY_ONLY:   # contains review quotes: store the summary lines only (no review text in git)
        open(OUT + cat + "/" + label, "w").write("".join(l for l in open(src) if not l.startswith((" ", "\t")) and "| '" not in l and "-> " not in l) + "\n[review excerpts omitted; full file stays local in " + src + "]\n")
    else:
        shutil.copyfile(src, OUT + cat + "/" + label)
    rows.append((cat, label, src, what, verdict))
with open(OUT + "INDEX.md", "w") as f:
    f.write(f"# Results index: Amazon Reviews 2023 Electronics (Phase 2)\n\nGenerated by `v2_reflective/e23/collect_results.py` on {datetime.date.today()}. Files here are labelled copies; "
            "the scripts write to `v2_reflective/e23/` (source of truth). Every run and its pass/fail rule is in `v2_reflective/LOG.md`. No review text or data is stored here.\n\n"
            "| Category | File | What it is | Result / status | Source |\n|---|---|---|---|---|\n")
    for cat, label, src, what, verdict in rows: f.write(f"| {cat} | `{cat}/{label}` | {what} | {verdict} | `{src}` |\n")
    if missing:
        f.write("\n## Pending (no output yet)\n\n| Category | File | What it is | Status |\n|---|---|---|---|\n")
        for cat, label, src, what, verdict in missing: f.write(f"| {cat} | `{label}` | {what} | {verdict} |\n")
print(f"copied {len(rows)}; pending {len(missing)}")
