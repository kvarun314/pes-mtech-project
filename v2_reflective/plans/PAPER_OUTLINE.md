# Paper outline + facts index (2026-10-21). OUTLINE ONLY: you write the prose (plagiarism/AI <5% rule). Every fact points to its source.

Working angle (decide Nov 1): B = leak audit + preprocessing ablation + modest, robust stacking gain + negative result for LLM adjudication. Angle A ("agentic verification helps") is NOT supported (Gate 0 FAIL).

| Section | Points to make (facts) | Source |
|---|---|---|
| 1 Intro | Phase 1 LLaMA-3-8B LoRA sentiment analyst; Phase 2 adds Evidence agent + Reflector; question: what does the second layer add, measured leak-free | PAPER_PLAN.md |
| 2 Related work | (you collect; Claude can check reference details on request) | - |
| 3 Method | Electronics 2023 only; splits evid_train 100K / dev 500 / natural_dev 500 / eval 1,000 balanced / natural_test 2,000 natural, seed 2023, disjoint by id+text; Analyst = LoRA arm R (Table 1 hyperparameters, 2,000 raw balanced rows); Evidence = TF-IDF 1-2gram 400K + LR C=3 on 100K; Reflector = LR C=1 on log-probs, fit on balanced dev | LOG 2026-10-11, 10-13, 10-17 |
| 4 Protocol | zero-shot prompt; dev only for choices; test read once; all configs reported; paired bootstrap 3,000 + exact McNemar; macro-F1 and per-class recall because natural_test is 67% 5* | LOG 10-17, 10-18 |
| 5 Main results | Table: `e23/results_frozen.md` (Analyst 54.0, Evidence 62.4, stack 63.9 on eval; 53.4/69.3/73.4 on natural_test) | results_frozen.md, phase2_e23.out |
| 6 Is the stack gain real? | +1.5 ns (frozen), 5 evidence/stacker resamples: +2.7±1.3 eval, +4.3±1.1 natural_test, 5/5 positive; adapter seed 2: +2.2 ns eval, +3.4 sig natural_test | variance.out, seed2_report.out |
| 7 Prior shift | natural_test exact dominated by 5* prior; prior-adjusted Evidence 76.3, best natural-prior stacker 79.0 (+2.7 [+1.4,+4.0]); minority recall collapses (2* ~.13); single analyst natural_test exact varies 53.4 vs 74.8 across seeds | stack_week1.out, prior_baseline.out, base_vs_lora.out, seed2_report.out |
| 8 Why LLaMA? | base no-LoRA 34.3 eval; LoRA +19.7; DistilBERT 2K level with LoRA (52.8 vs 54.0 ns); [base+E] ~ [R+E] (63.7 vs 63.9): value is LLM pretraining as a complementary signal, not the fine-tune; [enc2k+E] adds nothing over Evidence | base_vs_lora.out, encoder_2k.out |
| 9 Negative results | LLM adjudicator Gate 0 FAIL (54.2 vs stack 73.3, -19.1 [-25.3,-12.4]), also on Kaggle; routing ties C0; extra Reflector features add nothing; Visual Verifier excluded (constant output) | gate0_e23.out, stack_week1.out |
| 10 Leak audit + ablation | one-shot label leak ~8 pts; retracted calibrator +5.0 -> +0.1; TextBlob DQC -2.5 to -3.5, VGST +0.2 ns; arm P 52.2 vs R 55.2 dev | LOG 10-11, ablate_preproc.log |
| 11 Hallucination | 150 CoT generations (Kaggle adapter): 41% wrong, 16 severe, canned rationale; Electronics repeat is in halluc_p1.out | halluc_p1.out |
| 12 Limitations | single dataset; 2 adapter seeds; single encoder seed; bootstrap Evidence weaker than frozen fit; uncorrected p-values (5 stacker configs); Gate 0 one prompt only | LOG |
| Figures | `paper/figures/e23/`: main_bars, cm_stack_*, recall_* | - |

Do-not-claim list: agentic LLM verification gain; 70.7% Kaggle headline; +5.0 calibrator; natural_test exact of a single analyst as model quality.
