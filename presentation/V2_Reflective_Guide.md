# Presenter guide: the `v2_reflective/` folder

Use this when someone asks "where is the code?", "how did you run the experiments?" or "how do we know the numbers are real?".
- Deck slides 27–28 show this material. Slides 29–37 are historical backups.
- Companion files (same folder, `presentation/`): `Presenter_Notes.md` (slide-by-slide script) and `Project_Journey.md` (full history).

---

## 1. One-sentence answer

> "`v2_reflective/` is the experiment workspace for Phase 2. Every Phase 2 number in the deck and the paper comes from a script in this folder. Each script writes a saved output file, and each run has a dated entry in `LOG.md` that states the pass/fail rule before the run."

## 2. Why it is called "v2_reflective"

- **Oct 10:** the folder was created for **"Reflective Orchestration v2"** (historical slide 37). That design had an uncertainty gate, a Reflector and a pair adjudicator on top of the Phase 1 model.
- **Oct 11–12:** the first experiments here found:
  - the one-shot label leak,
  - that the recalibration gain was not real,
  - that the LLM adjudicator fails (Gate 0).
- **From Oct 13:** the project moved to Amazon Reviews 2023 Electronics. All new work went into the sub-folder `v2_reflective/e23/` ("e23" = Electronics 2023).
- The folder name stayed, so it now holds all Phase 2 experiments.
- The older `src/agentic_sentiment/` package (LangGraph agents, 129 tests) is the Phase 1 / Run 1 pipeline. Phase 2 v3 will plug the new agents back into its LangGraph graph.

**Say:** "The name is historical. The reflective idea was tested here first, failed its gate honestly, and the folder became the home of every later Phase 2 experiment."

## 3. Folder map

```
v2_reflective/
├── LOG.md                    lab notebook: every rule declared before a run, every result after (75 KB)
├── EXPERIMENTS_SUMMARY.md    one-page table: what worked (W1–W10), what failed (F1–F16), next steps
├── plans/                    PAPER_PLAN, PAPER_OUTLINE, PROJECT_ROADMAP, PHASE2_PLAN, CONTINGENCY, DAILY_PLAN, ...
├── progress/                 weekly notes, DAILY_STATUS.md, status updates for the supervisor
├── backup_artifacts.sh, verify_backup.sh   backups (checksummed; copied to Google Drive)
├── build_deck_e23.py         builds the one deck PES_Project_Phase1/Agentic_LLaMA3_Presentation.pptx
├── build_demo_notebook.py    builds presentation/Presentation_Demo.ipynb (to be rebuilt for Phase 2 v3)
│
├── (Kaggle close-out, Oct 9–12; historical but kept)
│   ├── phase1_final.py, phase1_final_score.py   leak audit: prompts A/B/C/D on 300 rows
│   ├── probs_zeroshot.py, zeroshot_all.sh       leak-free zero-shot probabilities
│   ├── reflective.py, feature_cv.py, recalibrate_eval.py   v2 Reflector / calibrator (retracted, F3)
│   ├── build_pair_data.py, pair_adapter/, gate0.py         LLM pair adjudicator, Gate 0 (failed, F1)
│   ├── build_p1_answeronly_data.py, eval_p1_answeronly.py, p1_answeronly_adapter/   answer-only retrain (F4)
│   ├── stack_experiment.py                      first proof that a second signal (TF-IDF) helps
│   └── score.py, classical_paper_protocol.py, build_paper_protocol_test.py   paper-protocol re-score (69.1%)
│
└── e23/   Amazon Reviews 2023 Electronics: ALL current results (177 files)
```

## 4. `e23/` in six groups (the pipeline order)

| Step | What | Main files | Output |
|---|---|---|---|
| 1. Data | Stream Electronics reviews; build frozen disjoint splits (seed 2023) | `fetch.py`, `fetch_big.py`, `build_splits.py`, `build_natural_dev.py`, `build_H.py`, `build_A1.py` | `evid_train`, `dev`, `dev2`, `eval`, `natural_dev`, `natural_test`, `fit2`, `fit3` (`.jsonl`) |
| 2. Agents | Each agent writes its 5 star probabilities for every split | `evidence.py` (E, 100K), `evidence_big.py` / `evid_H.py` (E6b, ~497K), `probe_agent*.py` (LLaMA hidden states, `hid*/`), `eval_p1.py` (LoRA Analyst, `p1/`), `encoder_2k.py` (DistilBERT), `embed_agent.py`, `char_agent.py` | `<agent>_<split>_probs.jsonl` |
| 3. Gates | Pass/fail rule on dev data only, written in `LOG.md` first | `gate0_e23.py`, `stack_gate_B.py`, `probe_gate_F.py`, `probe_gate_F2.py`, `phase_D_gate.py`, `stack_gate_G.py`, `gate_H.py`, `gate_A1.py` | `*.out` with PASS / FAIL |
| 4. Readout | Only a passing config touches eval and natural_test, once | `phase2_e23.py` (first stack), `readout_H.py` (SH), `phase_D_readout.py` | `phase2_e23.out`, `readout_H.out` |
| 5. Robustness | Variance, baselines, ablations, error analysis | `variance*.py`, `base_vs_lora.py`, `seed2_report.py`, `prior_baseline.py`, `polarity_ci.py`, `metrics_extra.py`, `halluc_p1.py`, `ablate_preproc.py`, `evid_curve.py`, `analysis_C.py` | `*.out`, `variance_*_seed*.json` |
| 6. Freeze | Recompute every table and figure from the stored probabilities | `freeze_tables.py` | `results_frozen.md`, `paper/figures/e23/*.png`, then the deck and `paper/e23_results.tex` |

**Probability file names (the agents):**

| Prefix | Agent | Status |
|---|---|---|
| `p1/probs_R_*` | LoRA Analyst (Phase 1 arm R, 2K rows) | ablation row |
| `evid` | Evidence E: TF-IDF + LR, 100K rows | baseline |
| `evid2` / `evid3` | MiniLM embedding agent / character n-gram agent | failed (F8) |
| `evid4` / `evid5` | Hidden-state probe, 10K fit / multi-layer | failed (F10, F11) |
| `evid6` / `evid6b` | Evidence on 500K rows / same with dev2 removed (~497K) | **E6b is in SH** |
| `evid7` | P3: probe on frozen LLaMA layer 20, 40K fit | **in SH** |
| `evid8` | P3b: probe on 120K fit (A1) | pending A1 gate |

**Run control:** `queue_*.sh` starts a long job. A `stage_*.txt` marker records how far it got. Hidden-state extraction saves in chunks, so a stopped run (sleep, battery, planned pause) resumes where it left off.

## 5. How one experiment flows (slide 28)

1. **Declare.** Write the candidate, the data and the exact pass rule in `LOG.md`. Example: "dev2 +1.0 point, paired CI > 0, macro-F1 not lower, natural_dev not lower".
2. **Run.** The script is resumable, logs to a `.log` file, and writes its result to a `.out` file.
3. **Gate.** The gate is checked on dev / dev2 / natural_dev only. A failure is logged and stops there; it never touches eval or natural_test.
4. **Readout.** A pass is read once on eval (1,000 balanced) and natural_test (2,000 natural), with paired bootstrap and McNemar tests.
5. **Freeze.** `freeze_tables.py` recomputes every number from the stored probabilities. The deck and paper copy from there.
6. **Record.** Add the `LOG.md` result, back up (checksum, then a Google Drive copy) and push to GitHub.

**Say:** "The rule is written before the run, so I cannot move the goalposts after seeing the result. Failed ideas never see the test sets. That is why I can report 16 failures and still trust the one pass."

## 6. Talking script (2 minutes, slides 27–28)

- **Slide 27:** "This is where all the Phase 2 work lives. The top level holds the October Kaggle close-out. That's where the label leak was found and the first LLM adjudicator failed. Everything current is in `e23`, Electronics 2023. The pipeline runs left to right: build frozen splits, let each agent write its probabilities, gate on dev, read the test sets once, freeze the tables. Every agent leaves a probability file per split, so any table can be recomputed in seconds without rerunning LLaMA."
- **Slide 28:** "This is the discipline behind every number. The rule goes in the log first, the run is resumable, the gate uses dev data only, a pass is read once, and the freeze script regenerates the tables. There are 177 files in `e23`, and every result in the deck traces back to one of them."

## 7. Questions to expect

- **"Can I rerun your main result?"** Yes. `python v2_reflective/e23/freeze_tables.py` recomputes every table and the bar chart from the stored probabilities in about 3 seconds. Rerunning an agent from raw text takes longer: about 0.67 s per review for the LLaMA probe, and minutes for TF-IDF.
- **"Where is the data?"** In `e23/*.jsonl`: Amazon Reviews 2023 Electronics, pulled with HuggingFace `datasets` (`fetch.py`). Review text is never committed to GitHub; only code, logs and outputs are.
- **"Why so many probe scripts?"** Each is one declared phase. `probe_agent.py` was Phase F (10K), `probe_agent2.py` F2 (multi-layer), `probe_agent3.py` Phase H (40K, which passed), and `probe_agent4.py` is A1 (120K, running).
- **"How do I know you didn't tune on the test set?"** Every gate in `LOG.md` predates its readout and names its splits. The one slip, Evidence-alone accuracy printed once before the Phase H gate, is disclosed and was not used for any choice.
- **"What is `LOG.md`?"** The lab notebook: 75 KB of dated entries from Oct 9 to today. Each one has a declaration, the command, the result, and the decision.
- **"Is this agentic?"** The current system is two agents plus a learned Reflector. Phase 2 v3 adds Router, Explainer and Critic agents into the existing LangGraph graph in `src/agentic_sentiment/agents/graph.py`. The new agents load the saved files from `e23/`.

## 8. Numbers to have ready

- 177 files in `e23/`. About 1.7 GB of saved LLaMA hidden states (`hid*/`).
- 16 failed or not-adopted ideas and 10 adopted ones (`EXPERIMENTS_SUMMARY.md`).
- 8 declared gates on Electronics. Two readouts of the main config family: the first stack (Oct 13) and SH (Oct 24).
- `freeze_tables.py` runs in about 3 seconds.
