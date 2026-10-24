# Phase 2 v3: accuracy, agentic layer, explainability, demo

Date: 2026-10-25. Status: draft for user review.
- Background and evidence: `Project_Journey.md`, `v2_reflective/EXPERIMENTS_SUMMARY.md`, `v2_reflective/LOG.md`.
- Scope is Phase 2 only. Phase 1 stays frozen as the baseline and an ablation row.

## 1. Goals

1. **Accuracy:** improve on SH (67.5 eval / 75.7 natural_test exact) only through pre-declared, gated experiments with a credible path to at least +1.5 points.
2. **Agentic layer:** add LLM agents whose benefit is measured. They route, explain, and self-correct explanations. The specialists keep deciding the star.
3. **Explainability:** every prediction comes with evidence and a rationale, and the rationale's faithfulness is measured.
4. **Paper rigor:** close the open disclosures and add an error analysis, calibration, cost, and an agent-audit table.
5. **Demo:** one local notebook that runs end to end in about 2 minutes.

### Constraints

- Everything must run locally on the M1 (16 GB) for the demo. A Colab-only model can be an ablation row, never the main system.
- Colab credits arrive in November. Colab is used only for work that cannot run locally in reasonable time and that produces value whatever the outcome.
- Every experiment gets a gate written in `LOG.md` before it runs. Failures never touch eval or natural_test.
- One GPU job at a time. A memory-heavy CPU job never runs alongside a GPU job, and nothing runs overnight on battery.
- Every step is logged, backed up (local + Google Drive `PES_backups`, `cp -n` only), and pushed to GitHub.

## 2. Architecture

The original 4-agent design (Analyst, Visual Verifier, RAG Prover, Critic) is kept as the frame. Each agent is re-tasked or dropped according to the evidence.

```
review -> Evidence agent (TF-IDF+LR, ~497K) --------------------+
            | confidence, agreement                             |
            v                                                   |
         Router -- confident --------------------------------+  |
            | unsure                                         |  |
            v                                                v  v
         Analyst (L20 probe on frozen LLaMA-3-8B Q4) ---> Reflector (LR) -> star + probs
         [DeBERTa agent, only if gate A2 passes]             |
                                                             v
                                       evidence pack (phrases, agent shares, agreement, star)
                                                             |
                                                             v
                                       Explainer (Llama-3-8B-Instruct Q4) -> rationale
                                                             |
                                                             v
                                       Critic -- faithful -> output
                                          | unfaithful: specific feedback
                                          +--> Explainer rewrite (max 2 rounds)
```

**Implementation:** rewire `src/agentic_sentiment/agents/graph.py`, which is already a real LangGraph `StateGraph` with tests. The rewrite loop is a conditional edge from Critic back to Explainer. The agents load saved artifacts:
- probe weights (`evid7`, or `evid8` if A1 passes)
- the E6b vectorizer and model, persisted with joblib
- Reflector coefficients
- the MLX models

| Agent | Job | Source |
|---|---|---|
| Evidence | Star probabilities, plus per-phrase contributions (coefficient × tf-idf weight for each star) | E6b, existing config, persisted |
| Router | Skip the Analyst when Evidence is confident (rule below) | threshold chosen on dev2 |
| Analyst | Star probabilities from the layer-20 last-token state | P3 now, P3b if A1 passes |
| DeBERTa (optional) | Third specialist plus token attributions | Colab A2 |
| Reflector | Final star = LR on agents' log-probabilities | dev2 fit, as SH |
| Explainer | 1–2 sentence rationale built only from the evidence pack | Llama-3-8B-Instruct 4-bit MLX (~4.5 GB download, **needs user OK**) |
| Critic | Rule checks + an LLM check; specific feedback; at most 2 rewrites | rules + Instruct model |

**Router rule:** call the Analyst unless Evidence's top probability is at least τ. τ is chosen on dev2 as the smallest threshold whose routed accuracy is within 0.5 points of full SH (5-fold CV). When the Analyst is skipped, the Reflector uses the Evidence-only fallback, an LR on Evidence log-probabilities fit on dev2.

**Evidence pack:**
- top 5 phrases toward the predicted star and top 3 against it, each with its weight
- each agent's share of the decision: the Reflector's coefficient contribution, normalised
- whether the agents agreed
- the final star and its probability

**Critic checks:**
1. The rationale's stated star equals the final star.
2. Every quoted phrase occurs in the review and in the evidence pack.
3. At least 2 of the top 3 phrases are covered.
4. If the agents disagreed, the rationale says so.
5. LLM check: no claim absent from the review (yes/no plus the offending span).

Failing any check produces specific feedback, e.g. "You said 4 stars; final is 3" or "'great battery' does not occur in the review".

## 3. Experiments and gates

| ID | What | Where | Gate (dev2 + natural_dev only) |
|---|---|---|---|
| A1 | Probe fit 40K → 120K (P3b) | local, **running** (declared 2026-10-24) | P3b+E6b vs SH: dev2 +1.0, CI > 0, macro-F1 ≥ on both, natural_dev exact ≥ |
| A2 | DeBERTa-v3-base, ~200K balanced Electronics reviews disjoint from all splits, 2 epochs; probabilities for all splits | Colab, Nov 1 | Best of {SH+D, SH-with-E6b-replaced-by-D} vs SH, same rule; 2 candidates disclosed |
| B1 | Router | local | Routed accuracy within 0.5 of SH on dev2 CV; report the share of Analyst calls saved |
| B2 | Explainer + Critic loop | local | Declared metrics only (below); no accuracy effect, since the star is fixed before explanation |
| C1 | SH variance with E6b retrained per seed | local, when GPU idle and plugged in | robustness only |
| C2–C5 | Holm, error analysis, calibration, cost | done 2026-10-25 | n/a |
| C6 | Agent-audit table (original agent → finding → new role) | writing | n/a |

### Explanation metrics (B2)

Measured on **all 3,000 test reviews** (eval 1,000 + natural_test 2,000), read once:
- **Faithfulness rate:** share of rationales that pass all Critic checks, before vs after the loop. Paired McNemar test.
- **Rewrite statistics:** how often the Critic asks for a rewrite, and the mean number of rounds.
- **Deletion test, comprehensiveness:** the drop in the predicted star's probability when the cited phrases are removed, vs removing the same number of random words (3 random draws). Paired test.
- **Deletion test, sufficiency:** the predicted star's probability when only the cited phrases are kept.
- **LLM calls and latency per review**, with and without the Router.

## 4. Demo notebook

`Phase2_Demo.ipynb` (local Jupyter, ~2 min):
1. Load the saved agents.
2. Run 5 chosen reviews live through the graph. For each, show:
   - highlighted evidence phrases
   - the agent votes and the Router decision
   - the final star
   - the rationale before and after the Critic, with the faithfulness verdict
3. Load the frozen tables and figures: main bars, reliability curves, deletion-test plot, agent-audit table.

The notebook never needs Colab or a network. The model files are already local; the Instruct download happens once, beforehand.

## 5. Schedule

| Date | Work | Compute |
|---|---|---|
| Oct 25 | Presentation. A1 pauses 08:00, resumes after | GPU: A1 |
| Oct 26–27 | B: graph rewire, persist agents, Router, evidence pack, Explainer, Critic; unit tests | GPU: A1 finish, then light dev runs |
| Oct 28 | A1 gate; B1 gate; B2 dev check on dev2 (500); C1 when GPU idle | CPU |
| Oct 29 night | B2 readout part 1: Explainer + Critic on all 3,000 (~8–9 h) | GPU |
| Oct 30 night | B2 readout part 2: deletion test on all 3,000 (~5 h) | GPU |
| Oct 31 | Demo v1, tables, figures, deck, paper section update | — |
| **Nov 1** | **Freeze stage 1.** A2 on Colab | Colab ~2–3 h |
| Nov 2–4 | A2 gate, then readout if it passes; demo final | CPU |
| **Nov 5** | **Freeze stage 2** | — |

Prose (the user writes it) starts Nov 2. Draft 1 goes to the supervisor on Nov 8. Submission by Nov 20.

## 6. Risks and fallbacks

- **A1 or A2 fails its gate:** logged as an ablation row; SH stays.
- **Explainer slow or Instruct download refused:** fall back to the base model with a strict template prompt. Faithfulness is measured either way.
- **An overnight run stops** (sleep, lid, battery): all runs are chunked and resumable, costing at most one catch-up day.
- **Memory:** never more than one heavy job at a time; swap is checked before each launch.
- **Multiplicity:** every candidate is listed in its gate entry; Holm correction is applied to the final family of comparisons.

## 7. Out of scope

- An LLM choosing the star (failed Gate 0 twice).
- Any vision agent.
- More LoRA data.
- Extra classical text views.
- Natural-prior Reflectors.
- Probing a 70B model (it cannot run locally for the demo).
