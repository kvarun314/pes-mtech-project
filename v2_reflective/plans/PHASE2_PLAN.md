# Phase 2 plan — reflective orchestration that must beat Phase 1 (or we say it didn't)

Baseline to beat: Phase 1 adapter, same rows, same metric. Phase 2 reuses Phase 1's probabilities, so any gain is attributable.

## Evidence from existing probabilities (dev 500 / eval 1000, balanced; computed 2026-10-10)
- Phase 1 top-1 exact: 59.4% / 60.2%. Correct star is in the model's top-2 on 88.4% / 89.0% of rows.
  So ~29 points of headroom exist, almost all between two neighbouring stars.
- Gate by top-2 probability margin < 0.3: covers 68% of rows. Top-1 is right on only 48–51% inside the gate and 79–83% outside it.
  So reflecting only inside the gate leaves the confident 32% of rows untouched.
- Inside the gate the baseline already picks the right one of the two candidates ~60% of the time (51% / 86%).
  **An adjudicator must beat ~60% pairwise to help at all.** Ceiling if it is right 70% / 80% of the time:
  ~66–67% / ~72% exact (gate 0.3). At 60% it adds nothing. This is the real risk, so it is tested first.

## Gate 0 (cheap, decides everything): can the adjudicator beat the Analyst on the pair?
- Rows: dev rows with margin < 0.3 and truth in top-2 (~170). Prompt: star-rubric + the two candidate ratings + (optional) retrieved lessons.
- Measure pairwise accuracy. Pass: >= 68% (so expected exact gain >= ~+5 points). Fail: stop, log it, do not build the rest.
- Variants, cheapest first: (a) rubric-only, (b) rubric + one contrastive example per pair, (c) rubric + Reflector critique,
  (d) combine with Analyst probability (log-odds fusion fit on dev, so it can never be worse than the Analyst alone).

## Build order (only if Gate 0 passes)
1. Uncertainty gate (margin threshold chosen on dev).
2. Reflector -> Adjudicator over the two candidates, <= 2 rounds, stop on agreement.
3. Lesson memory: mistakes on a training-side pool are distilled into short rules, retrieved by similarity.
4. Visual Verifier stays excluded (constant output, 3 vision models tried). RAG kNN vote stays out (fitted weight 0).

## Evaluation protocol (no cherry-picking)
- Fit everything on dev (500 rows). Touch eval (1,000 balanced) and the 4,000-row paper-protocol set once, for the read-out.
- Ablation rows: Phase 1 | + gate only | + reflection | + reflection + memory.
- Metrics each time: exact, within-1, MAE, QWK, polarity 3-class, with trivial baselines beside them.
- Success = exact gain over Phase 1 with a paired bootstrap 95% CI above 0 on the 1,000 eval rows. Otherwise report "no gain".

## Honest expectation
Realistic target is ~65–70% exact on balanced 5-class if the adjudicator reaches 70–80% pairwise; 80% exact is not reachable
this way. Polarity/within-1 numbers will stay high either way. Latency rises only on gated rows (~2/3 of rows).
