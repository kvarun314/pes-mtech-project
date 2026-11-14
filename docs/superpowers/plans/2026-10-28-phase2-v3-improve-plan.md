# Phase 2 v3 Improvement Round (Q1–Q5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four measured pieces to Phase 2 v3 before the Nov 1 freeze:
- Q1: star ranges with a coverage guarantee for the Hand-off agent.
- Q5: a breakdown by year and product popularity.
- Q4: robustness to small text changes.
- Q3: a second LLM judge for the Critic.

**Architecture:** Two small pure modules in `src/agentic_sentiment/v3/` (`conformal.py`, `perturb.py`) with unit tests, plus one experiment script per item in `v2_reflective/e23/`. Every script reads the stored probabilities or frozen agents. Nothing is refit except the declared out-of-fold Reflector refit on dev2, which is the same as I2. Each item is declared in `LOG.md` before it runs. Results go to `.out`/`.json`, then `collect_results.py`, the paper FACT lines, the deck bands and the presenter notes.

**Tech Stack:** Python 3.13 `.venv`, numpy, scikit-learn, scipy, mlx-lm (LLaMA-3-8B Q4, Llama-3-8B-Instruct Q4, Qwen2.5-7B-Instruct Q4 for Q3 only), pytest.

## Global Constraints

- Single dataset: Amazon Reviews 2023 Electronics. Run locally.
- Gates are declared in `v2_reflective/LOG.md` before each run and checked on dev2 / natural_dev only. A failure never touches eval or natural_test. Q4 and Q5 are reported analyses, not selection steps. They are still declared first and read once.
- One heavy (GPU) job at a time, on the charger only. Check `pgrep -fl python` and swap first.
- Never commit review text, data `.jsonl`, `.pptx`, `.tar.gz`, or files over 100 MB.
- Never write paper prose. Add tables, figures and `% FACT` lines only.
- Deck stays at 25 slides. No new slides: update existing bands and notes. Render changed slides with `soffice` and check them.
- After each logged step: `LOG.md` → `.venv/bin/python v2_reflective/e23/collect_results.py` → `sh v2_reflective/backup_artifacts.sh` → `verify_backup.sh <abs path>` → commit and push to `e23-leakfree-oct2026` with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Q3 needs a ~4.3 GB download. Ask the user and get a clear yes in chat before downloading.
- Wait loops must not `pgrep -f` a string that appears in their own command line (I3 lesson): match `python.*<script>` and run the loop from a script file.

## Dropped from the candidate list

- **Later-time holdout (Q2):** dropped. Every agent was fit on reviews from 1999–2023 (`pool.jsonl` years), so no clean "future" set exists without refitting everything. Q5 reports accuracy by year instead.

## Shared facts (used by several tasks)

- Main stack probabilities = `joblib.load(PT.REFLECTOR_JOBLIB)["full"].predict_proba(np.c_[logp(probe), logp(evidence)])`. Probe probabilities are `v2_reflective/e23/evid7_<split>_probs.jsonl` and Evidence probabilities are `evid6b_<split>_probs.jsonl`. Rows look like `{"review_id", "gt_rating", "probs"}`.
- The Reflector was fit on dev2, so dev2 main-stack probabilities must be out-of-fold: 5-fold `clone(full).fit`, seed 0, exactly as in `i2_handoff_learned.py`. natural_dev was never used for fitting.
- Phase 1 probabilities: `v2_reflective/e23/p1/probs_R_<split>.jsonl` (same row format).
- Split texts: `v2_reflective/e23/<split>.jsonl` rows look like `{"review_id", "gt_rating", "review_text", "title"}`. Timestamps and product ids live only in `v2_reflective/e23/pool.jsonl` rows: `{"id", "rating", "title", "text", "asin", "ts" (ms), "pool"}`. eval and natural_test ids are all in the pool; dev2 ids are not.
- Hand-off threshold: `[x for x in json.load(open(PT.D + "s1_selective.json"))["SH"]["eval"] if x["target"] == 0.4][0]["threshold"]`.

---

### Task 1: Q1 — Star ranges with a coverage guarantee (conformal prediction)

**Files:**
- Create: `src/agentic_sentiment/v3/conformal.py`
- Create: `tests/v3/test_conformal.py`
- Create: `v2_reflective/e23/q1_conformal.py` (outputs `q1_conformal.out`, `q1_conformal.json`)
- Modify: `v2_reflective/LOG.md` (declaration, then result)

**Interfaces:**
- Produces: `qhat(P_cal: np.ndarray, y_cal: array[int 1-5], alpha: float = 0.1) -> float`; `star_range(p: array[5], q: float) -> tuple[int, int]` (inclusive stars, always contiguous); `evaluate(P, y, q) -> dict(coverage, mean_size, single_star)`.

- [ ] **Step 1: Declare in LOG.md**

Append:

```markdown
### 2026-10-28 — Q1 DECLARED: star ranges with a 90% coverage guarantee (split conformal; dev2 gate)
- Score = 1 − probability of the true star. q = the ⌈(n+1)·0.9⌉/n quantile on a calibration set. Range = the smallest contiguous span of stars containing every star with probability ≥ 1 − q (contiguous hull, so coverage can only rise).
- Gate (dev2, out-of-fold main-stack probabilities, 5-fold: calibrate on 4 folds, score on 1): main stack coverage within [0.88, 0.92], and mean range size at least 0.2 stars smaller than Phase 1's under the same procedure.
- If PASS: calibrate once on all of dev2 → read eval; calibrate once on natural_dev (natural class mix) → read natural_test. Report coverage, mean size and single-star share for the main stack and Phase 1; a three-way Hand-off: answer (confidence ≥ the S1 threshold), answer with a range (range ≤ 2 stars), otherwise a person. Share and coverage of each path.
- Script `e23/q1_conformal.py`; CPU only, minutes.
```

- [ ] **Step 2: Write the failing tests**

`tests/v3/test_conformal.py`:

```python
import numpy as np
from agentic_sentiment.v3.conformal import qhat, star_range, evaluate


def test_star_range_is_contiguous_hull():
    assert star_range([.5, .0, .45, .03, .02], q=0.6) == (1, 3)


def test_star_range_falls_back_to_argmax():
    assert star_range([.1, .2, .4, .2, .1], q=-1.0) == (3, 3)


def test_coverage_close_to_target_on_fresh_data():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5) * 0.5, size=6000); y = np.array([rng.choice(5, p=p) + 1 for p in P])
    q = qhat(P[:3000], y[:3000], alpha=0.1); r = evaluate(P[3000:], y[3000:], q)
    assert 0.88 <= r["coverage"] <= 0.93 and 1 <= r["mean_size"] <= 5
```

- [ ] **Step 3: Run the tests and check they fail**

Run: `.venv/bin/python -m pytest -q tests/v3/test_conformal.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_sentiment.v3.conformal'`

- [ ] **Step 4: Implement**

`src/agentic_sentiment/v3/conformal.py`:

```python
"""Split conformal star ranges (score = 1 - p(true star)) for the Hand-off agent: a contiguous span of stars that holds the
true star with probability >= 1 - alpha on exchangeable data (Q1, LOG.md 2026-10-28)."""
import numpy as np


def qhat(P_cal, y_cal, alpha=0.1):
    s = 1 - np.asarray(P_cal)[np.arange(len(y_cal)), np.asarray(y_cal) - 1]; n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def star_range(p, q):
    keep = np.flatnonzero(1 - np.asarray(p) <= q)
    if keep.size == 0:
        keep = np.array([int(np.argmax(p))])
    return int(keep.min()) + 1, int(keep.max()) + 1


def evaluate(P, y, q):
    R = [star_range(p, q) for p in P]
    cover = np.array([a <= t <= b for (a, b), t in zip(R, y)]); size = np.array([b - a + 1 for a, b in R])
    return {"coverage": float(cover.mean()), "mean_size": float(size.mean()), "single_star": float((size == 1).mean())}
```

- [ ] **Step 5: Run the tests and check they pass**

Run: `.venv/bin/python -m pytest -q tests/v3`
Expected: all pass (31 old + 3 new).

- [ ] **Step 6: Write the experiment script**

`v2_reflective/e23/q1_conformal.py`:

```python
"""Q1 (declared LOG.md 2026-10-28): conformal star ranges. dev2 gate on out-of-fold main-stack probabilities; test readout only on PASS.
usage: q1_conformal.py   (tee to q1_conformal.out; writes q1_conformal.json)"""
import json, sys, numpy as np, joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
from agentic_sentiment.v3.conformal import qhat, star_range, evaluate
full = joblib.load(PT.REFLECTOR_JOBLIB)["full"]
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
htau = [x for x in json.load(open(PT.D + "s1_selective.json"))["SH"]["eval"] if x["target"] == 0.4][0]["threshold"]


def load(s):
    P, E, R = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"p1/probs_R_{s}.jsonl"); ids = list(P)
    X = np.c_[logp([P[i]["probs"] for i in ids]), logp([E[i]["probs"] for i in ids])]
    return X, np.array([R[i]["probs"] for i in ids]), np.array([P[i]["gt_rating"] for i in ids])


def cv(S, y):
    out = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2).split(S, y):
        out.append(evaluate(S[te], y[te], qhat(S[tr], y[tr])))
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


res = {}
X, P1, y = load("dev2"); S = np.zeros((len(y), 5))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    S[te] = clone(full).fit(X[tr], y[tr] - 1).predict_proba(X[te])
g = {"SH": cv(S, y), "P1": cv(P1, y)}
g["pass"] = bool(0.88 <= g["SH"]["coverage"] <= 0.92 and g["P1"]["mean_size"] - g["SH"]["mean_size"] >= 0.2)
res["gate_dev2"] = g; print("GATE dev2", json.dumps(g, indent=1))
if g["pass"]:
    res["readout"] = {}
    for cal, test in (("dev2", "eval"), ("natural_dev", "natural_test")):
        Xc, P1c, yc = load(cal); Sc = S if cal == "dev2" else full.predict_proba(Xc)
        Xt, P1t, yt = load(test); St = full.predict_proba(Xt)
        qs, qp = qhat(Sc, yc), qhat(P1c, yc)
        r = {"q_SH": qs, "q_P1": qp, "SH": evaluate(St, yt, qs), "P1": evaluate(P1t, yt, qp)}
        R = [star_range(p, qs) for p in St]; size = np.array([b - a + 1 for a, b in R]); cover = np.array([a <= t <= b for (a, b), t in zip(R, yt)])
        auto = St.max(1) >= htau; rng_ = ~auto & (size <= 2); human = ~auto & ~rng_
        r["three_way"] = {"auto_share": float(auto.mean()), "auto_exact": float((St.argmax(1) + 1 == yt)[auto].mean()),
                          "range_share": float(rng_.mean()), "range_coverage": float(cover[rng_].mean()) if rng_.any() else None,
                          "human_share": float(human.mean())}
        res["readout"][test] = r; print(test, json.dumps(r, indent=1))
else:
    print("FAIL: no test read")
json.dump(res, open(PT.D + "q1_conformal.json", "w"), indent=1)
```

- [ ] **Step 7: Run it**

Run: `.venv/bin/python v2_reflective/e23/q1_conformal.py 2>&1 | grep -v -i warn | tee v2_reflective/e23/q1_conformal.out`
Expected: a `GATE dev2` block with `"pass": true/false`. If true, there are eval and natural_test blocks. Check that coverage on natural_test is near 0.90. A large miss means natural_dev (500 reviews) is too small; report it as it is.

- [ ] **Step 8: Log the result, collect, back up, commit**

- Append a `Q1 RESULT` entry to `LOG.md` with the gate numbers and the readout.
- Add two rows to `collect_results.py`: `("07_phase2_v3", "conformal_ranges_Q1.out", E + "q1_conformal.out", "Q1: star ranges with 90% coverage guarantee", "<headline>")` and the matching `.json` row.
- Then run:

```bash
.venv/bin/python v2_reflective/e23/collect_results.py && sh v2_reflective/backup_artifacts.sh
sh v2_reflective/verify_backup.sh /Users/varunk/Documents/PES_Project_Phase1/results_archive/backups/<newest>.tar.gz
git add src/agentic_sentiment/v3/conformal.py tests/v3/test_conformal.py v2_reflective/e23/q1_conformal.* v2_reflective/LOG.md v2_reflective/e23/collect_results.py results/e23
git commit -m "Q1 conformal star ranges: <PASS/FAIL headline>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin e23-leakfree-oct2026
```

---

### Task 2: Q5 — Breakdown by year and product popularity

**Files:**
- Create: `v2_reflective/e23/q5_breakdown.py` (outputs `q5_breakdown.out`, `q5_breakdown.json`)
- Modify: `v2_reflective/LOG.md`

**Interfaces:**
- Consumes: stored probabilities (see Shared facts) and `pool.jsonl` `ts` / `asin`.
- Produces: `q5_breakdown.json` = `{split: {dimension: [{band, n, SH, P1, diff, ci_lo, ci_hi}]}}`.

- [ ] **Step 1: Declare in LOG.md**

```markdown
### 2026-10-28 — Q5 DECLARED: accuracy by review year and product popularity (reported analysis, no selection)
- Bands fixed now: year ≤2014 / 2015–2017 / 2018–2020 / 2021–2023 (review timestamp from pool.jsonl); product popularity = reviews of the same parent_asin in pool.jsonl: 1 / 2–5 / 6+.
- Per band, eval and natural_test: n, main-stack exact, Phase 1 exact, difference with a paired bootstrap 95% CI (1,000 resamples, seed 0). Length bands already exist (analysis_C) and are not repeated.
- A later-time holdout is not possible without refitting every agent (training covers 1999–2023), so year bands are the time check.
```

- [ ] **Step 2: Write the script**

`v2_reflective/e23/q5_breakdown.py`:

```python
"""Q5 (declared LOG.md 2026-10-28): main stack vs Phase 1 exact accuracy by review year and product popularity, eval and natural_test.
usage: q5_breakdown.py   (tee to q5_breakdown.out; writes q5_breakdown.json)"""
import collections, datetime as dt, json, sys, numpy as np, joblib
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
full = joblib.load(PT.REFLECTOR_JOBLIB)["full"]
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
pool = {r["id"]: r for r in map(json.loads, open(PT.D + "pool.jsonl"))}
per_asin = collections.Counter(r["asin"] for r in pool.values())
year = lambda i: dt.datetime.fromtimestamp(pool[i]["ts"] / 1000, dt.UTC).year
BANDS = {"year": (lambda i: "<=2014" if year(i) <= 2014 else "2015-2017" if year(i) <= 2017 else "2018-2020" if year(i) <= 2020 else "2021-2023",
                  ["<=2014", "2015-2017", "2018-2020", "2021-2023"]),
         "product_reviews": (lambda i: "1" if per_asin[pool[i]["asin"]] == 1 else "2-5" if per_asin[pool[i]["asin"]] <= 5 else "6+", ["1", "2-5", "6+"])}
rng = np.random.default_rng(0); res = {}
for s in ("eval", "natural_test"):
    P, E, R = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"p1/probs_R_{s}.jsonl"); ids = list(P)
    y = np.array([P[i]["gt_rating"] for i in ids])
    sh = full.predict(np.c_[logp([P[i]["probs"] for i in ids]), logp([E[i]["probs"] for i in ids])]) + 1 == y
    p1 = np.array([R[i]["probs"] for i in ids]).argmax(1) + 1 == y
    res[s] = {}; print(f"== {s} (n {len(ids)}; SH {sh.mean():.3f}, P1 {p1.mean():.3f})")
    for dim, (f, order) in BANDS.items():
        lab = np.array([f(i) for i in ids]); rows = []
        for b in order:
            m = np.flatnonzero(lab == b)
            if not len(m): continue
            d = sh[m].astype(float) - p1[m]; boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            rows.append({"band": b, "n": int(len(m)), "SH": float(sh[m].mean()), "P1": float(p1[m].mean()), "diff": float(d.mean()),
                         "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
            r = rows[-1]; print(f"{dim:16s} {b:10s} n {r['n']:5d}  SH {r['SH']:.3f}  P1 {r['P1']:.3f}  diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        res[s][dim] = rows
json.dump(res, open(PT.D + "q5_breakdown.json", "w"), indent=1)
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python v2_reflective/e23/q5_breakdown.py 2>&1 | grep -v -i warn | tee v2_reflective/e23/q5_breakdown.out`
Expected: per-split headers whose SH / P1 totals match the frozen numbers (eval .675 / .540, natural_test .757 / .534), then 7 band rows per split. If a total does not match, stop: the probability files or ids are wrong.

- [ ] **Step 4: Log the result, collect, back up, commit**

Append a `Q5 RESULT` entry to `LOG.md`: the bands where the gain is smallest and largest, and whether any CI crosses 0. Add `q5_breakdown.out` / `.json` to `collect_results.py` in group `07_phase2_v3`. Then run the same collect / backup / verify / commit / push commands as in Task 1, Step 8, with message `Q5 breakdown by year and product popularity`.

---

### Task 3: Q4 — Robustness to small text changes (GPU, ~30 min)

**Files:**
- Create: `src/agentic_sentiment/v3/perturb.py`
- Create: `tests/v3/test_perturb.py`
- Create: `v2_reflective/e23/q4_robustness.py` (outputs `q4_robustness.out`, `q4_robustness.json`; rows in `q4/robustness.jsonl`, which is gitignored as `.jsonl`)
- Modify: `v2_reflective/LOG.md`

**Interfaces:**
- Produces: `typos(text: str, rate: float = 0.1, seed: int = 0) -> str`; `flat(text: str) -> str`.
- Consumes: `SharedLlama`, `load_probe`, `phase1_probs` from `agentic_sentiment.v3.llama`; `EvidenceAgent` from `agentic_sentiment.v3.evidence`; `Reflector` from `agentic_sentiment.v3.reflector`.

- [ ] **Step 1: Declare in LOG.md**

```markdown
### 2026-10-28 — Q4 DECLARED: robustness to label-preserving text changes (reported analysis, no selection)
- Sample fixed now: the first 60 eval reviews of each star in file order (300).
- Variants: original (recomputed in the same run), typos (adjacent letters swapped in 10% of words longer than 3 letters, seed 0), flat (lowercase, punctuation removed), no title (Evidence agent only reads the title; probe and Phase 1 never do).
- Per variant: exact accuracy and flip rate (star differs from the same system's original star) for the main stack (probe + Evidence + Reflector) and Phase 1 (LLaMA + LoRA digit probabilities). Paired comparison of flips: exact McNemar.
- Script `e23/q4_robustness.py`; one LLaMA in memory (SharedLlama), ~1.4 s per review-variant, ~30 min.
```

- [ ] **Step 2: Write the failing tests**

`tests/v3/test_perturb.py`:

```python
from agentic_sentiment.v3.perturb import typos, flat


def test_typos_deterministic_and_keeps_words():
    t = "This charger stopped working after three weeks of normal use"
    a, b = typos(t, rate=0.5, seed=1), typos(t, rate=0.5, seed=1)
    assert a == b and a != t and len(a.split()) == len(t.split()) and sorted(a) == sorted(t)


def test_typos_rate_zero_is_identity():
    assert typos("Great sound, works well.", rate=0.0) == "Great sound, works well."


def test_flat_lowercases_and_strips_punctuation():
    assert flat("Great!!! Doesn't work, though.") == "great doesn't work though"
```

- [ ] **Step 3: Run the tests and check they fail**

Run: `.venv/bin/python -m pytest -q tests/v3/test_perturb.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_sentiment.v3.perturb'`

- [ ] **Step 4: Implement**

`src/agentic_sentiment/v3/perturb.py`:

```python
"""Label-preserving text changes for the robustness check (Q4, LOG.md 2026-10-28). Deterministic for a given seed."""
import random
import re


def typos(text: str, rate: float = 0.1, seed: int = 0) -> str:
    rng, out = random.Random(seed), []
    for w in text.split(" "):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            i = rng.randrange(len(w) - 1); w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def flat(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
```

- [ ] **Step 5: Run the tests and check they pass**

Run: `.venv/bin/python -m pytest -q tests/v3`
Expected: all pass.

- [ ] **Step 6: Write the experiment script**

`v2_reflective/e23/q4_robustness.py`:

```python
"""Q4 (declared LOG.md 2026-10-28): flip rate and accuracy under typos / flat text / no title, main stack vs Phase 1, 300 eval reviews.
usage: q4_robustness.py   resumable (q4/robustness.jsonl); tee to q4_robustness.out; writes q4_robustness.json"""
import json, os, sys, time, numpy as np, joblib
from scipy.stats import binomtest
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.evidence import EvidenceAgent
from agentic_sentiment.v3.reflector import Reflector
from agentic_sentiment.v3.perturb import typos, flat
V = {"original": lambda t, x: (t, x), "typos": lambda t, x: (t, typos(x)), "flat": lambda t, x: (flat(t), flat(x)), "no_title": lambda t, x: ("", x)}
rows = [json.loads(l) for l in open(PT.D + "eval.jsonl")]
sample = [r for k in range(1, 6) for r in [r for r in rows if r["gt_rating"] == k][:60]]
os.makedirs(PT.D + "q4", exist_ok=True); out = PT.D + "q4/robustness.jsonl"
done = {(d["review_id"], d["variant"]) for d in map(json.loads, open(out))} if os.path.exists(out) else set()
if len(done) < len(sample) * len(V):
    from agentic_sentiment.v3.llama import SharedLlama, load_probe, phase1_probs
    e, r = joblib.load(PT.EVIDENCE_JOBLIB), joblib.load(PT.REFLECTOR_JOBLIB)
    ev, rf = EvidenceAgent(e["vectorizer"], e["model"]), Reflector(r["full"], r["fallback"])
    llama = SharedLlama(); probe = load_probe(llama); t0 = time.time()
    with open(out, "a") as f:
        for k, row in enumerate(sample):
            for v, fn in V.items():
                if (row["review_id"], v) in done: continue
                title, text = fn(row["title"], row["review_text"])
                sh, _, _ = rf.combine(probe.probs(text), ev.probs(title, text))
                p1 = int(np.argmax(phase1_probs(llama, text))) + 1
                f.write(json.dumps({"review_id": row["review_id"], "variant": v, "gt": row["gt_rating"], "SH": int(sh), "P1": p1}) + "\n"); f.flush()
            if (k + 1) % 25 == 0: print(f"{k + 1}/{len(sample)} {time.time() - t0:.0f}s", flush=True)
D = {(d["review_id"], d["variant"]): d for d in map(json.loads, open(out))}; ids = [r["review_id"] for r in sample]; res = {}
for v in V:
    a = {m: np.array([D[i, v][m] == D[i, v]["gt"] for i in ids]) for m in ("SH", "P1")}
    fl = {m: np.array([D[i, v][m] != D[i, "original"][m] for i in ids]) for m in ("SH", "P1")}
    b, c = int((fl["SH"] & ~fl["P1"]).sum()), int((fl["P1"] & ~fl["SH"]).sum())
    res[v] = {"SH_exact": float(a["SH"].mean()), "P1_exact": float(a["P1"].mean()), "SH_flip": float(fl["SH"].mean()), "P1_flip": float(fl["P1"].mean()),
              "only_SH_flips": b, "only_P1_flips": c, "p": binomtest(b, b + c).pvalue if b + c else 1.0}
    print(f"{v:9s} exact SH {res[v]['SH_exact']:.3f} P1 {res[v]['P1_exact']:.3f} | flips SH {res[v]['SH_flip']:.3f} P1 {res[v]['P1_flip']:.3f} | only-SH {b} only-P1 {c} p {res[v]['p']:.2g}")
json.dump(res, open(PT.D + "q4_robustness.json", "w"), indent=1)
```

- [ ] **Step 7: Run it (GPU; charger; nothing else heavy running)**

```bash
pgrep -fl "python.*(i1_|i3_|q3_|b2_)" ; sysctl vm.swapusage ; pmset -g batt | head -1
caffeinate -is .venv/bin/python v2_reflective/e23/q4_robustness.py 2>&1 | grep -v -i warn | tee v2_reflective/e23/q4_robustness.out
```

Expected: progress lines, then 4 variant lines. The `original` line must show flips 0.000 for both systems, and SH exact near the eval figure on this subset. If original flips are non-zero, stop: there is a nondeterminism bug.

- [ ] **Step 8: Log the result, collect, back up, commit**

Append a `Q4 RESULT` entry to `LOG.md`. Add `q4_robustness.out` / `.json` to `collect_results.py`. Run the collect / backup / verify / commit / push commands from Task 1, Step 8, adding `src/agentic_sentiment/v3/perturb.py tests/v3/test_perturb.py v2_reflective/e23/q4_robustness.*` and using message `Q4 robustness to text changes`.

---

### Task 4: Q3 — Second LLM judge for the Critic (download needs user approval; GPU ~30 min)

**Files:**
- Create: `v2_reflective/e23/q3_second_judge.py` (outputs `q3_second_judge_<part>.out`, `q3_second_judge.json`; rows in `b2/q3_<split>.jsonl`, gitignored)
- Modify: `src/agentic_sentiment/v3/paths.py` (add `JUDGE2`)
- Modify: `v2_reflective/LOG.md`

**Interfaces:**
- Consumes: `JUDGE_SYSTEM`, `judge_user`, `parse_judge` from `agentic_sentiment.v3.critic`; `make_chat_fn` from `agentic_sentiment.v3.explainer`; `b2/<split>.jsonl`, `b2/i1_<split>.jsonl`, `b2/i3_dev2.jsonl`.
- Produces: `PT.JUDGE2 = ".scratch/mlx-qwen2.5-7b-instruct-q4"`.

- [ ] **Step 1: Ask the user for the download (stop until a clear yes)**

Ask in chat: "Q3 needs `mlx-community/Qwen2.5-7B-Instruct-4bit` from huggingface.co, about 4.3 GB, saved to `.scratch/mlx-qwen2.5-7b-instruct-q4` (gitignored). OK to download?" If the answer is no, skip Task 4 and log it as not run.

- [ ] **Step 2: Declare in LOG.md**

```markdown
### 2026-10-28 — Q3 DECLARED: second LLM judge (Qwen2.5-7B-Instruct 4-bit) for Critic check 5 (dev2 gate)
- Same prompt, parser and greedy decoding as check 5; only the model differs.
- dev2 sets: A = the 16 I3 fallbacks the Llama judge rejected; B = the 22 dev2 explanations still failing after 4 rewrites only on check 5; C = 60 dev2 final explanations the Llama judge accepted (random, seed 0).
- Gate: Qwen accepts at least 50% of A∪B (38) and rejects at most 10% of C.
- If PASS: test readout on the eval and natural_test explanations whose last failure after 4 rewrites is check 5 only (b2/i1_<split>.jsonl), plus 60 accepted ones per split as a control. Report faithful under "reject only if both judges reject" next to the current single-judge figure, and the control rejection rate. This loosens check 5, so it is reported as a variant; the human check (pending, user) decides which judge to trust.
```

- [ ] **Step 3: Download (only after the yes)**

```bash
.venv/bin/huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit --local-dir .scratch/mlx-qwen2.5-7b-instruct-q4
du -sh .scratch/mlx-qwen2.5-7b-instruct-q4
```

Expected: about 4.3G. Then append `JUDGE2 = ".scratch/mlx-qwen2.5-7b-instruct-q4"   # Q3 second judge (LOG.md 2026-10-28)` to `src/agentic_sentiment/v3/paths.py`.

- [ ] **Step 4: Write the script**

`v2_reflective/e23/q3_second_judge.py`:

```python
"""Q3 (declared LOG.md 2026-10-28): second judge (Qwen2.5-7B-Instruct 4-bit) on explanations the Llama judge rejected / accepted.
usage: q3_second_judge.py gate | q3_second_judge.py test   (tee to q3_second_judge_<part>.out; updates q3_second_judge.json)"""
import json, os, random, sys
sys.path.insert(0, "src")
from mlx_lm import load
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.critic import JUDGE_SYSTEM, judge_user, parse_judge
from agentic_sentiment.v3.explainer import make_chat_fn
part = sys.argv[1]; J = PT.D + "q3_second_judge.json"; res = json.load(open(J)) if os.path.exists(J) else {}
L = lambda f: [json.loads(l) for l in open(PT.B2 + f)]
T = lambda s: {x["review_id"]: x for x in map(json.loads, open(f"{PT.D}{s}.jsonl"))}


def sets(split):
    B, I = L(f"{split}.jsonl"), {x["review_id"]: x for x in L(f"i1_{split}.jsonl")}
    rej = [(b["review_id"], I[b["review_id"]]["drafts"][-1]) for b in B if not b["faithful"] and not I[b["review_id"]]["faithful"] and I[b["review_id"]]["failed_checks"][-1] == [5]]
    acc = random.Random(0).sample([(b["review_id"], b["drafts"][-1]) for b in B if b["faithful"]], 60)
    return rej, acc


model, tok = load(PT.JUDGE2); chat = make_chat_fn(model, tok)
def judge(split, items):
    t = T(split); return [parse_judge(chat(JUDGE_SYSTEM, judge_user(x, t[i]["title"], t[i]["review_text"])))[0] for i, x in items]


if part == "gate":
    rej, acc = sets("dev2")
    A = [(x["review_id"], x["fallback"]) for x in L("i3_dev2.jsonl") if not x["faithful"]]
    ra, rc = judge("dev2", A + rej), judge("dev2", acc)
    g = {"n_rejected": len(ra), "qwen_accepts_rejected": sum(v is True for v in ra) / len(ra), "n_accepted": len(rc),
         "qwen_rejects_accepted": sum(v is False for v in rc) / len(rc), "unparsed": sum(v is None for v in ra + rc)}
    g["pass"] = bool(g["qwen_accepts_rejected"] >= 0.5 and g["qwen_rejects_accepted"] <= 0.10)
    res["gate_dev2"] = g; print("GATE dev2", json.dumps(g, indent=1))
elif part == "test":
    assert res.get("gate_dev2", {}).get("pass"), "gate not passed: no test read"
    for s in ("eval", "natural_test"):
        rej, acc = sets(s); n = len(L(f"{s}.jsonl")); ra, rc = judge(s, rej), judge(s, acc)
        f4 = sum(x["faithful"] for x in L(f"{s}.jsonl")) + sum(x["faithful"] for x in L(f"i1_{s}.jsonl"))
        r = {"judge_only_failures": len(rej), "qwen_accepts": sum(v is True for v in ra), "control_rejected": sum(v is False for v in rc) / len(rc),
             "faithful_single_judge": f4 / n, "faithful_two_judge": (f4 + sum(v is True for v in ra)) / n}
        res[s] = r; print(s, json.dumps(r, indent=1))
json.dump(res, open(J, "w"), indent=1)
```

- [ ] **Step 5: Run the gate**

Run: `caffeinate -is .venv/bin/python v2_reflective/e23/q3_second_judge.py gate 2>&1 | grep -v -i warn | tee v2_reflective/e23/q3_second_judge_gate.out`
Expected: `n_rejected` 38 (16 + 22) and `n_accepted` 60. If the counts differ, stop and check the set definitions against the I1 / I3 outputs before reading anything else.

- [ ] **Step 6: On PASS only, run the test readout**

Run: `caffeinate -is .venv/bin/python v2_reflective/e23/q3_second_judge.py test 2>&1 | grep -v -i warn | tee v2_reflective/e23/q3_second_judge_test.out`
Expected: `judge_only_failures` ≈ 51 (eval) and ≈ 110 (natural_test), matching the I1 "still failing by check" counts for check 5 alone (lower if some of those also failed another check).

- [ ] **Step 7: Log the result, collect, back up, commit**

Append a `Q3 RESULT` entry to `LOG.md`. Add the `.out` files and `q3_second_judge.json` to `collect_results.py`. Run the collect / backup / verify / commit / push commands from Task 1, Step 8, adding `src/agentic_sentiment/v3/paths.py v2_reflective/e23/q3_second_judge*`.

---

### Task 5: Fold results into tables, paper, deck and notes

**Files:**
- Modify: `v2_reflective/EXPERIMENTS_SUMMARY.md` (section 4 robustness rows: Q1, Q4, Q5; W/F row for Q3 depending on outcome)
- Modify: `paper/e23_results.tex` (`% FACT` lines for Q1–Q5; one new table `tab:e23conformal` if Q1 passes, rows = split × system, columns = coverage, mean size, single-star share)
- Modify: `v2_reflective/build_deck_e23.py`. No new slides. Hand-off slide band (`nho`): add the range path if Q1 passes. Robustness slide band (`rob`-equivalent, slide 21): one line each for Q4 and Q5. Conclusions "Limits" card: replace "One dataset" with "One dataset (stable across years)" only if Q5 shows no band with a CI below 0. Speaker notes for the changed slides.
- Modify: `presentation/Presenter_Notes.md` Part 0 (same slides) and Q&A (judge reliability if Q3 ran)
- Modify: `v2_reflective/e23/agent_audit.md` (Hand-off row if Q1 passes; Critic row if Q3 passes)

- [ ] **Step 1: Update `EXPERIMENTS_SUMMARY.md`, `agent_audit.md` and the paper FACT lines / table from the `.out` files.** Copy numbers only from `.out` / `.json`.
- [ ] **Step 2: Build the paper**

Run: `cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex | grep -c Overfull; grep -c Error main.log`
Expected: overfull boxes not above 10 (current count); 0 errors. Then run `.venv/bin/python paper/integrity/audit.py`.

- [ ] **Step 3: Update the deck builder and notes, rebuild, render**

```bash
/opt/homebrew/bin/python3 v2_reflective/build_deck_e23.py | tail -2
cd /private/tmp/claude-502/-Users-varunk-Documents-PES-Project-Phase1/90ee5fd8-46e6-494d-a545-8dfafb05808e/scratchpad/r2 && soffice --headless --convert-to pdf /Users/varunk/Documents/PES_Project_Phase1/Agentic_LLaMA3_Presentation.pptx >/dev/null 2>&1 && for p in 16 21 25; do pdftoppm -r 80 -png -f $p -l $p Agentic_LLaMA3_Presentation.pdf q; done
```

Expected: `saved ... 25`, Drive sync line. Open each PNG and check for text overflow or wrapped headers; fix and rebuild if needed.

- [ ] **Step 4: Log, collect, back up, commit**

Append a `LOG.md` entry listing every file changed. Then run the collect / backup / verify / commit / push commands from Task 1, Step 8, with message `Q1-Q5 folded into tables, paper, deck and notes`.

---

## Order and time

| Task | Needs | Time |
|---|---|---|
| 1 Q1 star ranges | CPU | ~20 min |
| 2 Q5 breakdown | CPU | ~15 min |
| 3 Q4 robustness | GPU, charger | ~30 min run + 15 min |
| 4 Q3 second judge | user OK for 4.3 GB; GPU | ~10 min download + ~30 min |
| 5 Fold in | — | ~45 min |

Tasks 1 and 2 can run while nothing else is on the GPU. Tasks 3 and 4 run one after the other, never together.
