"""H1 hard-case analysis (declared LOG.md 2026-10-27): Phase 1 LoRA vs main stack (SH) on subsets defined by TextBlob polarity and cue words.
usage: hard_cases.py   -> prints the report (tee to hard_cases.out), writes hard_cases.json. Numbers only."""
import json, re, sys, joblib, numpy as np
from scipy.stats import binomtest
from textblob import TextBlob
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import Reflector
r = joblib.load(PT.REFLECTOR_JOBLIB); rf = Reflector(r["full"], r["fallback"])
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
CONTRAST = re.compile(r"\b(but|however|although|though|except|yet)\b", re.I)
SARC = re.compile(r"yeah,? right|thanks a lot|just great|so much for|what a joke|/s\b|\bnot!", re.I)
SUBSETS = {"surface_positive_negatives": lambda pol, t, y: pol >= 0.2 and y <= 2, "surface_negative_positives": lambda pol, t, y: pol <= 0 and y >= 4,
           "mixed_with_contrast": lambda pol, t, y: bool(CONTRAST.search(t)) and 2 <= y <= 4, "sarcasm_cues": lambda pol, t, y: bool(SARC.search(t)),
           "all": lambda pol, t, y: True}
res = {}
for s in ("eval", "natural_test"):
    T, P, E, Q = L(f"{s}.jsonl"), L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"p1/probs_R_{s}.jsonl")
    ids = list(T); y = np.array([T[i]["gt_rating"] for i in ids]); pol = [TextBlob(T[i]["review_text"]).sentiment.polarity for i in ids]
    sh = np.array([rf.combine(np.array(P[i]["probs"]), np.array(E[i]["probs"]))[0] for i in ids]); p1 = np.array([int(np.argmax(Q[i]["probs"])) + 1 for i in ids])
    res[s] = {}
    for name, f in SUBSETS.items():
        m = np.array([f(pol[k], T[i]["review_text"], y[k]) for k, i in enumerate(ids)])
        a, b = sh[m] == y[m], p1[m] == y[m]; n = int(m.sum())
        if n == 0: res[s][name] = {"n": 0}; continue
        rng = np.random.default_rng(0); d = a.astype(float) - b
        ci = [float(x) for x in np.percentile([d[rng.integers(0, n, n)].mean() for _ in range(2000)], [2.5, 97.5])]
        u, v = int((a & ~b).sum()), int((b & ~a).sum())
        res[s][name] = {"n": n, "sh_exact": float(a.mean()), "p1_exact": float(b.mean()), "diff_ci": ci, "sh_within1": float((abs(sh[m] - y[m]) <= 1).mean()),
                        "p1_within1": float((abs(p1[m] - y[m]) <= 1).mean()), "rescued": u, "lost": v, "mcnemar_p": float(binomtest(u, u + v).pvalue) if u + v else 1.0}
        x = res[s][name]
        print(f"{s:12s} {name:28s} n={n:4d}  SH {x['sh_exact']:.3f}  P1 {x['p1_exact']:.3f}  diff {x['sh_exact']-x['p1_exact']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
              f"within1 {x['sh_within1']:.3f}/{x['p1_within1']:.3f}  rescued {u} lost {v}  p {x['mcnemar_p']:.2g}")
json.dump(res, open(PT.D + "hard_cases.json", "w"), indent=1)
