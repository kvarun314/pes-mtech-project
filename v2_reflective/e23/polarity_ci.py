"""Polarity paired CIs (declared LOG.md 2026-10-22)."""
import json, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
D = "v2_reflective/e23/"; rng = np.random.default_rng(0)
G = lambda p, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}evid_{s}_probs.jsonl" if p == "evid" else f"{D}p1/probs_{p}_{s}.jsonl"))}
S = ("dev", "eval", "natural_test"); ids = {s: list(G("R", s)) for s in S}; arr = lambda p, s: np.clip([G(p, s)[i]["probs"] for i in ids[s]], 1e-6, 1)
y = {s: np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids[s]]) for s in S}; X = lambda p, s: np.c_[np.log(arr(p, s)), np.log(arr("evid", s))]
stk = {p: LogisticRegression(C=1, max_iter=3000).fit(X(p, "dev"), y["dev"]) for p in ("R", "R2")}
pol = lambda v: np.where(v <= 1, 0, np.where(v == 2, 1, 2))
def cmp(c1, c2, name):
    d = c1.astype(float) - c2; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]; f, r = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    print(f"   {name}: {d.mean()*100:+.1f} [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}] fixed {f}/regressed {r} p={binomtest(f,f+r,.5).pvalue if f+r else 1:.2g}")
for s in ("eval", "natural_test"):
    print(f"===== {s} n={len(ids[s])}"); P = {"E": arr("evid", s).argmax(1), "A": arr("R", s).argmax(1), "R+E": stk["R"].predict(X("R", s)), "R2+E": stk["R2"].predict(X("R2", s))}
    for metric in ("polarity3", "binary"):
        print(metric); nb = (y[s] != 2) if metric == "binary" else np.ones(len(y[s]), bool)
        ok = {k: ((pol(v) == pol(y[s])) if metric == "polarity3" else ((v >= 3) == (y[s] >= 3)))[nb] for k, v in P.items()}
        print("   acc: " + ", ".join(f"{k} {v.mean():.3f}" for k, v in ok.items()))
        cmp(ok["R+E"], ok["E"], "[R+E] vs Evidence"); cmp(ok["R2+E"], ok["E"], "[R2+E] vs Evidence"); cmp(ok["R+E"], ok["A"], "[R+E] vs Analyst R")
