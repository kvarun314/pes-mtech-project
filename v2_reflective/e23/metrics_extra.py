"""Phase A (declared in LOG.md 2026-10-21): polarity/binary/ordinal metrics from stored probs. Reporting only."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"
G = lambda p, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}evid_{s}_probs.jsonl" if p == "evid" else f"{D}p1/probs_{p}_{s}.jsonl"))}
S = ("dev", "eval", "natural_test"); ids = {s: list(G("R", s)) for s in S}; arr = lambda p, s: np.clip([G(p, s)[i]["probs"] for i in ids[s]], 1e-6, 1)
y = {s: np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids[s]]) for s in S}; X = lambda p, s: np.c_[np.log(arr(p, s)), np.log(arr("evid", s))]
stk = {p: LogisticRegression(C=1, max_iter=3000).fit(X(p, "dev"), y["dev"]) for p in ("R", "R2", "base", "enc2k")}
pol = lambda v: np.where(v <= 1, 0, np.where(v == 2, 1, 2))
M = {"Base LLaMA": lambda s: arr("base", s), "DistilBERT 2K": lambda s: arr("enc2k", s), "Analyst R": lambda s: arr("R", s), "Analyst R2": lambda s: arr("R2", s), "Evidence": lambda s: arr("evid", s)}
for k, n in (("R", "Stack R+E"), ("R2", "Stack R2+E"), ("base", "Stack base+E"), ("enc2k", "Stack enc2k+E")): M[n] = (lambda k: lambda s: stk[k].predict_proba(X(k, s)))(k)
for s in ("eval", "natural_test"):
    print(f"===== {s} n={len(ids[s])}"); print(f"{'model':16s} exact  pol3acc pol3F1  bin_acc within1 MAE   expMAE"); nb = y[s] != 2
    for n, f in M.items():
        P = f(s); p = P.argmax(1); b = (p >= 3)[nb] == (y[s] >= 3)[nb]; em = abs((P * np.arange(5)).sum(1) - y[s]).mean()
        print(f"{n:16s} {(p==y[s]).mean():.3f}  {(pol(p)==pol(y[s])).mean():.3f}   {f1_score(pol(y[s]),pol(p),average='macro'):.3f}   {b.mean():.3f}   {(abs(p-y[s])<=1).mean():.3f}   {abs(p-y[s]).mean():.3f} {em:.3f}")
