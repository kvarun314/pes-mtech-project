"""Adapter-seed variance (declared in LOG.md 2026-10-21): R2 vs R, stacks [R2+E] vs [R+E], both fit on balanced dev."""
import json, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"; rng = np.random.default_rng(0)
G = lambda pref, s: {x["review_id"]: x for x in map(json.loads, open(f"{D}evid_{s}_probs.jsonl" if pref == "evid" else f"{D}p1/probs_{pref}_{s}.jsonl"))}
ids = {s: list(G("R", s)) for s in ("dev", "eval", "natural_test")}; arr = lambda p, s: np.clip([G(p, s)[i]["probs"] for i in ids[s]], 1e-6, 1)
y = {s: np.array([G("R", s)[i]["gt_rating"] - 1 for i in ids[s]]) for s in ids}; X = lambda p, s: np.c_[np.log(arr(p, s)), np.log(arr("evid", s))]
met = lambda yy, p: f"exact {(p==yy).mean():.1%} macroF1 {f1_score(yy,p,average='macro'):.3f}"
def cmp(yy, a, b, name):
    ca, cb = a == yy, b == yy; d = ca.astype(float) - cb; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]; f, r = int((ca & ~cb).sum()), int((~ca & cb).sum())
    print(f"   {name}: {d.mean()*100:+.1f} [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}] fixed {f}/regressed {r} p={binomtest(f,f+r,.5).pvalue if f+r else 1:.2g}")
st = {p: LogisticRegression(C=1, max_iter=3000).fit(X(p, "dev"), y["dev"]) for p in ("R", "R2")}
for s in ids:
    print(f"===== {s} n={len(ids[s])}"); a, b, e = arr("R", s).argmax(1), arr("R2", s).argmax(1), arr("evid", s).argmax(1)
    print(f"R : {met(y[s],a)}\nR2: {met(y[s],b)}\nagreement R vs R2: {(a==b).mean():.1%}"); sR, sR2 = st["R"].predict(X("R", s)), st["R2"].predict(X("R2", s))
    print(f"[R+E] : {met(y[s],sR)}\n[R2+E]: {met(y[s],sR2)}\nEvidence: {met(y[s],e)}")
    if s != "dev": cmp(y[s], b, a, "R2 vs R"); cmp(y[s], sR, e, "[R+E] vs Evidence"); cmp(y[s], sR2, e, "[R2+E] vs Evidence"); cmp(y[s], sR2, sR, "[R2+E] vs [R+E]")
