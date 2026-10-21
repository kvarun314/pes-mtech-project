"""Read-out for the DistilBERT 2K baseline (declared in LOG.md 2026-10-21): metrics, paired CI vs arm R, stack [enc2k+E] vs [R+E] (both fit on balanced dev)."""
import json, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"; rng = np.random.default_rng(0)
def get(pref, sp):
    f = f"{D}evid_{sp}_probs.jsonl" if pref == "evid" else f"{D}{pref}_{sp}.jsonl"; return {x["review_id"]: x for x in map(json.loads, open(f))}
def arr(pref, sp, ids): g = get(pref, sp); return np.clip([g[i]["probs"] for i in ids], 1e-6, 1)
def met(y, p): return f"exact {(p==y).mean():.1%} macroF1 {f1_score(y,p,average='macro'):.3f} recall {[round(float(((p==c)&(y==c)).sum()/max((y==c).sum(),1)),2) for c in range(5)]}"
def cmp(y, a, b, name):
    ca, cb = a == y, b == y; d = ca.astype(float) - cb; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
    f, r = int((ca & ~cb).sum()), int((~ca & cb).sum()); p = binomtest(f, f + r, .5).pvalue if f + r else 1
    print(f"   {name}: {d.mean()*100:+.1f} [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}] fixed {f}/regressed {r} p={p:.2g}")
ids = {s: list(get("p1/probs_R", s)) for s in ("dev", "eval", "natural_test")}
y = {s: np.array([get("p1/probs_R", s)[i]["gt_rating"] - 1 for i in ids[s]]) for s in ids}
Xd = lambda pref, s, ev: np.c_[np.log(arr(pref, s, ids[s])), np.log(arr("evid", s, ids[s]))]
stk = {k: LogisticRegression(C=1, max_iter=3000).fit(Xd(p, "dev", 0), y["dev"]) for k, p in (("R", "p1/probs_R"), ("enc", "p1/probs_enc2k"))}
for s in ("dev", "eval", "natural_test"):
    print(f"===== {s} n={len(ids[s])}"); P = {k: arr(p, s, ids[s]).argmax(1) for k, p in (("enc2k", "p1/probs_enc2k"), ("R", "p1/probs_R"), ("base", "p1/probs_base"))}
    P["Evidence"] = arr("evid", s, ids[s]).argmax(1)
    for k in ("enc2k", "R", "base", "Evidence"): print(f"{k:9s}: {met(y[s], P[k])}")
    if s != "dev":
        cmp(y[s], P["enc2k"], P["R"], "enc2k vs arm R")
        pe = stk["enc"].predict(Xd("p1/probs_enc2k", s, 0)); pr = stk["R"].predict(Xd("p1/probs_R", s, 0))
        print(f"[enc2k+E]: {met(y[s], pe)}"); print(f"[R+E]    : {met(y[s], pr)}"); cmp(y[s], pe, P["Evidence"], "[enc2k+E] vs Evidence"); cmp(y[s], pe, pr, "[enc2k+E] vs [R+E]")
