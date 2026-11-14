"""Phase 2 on Electronics 2023: Analyst (Phase 1 arm R) + Evidence agent -> Reflector (stacker fit on dev) -> trace.
Read-out on eval (1,000 balanced) and natural_test (2,000 natural). Baselines on the same rows. Run ONCE."""
import json, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"
def L(p): return {r["review_id"]: r for r in map(json.loads, open(p))}
def get(split):
    a, e = L(f"{D}p1/probs_R_{split}.jsonl"), L(f"{D}evid_{split}_probs.jsonl"); ids = [i for i in a if i in e]
    rows = L(f"{D}{split}.jsonl")
    y = np.array([a[i]["gt_rating"] - 1 for i in ids]); A = np.log(np.clip([a[i]["probs"] for i in ids], 1e-6, 1)); E = np.log(np.clip([e[i]["probs"] for i in ids], 1e-6, 1))
    N = np.log1p([len(rows[i]["review_text"]) for i in ids]); return ids, y, A, E, N, rows
_, yd, Ad, Ed, Nd, _ = get("dev")
rng = np.random.default_rng(0)
def metrics(y, p):
    pol = lambda x: np.where(x <= 1, 0, np.where(x == 2, 1, 2)); k = y != 2
    return dict(exact=(p == y).mean(), within1=(abs(p - y) <= 1).mean(), mae=abs(p - y).mean(), polarity=(pol(p) == pol(y)).mean(),
                binary=((p[k] >= 3) == (y[k] >= 3)).mean(), macroF1=f1_score(y, p, average="macro"), recall=[round(float(((p == c) & (y == c)).sum() / max((y == c).sum(), 1)), 2) for c in range(5)])
def fmt(m): return f"exact {m['exact']:.1%} within1 {m['within1']:.1%} MAE {m['mae']:.3f} pol {m['polarity']:.1%} bin {m['binary']:.1%} macroF1 {m['macroF1']:.3f} recall {m['recall']}"
cfgs = {"stack A+E": lambda A, E, N: np.c_[A, E], "stack A+E+len": lambda A, E, N: np.c_[A, E, N]}
for split in ("eval", "natural_test"):
    ids, y, A, E, N, rows = get(split); print(f"\n===== {split} n={len(y)}")
    base = A.argmax(1) == y; ev = E.argmax(1) == y
    print("Analyst (LLaMA arm R) :", fmt(metrics(y, A.argmax(1)))); print("Evidence alone        :", fmt(metrics(y, E.argmax(1))))
    for n, f in cfgs.items():
        p = LogisticRegression(C=1, max_iter=3000).fit(f(Ad, Ed, Nd), yd).predict(f(A, E, N)); ok = p == y
        print(f"{n:22s}:", fmt(metrics(y, p)))
        for nm, b in (("vs Analyst", base), ("vs Evidence", ev)):
            d = ok.astype(float) - b; ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5])
            fx, rg = int((ok & ~b).sum()), int((b & ~ok).sum()); pv = binomtest(fx, fx + rg, .5).pvalue if fx + rg else 1
            print(f"     {nm:12s} {d.mean()*100:+.1f} pts CI [{ci[0]*100:+.1f},{ci[1]*100:+.1f}] fixed {fx} regressed {rg} McNemar p={pv:.2g}")
        if n == "stack A+E": 
            json.dump([{"id": i, "gt": int(g) + 1, "analyst": int(a) + 1, "evidence": int(e_) + 1, "final": int(q) + 1} for i, g, a, e_, q in zip(ids, y, A.argmax(1), E.argmax(1), p)], open(f"{D}phase2_trace_{split}.json", "w"))
            dis = (A.argmax(1) != E.argmax(1)); print(f"     agents disagree on {dis.mean():.0%} rows: analyst right {base[dis].mean():.0%}, evidence right {ev[dis].mean():.0%}, stack right {ok[dis].mean():.0%};  agree rows: stack right {ok[~dis].mean():.0%}")
