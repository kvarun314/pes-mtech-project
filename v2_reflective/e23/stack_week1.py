"""Week 1 (S1 + P2-1 + P2-3): Reflector variants. All 5 configs below are PRE-DECLARED (2026-10-17, LOG.md) and ALL are read out once on eval + natural_test.
Dev-only: 5-fold CV exact/macroF1 on the fit sets. Fit sets: bal dev (500 balanced, existing) and natural_dev (500 natural, new, e23/build_natural_dev.py).
C0 A+E fit bal-dev (reference = phase2_e23 stack) | C1 A+E fit natural_dev | C2 A+E fit bal-dev reweighted to natural prior
C3 A+E+[margin,entropy,disagree] fit natural_dev | C4 routing: agree rows keep the shared label, disagree rows get a stacker (A+E+feats) fit on natural_dev disagree rows.
Reports exact, macro-F1, within1, MAE, recall per class; paired bootstrap + McNemar vs C0 and vs Evidence alone."""
import json, numpy as np, collections
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
D = "v2_reflective/e23/"; rng = np.random.default_rng(0)
Lj = lambda p: {r["review_id"]: r for r in map(json.loads, open(p))}
def get(split, evsplit=None):
    a, e = Lj(f"{D}p1/probs_R_{split}.jsonl"), Lj(f"{D}evid_{split}_probs.jsonl"); ids = [i for i in a if i in e]
    y = np.array([a[i]["gt_rating"] - 1 for i in ids]); A = np.clip([a[i]["probs"] for i in ids], 1e-6, 1); E = np.clip([e[i]["probs"] for i in ids], 1e-6, 1)
    return y, A / A.sum(1, keepdims=True), E / E.sum(1, keepdims=True)
ent = lambda P: -(P * np.log(P)).sum(1); marg = lambda P: np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2]
def feats(A, E, extra):
    base = np.c_[np.log(A), np.log(E)]
    return np.c_[base, marg(A), marg(E), ent(A), ent(E), (A.argmax(1) != E.argmax(1)).astype(float)] if extra else base
def met(y, p):
    return dict(exact=(p == y).mean(), macroF1=f1_score(y, p, average="macro"), within1=(abs(p - y) <= 1).mean(), mae=abs(p - y).mean(),
                recall=[round(float(((p == c) & (y == c)).sum() / max((y == c).sum(), 1)), 2) for c in range(5)])
fmt = lambda m: f"exact {m['exact']:.1%} macroF1 {m['macroF1']:.3f} within1 {m['within1']:.1%} MAE {m['mae']:.3f} recall {m['recall']}"
yb, Ab, Eb = get("dev"); yn, An, En = get("natural_dev")
nat_prior = np.bincount(yn, minlength=5) / len(yn)
fit = lambda X, y, w=None: LogisticRegression(C=1, max_iter=3000).fit(X, y, sample_weight=w)
w_bal = (nat_prior / (np.bincount(yb, minlength=5) / len(yb)))[yb]
# ---- dev CV (selection evidence only)
print("=== DEV 5-fold CV (fit sets only; no eval/natural_test touched)")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
strat = np.where(np.bincount(yn)[yn] < 5, 4, yn)  # tiny classes folded for stratification only
for nm, extra in (("natural_dev A+E", False), ("natural_dev A+E+feats", True)):
    X = feats(An, En, extra); p = np.zeros(len(yn), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, strat): p[te] = fit(X[tr], yn[tr]).predict(X[te])
    print(f"{nm:24s}", fmt(met(yn, p)))
print("natural_dev Analyst :", fmt(met(yn, An.argmax(1)))); print("natural_dev Evidence:", fmt(met(yn, En.argmax(1))))
# ---- configs
def build():
    m0 = fit(feats(Ab, Eb, False), yb); m1 = fit(feats(An, En, False), yn); m2 = fit(feats(Ab, Eb, False), yb, w_bal); m3 = fit(feats(An, En, True), yn)
    dis = An.argmax(1) != En.argmax(1); m4 = fit(feats(An, En, True)[dis], yn[dis])
    def C4(A, E):
        p = E.argmax(1).copy(); d = A.argmax(1) != E.argmax(1)
        if d.any(): p[d] = m4.predict(feats(A, E, True)[d])
        return p
    return {"C0 A+E bal-dev": lambda A, E: m0.predict(feats(A, E, False)), "C1 A+E natural-dev": lambda A, E: m1.predict(feats(A, E, False)),
            "C2 A+E bal-dev reweighted": lambda A, E: m2.predict(feats(A, E, False)), "C3 A+E+feats natural-dev": lambda A, E: m3.predict(feats(A, E, True)), "C4 routing": C4}
cfg = build()
def cmp(ok, b):
    d = ok.astype(float) - b; ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)], [2.5, 97.5])
    fx, rg = int((ok & ~b).sum()), int((b & ~ok).sum()); pv = binomtest(fx, fx + rg, .5).pvalue if fx + rg else 1
    return f"{d.mean()*100:+.1f} [{ci[0]*100:+.1f},{ci[1]*100:+.1f}] fixed {fx}/regressed {rg} p={pv:.2g}"
res = {}
for split in ("eval", "natural_test"):
    y, A, E = get(split); print(f"\n===== {split} n={len(y)}"); print("Analyst :", fmt(met(y, A.argmax(1)))); print("Evidence:", fmt(met(y, E.argmax(1)))); pe = E.argmax(1) == y; p0 = cfg["C0 A+E bal-dev"](A, E) == y
    for n, f in cfg.items():
        p = f(A, E); ok = p == y; print(f"{n:27s}", fmt(met(y, p))); print(f"{'':27s}vs Evidence {cmp(ok, pe)}")
        if n != "C0 A+E bal-dev": print(f"{'':27s}vs C0       {cmp(ok, p0)}")
        res[f"{split}|{n}"] = {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in met(y, p).items()}
json.dump(res, open(f"{D}stack_week1_results.json", "w"), indent=1, default=float)
