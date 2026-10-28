"""I2 (declared LOG.md 2026-10-28): learned Hand-off (LR on 8 uncertainty features predicting 'main stack correct') vs max probability.
dev2: out-of-fold main-stack probabilities (5-fold Reflector refit), deferral model scored by 5-fold CV. Gate first; test readout only on PASS.
usage: i2_handoff_learned.py   (tee to i2_handoff_learned.out; writes i2_handoff_learned.json)"""
import json, sys, numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
import joblib
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
full = joblib.load(PT.REFLECTOR_JOBLIB)["full"]
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
def load(s):
    P, E, T = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"{s}.jsonl"); ids = list(P)
    p = np.array([P[i]["probs"] for i in ids]); e = np.array([E[i]["probs"] for i in ids]); y = np.array([P[i]["gt_rating"] for i in ids])
    return p, e, y, np.array([len(T[i]["review_text"].split()) for i in ids]), np.c_[logp(p), logp(e)]
def feats(S, p, e, n):
    s = np.sort(S, 1); ent = -(S * np.log(np.clip(S, 1e-9, 1))).sum(1)
    return np.c_[s[:, -1], s[:, -1] - s[:, -2], ent, p.max(1), e.max(1), p.argmax(1) == e.argmax(1), np.abs(p.argmax(1) - e.argmax(1)), np.log1p(n)]
def acc_at(conf, ok, cov):
    k = int(round(cov * len(ok))); return float(ok[np.argsort(-conf)[:k]].mean())
res = {}
p, e, y, n, X = load("dev2")
S = np.zeros((len(y), 5))                                                     # out-of-fold main-stack probabilities on dev2
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y): S[te] = clone(full).fit(X[tr], y[tr] - 1).predict_proba(X[te])
ok = S.argmax(1) + 1 == y; F = feats(S, p, e, n)
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
cv = cross_val_predict(model, F, ok, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_proba")[:, 1]
base, learn = S.max(1), cv
g = {"dev2_oof_exact": float(ok.mean()), "auroc_maxprob": roc_auc_score(ok, base), "auroc_learned": roc_auc_score(ok, learn),
     "acc50_maxprob": acc_at(base, ok, .5), "acc50_learned": acc_at(learn, ok, .5), "acc40_maxprob": acc_at(base, ok, .4), "acc40_learned": acc_at(learn, ok, .4)}
g["pass"] = bool(g["auroc_learned"] - g["auroc_maxprob"] >= .02 and g["acc50_learned"] - g["acc50_maxprob"] >= .01)
res["gate_dev2"] = g; print("GATE dev2", json.dumps(g, indent=1))
if g["pass"]:
    m = model.fit(F, ok); c2 = m.predict_proba(F)[:, 1]; res["readout"] = {}
    for s in ("eval", "natural_test"):
        p, e, y, n, X = load(s); St = full.predict_proba(X); okt = St.argmax(1) + 1 == y; ct = m.predict_proba(feats(St, p, e, n))[:, 1]; rows = []
        for t in (.9, .8, .7, .6, .5, .4):
            thb, thl = np.quantile(S.max(1), 1 - t), np.quantile(c2, 1 - t); mb, ml = St.max(1) >= thb, ct >= thl
            rows.append({"target": t, "cov_maxprob": float(mb.mean()), "acc_maxprob": float(okt[mb].mean()), "cov_learned": float(ml.mean()), "acc_learned": float(okt[ml].mean())})
        rng = np.random.default_rng(0); d = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y)); d.append(acc_at(ct[i], okt[i], .4) - acc_at(St.max(1)[i], okt[i], .4))
        res["readout"][s] = {"auroc_maxprob": roc_auc_score(okt, St.max(1)), "auroc_learned": roc_auc_score(okt, ct), "rows": rows,
                             "acc40_matched_diff": acc_at(ct, okt, .4) - acc_at(St.max(1), okt, .4), "acc40_diff_ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
        print(s, json.dumps(res["readout"][s], indent=1))
else:
    print("FAIL: no test read")
json.dump(res, open(PT.D + "i2_handoff_learned.json", "w"), indent=1)
