"""Variance of stack-minus-evidence over evidence-train bootstrap + dev bootstrap for the stacker. Pre-declared in LOG.md 2026-10-20. Resumable per seed."""
import json, os, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"; L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
tr, S = L("evid_train"), {n: L(n) for n in ("dev", "eval", "natural_test")}; txt = lambda r: r["title"] + " . " + r["review_text"]
A = {n: {r["review_id"]: r["probs"] for r in map(json.loads, open(f"{D}p1/probs_R_{n}.jsonl"))} for n in S}
Aa = {n: np.clip([A[n][r["review_id"]] for r in S[n]], 1e-6, 1) for n in S}; y = {n: np.array([r["gt_rating"] - 1 for r in S[n]]) for n in S}
for seed in range(5):
    out = f"{D}variance_seed{seed}.json"
    if os.path.exists(out): continue
    rng = np.random.default_rng(seed); b = rng.integers(0, len(tr), len(tr)); sub = [tr[i] for i in b]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True)
    m = LogisticRegression(C=3, max_iter=400).fit(vec.fit_transform([txt(r) for r in sub]), [r["gt_rating"] for r in sub])
    E = {n: np.clip(m.predict_proba(vec.transform([txt(r) for r in S[n]])), 1e-6, 1) for n in S}
    F = lambda n: np.c_[np.log(Aa[n] / Aa[n].sum(1, keepdims=True)), np.log(E[n])]
    bd = rng.integers(0, len(S["dev"]), len(S["dev"])); st = LogisticRegression(C=1, max_iter=3000).fit(F("dev")[bd], y["dev"][bd]); res = {}
    for n in ("eval", "natural_test"):
        pe, ps = E[n].argmax(1), st.predict(F(n)); res[n] = {"ev_exact": float((pe == y[n]).mean()), "st_exact": float((ps == y[n]).mean()),
            "ev_f1": float(f1_score(y[n], pe, average="macro")), "st_f1": float(f1_score(y[n], ps, average="macro"))}
    json.dump(res, open(out, "w")); print("seed", seed, res, flush=True)
print("=== summary (mean +/- sd over 5 seeds)")
for n in ("eval", "natural_test"):
    R = [json.load(open(f"{D}variance_seed{s}.json"))[n] for s in range(5)]
    for k, lab in (("ev_exact", "Evidence exact"), ("st_exact", "Stack exact"), ("ev_f1", "Evidence macroF1"), ("st_f1", "Stack macroF1")):
        v = np.array([r[k] for r in R]); print(f"{n} {lab}: {v.mean():.4f} +/- {v.std(ddof=1):.4f}")
    d = np.array([r["st_exact"] - r["ev_exact"] for r in R]); f = np.array([r["st_f1"] - r["ev_f1"] for r in R])
    print(f"{n} stack-minus-Evidence exact: {d.mean()*100:+.2f} +/- {d.std(ddof=1)*100:.2f} pts (per seed {np.round(d*100,2).tolist()}); macroF1 {f.mean():+.4f} +/- {f.std(ddof=1):.4f}")
