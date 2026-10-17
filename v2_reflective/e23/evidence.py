"""Evidence model on Electronics 2023: TF-IDF(1-2gram)+LR on 100K balanced evid_train. Variants on dev only: text / title+text. Writes probs for dev, eval, natural_test."""
import json, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
L = lambda p: [json.loads(l) for l in open(f"v2_reflective/e23/{p}.jsonl")]
tr, dv, ev, nt = L("evid_train"), L("dev"), L("eval"), L("natural_test")
txt = lambda r, t: (r["title"] + " . " if t else "") + r["review_text"]
def fit(use_title, C=3):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True)
    X = vec.fit_transform([txt(r, use_title) for r in tr]); m = LogisticRegression(C=C, max_iter=400).fit(X, [r["gt_rating"] for r in tr])
    return lambda rs: m.predict_proba(vec.transform([txt(r, use_title) for r in rs]))
best = None
for t in (False, True):
    f = fit(t); P = f(dv); y = np.array([r["gt_rating"] for r in dv]); acc = ((P.argmax(1) + 1) == y).mean(); print(f"title={t}: dev exact {acc:.3f}", flush=True)
    if best is None or acc > best[0]: best = (acc, t, f)
_, t, f = best; print("chosen on dev: title =", t)
for name, rs in (("dev", dv), ("eval", ev), ("natural_test", nt)):
    P = f(rs)
    with open(f"v2_reflective/e23/evid_{name}_probs.jsonl", "w") as o:
        for r, p in zip(rs, P): o.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in p]}) + "\n")
json.dump({"title": t}, open("v2_reflective/e23/evidence_config.json", "w"))
