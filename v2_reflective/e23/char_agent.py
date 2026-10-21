"""B2 (declared LOG.md 2026-10-21): char_wb 2-5 TF-IDF + LR on evid_train -> evid3_<split>_probs.jsonl."""
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D = "v2_reflective/e23/"; L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]; txt = lambda r: r["title"] + " . " + r["review_text"]
tr = L("evid_train"); vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
clf = LogisticRegression(C=3, max_iter=300).fit(vec.fit_transform([txt(r) for r in tr]), [r["gt_rating"] for r in tr]); print("fit done", flush=True)
for s in ("dev", "natural_dev", "eval", "natural_test"):
    rs = L(s); P = clf.predict_proba(vec.transform([txt(r) for r in rs])); y = np.array([r["gt_rating"] for r in rs]); print(f"{s}: exact {((P.argmax(1)+1)==y).mean():.3f}", flush=True)
    with open(f"{D}evid3_{s}_probs.jsonl", "w") as o:
        for r, p in zip(rs, P): o.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in p]}) + "\n")
