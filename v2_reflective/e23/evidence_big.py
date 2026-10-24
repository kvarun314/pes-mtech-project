"""Phase G2 (declared LOG.md 2026-10-24): Evidence agent E6 = same TF-IDF+LR config as evidence.py, trained on evid_train + evid_big_extra (500K balanced)."""
import json, time, collections, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D = "v2_reflective/e23/"; L = lambda p: [json.loads(l) for l in open(f"{D}{p}.jsonl")]
tr = L("evid_train") + L("evid_big_extra"); t = json.load(open(D + "evidence_config.json"))["title"]
print("train rows", len(tr), sorted(collections.Counter(r["gt_rating"] for r in tr).items()), flush=True)
held = set()
for n in ("dev", "eval", "natural_test", "natural_dev"):
    for r in L(n): held.add(r["review_id"]); held.add(r["review_text"])
assert not any(r["review_id"] in held or r["review_text"] in held for r in tr), "leak"
txt = lambda r: (r["title"] + " . " if t else "") + r["review_text"]; t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True); X = vec.fit_transform(map(txt, tr)); print("tfidf", X.shape, f"{time.time()-t0:.0f}s", flush=True)
m = LogisticRegression(C=3, max_iter=400).fit(X, [r["gt_rating"] for r in tr]); print("fit", f"{time.time()-t0:.0f}s", flush=True)
for name in ("dev", "natural_dev", "eval", "natural_test"):
    rs = L(name); P = m.predict_proba(vec.transform(map(txt, rs)))
    with open(f"{D}evid6_{name}_probs.jsonl", "w") as o:
        for r, p in zip(rs, P): o.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in p]}) + "\n")
    if name in ("dev", "natural_dev"): print(name, "E6 alone exact", f"{((P.argmax(1) + 1) == np.array([r['gt_rating'] for r in rs])).mean():.3f}", flush=True)
print("done")
