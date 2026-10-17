"""Evidence-agent probs for natural_dev. Refits the SAME model as evidence.py (title+text, C=3, 400K feats) on evid_train and first checks it
reproduces the stored eval probs (sanity), then writes evid_natural_dev_probs.jsonl."""
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D = "v2_reflective/e23/"; L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
tr, ev, nd = L("evid_train"), L("eval"), L("natural_dev"); txt = lambda r: r["title"] + " . " + r["review_text"]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True)
m = LogisticRegression(C=3, max_iter=400).fit(vec.fit_transform([txt(r) for r in tr]), [r["gt_rating"] for r in tr])
f = lambda rs: m.predict_proba(vec.transform([txt(r) for r in rs]))
old = {json.loads(l)["review_id"]: json.loads(l)["probs"] for l in open(f"{D}evid_eval_probs.jsonl")}
P = f(ev[:100]); dev_max = max(abs(P[i, k] - old[ev[i]["review_id"]][k]) for i in range(100) for k in range(5)); print(f"reproduction check: max |diff| vs stored eval probs = {dev_max:.2e}")
assert dev_max < 1e-4, "evidence model does not reproduce stored probs"
with open(f"{D}evid_natural_dev_probs.jsonl", "w") as o:
    for r, p in zip(nd, f(nd)): o.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in p]}) + "\n")
print("wrote evid_natural_dev_probs.jsonl", len(nd))
