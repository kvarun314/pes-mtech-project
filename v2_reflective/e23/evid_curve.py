"""Phase G (declared LOG.md 2026-10-24): Evidence learning curve on held-out evid_train rows. eval/natural_test untouched."""
import json, random, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D = "v2_reflective/e23/"; tr = [json.loads(l) for l in open(D + "evid_train.jsonl")]; t = json.load(open(D + "evidence_config.json"))["title"]
txt = lambda r: (r["title"] + " . " if t else "") + r["review_text"]
by = {k: [r for r in tr if r["gt_rating"] == k] for k in range(1, 6)}; rng = random.Random(5)
for k in by: rng.shuffle(by[k])
val = [r for k in by for r in by[k][:2000]]; rest = {k: by[k][2000:] for k in by}; yv = [r["gt_rating"] for r in val]
for n in (2250, 4500, 9000, 18000):
    rs = [r for k in rest for r in rest[k][:n]]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True); X = vec.fit_transform(map(txt, rs))
    p = LogisticRegression(C=3, max_iter=400).fit(X, [r["gt_rating"] for r in rs]).predict(vec.transform(map(txt, val)))
    print(f"train {5*n:6d}: val exact {(p == np.array(yv)).mean():.4f} macroF1 {f1_score(yv, p, average='macro'):.4f}", flush=True)
