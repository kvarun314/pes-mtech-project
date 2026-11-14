"""Phase H (declared LOG.md 2026-10-24): E probs on dev2 (evidence.py config, 100K) and E6b (500K config minus dev2, max_iter 1000) on all gate/readout splits."""
import json, time, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D = "v2_reflective/e23/"; L = lambda p: [json.loads(l) for l in open(f"{D}{p}.jsonl")]; t = json.load(open(D + "evidence_config.json"))["title"]
txt = lambda r: (r["title"] + " . " if t else "") + r["review_text"]
def train(rows, it):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True); X = vec.fit_transform(map(txt, rows))
    m = LogisticRegression(C=3, max_iter=it).fit(X, [r["gt_rating"] for r in rows]); return lambda rs: m.predict_proba(vec.transform(map(txt, rs)))
def write(pref, f, name):
    rs = L(name); P = f(rs)
    with open(f"{D}{pref}_{name}_probs.jsonl", "w") as o:
        for r, p in zip(rs, P): o.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in p]}) + "\n")
    print(pref, name, "alone exact", f"{((P.argmax(1) + 1) == np.array([r['gt_rating'] for r in rs])).mean():.3f}", flush=True)
t0 = time.time(); fE = train(L("evid_train"), 400); print("E fit", f"{time.time()-t0:.0f}s", flush=True)
write("evid", fE, "dev2"); write("evidchk", fE, "dev")  # evidchk_dev must reproduce evid_dev (determinism check)
d2 = {r["review_id"] for r in L("dev2")}; tr = L("evid_train") + L("evid_big_extra_nodev2"); assert not d2 & {r["review_id"] for r in tr}
fB = train(tr, 1000); print("E6b fit", f"{time.time()-t0:.0f}s", flush=True)
for n in ("dev2", "dev", "natural_dev", "eval", "natural_test"): write("evid6b", fB, n)
a = [json.loads(l)["probs"] for l in open(D + "evid_dev_probs.jsonl")]; b = [json.loads(l)["probs"] for l in open(D + "evidchk_dev_probs.jsonl")]
print("determinism max |diff| vs evid_dev:", float(np.abs(np.array(a) - np.array(b)).max()))
