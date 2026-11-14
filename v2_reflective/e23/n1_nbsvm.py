"""N1 (declared LOG.md 2026-11-02): NB-SVM text agent, logistic version (Wang & Manning 2012), same training rows as E6b. Prints dev, dev2 and natural_dev only.
usage: n1_nbsvm.py   (tee to n1_nbsvm.out; writes nb1_<split>_probs.jsonl). SMOKE=1: 20K rows, dev only, writes nothing."""
import json, os, time, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
D = "v2_reflective/e23/"; L = lambda p: [json.loads(l) for l in open(f"{D}{p}.jsonl")]; t = json.load(open(D + "evidence_config.json"))["title"]
txt = lambda r: (r["title"] + " . " if t else "") + r["review_text"]
SMOKE = os.environ.get("SMOKE") == "1"; SEEN = ("dev", "dev2", "natural_dev")


def ratio(X, pos):
    a, b = np.asarray(X[pos].sum(0)).ravel() + 1, np.asarray(X[~pos].sum(0)).ravel() + 1
    return np.log(a / a.sum()) - np.log(b / b.sum())


t0 = time.time(); d2 = {r["review_id"] for r in L("dev2")}
tr = L("evid_train") + ([] if SMOKE else L("evid_big_extra_nodev2")); assert not d2 & {r["review_id"] for r in tr}
if SMOKE: tr = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:20000]]
vec = CountVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, binary=True); X = vec.fit_transform(map(txt, tr)).tocsr()
y = np.array([r["gt_rating"] for r in tr]); print("rows", len(tr), "features", X.shape[1], f"{time.time()-t0:.0f}s", flush=True)
R, M = {}, {}
for c in range(1, 6):
    R[c] = ratio(X, y == c); M[c] = LogisticRegression(C=1, max_iter=1000).fit(X.multiply(R[c]).tocsr(), y == c)
    print("class", c, "iters", int(M[c].n_iter_[0]), f"{time.time()-t0:.0f}s", flush=True)


def probs(rs):
    A = vec.transform(map(txt, rs)); P = np.c_[[M[c].predict_proba(A.multiply(R[c]).tocsr())[:, 1] for c in range(1, 6)]].T
    return P / P.sum(1, keepdims=True)


for n in (("dev",) if SMOKE else ("dev2", "dev", "natural_dev", "eval", "natural_test")):
    rs = L(n); P = probs(rs); assert np.isfinite(P).all() and np.allclose(P.sum(1), 1)
    if n in SEEN: print(n, "alone exact", f"{((P.argmax(1) + 1) == np.array([r['gt_rating'] for r in rs])).mean():.3f}", flush=True)
    if not SMOKE:
        with open(f"{D}nb1_{n}_probs.jsonl", "w") as o:
            for r, p in zip(rs, P): o.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in p]}) + "\n")
print("done", f"{time.time()-t0:.0f}s")
