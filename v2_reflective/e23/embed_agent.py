"""B1 (declared LOG.md 2026-10-21): MiniLM sentence-embedding agent. Chunked + resumable embeddings, then LR on evid_train -> evid2_<split>_probs.jsonl."""
import json, os, numpy as np, time
from sentence_transformers import SentenceTransformer
D = "v2_reflective/e23/"; os.makedirs(D + "emb", exist_ok=True); CH = 5000
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps"); m.max_seq_length = 256
rows = {s: [json.loads(l) for l in open(f"{D}{s}.jsonl")] for s in ("evid_train", "dev", "natural_dev", "eval", "natural_test")}
t0 = time.time(); E = {}
for s, rs in rows.items():
    parts = []
    for c in range(0, len(rs), CH):
        f = f"{D}emb/{s}_{c//CH:03d}.npy"
        if not os.path.exists(f): np.save(f, m.encode([r["title"] + " . " + r["review_text"] for r in rs[c:c + CH]], batch_size=64, normalize_embeddings=True))
        parts.append(np.load(f)); print(s, c, f"{time.time()-t0:.0f}s", flush=True)
    E[s] = np.vstack(parts)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1, max_iter=1000).fit(E["evid_train"], [r["gt_rating"] for r in rows["evid_train"]])
for s in ("dev", "natural_dev", "eval", "natural_test"):
    P = clf.predict_proba(E[s]); y = np.array([r["gt_rating"] for r in rows[s]]); print(f"{s}: exact {((P.argmax(1)+1)==y).mean():.3f}")
    with open(f"{D}evid2_{s}_probs.jsonl", "w") as o:
        for r, p in zip(rows[s], P): o.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(x) for x in p]}) + "\n")
