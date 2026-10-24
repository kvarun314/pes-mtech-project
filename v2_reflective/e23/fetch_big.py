"""Phase G2 (declared LOG.md 2026-10-24): 400K extra balanced Electronics rows, disjoint from every frozen split and pool."""
import json, random, sys, time
from datasets import load_dataset
URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl"
D = "v2_reflective/e23/"; N = 6_000_000; K = 80000; MIN = 20; rng = random.Random(2025)
bad_id = {json.loads(l)["id"] for l in open(D + "pool.jsonl")}; bad_txt = set()
for n in ("evid_train", "dev", "eval", "natural_test", "natural_dev"):
    for l in open(f"{D}{n}.jsonl"): r = json.loads(l); bad_id.add(r["review_id"]); bad_txt.add(r["review_text"])
res = {r: [] for r in range(1, 6)}; seen = {r: 0 for r in range(1, 6)}; t0 = time.time(); skipped = 0
for i, row in enumerate(load_dataset("json", data_files=URL, split="train", streaming=True)):
    if i >= N: break
    t = (row.get("text") or "").replace("\n", " ").strip(); r = row.get("rating")
    if r is None or int(r) not in res or len(t) < MIN: continue
    r = int(r)
    if f"e23_{i}" in bad_id or t in bad_txt: skipped += 1; continue
    rec = {"review_id": f"e23_{i}", "gt_rating": r, "review_text": t, "title": row.get("title") or ""}
    seen[r] += 1
    if len(res[r]) < K: res[r].append(rec)
    else:
        j = rng.randrange(seen[r])
        if j < K: res[r][j] = rec
    if i % 500000 == 0: print(i, seen, f"skipped {skipped}", f"{time.time()-t0:.0f}s", flush=True)
# exact-text dedup inside the new set (keep first); duplicates are rare
out, txt = [], set()
for k in res:
    for rec in res[k]:
        if rec["review_text"] in txt: continue
        txt.add(rec["review_text"]); out.append(rec)
with open(D + "evid_big_extra.jsonl", "w") as f:
    for rec in out: f.write(json.dumps(rec) + "\n")
print("DONE", {k: len(v) for k, v in res.items()}, "after dedup", len(out), "seen", seen, "skipped", skipped, flush=True)
