"""Amazon Reviews 2023, Electronics. Stream the first N review rows, keep reservoir samples: 21,000 per star (balanced pool) + 3,000 natural.
Frozen for the paper. Output: v2_reflective/e23/pool.jsonl. Rows keep file index as id."""
import json, random, sys, time
from datasets import load_dataset
URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl"
N = int(sys.argv[1]); K = 21000; KN = 3000; MIN = 20
rng = random.Random(2023)
res = {r: [] for r in range(1, 6)}; seen = {r: 0 for r in range(1, 6)}; nat = []; nseen = 0
t0 = time.time()
for i, row in enumerate(load_dataset("json", data_files=URL, split="train", streaming=True)):
    if i >= N: break
    t = (row.get("text") or "").replace("\n", " ").strip(); r = row.get("rating")
    if r is None: continue
    r = int(r)
    if r not in res or len(t) < MIN: continue
    rec = {"id": f"e23_{i}", "rating": r, "title": row.get("title") or "", "text": t, "asin": row.get("parent_asin"), "ts": row.get("timestamp")}
    seen[r] += 1
    if len(res[r]) < K: res[r].append(rec)
    else:
        j = rng.randrange(seen[r])
        if j < K: res[r][j] = rec
    nseen += 1
    if len(nat) < KN: nat.append(rec)
    else:
        j = rng.randrange(nseen)
        if j < KN: nat[j] = rec
    if i % 250000 == 0: print(i, {k: seen[k] for k in seen}, f"{time.time()-t0:.0f}s", flush=True)
with open("v2_reflective/e23/pool.jsonl", "w") as f:
    for r in res:
        for rec in res[r]: f.write(json.dumps({**rec, "pool": "balanced"}) + "\n")
    for rec in nat: f.write(json.dumps({**rec, "pool": "natural"}) + "\n")
print("DONE", {k: len(v) for k, v in res.items()}, len(nat), seen, flush=True)
