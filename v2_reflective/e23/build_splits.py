"""Frozen splits, Amazon Reviews 2023 Electronics, disjoint by id AND by exact text. seed 2023.
natural_test 2,000 (natural rating mix) | eval 1,000 balanced (200/star) | dev 500 balanced (100/star) | evid_train 20,000/star balanced."""
import json, random, collections
rng = random.Random(2023)
P = [json.loads(l) for l in open("v2_reflective/e23/pool.jsonl")]
bal = [r for r in P if r["pool"] == "balanced"]; nat = [r for r in P if r["pool"] == "natural"]
rng.shuffle(nat); test = nat[:2000]; tids = {r["id"] for r in test}; ttexts = {r["text"] for r in test}
by = collections.defaultdict(list)
for r in bal:
    if r["id"] in tids or r["text"] in ttexts: continue
    by[r["rating"]].append(r)
ev, dv, tr = [], [], []; used = set(ttexts)
for s in range(1, 6):
    rs = by[s]; rng.shuffle(rs); seen = []
    for r in rs:
        if r["text"] in used: continue      # dedupe: same text never in two splits (and never twice)
        used.add(r["text"]); seen.append(r)
    ev += seen[:200]; dv += seen[200:300]; tr += seen[300:300 + 20000]
def w(name, rows):
    with open(f"v2_reflective/e23/{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps({"review_id": r["id"], "gt_rating": r["rating"], "review_text": r["text"], "title": r["title"]}) + "\n")
    print(name, len(rows), sorted(collections.Counter(r["rating"] for r in rows).items()))
for n, rs in (("natural_test", test), ("eval", ev), ("dev", dv), ("evid_train", tr)): rng.shuffle(rs); w(n, rs)
import statistics
for n, rs in (("eval", ev),): print("median chars", statistics.median(len(r["text"]) for r in rs), "p95", sorted(len(r["text"]) for r in rs)[int(.95 * len(rs))])
