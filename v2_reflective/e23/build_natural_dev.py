"""S1: natural-distribution dev set (Week 1, PROJECT_ROADMAP). 500 rows drawn from the unused part of the NATURAL pool, seed 2023+1,
disjoint by id AND exact text from natural_test, eval, dev, evid_train (so the evidence model never saw them). Frozen splits are not modified."""
import json, random, collections
D = "v2_reflective/e23/"; L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
used_ids, used_txt = set(), set()
for n in ("natural_test", "eval", "dev", "evid_train"):
    for r in L(n): used_ids.add(r["review_id"]); used_txt.add(r["review_text"])
P = [json.loads(l) for l in open(f"{D}pool.jsonl")]; nat = [r for r in P if r["pool"] == "natural"]
cand = []; seen = set()
for r in nat:
    if r["id"] in used_ids or r["text"] in used_txt or r["text"] in seen: continue
    seen.add(r["text"]); cand.append(r)
random.Random(2024).shuffle(cand); pick = cand[:500]
print("natural pool", len(nat), "candidates after disjointness", len(cand))
with open(f"{D}natural_dev.jsonl", "w") as f:
    for r in pick: f.write(json.dumps({"review_id": r["id"], "gt_rating": r["rating"], "review_text": r["text"], "title": r["title"]}) + "\n")
print("natural_dev", len(pick), sorted(collections.Counter(r["rating"] for r in pick).items()))
