"""Phase H (declared LOG.md 2026-10-24): dev2 (2,000 balanced, from evid_big_extra) and fit2 (30K balanced from evid_train, disjoint from Phase F fit)."""
import json, random, collections
D = "v2_reflective/e23/"; L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
ex = L("evid_big_extra"); rng = random.Random(2026); dev2 = []
for k in range(1, 6):
    c = [r for r in ex if r["gt_rating"] == k]; rng.shuffle(c); dev2 += c[:400]
d2 = {r["review_id"] for r in dev2}; rng.shuffle(dev2)
old = set(json.load(open(D + "hid/ids.json"))["fit"]); tr = [r for r in L("evid_train") if r["review_id"] not in old]; rng = random.Random(12); fit2 = []
for k in range(1, 6):
    c = [r for r in tr if r["gt_rating"] == k]; rng.shuffle(c); fit2 += c[:6000]
for name, rows in (("dev2", dev2), ("fit2", fit2), ("evid_big_extra_nodev2", [r for r in ex if r["review_id"] not in d2])):
    with open(f"{D}{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), sorted(collections.Counter(r["gt_rating"] for r in rows).items()))
assert not d2 & old and not d2 & {r["review_id"] for r in fit2}
