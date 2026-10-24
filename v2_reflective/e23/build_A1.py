"""Phase A1 (declared LOG.md 2026-10-24 night): fit3 = 80K balanced (16K/star) from evid_big_extra_nodev2, for a bigger probe fit (P3b)."""
import json, random, collections
D = "v2_reflective/e23/"; L = lambda n: [json.loads(l) for l in open(f"{D}{n}.jsonl")]
ex = L("evid_big_extra_nodev2"); rng = random.Random(2027); fit3 = []
for k in range(1, 6):
    c = [r for r in ex if r["gt_rating"] == k]; rng.shuffle(c); fit3 += c[:16000]
rng.shuffle(fit3)
used = {r["review_id"] for n in ("dev2", "fit2", "dev", "eval", "natural_test", "natural_dev") for r in L(n)}
assert not used & {r["review_id"] for r in fit3}, "fit3 overlaps an existing split"
with open(D + "fit3.jsonl", "w") as f:
    for r in fit3: f.write(json.dumps(r) + "\n")
print("fit3", len(fit3), sorted(collections.Counter(r["gt_rating"] for r in fit3).items()))
