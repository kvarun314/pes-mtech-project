"""Q5 (declared LOG.md 2026-10-28): main stack vs Phase 1 exact accuracy by review year and product popularity, eval and natural_test.
usage: q5_breakdown.py   (tee to q5_breakdown.out; writes q5_breakdown.json)"""
import collections, datetime as dt, json, sys, numpy as np, joblib
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import logp
full = joblib.load(PT.REFLECTOR_JOBLIB)["full"]
L = lambda f: {x["review_id"]: x for x in map(json.loads, open(PT.D + f))}
pool = {r["id"]: r for r in map(json.loads, open(PT.D + "pool.jsonl"))}
per_asin = collections.Counter(r["asin"] for r in pool.values())
year = lambda i: dt.datetime.fromtimestamp(pool[i]["ts"] / 1000, dt.UTC).year
BANDS = {"year": (lambda i: "<=2014" if year(i) <= 2014 else "2015-2017" if year(i) <= 2017 else "2018-2020" if year(i) <= 2020 else "2021-2023",
                  ["<=2014", "2015-2017", "2018-2020", "2021-2023"]),
         "product_reviews": (lambda i: "1" if per_asin[pool[i]["asin"]] == 1 else "2-5" if per_asin[pool[i]["asin"]] <= 5 else "6+", ["1", "2-5", "6+"])}
rng = np.random.default_rng(0); res = {}
for s in ("eval", "natural_test"):
    P, E, R = L(f"evid7_{s}_probs.jsonl"), L(f"evid6b_{s}_probs.jsonl"), L(f"p1/probs_R_{s}.jsonl"); ids = list(P)
    y = np.array([P[i]["gt_rating"] for i in ids])
    sh = full.predict(np.c_[logp([P[i]["probs"] for i in ids]), logp([E[i]["probs"] for i in ids])]) + 1 == y
    p1 = np.array([R[i]["probs"] for i in ids]).argmax(1) + 1 == y
    res[s] = {}; print(f"== {s} (n {len(ids)}; SH {sh.mean():.3f}, P1 {p1.mean():.3f})")
    for dim, (f, order) in BANDS.items():
        lab = np.array([f(i) for i in ids]); rows = []
        for b in order:
            m = np.flatnonzero(lab == b)
            if not len(m): continue
            d = sh[m].astype(float) - p1[m]; boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            rows.append({"band": b, "n": int(len(m)), "SH": float(sh[m].mean()), "P1": float(p1[m].mean()), "diff": float(d.mean()),
                         "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
            r = rows[-1]; print(f"{dim:16s} {b:10s} n {r['n']:5d}  SH {r['SH']:.3f}  P1 {r['P1']:.3f}  diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        res[s][dim] = rows
json.dump(res, open(PT.D + "q5_breakdown.json", "w"), indent=1)
