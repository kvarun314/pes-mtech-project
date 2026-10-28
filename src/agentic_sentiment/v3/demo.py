"""Demo helpers: live call with cached fallback (never break on stage) and tagged paper numbers for check_demo.py."""
import json, os

CACHE = "presentation/demo_cache.json"


def cached(key, fn, cache_path=CACHE):
    store = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    if os.environ.get("DEMO_OFFLINE") and key in store:
        print(f"[cached result: {key}]"); return store[key]
    try:
        val = fn()
    except Exception as ex:                       # model missing / too slow: fall back, say so
        if key not in store:
            raise
        print(f"[live run failed ({type(ex).__name__}); showing cached result: {key}]"); return store[key]
    if os.environ.get("DEMO_RECORD"):
        store[key] = val; json.dump(store, open(cache_path, "w"), indent=1)
    return val


def paper_line(label, value):
    print(f"PAPER: {label} {value}")


def pick_demo_ids(path="v2_reflective/e23/eval.jsonl", max_words=60):
    """First eval review per star with 15-60 words and where SH is correct (stored probs): short, readable, honest about errors elsewhere."""
    import numpy as np
    import joblib
    from agentic_sentiment.v3 import paths as PT
    G = lambda p: {x["review_id"]: x["probs"] for x in map(json.loads, open(f"{PT.D}{p}_eval_probs.jsonl"))}
    r = joblib.load(PT.REFLECTOR_JOBLIB)["full"]; P7, E6 = G(PT.PROBE_PREFIX), G("evid6b"); out = {}
    for row in map(json.loads, open(path)):
        s, n = row["gt_rating"], len(row["review_text"].split())
        if s in out or not 15 <= n <= max_words: continue
        x = np.r_[np.log(np.clip(P7[row["review_id"]], 1e-6, 1)), np.log(np.clip(E6[row["review_id"]], 1e-6, 1))][None]
        if r.predict(x)[0] + 1 == s: out[s] = row["review_id"]
    return [out[s] for s in range(1, 6)]
