"""Gate 0: can the pair adjudicator beat Phase 1 on the (top-1, top-2) pair? Dev rows only (500, balanced, fit set).
Both display orders are scored and averaged, so position bias cancels. Then simulates Phase 2 on dev: inside the
uncertainty gate (top-2 probability margin < th) the adjudicator picks between Phase 1's two candidates."""
import json, math, sys
sys.path.insert(0, "v2_reflective")
from mlx_lm import load
import mlx.core as mx
from build_pair_data import prompt

model, tok = load("v2_reflective/mlx_model", adapter_path="v2_reflective/pair_adapter")
dig = {d: tok.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)}
space = tok.encode(" ", add_special_tokens=False)

def score(text, a, b):  # log-odds that the true rating is a (vs b), prompt shows "a or b"
    ids = tok.encode(prompt(text, a, b), add_special_tokens=False) + space
    lg = model(mx.array(ids)[None])[0, -1]  # no BOS: matches training (plain-concatenation template)
    return float(lg[dig[a]] - lg[dig[b]])

rows = {json.loads(l)["review_id"]: json.loads(l) for l in open(".scratch/kaggle_dev_rows.jsonl")}
out = []
for l in open(".scratch/kaggle_dev_probs.jsonl"):
    r = json.loads(l); p = r["probs"]
    o = sorted(range(5), key=lambda k: -p[k]); a, b = o[0] + 1, o[1] + 1
    text = rows[r["review_id"]]["review_text"][:600].replace("<br />", " ")
    d = (score(text, a, b) - score(text, b, a)) / 2   # >0: adjudicator says a (Phase 1's top-1)
    out.append({"id": r["review_id"], "gt": r["gt_rating"], "a": a, "b": b, "margin": p[o[0]] - p[o[1]], "adj": d, "probs": p})
json.dump(out, open("v2_reflective/gate0_dev.json", "w"))
n = len(out)
print(f"dev rows {n}; Phase 1 top-1 exact {sum(x['gt']==x['a'] for x in out)/n:.1%}")
for th in (0.2, 0.3, 0.4, 0.5, 1.01):
    g = [x for x in out if x["margin"] < th and x["gt"] in (x["a"], x["b"])]
    if g:
        an = sum(x["gt"] == x["a"] for x in g) / len(g)
        ad = sum((x["gt"] == x["a"]) == (x["adj"] > 0) for x in g) / len(g)
        print(f"margin<{th}: pairs {len(g)}  Phase1 picks right {an:.1%}  adjudicator picks right {ad:.1%}")
    ex = sum((x["a"] if (x["margin"] >= th or x["adj"] > 0) else x["b"]) == x["gt"] for x in out) / n
    print(f"   simulated Phase 2 exact (adjudicator replaces top-1 inside gate): {ex:.1%}")
