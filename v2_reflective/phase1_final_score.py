import json, numpy as np, re, sys
sys.path.insert(0, "src")
from agentic_sentiment.agents.parsing import parse_rating
def L(p): return {json.loads(l)["review_id"]: json.loads(l) for l in open(p)}
A = L(".scratch/kaggle_eval_probs.jsonl")
V = {"A leak (as reported)": A}
for n in ("B_random_shot", "C_train_format", "D_zero_shot"): V[n] = L(f"v2_reflective/p1final_{n}.jsonl")
ids = list(V["B_random_shot"]); y = np.array([A[i]["gt_rating"] for i in ids])
base = None; rng = np.random.default_rng(0)
for n, d in V.items():
    p = np.array([np.argmax(d[i]["probs"]) + 1 for i in ids]); ok = p == y
    s = f"{n:24s} exact {ok.mean():.1%}  within1 {(abs(p-y)<=1).mean():.1%}  MAE {abs(p-y).mean():.3f}"
    if base is None: base = ok
    else:
        dd = ok.astype(float) - base; ci = np.percentile([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(3000)], [2.5, 97.5])
        s += f"  vs A: {dd.mean()*100:+.1f} [{ci[0]*100:+.1f},{ci[1]*100:+.1f}]"
    print(s)
print("\n--- hallucination study (150 generations) ---")
G = [json.loads(l) for l in open("v2_reflective/p1final_H_generations.jsonl")]
n = len(G); parsed = [parse_rating(g["output"]) for g in G]
fmt_fail = sum(p is None for p in parsed)
desc = {1:"dissatisfaction",2:"softened",3:"neutral",4:"positive experience",5:"satisfaction"}
mism = 0; invent = 0; wrong = 0; conf_wrong = 0
for g, p in zip(G, parsed):
    o = g["output"]
    if p is not None:
        wrong += p != g["gt_rating"]
        m = re.search(r"\d\.\s*(.*)", o)
        if m and p in desc and desc[p] not in o.lower().split("(rating")[0] and "rating " + str(p) not in o.lower(): mism += 1
    if re.search(r"\n\s*(Review|Example)\s*:", o): invent += 1
print(f"n={n} unparseable {fmt_fail}  wrong rating {wrong} ({wrong/n:.0%})  description doesn't match own rating {mism}  runs on into invented 'Review:/Example:' {invent}")
bad = [(g, p) for g, p in zip(G, parsed) if p is not None and abs(p - g["gt_rating"]) >= 2][:5]
print("severe misses (>=2 stars):", sum(p is not None and abs(p-g['gt_rating'])>=2 for g,p in zip(G,parsed)))
for g, p in bad: print(f" gt={g['gt_rating']} pred={p} | {g['review_text'][:160]!r}\n   -> {g['output'][:200]!r}")
print("\nsample raw outputs:"); [print(" ", repr(g["output"][:220])) for g in G[:4]]
