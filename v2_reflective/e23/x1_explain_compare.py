"""X1 (declared LOG.md 2026-10-27): Phase 1 justification vs Phase 2 final explanation on the same 150 balanced eval reviews
(the saved Phase 1 generations from halluc_p1.py; Phase 2 = last draft in b2/eval.jsonl). Phase 1 text = output up to its first '!'
(the point where the template starts repeating). Same Instruct judge and prompt as Critic check 5, run on both.
usage: x1_explain_compare.py   -> prints the report (tee to x1_explain_compare.out), writes x1_explain_compare.json"""
import json, re, sys
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, "src")
from agentic_sentiment.agents.parsing import parse_rating
from agentic_sentiment.v3.text import tokens, contains
from agentic_sentiment.v3.evidence import WEAK
from agentic_sentiment.v3.critic import STAR, JUDGE_SYSTEM, judge_user, parse_judge
from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn
D = "v2_reflective/e23/"
G = [json.loads(l) for l in open(D + "p1/halluc_R_generations.jsonl")]
B = {r["review_id"]: r for r in map(json.loads, open(D + "b2/eval.jsonl"))}
T = {r["review_id"]: r for r in map(json.loads, open(D + "eval.jsonl"))}
def grounded(rat, text):  # some 3-token window of the explanation, with a non-stop word, occurs in the review
    rt, hay = tokens(rat), tokens(text)
    return any(contains(hay, " ".join(rt[i:i + 3])) and set(rt[i:i + 3]) - WEAK for i in range(len(rt) - 2))
norm = lambda s: re.sub(r"[\d\W]+", " ", s.lower()).strip()
model, tok = load_instruct(); judge = make_chat_fn(model, tok)
out = {"phase1": [], "phase2": []}
for g in G:
    rid = g["review_id"]; t = T[rid]; b = B[rid]
    p1 = g["output"].split("!")[0].strip(); p2 = b["drafts"][-1]
    for k, rat, star in (("phase1", p1, parse_rating(g["output"])), ("phase2", p2, b["star"])):
        ok, span = parse_judge(judge(JUDGE_SYSTEM, judge_user(rat, t["title"], t["review_text"])))
        st = {int(x) for x in STAR.findall(rat)} | ({int(m) for m in re.findall(r"rating ([1-5])", rat.lower())} if k == "phase1" else set())
        out[k].append({"review_id": rid, "text": norm(rat), "star_ok": st == {star} if star else False, "grounded": grounded(rat, t["review_text"]),
                       "judge_ok": ok, "words": len(rat.split()), "repeats": k == "phase1" and g["output"].count(p1[:40]) > 1})
res = {}
for k, R in out.items():
    n = len(R); res[k] = {"n": n, "distinct_share": len({r["text"] for r in R}) / n, "star_consistent": np.mean([r["star_ok"] for r in R]),
        "grounded_in_review": np.mean([r["grounded"] for r in R]), "judge_no_unsupported": np.mean([r["judge_ok"] is True for r in R]),
        "judge_unparsed": sum(r["judge_ok"] is None for r in R), "mean_words": np.mean([r["words"] for r in R]), "template_repeats": np.mean([r["repeats"] for r in R])}
    res[k] = {a: float(v) for a, v in res[k].items()}
for m in ("grounded", "judge_ok"):
    x = np.array([r[m] is True for r in out["phase1"]]); y = np.array([r[m] is True for r in out["phase2"]])
    a, b = int((y & ~x).sum()), int((x & ~y).sum()); res[f"mcnemar_{m}"] = {"p2_only": a, "p1_only": b, "p": float(binomtest(a, a + b).pvalue) if a + b else 1.0}
print(json.dumps(res, indent=1)); json.dump(res, open(D + "x1_explain_compare.json", "w"), indent=1)
