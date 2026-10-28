"""Q3 (declared LOG.md 2026-10-28): second judge (Qwen2.5-7B-Instruct 4-bit) on explanations the Llama judge rejected / accepted.
usage: q3_second_judge.py gate | q3_second_judge.py test   (tee to q3_second_judge_<part>.out; updates q3_second_judge.json)"""
import json, os, random, sys
sys.path.insert(0, "src")
from mlx_lm import load
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.critic import JUDGE_SYSTEM, judge_user, parse_judge
from agentic_sentiment.v3.explainer import make_chat_fn
part = sys.argv[1]; J = PT.D + "q3_second_judge.json"; res = json.load(open(J)) if os.path.exists(J) else {}
L = lambda f: [json.loads(l) for l in open(PT.B2 + f)]
T = lambda s: {x["review_id"]: x for x in map(json.loads, open(f"{PT.D}{s}.jsonl"))}


def sets(split):
    B, I = L(f"{split}.jsonl"), {x["review_id"]: x for x in L(f"i1_{split}.jsonl")}
    rej = [(b["review_id"], I[b["review_id"]]["drafts"][-1]) for b in B if not b["faithful"] and not I[b["review_id"]]["faithful"] and I[b["review_id"]]["failed_checks"][-1] == [5]]
    acc = random.Random(0).sample([(b["review_id"], b["drafts"][-1]) for b in B if b["faithful"]], 60)
    return rej, acc


if part == "test":
    assert res.get("gate_dev2", {}).get("pass"), "gate not passed: no test read"
model, tok = load(PT.JUDGE2); chat = make_chat_fn(model, tok)
def judge(split, items):
    t = T(split); return [parse_judge(chat(JUDGE_SYSTEM, judge_user(x, t[i]["title"], t[i]["review_text"])))[0] for i, x in items]


if part == "gate":
    rej, acc = sets("dev2")
    A = [(x["review_id"], x["fallback"]) for x in L("i3_dev2.jsonl") if not x["faithful"]]
    ra, rc = judge("dev2", A + rej), judge("dev2", acc)
    g = {"n_rejected": len(ra), "qwen_accepts_rejected": sum(v is True for v in ra) / len(ra), "n_accepted": len(rc),
         "qwen_rejects_accepted": sum(v is False for v in rc) / len(rc), "unparsed": sum(v is None for v in ra + rc),
         "qwen_accepts_A_fallbacks": sum(v is True for v in ra[:len(A)]), "qwen_accepts_B_explanations": sum(v is True for v in ra[len(A):])}
    g["pass"] = bool(g["qwen_accepts_rejected"] >= 0.5 and g["qwen_rejects_accepted"] <= 0.10)
    res["gate_dev2"] = g; print("GATE dev2", json.dumps(g, indent=1))
elif part == "test":
    for s in ("eval", "natural_test"):
        rej, acc = sets(s); n = len(L(f"{s}.jsonl")); ra, rc = judge(s, rej), judge(s, acc)
        f4 = sum(x["faithful"] for x in L(f"{s}.jsonl")) + sum(x["faithful"] for x in L(f"i1_{s}.jsonl"))
        r = {"judge_only_failures": len(rej), "qwen_accepts": sum(v is True for v in ra), "control_rejected": sum(v is False for v in rc) / len(rc),
             "faithful_single_judge": f4 / n, "faithful_two_judge": (f4 + sum(v is True for v in ra)) / n}
        res[s] = r; print(s, json.dumps(r, indent=1))
json.dump(res, open(J, "w"), indent=1)
