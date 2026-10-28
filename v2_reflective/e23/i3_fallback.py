"""I3 (declared LOG.md 2026-10-28): deterministic fallback explanation when the Critic loop still fails after max_rewrites=4.
The fallback is built from the evidence pack only (final star + top-2 supporting phrases as they appear in the review) and goes
through the same Critic (rules 1-4 + LLM judge). Rows that use it are flagged, and the share of explanations from the fallback is reported.
usage: i3_fallback.py SPLIT   (needs b2/SPLIT.jsonl and b2/i1_SPLIT.jsonl; writes b2/i3_SPLIT.jsonl; tee to i3_fallback_SPLIT.out)"""
import json, sys
from scipy.stats import binomtest
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.critic import critique
from agentic_sentiment.v3.explainer import load_instruct, make_chat_fn


def fallback(pack):
    s, top = pack["star"], [p["display"] for p in pack["for"][:2]]
    pre = "The two agents disagreed on this review. " if pack["agreed"] is False else ""
    because = " because it says " + " and ".join(f'"{d}"' for d in top) if top else ""
    return f"{pre}The system rated this review {s} star{"s" if s > 1 else ""}{because}."


if __name__ == "__main__":
    split = sys.argv[1]
    B = [json.loads(l) for l in open(f"{PT.B2}{split}.jsonl")]
    I = {x["review_id"]: x for x in map(json.loads, open(f"{PT.B2}i1_{split}.jsonl"))}
    todo = [b for b in B if not b["faithful"] and not I[b["review_id"]]["faithful"]]
    T = {x["review_id"]: x for x in map(json.loads, open(f"{PT.D}{split}.jsonl"))}
    model, tok = load_instruct(); chat = make_chat_fn(model, tok)
    rows = []
    with open(f"{PT.B2}i3_{split}.jsonl", "w") as f:
        for b in todo:
            t = T[b["review_id"]]; text = fallback(b["pack"])
            fb, info = critique(text, b["pack"], t["title"], t["review_text"], chat)
            rows.append({"review_id": b["review_id"], "fallback": text, "faithful": not fb, "failed_checks": info["failed_checks"],
                         "judge_unparsed": info["judge_unparsed"]})
            f.write(json.dumps(rows[-1]) + "\n")
    n, ok = len(B), sum(r["faithful"] for r in rows)
    f4 = sum(b["faithful"] or I.get(b["review_id"], {}).get("faithful", False) for b in B)
    print(f"{split}: still unfaithful after 4 rounds {len(todo)}; fallback passes the Critic {ok} ({ok / max(len(todo), 1):.1%}); judge unparsed {sum(r['judge_unparsed'] for r in rows)}")
    print(f"faithful: max-4 {f4 / n:.3f}  max-4 + fallback {(f4 + ok) / n:.3f}  | McNemar fixed {ok} broken 0 p {binomtest(ok, ok).pvalue if ok else 1.0:.2g}")
    print(f"explanations from the fallback: {ok}/{n} ({ok / n:.1%}); still failing, by check:",
          {c: sum(c in r["failed_checks"] for r in rows if not r["faithful"]) for c in range(1, 6)})
    if split == "dev2": print("GATE (fallback passes >= 30 of the 43):", "PASS" if ok >= 30 else "FAIL")
