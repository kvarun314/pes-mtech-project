"""Phase 2 v3 LangGraph: evidence -> router -> [probe] -> reflector (+ hand-off decision) -> pack -> explainer -> critic -(rewrite, max 4; I1, LOG.md 2026-10-28)-> END.
The specialists decide the star; LLM agents only explain it and check the explanation."""
import time
from typing import TypedDict
import numpy as np
from langgraph.graph import END, StateGraph
from agentic_sentiment.v3.critic import critique
from agentic_sentiment.v3.explainer import EXPLAIN_SYSTEM, explainer_user
from agentic_sentiment.v3.pack import build_pack


class V3State(TypedDict, total=False):
    review_id: str; title: str; review_text: str
    evidence_probs: list; route: str; probe_probs: list | None
    star: int; probs: list; shares: dict; agreed: bool | None; handoff: str; pack: dict
    drafts: list; feedback: list; rounds: int; faithful_first: bool; faithful: bool
    failed_checks: list; judge_unparsed: int; llm_calls: int; timings: dict


def build_graph_v3(evidence, reflector, probe_fn, explain_fn=None, judge_fn=None, tau=None, max_rewrites=4, handoff_tau=None):
    def timed(state, key, t0):
        return {**state.get("timings", {}), key: state.get("timings", {}).get(key, 0.0) + time.time() - t0}

    def evidence_node(s):
        t0 = time.time(); p = evidence.probs(s["title"], s["review_text"])
        return {"evidence_probs": list(map(float, p)), "timings": timed(s, "evidence_s", t0)}

    def router_node(s):
        skip = s.get("probe_probs") is None and tau is not None and max(s["evidence_probs"]) >= tau
        return {"route": "skip" if skip else "probe"}

    def probe_node(s):
        if s.get("probe_probs") is not None:
            return {}
        t0 = time.time(); return {"probe_probs": list(map(float, probe_fn(s["review_text"]))), "timings": timed(s, "probe_s", t0)}

    def reflector_node(s):
        pp = s.get("probe_probs") if s["route"] == "probe" else None
        star, P, shares = reflector.combine(pp, np.array(s["evidence_probs"]))
        agreed = None if pp is None else int(np.argmax(pp)) == int(np.argmax(s["evidence_probs"]))
        handoff = "auto" if handoff_tau is None or max(P) >= handoff_tau else "human"   # hand-off agent: unsure reviews go to a person
        return {"star": star, "probs": list(map(float, P)), "shares": shares, "agreed": agreed, "handoff": handoff}

    def pack_node(s):
        pos, neg = evidence.phrases(s["title"], s["review_text"], s["star"])
        return {"pack": build_pack(s["star"], s["probs"], s["shares"], s["agreed"], pos, neg), "drafts": [], "feedback": [], "rounds": 0,
                "failed_checks": [], "judge_unparsed": 0, "llm_calls": 0}

    def explainer_node(s):
        t0 = time.time(); prev = s["drafts"][-1] if s["drafts"] else None
        text = explain_fn(EXPLAIN_SYSTEM, explainer_user(s["pack"], s["title"], s["review_text"], prev, s["feedback"][-1] if prev else ()))
        return {"drafts": s["drafts"] + [text], "rounds": s["rounds"] + (1 if prev else 0), "llm_calls": s["llm_calls"] + 1,
                "timings": timed(s, "explain_s", t0)}

    def critic_node(s):
        t0 = time.time(); fb, info = critique(s["drafts"][-1], s["pack"], s["title"], s["review_text"], judge_fn)
        upd = {"feedback": s["feedback"] + [fb], "failed_checks": s["failed_checks"] + [info["failed_checks"]], "faithful": not fb,
               "judge_unparsed": s["judge_unparsed"] + int(info["judge_unparsed"]),
               "llm_calls": s["llm_calls"] + int(not info["failed_checks"] or info["failed_checks"] == [5] or info["judge_unparsed"]),
               "timings": timed(s, "critic_s", t0)}
        if len(s["drafts"]) == 1:
            upd["faithful_first"] = not fb
        return upd

    g = StateGraph(V3State)
    for n, f in [("evidence", evidence_node), ("router", router_node), ("probe", probe_node), ("reflector", reflector_node)]:
        g.add_node(n, f)
    g.set_entry_point("evidence"); g.add_edge("evidence", "router")
    g.add_conditional_edges("router", lambda s: s["route"], {"probe": "probe", "skip": "reflector"}); g.add_edge("probe", "reflector")
    if explain_fn is None:
        g.add_edge("reflector", END); return g.compile()
    for n, f in [("pack", pack_node), ("explainer", explainer_node), ("critic", critic_node)]:
        g.add_node(n, f)
    g.add_edge("reflector", "pack"); g.add_edge("pack", "explainer"); g.add_edge("explainer", "critic")
    g.add_conditional_edges("critic", lambda s: "explainer" if not s["faithful"] and s["rounds"] < max_rewrites else END,
                            {"explainer": "explainer", END: END})
    return g.compile()
