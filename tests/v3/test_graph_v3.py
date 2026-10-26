import numpy as np
from agentic_sentiment.v3.graph import build_graph_v3

POS = [{"phrase": "great screen", "display": "Great screen", "weight": .4}, {"phrase": "screen", "display": "screen", "weight": .2}]


class Ev:
    def __init__(self, p): self.p = np.array(p)
    def probs(self, title, text): return self.p
    def phrases(self, title, text, star): return POS, []


class Rf:
    def combine(self, probe_p, evid_p):
        p = evid_p if probe_p is None else (np.asarray(probe_p) + evid_p) / 2
        return int(np.argmax(p)) + 1, p, {"probe": 0.0 if probe_p is None else .5, "evidence": 1.0 if probe_p is None else .5}


def run(graph, **kw): return graph.invoke({"review_id": "r", "title": "", "review_text": "Great screen, lovely.", **kw})


def test_router_skips_probe_when_evidence_confident():
    called = []
    g = build_graph_v3(Ev([0, 0, 0, .1, .9]), Rf(), lambda t: called.append(t) or [0, 0, 0, .5, .5], tau=0.8)
    out = run(g)
    assert out["route"] == "skip" and called == [] and out["star"] == 5 and "pack" not in out


def test_stored_probe_probs_are_used():
    g = build_graph_v3(Ev([0, 0, .5, .5, 0]), Rf(), lambda t: 1 / 0)
    out = run(g, probe_probs=[0, 0, 0, 1, 0])
    assert out["route"] == "probe" and out["star"] == 4 and out["agreed"] is False


def test_rewrite_loop_until_faithful():
    drafts = iter(["Rated 3 stars.", 'Rated 5 stars: "Great screen" and "screen".'])
    g = build_graph_v3(Ev([0, 0, 0, .1, .9]), Rf(), lambda t: [0, 0, 0, .1, .9], explain_fn=lambda s, u: next(drafts), judge_fn=lambda s, u: "NO")
    out = run(g)
    assert out["faithful_first"] is False and out["faithful"] is True and out["rounds"] == 1
    assert out["feedback"][0][0] == "You said 3 stars; the final rating is 5 stars."
    assert out["llm_calls"] == 3                       # 2 explainer + 1 judge (judge skipped on the failing first draft)


def test_loop_stops_after_max_rewrites():
    g = build_graph_v3(Ev([0, 0, 0, .1, .9]), Rf(), lambda t: [0, 0, 0, .1, .9], explain_fn=lambda s, u: "Rated 1 stars.", judge_fn=lambda s, u: "NO", max_rewrites=2)
    out = run(g)
    assert out["rounds"] == 2 and out["faithful"] is False and len(out["drafts"]) == 3


def test_handoff_agent_routes_unsure_reviews_to_human():
    g = build_graph_v3(Ev([0, 0, .4, .6, 0]), Rf(), lambda t: [0, 0, .4, .6, 0], handoff_tau=0.7)
    assert run(g)["handoff"] == "human"
    g = build_graph_v3(Ev([0, 0, 0, .1, .9]), Rf(), lambda t: [0, 0, 0, .1, .9], handoff_tau=0.7)
    assert run(g)["handoff"] == "auto"
