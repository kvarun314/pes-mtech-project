from agentic_sentiment.v3.explainer import explainer_user, EXPLAIN_SYSTEM

PACK = {"star": 2, "confidence": .55, "shares": {"probe": .3, "evidence": .7}, "agreed": False,
        "for": [{"phrase": "stopped working", "display": "stopped working", "weight": .5}], "against": [{"phrase": "nice", "display": "nice", "weight": -.2}]}


def test_prompt_contains_pack_and_instructions():
    u = explainer_user(PACK, "Meh", "Looks nice but stopped working.")
    assert '"stopped working"' in u and '"nice"' in u and "2 stars" in u and "70%" in u
    assert "The two agents disagreed on this review." in u and "never call them support" in u
    assert "Previous explanation" not in u


def test_rewrite_prompt_includes_feedback():
    u = explainer_user(PACK, "", "x", previous="Rated 3 stars.", feedback=["You said 3 stars; the final rating is 2 stars."])
    assert "Previous explanation: Rated 3 stars." in u and "- You said 3 stars" in u


def test_system_prompt_forbids_outside_claims():
    assert "only the evidence" in EXPLAIN_SYSTEM
