import pytest

from agentic_sentiment.agents.graph import build_graph


def _make_stub_llm(ratings_by_prompt_substring):
    """Returns an llm_fn that looks for a substring in the prompt and
    returns a canned rating string for it."""
    def llm_fn(prompt: str) -> str:
        for substr, rating in ratings_by_prompt_substring.items():
            if substr in prompt:
                return f"Sentiment (1-5): {rating}. stub."
        return "Sentiment (1-5): 3. default."
    return llm_fn


def test_graph_agrees_no_correction_needed():
    llm = _make_stub_llm({"Analyst": 5, "Visual": 5, "RAG": 5, "Critique": 5})
    graph = build_graph(llm_fn=llm)

    result = graph.invoke({
        "review_id": "r1", "review_text": "Loved it", "gt_rating": 5,
        "meta_title": "Widget", "image_caption": "a clean widget",
        "use_metadata": True, "use_image": True, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert result["final_rating"] == 5
    assert result["self_corrected"] is False
    assert result["dissonance"] < 0.4


def test_graph_triggers_self_correction_on_high_dissonance():
    calls = {"analyst_count": 0}
    critique_seen_in_analyst_prompt = []

    def llm_fn(prompt: str) -> str:
        if "Visual" in prompt:
            return "Sentiment (1-5): 1. stub."
        if "RAG" in prompt:
            return "Sentiment (1-5): 1. stub."
        if "Analyst" in prompt:
            calls["analyst_count"] += 1
            if "Critic feedback" in prompt:
                critique_seen_in_analyst_prompt.append(True)
            # First Analyst call disagrees wildly; after critique, it should
            # be called again (that's the self-correction loop).
            return "Sentiment (1-5): 5. stub."
        return "Sentiment (1-5): 3. stub."

    graph = build_graph(llm_fn=llm_fn, max_correction_iters=2)
    result = graph.invoke({
        "review_id": "r2", "review_text": "mixed feelings", "gt_rating": 1,
        "meta_title": "Widget", "image_caption": "a broken widget",
        "use_metadata": True, "use_image": True, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert calls["analyst_count"] >= 2  # Analyst was re-run after critique
    assert result["self_corrected"] is True
    assert result["correction_iters"] <= 2
    # Proves the loop's DATA flows: the second Analyst call actually
    # received the critique text in its prompt, not just that the node re-ran.
    assert len(critique_seen_in_analyst_prompt) >= 1


class _FakeSpecStore:
    """Stubs grounding_score() to simulate a review claim that does NOT
    match the product's spec (e.g. "USB-C" claimed vs. "Micro-USB" spec'd)."""

    def query(self, claim, asin, k=3):
        return []

    def grounding_score(self, claims, asin, threshold=0.5):
        return 0.1


def test_graph_uses_real_grounding_score_for_dissonance_when_spec_store_present():
    # Everyone (Analyst, Visual, RAG-sentiment-vote) agrees on a 5 -> if Gf
    # came from normalize_rating(rag_rating) as before, dissonance would be
    # norm(1.0 - (1.0 + 1.0)/2) = 0.0, well under DISSONANCE_THRESHOLD=0.4.
    llm = _make_stub_llm({"Analyst": 5, "Visual": 5, "RAG": 5, "Critique": 5})
    graph = build_graph(llm_fn=llm, spec_store=_FakeSpecStore(), max_correction_iters=0)

    result = graph.invoke({
        "review_id": "r4", "review_text": "Great USB-C port", "gt_rating": 5,
        "meta_title": "Widget", "image_caption": "a clean widget", "asin": "B001",
        "use_metadata": True, "use_image": True, "use_rag": True,
        "correction_iters": 0, "max_correction_iters": 0,
    })

    # With the spec store wired in, Gf must come from grounding_score()=0.1
    # instead: norm(1.0 - (1.0 + 0.1)/2) = 0.45 >= DISSONANCE_THRESHOLD.
    assert result["rag_grounding"] == 0.1
    assert result["dissonance"] == pytest.approx(0.45)
    assert result["dissonance"] >= 0.4


def test_graph_skips_visual_when_use_image_false():
    seen_visual = {"called": False}

    def llm_fn(prompt: str) -> str:
        if "Visual" in prompt:
            seen_visual["called"] = True
        return "Sentiment (1-5): 4. stub."

    graph = build_graph(llm_fn=llm_fn)
    graph.invoke({
        "review_id": "r3", "review_text": "fine", "gt_rating": 4,
        "meta_title": "Widget", "image_caption": None,
        "use_metadata": True, "use_image": False, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert seen_visual["called"] is False
