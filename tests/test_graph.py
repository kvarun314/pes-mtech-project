import pytest

from agentic_sentiment.agents.graph import SENTIMENT_INSTRUCTION, build_graph


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
    # Everyone (Analyst, Visual, RAG-sentiment-vote) agrees on a 5, so the
    # sentiment-agreement term is 0. The ungrounded claim (grounding_score=0.1)
    # must still force dissonance up via the independent penalty max(d, 1-Gf),
    # not get averaged away: max(0.0, 1 - 0.1) = 0.9.
    llm = _make_stub_llm({"Analyst": 5, "Visual": 5, "RAG": 5, "Critique": 5})
    graph = build_graph(llm_fn=llm, spec_store=_FakeSpecStore(), max_correction_iters=0)

    result = graph.invoke({
        "review_id": "r4", "review_text": "Great USB-C port", "gt_rating": 5,
        "meta_title": "Widget", "image_caption": "a clean widget", "asin": "B001",
        "use_metadata": True, "use_image": True, "use_rag": True,
        "correction_iters": 0, "max_correction_iters": 0,
    })

    assert result["rag_grounding"] == 0.1
    assert result["dissonance"] == pytest.approx(0.9)
    assert result["dissonance"] >= 0.4


def test_graph_grounding_penalty_fires_even_on_unanimous_negative_rating():
    # Regression: averaging Gf with H/Ev (the pre-fix formula) made a
    # unanimous NEGATIVE rating with an UNGROUNDED claim score dissonance
    # 0.0 (backwards -- an ungrounded claim should raise suspicion
    # regardless of which direction the rating leans). With the fix, the
    # grounding penalty is independent of sentiment direction.
    llm = _make_stub_llm({"Analyst": 1, "Visual": 1, "RAG": 1, "Critique": 1})
    graph = build_graph(llm_fn=llm, spec_store=_FakeSpecStore(), max_correction_iters=0)

    result = graph.invoke({
        "review_id": "r7", "review_text": "Broken USB-C port", "gt_rating": 1,
        "meta_title": "Widget", "image_caption": "a clean widget", "asin": "B001",
        "use_metadata": True, "use_image": True, "use_rag": True,
        "correction_iters": 0, "max_correction_iters": 0,
    })

    assert result["dissonance"] == pytest.approx(0.9)
    assert result["dissonance"] >= 0.4


def test_graph_grounding_does_not_mask_real_vote_disagreement():
    # A well-grounded claim (Gf=1.0) must not suppress genuine disagreement
    # between the agents' votes -- the max() combinator should keep the
    # higher of the two signals, not let a good grounding score cancel it.
    class _FullyGroundedStore:
        def query(self, claim, asin, k=3):
            return []

        def grounding_score(self, claims, asin, threshold=0.5):
            return 1.0

    llm = _make_stub_llm({"Analyst": 5, "Visual": 1, "RAG": 1, "Critique": 3})
    graph = build_graph(llm_fn=llm, spec_store=_FullyGroundedStore(), max_correction_iters=0)

    result = graph.invoke({
        "review_id": "r8", "review_text": "Mixed signals", "gt_rating": 3,
        "meta_title": "Widget", "image_caption": "a clean widget", "asin": "B001",
        "use_metadata": True, "use_image": True, "use_rag": True,
        "correction_iters": 0, "max_correction_iters": 0,
    })

    # dissonance_norm(h=1.0, ev=0.0, rag_sentiment=0.0) = |1.0 - 0.0| = 1.0
    assert result["dissonance"] == pytest.approx(1.0)


def test_critique_mentions_grounding_mismatch_when_ungrounded():
    seen_critique_prompt = {"text": None}

    def llm_fn(prompt: str) -> str:
        if "Critique" in prompt:
            seen_critique_prompt["text"] = prompt
            return "Reconsider: this may be a user-expectation mismatch, not a defect."
        return "Sentiment (1-5): 5. stub."

    graph = build_graph(llm_fn=llm_fn, spec_store=_FakeSpecStore(), max_correction_iters=1)
    graph.invoke({
        "review_id": "r9", "review_text": "Broken USB-C port", "gt_rating": 1,
        "meta_title": "Widget", "image_caption": "a clean widget", "asin": "B001",
        "use_metadata": True, "use_image": True, "use_rag": True,
        "correction_iters": 0, "max_correction_iters": 1,
    })

    assert seen_critique_prompt["text"] is not None
    assert "does not match the product's spec" in seen_critique_prompt["text"]
    assert "0.10" in seen_critique_prompt["text"]


def test_analyst_prompt_includes_metadata_iff_use_metadata():
    marker = "UNIQUE-MARKER-WIDGET-9000"
    seen_in_analyst_prompt = {"present": False}

    def llm_fn(prompt: str) -> str:
        if "Analyst" in prompt and marker in prompt:
            seen_in_analyst_prompt["present"] = True
        return "Sentiment (1-5): 4. stub."

    graph = build_graph(llm_fn=llm_fn)

    # use_metadata True + real metadata -> marker must appear in Analyst prompt.
    graph.invoke({
        "review_id": "r5", "review_text": "fine", "gt_rating": 4,
        "meta_title": marker, "meta_description": "a widget", "image_caption": None,
        "use_metadata": True, "use_image": False, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })
    assert seen_in_analyst_prompt["present"] is True

    # use_metadata False (same metadata present in state) -> marker must NOT appear.
    seen_in_analyst_prompt["present"] = False
    graph.invoke({
        "review_id": "r6", "review_text": "fine", "gt_rating": 4,
        "meta_title": marker, "meta_description": "a widget", "image_caption": None,
        "use_metadata": False, "use_image": False, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })
    assert seen_in_analyst_prompt["present"] is False


def test_every_agent_prompt_includes_the_sentiment_instruction():
    # Regression: the fine-tuned Phase 1 adapter (and Run 1's proven Phase 2
    # notebook, colab/phase2_agentic_full_comparison.ipynb) always opens
    # every agent's prompt with this task framing. Dropping it isn't one
    # of the mission spec's intentional upgrades.
    seen = {"analyst": False, "visual": False, "rag": False, "critique": False}

    def llm_fn(prompt: str) -> str:
        has_instruction = SENTIMENT_INSTRUCTION in prompt
        # CRITIQUE_PROMPT contains the literal word "Analyst" (e.g. "Analyst
        # said 1") -- check it first so that branch isn't shadowed.
        if "Critique" in prompt:
            seen["critique"] = seen["critique"] or has_instruction
            return "feedback stub."
        if "Analyst" in prompt:
            seen["analyst"] = seen["analyst"] or has_instruction
            return "Sentiment (1-5): 1. stub."
        if "Visual" in prompt:
            seen["visual"] = seen["visual"] or has_instruction
            return "Sentiment (1-5): 5. stub."
        if "RAG" in prompt:
            seen["rag"] = seen["rag"] or has_instruction
            return "Sentiment (1-5): 5. stub."
        return "Sentiment (1-5): 3. stub."

    graph = build_graph(llm_fn=llm_fn, max_correction_iters=1)
    graph.invoke({
        "review_id": "r10", "review_text": "mixed", "gt_rating": 3,
        "meta_title": "Widget", "image_caption": "a widget",
        "use_metadata": True, "use_image": True, "use_rag": True,
        "correction_iters": 0, "max_correction_iters": 1,
    })

    assert all(seen.values()), seen


def test_analyst_prompt_includes_one_shot_and_cot_when_provided():
    seen = {"one_shot": False, "cot": False}

    def llm_fn(prompt: str) -> str:
        if "Analyst" in prompt:
            seen["one_shot"] = "Example:\nReview: a prior review" in prompt
            seen["cot"] = "Let's take it one step at a time." in prompt
        return "Sentiment (1-5): 4. stub."

    graph = build_graph(llm_fn=llm_fn)
    graph.invoke({
        "review_id": "r11", "review_text": "fine", "gt_rating": 4,
        "meta_title": "Widget", "image_caption": None,
        "use_metadata": False, "use_image": False, "use_rag": False,
        "use_cot": True, "one_shot_example": "Review: a prior review\nSentiment (1-5): 2. meh.",
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert seen["one_shot"] is True
    assert seen["cot"] is True


def test_analyst_prompt_omits_one_shot_and_cot_when_disabled():
    seen = {"one_shot": True, "cot": True}  # start True so a missed check would fail loudly

    def llm_fn(prompt: str) -> str:
        if "Analyst" in prompt:
            seen["one_shot"] = "Example:" in prompt
            seen["cot"] = "Let's take it one step at a time." in prompt
        return "Sentiment (1-5): 4. stub."

    graph = build_graph(llm_fn=llm_fn)
    graph.invoke({
        "review_id": "r12", "review_text": "fine", "gt_rating": 4,
        "meta_title": "Widget", "image_caption": None,
        "use_metadata": False, "use_image": False, "use_rag": False,
        "use_cot": False, "one_shot_example": None,
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert seen["one_shot"] is False
    assert seen["cot"] is False


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
