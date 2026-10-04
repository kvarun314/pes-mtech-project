"""Real langgraph.graph.StateGraph: Analyst -> {Visual Verifier, RAG Prover}
-> Critic -> (loop to Analyst with critique, or END). Replaces Run 1's plain
-Python-functions pipeline (colab/phase2_agentic_full_comparison.ipynb)."""

import re
from typing import Callable

from langgraph.graph import END, StateGraph

from agentic_sentiment.agents.critic import (
    DISSONANCE_THRESHOLD,
    dissonance_norm,
    normalize_rating,
)
from agentic_sentiment.agents.parsing import parse_rating
from agentic_sentiment.agents.state import AgentState
from agentic_sentiment.eval.one_shot import COT_PHRASE

# Same task framing the Phase 1 LoRA adapter was fine-tuned on
# (agentic_sentiment.phase1.data.preprocessing.SENTIMENT_INSTRUCTION) and
# that Run 1's proven Phase 2 notebook
# (colab/phase2_agentic_full_comparison.ipynb cell 12) includes in every
# agent's prompt. Every prompt below must open with it -- dropping it is
# not one of the mission spec's intentional upgrades (real StateGraph,
# spec-store RAG, Norm dissonance); it just means the adapter never sees
# the rubric it was trained to respond to.
SENTIMENT_INSTRUCTION = (
    "Evaluate the sentiment expressed in user reviews and classify each one "
    "according to its sentiment rating. Use a five-point scale: "
    "1-2 negative, 3 neutral, 4-5 positive."
)

ANALYST_PROMPT = """{instruction}
{cot_block}
Analyst: read this review and rate its sentiment 1-5.
{one_shot_block}{metadata_block}Review: {review_text}
{critique_block}
Sentiment (1-5):"""

VISUAL_PROMPT = """{instruction}

Visual Verifier: the product image shows: {image_caption}
Product: {meta_title}
Based only on the image, rate expected sentiment 1-5.
Sentiment (1-5):"""

RAG_PROMPT = """{instruction}

RAG Prover: similar context: {context}
Review: {review_text}
Rate sentiment 1-5 given this grounding.
Sentiment (1-5):"""

CRITIQUE_PROMPT = """{instruction}

Critique: Analyst said {analyst}, Visual said {visual}, RAG said {rag}.
{grounding_block}Give one sentence of feedback for the Analyst to reconsider."""


def _split_into_claims(review_text: str) -> list[str]:
    """Sentence-split a review into separate claims for grounding_score(),
    rather than embedding the whole multi-sentence review as one claim.
    A simple punctuation split, not full NLP sentence segmentation --
    good enough to shrink each claim toward spec-snippet length, which is
    the thing that actually matters for the embedding comparison."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", review_text) if s.strip()]
    return sentences or [review_text]


def build_graph(llm_fn: Callable[[str], str], spec_store=None, max_correction_iters: int = 2):
    def analyst_node(state: AgentState) -> AgentState:
        metadata_block = ""
        if state.get("use_metadata"):
            title = state.get("meta_title", "")
            description = state.get("meta_description", "")
            if title or description:
                metadata_block = f"Product: {title}. {description}\n"
        critique_block = f"Critic feedback: {state['critique']}" if state.get("critique") else ""
        cot_block = COT_PHRASE if state.get("use_cot", True) else ""
        one_shot_block = f"Example:\n{state['one_shot_example']}\n\n" if state.get("one_shot_example") else ""
        text = llm_fn(ANALYST_PROMPT.format(
            instruction=SENTIMENT_INSTRUCTION, cot_block=cot_block, one_shot_block=one_shot_block,
            metadata_block=metadata_block, review_text=state["review_text"], critique_block=critique_block
        ))
        rating = parse_rating(text) or 3
        return {**state, "analyst_rating": rating}

    def visual_node(state: AgentState) -> AgentState:
        if not state.get("use_image") or not state.get("image_caption"):
            return {**state, "visual_rating": None}
        text = llm_fn(VISUAL_PROMPT.format(
            instruction=SENTIMENT_INSTRUCTION,
            image_caption=state["image_caption"], meta_title=state.get("meta_title", "")
        ))
        return {**state, "visual_rating": parse_rating(text)}

    def rag_node(state: AgentState) -> AgentState:
        if not state.get("use_rag"):
            return {**state, "rag_rating": None, "rag_grounding": None}
        context = state.get("meta_description", "")
        has_spec_store = spec_store is not None and state.get("asin")
        if has_spec_store:
            context += " " + " ".join(r["text"] for r in spec_store.query(state["review_text"], asin=state["asin"]))
        text = llm_fn(RAG_PROMPT.format(
            instruction=SENTIMENT_INSTRUCTION, context=context, review_text=state["review_text"]
        ))
        rag_rating = parse_rating(text)
        # Real factual grounding (catches e.g. "claims USB-C, spec says
        # Micro-USB"), independent of the sentiment vote above. Compared
        # per-sentence, not as one giant claim: a multi-sentence review
        # embedded whole against a short "Connector Type: USB-C"-style spec
        # snippet is systematically far apart in a sentence embedding space
        # on length/style alone, regardless of factual consistency -- that
        # would force grounding near 0 (dissonance near 1) for nearly every
        # review, triggering the self-correction loop's max iterations on
        # almost every row. Splitting into sentences keeps each claim's
        # length closer to the spec snippets it's compared against.
        rag_grounding = (
            spec_store.grounding_score(claims=_split_into_claims(state["review_text"]), asin=state["asin"])
            if has_spec_store else None
        )
        return {**state, "rag_rating": rag_rating, "rag_grounding": rag_grounding}

    def critic_node(state: AgentState) -> AgentState:
        h = normalize_rating(state["analyst_rating"])
        ev = normalize_rating(state["visual_rating"]) if state.get("visual_rating") else h
        # Sentiment-axis term: when a spec store grounded the claim, Gf is a
        # support score (0..1), not a sentiment estimate — it must not be
        # averaged with H/Ev on the same axis (doing so makes a confidently
        # *ungrounded but unanimous* review score D=0, and a well-grounded
        # negative review score high, which is backwards). So the sentiment
        # term here always uses rag_rating (another vote on the same axis
        # as H/Ev); grounding is folded in afterwards as an independent
        # penalty via max(), so it can only raise D, never mask a real vote
        # disagreement by cancelling it out.
        rag_sentiment = normalize_rating(state["rag_rating"]) if state.get("rag_rating") else h

        if state.get("dissonance_method") == "stdev":
            from agentic_sentiment.agents.critic import dissonance_stdev
            votes = [v for v in (state["analyst_rating"], state.get("visual_rating"), state.get("rag_rating")) if v]
            d = dissonance_stdev(votes)
        else:
            d = dissonance_norm(h, ev, rag_sentiment)
            if state.get("rag_grounding") is not None:
                d = max(d, 1.0 - state["rag_grounding"])

        iters = state.get("correction_iters", 0)
        max_iters = state.get("max_correction_iters", max_correction_iters)
        if d >= DISSONANCE_THRESHOLD and iters < max_iters:
            grounding_block = ""
            if state.get("rag_grounding") is not None and state["rag_grounding"] < 0.5:
                # Mission spec §13.5 example: "User is complaining about a
                # feature this product never claimed to have." -- make the
                # grounding mismatch explicit, not just a bare disagreement.
                grounding_block = (
                    "The review's claim does not match the product's spec "
                    f"(grounding score {state['rag_grounding']:.2f} of 1.0) -- "
                    "this may be a user-expectation mismatch, not a genuine defect.\n"
                )
            critique = llm_fn(CRITIQUE_PROMPT.format(
                instruction=SENTIMENT_INSTRUCTION,
                analyst=state["analyst_rating"], visual=state.get("visual_rating"), rag=state.get("rag_rating"),
                grounding_block=grounding_block,
            ))
            return {
                **state, "dissonance": d, "critique": critique,
                "correction_iters": iters + 1, "self_corrected": True,
            }

        votes = [v for v in (state["analyst_rating"], state.get("visual_rating"), state.get("rag_rating")) if v]
        final = max(set(votes), key=votes.count) if votes else state["analyst_rating"]
        return {
            **state, "dissonance": d, "final_rating": final,
            "self_corrected": state.get("self_corrected", False),
            "rationale": f"Analyst={state['analyst_rating']}, Visual={state.get('visual_rating')}, "
                         f"RAG={state.get('rag_rating')}, Dissonance={d:.3f}",
        }

    def route_after_critic(state: AgentState) -> str:
        return "analyst" if state.get("final_rating") is None else END

    graph = StateGraph(AgentState)
    graph.add_node("analyst", analyst_node)
    graph.add_node("visual", visual_node)
    graph.add_node("rag", rag_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "visual")
    graph.add_edge("visual", "rag")
    graph.add_edge("rag", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"analyst": "analyst", END: END})

    return graph.compile()
