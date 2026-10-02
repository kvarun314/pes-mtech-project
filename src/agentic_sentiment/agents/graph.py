"""Real langgraph.graph.StateGraph: Analyst -> {Visual Verifier, RAG Prover}
-> Critic -> (loop to Analyst with critique, or END). Replaces Run 1's plain
-Python-functions pipeline (colab/phase2_agentic_full_comparison.ipynb)."""

from typing import Callable

from langgraph.graph import END, StateGraph

from agentic_sentiment.agents.critic import (
    DISSONANCE_THRESHOLD,
    dissonance_norm,
    normalize_rating,
)
from agentic_sentiment.agents.parsing import parse_rating
from agentic_sentiment.agents.state import AgentState

ANALYST_PROMPT = """Analyst: read this review and rate its sentiment 1-5.
Review: {review_text}
{critique_block}
Sentiment (1-5):"""

VISUAL_PROMPT = """Visual Verifier: the product image shows: {image_caption}
Product: {meta_title}
Based only on the image, rate expected sentiment 1-5.
Sentiment (1-5):"""

RAG_PROMPT = """RAG Prover: similar context: {context}
Review: {review_text}
Rate sentiment 1-5 given this grounding.
Sentiment (1-5):"""

CRITIQUE_PROMPT = """Critique: Analyst said {analyst}, Visual said {visual}, RAG said {rag}.
They disagree. Give one sentence of feedback for the Analyst to reconsider."""


def build_graph(llm_fn: Callable[[str], str], spec_store=None, max_correction_iters: int = 2):
    def analyst_node(state: AgentState) -> AgentState:
        critique_block = f"Critic feedback: {state['critique']}" if state.get("critique") else ""
        text = llm_fn(ANALYST_PROMPT.format(review_text=state["review_text"], critique_block=critique_block))
        rating = parse_rating(text) or 3
        return {**state, "analyst_rating": rating}

    def visual_node(state: AgentState) -> AgentState:
        if not state.get("use_image") or not state.get("image_caption"):
            return {**state, "visual_rating": None}
        text = llm_fn(VISUAL_PROMPT.format(
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
        text = llm_fn(RAG_PROMPT.format(context=context, review_text=state["review_text"]))
        rag_rating = parse_rating(text)
        # Real factual grounding (catches e.g. "claims USB-C, spec says
        # Micro-USB"), independent of the sentiment vote above.
        rag_grounding = (
            spec_store.grounding_score(claims=[state["review_text"]], asin=state["asin"])
            if has_spec_store else None
        )
        return {**state, "rag_rating": rag_rating, "rag_grounding": rag_grounding}

    def critic_node(state: AgentState) -> AgentState:
        h = normalize_rating(state["analyst_rating"])
        ev = normalize_rating(state["visual_rating"]) if state.get("visual_rating") else h
        if state.get("rag_grounding") is not None:
            gf = state["rag_grounding"]
        elif state.get("rag_rating"):
            gf = normalize_rating(state["rag_rating"])
        else:
            gf = h

        if state.get("dissonance_method") == "stdev":
            from agentic_sentiment.agents.critic import dissonance_stdev
            votes = [v for v in (state["analyst_rating"], state.get("visual_rating"), state.get("rag_rating")) if v]
            d = dissonance_stdev(votes)
        else:
            d = dissonance_norm(h, ev, gf)

        iters = state.get("correction_iters", 0)
        max_iters = state.get("max_correction_iters", max_correction_iters)
        if d >= DISSONANCE_THRESHOLD and iters < max_iters:
            critique = llm_fn(CRITIQUE_PROMPT.format(
                analyst=state["analyst_rating"], visual=state.get("visual_rating"), rag=state.get("rag_rating"),
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
