"""Shared state threaded through every LangGraph node."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    review_id: str
    review_text: str
    gt_rating: int | None
    meta_title: str
    meta_description: str
    image_caption: str | None
    asin: str

    use_metadata: bool
    use_image: bool
    use_rag: bool
    dissonance_method: str  # "norm" | "stdev"
    max_correction_iters: int

    analyst_rating: int | None
    visual_rating: int | None
    rag_rating: int | None
    critique: str | None
    dissonance: float
    correction_iters: int
    final_rating: int | None
    self_corrected: bool
    rationale: str
