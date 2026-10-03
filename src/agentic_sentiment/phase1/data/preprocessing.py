"""
Prompt formatting and preprocessing (paper Section 2.2).
Five-point scale, one-shot, chain-of-thought.
"""

import random
from typing import Optional

from agentic_sentiment.phase1.config import DataConfig

SENTIMENT_INSTRUCTION = (
    "Evaluate the sentiment expressed in user reviews and classify each one "
    "according to its sentiment rating. Use a five-point scale: "
    "1-2 negative, 3 neutral, 4-5 positive."
)

RATING_DESCRIPTIONS = {
    1: "Comments show a high level of dissatisfaction and negativity (rating 1).",
    2: "Although still negative, the user's sentiment may be slightly softened (rating 2).",
    3: "The review is generally neutral, with both positive and negative aspects (rating 3).",
    4: "Users have a positive experience overall, with room for improvement (rating 4).",
    5: "High level of satisfaction and positive emotions (rating 5).",
}

COT_PHRASE = "Let's take it one step at a time."


def format_one_shot_example(review: str, rating: int, cfg: DataConfig) -> str:
    """Format a single one-shot example (review -> rating)."""
    desc = RATING_DESCRIPTIONS.get(rating, f"Rating {rating}.")
    return f"Review: {review}\nSentiment (1-5): {rating}. {desc}"


def build_prompt(
    review: str, cfg: DataConfig, one_shot_example: Optional[str] = None
) -> str:
    """Build input prompt (Section 2.2). Optionally add one-shot and CoT."""
    parts = [SENTIMENT_INSTRUCTION]
    if cfg.use_cot:
        parts.append(cfg.cot_phrase)
    if one_shot_example and cfg.use_one_shot:
        parts.append("\n\nExample:\n" + one_shot_example)
    parts.append("\n\nReview to classify:\n" + review)
    parts.append(
        "\nAfter your reasoning, end with exactly one line starting with \"Sentiment (1-5):\" "
        "followed by the rating 1-5 and a short justification (paper-style output)."
    )
    parts.append("\nSentiment (1-5):")
    return "\n".join(parts)


def rating_to_sentiment_class(rating: int, cfg: DataConfig) -> str:
    """Map 1-5 rating to negative/neutral/positive."""
    if rating in cfg.negative_labels:
        return "negative"
    if rating in cfg.neutral_labels:
        return "neutral"
    if rating in cfg.positive_labels:
        return "positive"
    return "neutral"


def prepare_conversation_format(
    text: str, rating: int, cfg: DataConfig, one_shot: Optional[str] = None
) -> dict:
    """Single example in chat format for SFT (prompt + answer).

    The SFT target is the full "{rating}. {description}" line, not a bare
    digit -- it must match the one-shot example's format and the eval
    parser's expectation. Matches
    colab/llama_sentiment_baseline_train.ipynb cell 11's prepare_conversation
    exactly (its own comment: "Target matches the one-shot line after
    'Sentiment (1-5):' (not digit-only) so train matches inference")."""
    prompt = build_prompt(text, cfg, one_shot_example=one_shot)
    desc = RATING_DESCRIPTIONS.get(rating, f"Rating {rating}.")
    return {
        "prompt": prompt,
        "answer": f"{rating}. {desc}",
        "rating": rating,
        "text": text,
    }


def get_one_shot_from_pool(
    pool: list[dict], cfg: DataConfig, rng: random.Random
) -> Optional[str]:
    """Pick a random one-shot example from pool."""
    if not pool or not cfg.use_one_shot:
        return None
    ex = rng.choice(pool)
    return format_one_shot_example(ex["text"], ex["rating"], cfg)


def create_one_shot_pool(
    samples: list[dict], cfg: DataConfig, pool_size: int = 5
) -> list[dict]:
    """Create pool of one-shot examples (one per rating 1-5 if possible)."""
    by_rating: dict[int, list[dict]] = {r: [] for r in range(1, 6)}
    for s in samples:
        r = s.get("rating", s.get(cfg.label_column))
        if r is not None and r in by_rating:
            text = s.get("text", s.get(cfg.text_column, ""))
            by_rating[r].append({"text": text, "rating": int(r)})
    pool = []
    for r in range(1, 6):
        if by_rating[r]:
            pool.append(random.choice(by_rating[r]))
    if not pool and samples:
        s = samples[0]
        pool = [
            {
                "text": s.get("text", s.get(cfg.text_column, "")),
                "rating": int(s.get("rating", s.get(cfg.label_column, 3))),
            }
        ]
    return pool[:pool_size]
