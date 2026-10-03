"""One-shot example pool for the Phase 2 Analyst -- both proven Phase 2
notebooks (colab/phase2_agentic_full_comparison.ipynb's
create_one_shot_pool/pick_one_shot, and
colab/phase2_agentic_vs_phase1_amazon2023.ipynb via build_prompt_phase1)
always give the Analyst a one-shot example, matching how the Phase 1
adapter was fine-tuned (one-shot + CoT). graph.py's Analyst dropped this
when rebuilt; this module ports the proven pool/pick logic so run_eval.py
can wire it back in."""

import random

from agentic_sentiment.phase1.data.preprocessing import RATING_DESCRIPTIONS

COT_PHRASE = "Let's take it one step at a time."


def format_one_shot_example(review_text: str, rating: int) -> str:
    desc = RATING_DESCRIPTIONS.get(rating, f"Rating {rating}.")
    return f"Review: {review_text}\nSentiment (1-5): {rating}. {desc}"


def build_one_shot_pool(rows: list[dict], pool_size: int = 5, seed: int = 42) -> list[dict]:
    """One row per rating 1-5 (if available among `rows`), for use as
    one-shot examples. Matches the proven notebooks' approach of building
    the pool from the eval set itself.

    `rows` is sorted by review_id before pooling -- rng.choice() picks by
    position, so an unsorted/differently-ordered `rows` (e.g. a caller that
    reloads the eval set fresh on a resumed run, in a different order)
    would otherwise silently produce a different pool. Sorting first makes
    the result depend only on which rows and ratings are present, not on
    the order they arrived in."""
    rng = random.Random(seed)
    by_rating: dict[int, list[dict]] = {r: [] for r in range(1, 6)}
    for row in sorted(rows, key=lambda r: r.get("review_id", "")):
        rating = row.get("gt_rating")
        if rating in by_rating:
            by_rating[rating].append(row)
    pool = [rng.choice(candidates) for candidates in by_rating.values() if candidates]
    rng.shuffle(pool)
    return pool[:pool_size]


def pick_one_shot_text(row: dict, pool: list[dict], rng: random.Random) -> str | None:
    """Pick a formatted one-shot example from the pool, preferring one with
    a DIFFERENT rating than `row`'s own -- so the example doesn't trivially
    hand the model the answer. Matches
    colab/phase2_agentic_full_comparison.ipynb's pick_one_shot."""
    if not pool:
        return None
    gt = row.get("gt_rating")
    candidates = [p for p in pool if p.get("gt_rating") != gt] or pool
    chosen = rng.choice(candidates)
    return format_one_shot_example(chosen["review_text"], chosen["gt_rating"])
