import random

from agentic_sentiment.eval.one_shot import (
    build_one_shot_pool,
    format_one_shot_example,
    pick_one_shot_text,
)


def test_format_one_shot_example_includes_review_and_labeled_rating():
    text = format_one_shot_example("Loved it.", 5)
    assert "Loved it." in text
    assert "Sentiment (1-5): 5." in text


def test_build_one_shot_pool_has_one_per_rating():
    rows = [{"review_text": f"review {i}", "gt_rating": (i % 5) + 1} for i in range(20)]
    pool = build_one_shot_pool(rows, pool_size=5, seed=42)
    assert len(pool) == 5
    assert {p["gt_rating"] for p in pool} == {1, 2, 3, 4, 5}


def test_build_one_shot_pool_caps_at_pool_size():
    rows = [{"review_text": f"review {i}", "gt_rating": (i % 5) + 1} for i in range(20)]
    pool = build_one_shot_pool(rows, pool_size=2, seed=42)
    assert len(pool) == 2


def test_build_one_shot_pool_skips_ratings_with_no_rows():
    rows = [{"review_text": "only fives", "gt_rating": 5}] * 3
    pool = build_one_shot_pool(rows, pool_size=5, seed=42)
    assert len(pool) == 1
    assert pool[0]["gt_rating"] == 5


def test_pick_one_shot_text_prefers_a_different_rating_than_the_row():
    pool = [
        {"review_text": "bad one", "gt_rating": 1},
        {"review_text": "great one", "gt_rating": 5},
    ]
    row = {"review_text": "mine", "gt_rating": 5}
    rng = random.Random(0)

    text = pick_one_shot_text(row, pool, rng)

    assert "bad one" in text  # the rating-1 example, not the same-rating-5 one
    assert "Sentiment (1-5): 1." in text


def test_pick_one_shot_text_falls_back_to_pool_when_all_ratings_match():
    pool = [{"review_text": "same rating", "gt_rating": 3}]
    row = {"review_text": "mine", "gt_rating": 3}
    rng = random.Random(0)

    text = pick_one_shot_text(row, pool, rng)

    assert "same rating" in text


def test_pick_one_shot_text_returns_none_for_empty_pool():
    rng = random.Random(0)
    assert pick_one_shot_text({"gt_rating": 3}, [], rng) is None
