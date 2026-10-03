"""Tests for data preprocessing (prompts, one-shot, CoT)."""

import random

from agentic_sentiment.phase1.data.preprocessing import (
    build_prompt,
    create_one_shot_pool,
    format_one_shot_example,
    get_one_shot_from_pool,
    prepare_conversation_format,
    rating_to_sentiment_class,
)


def test_build_prompt_includes_review_and_instruction(data_config):
    prompt = build_prompt("Great product!", data_config, one_shot_example=None)
    assert "Great product!" in prompt
    assert "Sentiment (1-5)" in prompt
    assert "1-2 negative" in prompt or "five-point" in prompt.lower()


def test_build_prompt_with_cot_includes_phrase(data_config):
    data_config.use_cot = True
    prompt = build_prompt("Okay.", data_config, one_shot_example=None)
    assert "step" in prompt.lower() or "time" in prompt.lower()


def test_format_one_shot_example(data_config):
    s = format_one_shot_example("It was bad.", 1, data_config)
    assert "It was bad." in s
    assert "1" in s


def test_prepare_conversation_format(data_config):
    # The SFT target is "{rating}. {description}", not a bare digit --
    # must match the one-shot example's format and the eval parser's
    # expectation (colab/llama_sentiment_baseline_train.ipynb cell 11).
    out = prepare_conversation_format("Nice.", 5, data_config, one_shot=None)
    assert out["text"] == "Nice."
    assert out["rating"] == 5
    assert out["answer"].startswith("5. ")
    assert len(out["answer"]) > len("5. ")
    assert "Nice." in out["prompt"]


def test_rating_to_sentiment_class(data_config):
    assert rating_to_sentiment_class(1, data_config) == "negative"
    assert rating_to_sentiment_class(2, data_config) == "negative"
    assert rating_to_sentiment_class(3, data_config) == "neutral"
    assert rating_to_sentiment_class(4, data_config) == "positive"
    assert rating_to_sentiment_class(5, data_config) == "positive"


def test_create_one_shot_pool_empty_returns_empty(data_config):
    pool = create_one_shot_pool([], data_config, pool_size=5)
    assert pool == []


def test_create_one_shot_pool_from_samples(data_config):
    samples = [
        {"text": "Good", "rating": 5},
        {"text": "Bad", "rating": 1},
    ]
    pool = create_one_shot_pool(samples, data_config, pool_size=5)
    assert len(pool) >= 1
    assert all("text" in p and "rating" in p for p in pool)


def test_get_one_shot_from_pool_none_when_disabled(data_config):
    data_config.use_one_shot = False
    rng = random.Random(42)
    assert get_one_shot_from_pool([{"text": "x", "rating": 1}], data_config, rng) is None
