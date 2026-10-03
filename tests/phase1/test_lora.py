"""Regression test: get_lora_config's target_modules fallback must match
colab/llama_sentiment_baseline_train.ipynb cell 14's get_lora_config
exactly (q/k/v/o, not just q/v)."""

import pytest

pytest.importorskip("peft")

from agentic_sentiment.phase1.config import ModelConfig
from agentic_sentiment.phase1.models.lora import get_lora_config


def test_target_modules_fallback_is_all_four_attention_projections():
    cfg = ModelConfig(lora_target_modules=None)
    lora_cfg = get_lora_config(cfg)
    assert lora_cfg.target_modules == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_explicit_target_modules_are_not_overridden():
    cfg = ModelConfig(lora_target_modules=["q_proj"])
    lora_cfg = get_lora_config(cfg)
    assert lora_cfg.target_modules == {"q_proj"}
