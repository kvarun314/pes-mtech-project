"""detect_use_4bit must match colab/llama_sentiment_baseline_train.ipynb
cell 14's own bitsandbytes/GPU-memory detection exactly -- the proven run
trained QLoRA (nf4), not full-fp16; silently falling back to fp16 changes
the training's numerical regime and OOMs on a T4."""

from unittest.mock import patch

import pytest

from agentic_sentiment.phase1.models.lora import detect_fast_training_config, detect_use_4bit


def test_returns_true_when_bitsandbytes_importable():
    with patch.dict("sys.modules", {"bitsandbytes": object()}):
        assert detect_use_4bit() is True


def test_raises_when_bitsandbytes_missing_and_gpu_too_small():
    with patch.dict("sys.modules", {"bitsandbytes": None}), \
         patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="would OOM"):
            detect_use_4bit(min_fp16_gpu_gb=18.0)


def test_falls_back_to_fp16_when_bitsandbytes_missing_but_gpu_is_large_enough():
    class _FakeProps:
        total_memory = 24 * 1024**3  # 24GB, e.g. an A100

    with patch.dict("sys.modules", {"bitsandbytes": None}), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=_FakeProps()):
        assert detect_use_4bit(min_fp16_gpu_gb=18.0) is False


class _FakeProps:
    def __init__(self, gb):
        self.total_memory = gb * 1024**3


def test_detect_fast_training_config_empty_on_small_gpu():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=_FakeProps(16)):
        assert detect_fast_training_config(min_gpu_gb=20.0) == {}


def test_detect_fast_training_config_keeps_effective_batch_on_large_gpu():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=_FakeProps(24)):
        cfg = detect_fast_training_config(min_gpu_gb=20.0)
        assert cfg["gradient_checkpointing"] is False
        assert cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] == 12
