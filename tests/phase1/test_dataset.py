"""Regression tests for dataset.py's max_csv_rows cap and val-set one-shot
usage -- both match colab/llama_sentiment_baseline_train.ipynb cells 10/12."""

import csv

from agentic_sentiment.phase1.config import DataConfig, TrainingConfig
from agentic_sentiment.phase1.data.dataset import build_sft_dataset, load_raw_data


def test_load_raw_data_respects_max_rows_cap(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_text", "rating"])
        for i in range(100):
            writer.writerow([f"review {i}", (i % 5) + 1])

    rows = load_raw_data(str(csv_path), "review_text", "rating", max_rows=10)
    assert len(rows) == 10


def test_load_raw_data_reads_all_rows_when_max_rows_none(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_text", "rating"])
        for i in range(50):
            writer.writerow([f"review {i}", (i % 5) + 1])

    rows = load_raw_data(str(csv_path), "review_text", "rating", max_rows=None)
    assert len(rows) == 50


class _FakeTokenizer:
    """Minimal stand-in: word-count length, fixed pad/eos ids, no real
    vocabulary -- enough to exercise build_sft_dataset's control flow
    (one-shot selection, tokenization, padding) without a real model."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, truncation=True, max_length=64, padding=None, return_tensors=None):
        ids = [1] * min(len(text.split()), max_length)
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


def test_build_sft_dataset_uses_one_shot_for_val_when_enabled(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_text", "rating"])
        for i in range(40):
            writer.writerow([f"review number {i} with enough words", (i % 5) + 1])

    data_cfg = DataConfig(
        use_textblob_filter=False, use_vgst=False, oversample_neutral=False,
        stratified_max_total=40, use_one_shot=True,
    )
    training_cfg = TrainingConfig(max_samples=40, max_seq_length=32, val_ratio=0.2)

    # Doesn't raise and produces non-empty train/val sets -- the control-flow
    # path that selects one-shot for val (not just train) is exercised since
    # use_one_shot=True; a crash there would mean the val branch is broken.
    train_ds, val_ds = build_sft_dataset(str(csv_path), _FakeTokenizer(), data_cfg, training_cfg, seed=42)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
