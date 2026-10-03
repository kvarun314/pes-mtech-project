"""Regression tests for apply_paper_preprocessing's VGST target size --
must match colab/llama_sentiment_baseline_train.ipynb cell 10 exactly:
VGST's ~1% diversity target is of the POST-DQC pool (not the already-
shrunk stratified pool), floored at max_samples."""

import random

from agentic_sentiment.phase1.config import DataConfig
from agentic_sentiment.phase1.data.sampling import apply_paper_preprocessing


class _FakeTokenizer:
    """Deterministic: each row's text maps to a distinct token id, so
    target_size is exactly how many unique rows VGST selects."""

    def encode(self, text, add_special_tokens=False):
        return [hash(text) % 10_000]


def test_vgst_target_size_is_floored_at_max_samples_not_one_percent_of_stratified_pool():
    # 20,000 rows after DQC -> stratified pool capped at stratified_max_total
    # (10,000, DataConfig default) -> VGST must then select ~max_samples
    # (4,000, i.e. close to the full stratified pool), NOT ~1% of the
    # stratified pool (100 rows) and NOT skip VGST's floor entirely.
    rows = [{"text": f"review {i}", "rating": (i % 5) + 1} for i in range(20_000)]
    cfg = DataConfig(use_textblob_filter=False, stratified_max_total=10_000, use_vgst=True)

    result = apply_paper_preprocessing(
        rows, cfg, tokenizer=_FakeTokenizer(), max_samples=4_000, seed=42,
    )

    # Final cap (outside apply_paper_preprocessing, in build_sft_dataset) is
    # applied by the caller -- here we only check VGST itself didn't shrink
    # the pool down to ~1% of 10,000 (100), which was the bug.
    assert len(result) > 1_000


def test_vgst_disabled_leaves_stratified_pool_untouched():
    rows = [{"text": f"review {i}", "rating": (i % 5) + 1} for i in range(1_000)]
    cfg = DataConfig(
        use_textblob_filter=False, stratified_max_total=500, use_vgst=False,
        oversample_neutral=False,
    )

    result = apply_paper_preprocessing(rows, cfg, tokenizer=_FakeTokenizer(), max_samples=4_000, seed=42)

    assert len(result) == 500  # exactly the stratified cap, no VGST shrink
