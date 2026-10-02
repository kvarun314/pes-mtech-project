"""Data loading, sampling (paper 2.1–2.2), and preprocessing."""

from agentic_sentiment.phase1.data.dataset import build_sft_dataset, load_raw_data
from agentic_sentiment.phase1.data.preprocessing import (
    build_prompt,
    create_one_shot_pool,
    get_one_shot_from_pool,
    prepare_conversation_format,
)
from agentic_sentiment.phase1.data.sampling import (
    apply_paper_preprocessing,
    filter_by_textblob_polarity,
    oversample_neutral,
    stratified_sample,
    vgst_sample,
)

__all__ = [
    "build_sft_dataset",
    "load_raw_data",
    "build_prompt",
    "prepare_conversation_format",
    "create_one_shot_pool",
    "get_one_shot_from_pool",
    "apply_paper_preprocessing",
    "filter_by_textblob_polarity",
    "stratified_sample",
    "vgst_sample",
    "oversample_neutral",
]
