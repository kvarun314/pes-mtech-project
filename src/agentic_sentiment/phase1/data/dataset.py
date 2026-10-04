"""
Dataset loading: CSV/JSON -> train/val HuggingFace Datasets for SFT.
"""

import os
import random
from typing import Optional

from datasets import Dataset

from agentic_sentiment.phase1.config import DataConfig, TrainingConfig
from agentic_sentiment.phase1.data.preprocessing import (
    create_one_shot_pool,
    get_one_shot_from_pool,
    prepare_conversation_format,
)
from agentic_sentiment.phase1.data.sampling import apply_paper_preprocessing

TEXT_COLUMN_CANDIDATES = (
    "review_text",
    "text",
    "Text",
    "review",
    "Review",
    "content",
    "body",
)
RATING_COLUMN_CANDIDATES = ("rating", "Rating", "score", "Score", "overall", "rating_star")


def _infer_columns(df_columns: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Infer text and rating column from DataFrame columns."""
    text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in df_columns), None)
    rating_col = next((c for c in RATING_COLUMN_CANDIDATES if c in df_columns), None)
    return text_col, rating_col


def load_raw_data(
    path: str, text_column: str, label_column: str, max_rows: Optional[int] = None
) -> list[dict]:
    """Load CSV or JSON into list of dicts with 'text' and 'rating'.

    `max_rows` caps the CSV read (500k-row Kaggle pool; DataConfig.max_csv_rows
    default 250k balances RAM vs diversity on Colab) -- matches
    colab/llama_sentiment_baseline_train.ipynb cell 10's load_raw_data."""
    ext = os.path.splitext(path)[1].lower()
    rows: list[dict] = []
    if ext == ".csv":
        import pandas as pd

        df = pd.read_csv(path, nrows=max_rows)
        infer_text, infer_rating = _infer_columns(df.columns.tolist())
        use_text = (
            text_column
            if text_column in df.columns
            else (infer_text or "review_text")
        )
        use_rating = (
            label_column
            if label_column in df.columns
            else (infer_rating or "rating")
        )
        for _, r in df.iterrows():
            text = (
                str(r[use_text])
                if use_text and use_text in r
                else str(r.get("review_text", r.get("review", "")))
            )
            rating = (
                r[use_rating]
                if use_rating and use_rating in r
                else r.get("rating", 3)
            )
            try:
                rating = int(float(rating))
            except (ValueError, TypeError):
                rating = 3
            rating = max(1, min(5, rating))
            rows.append({"text": text, "rating": rating})
    elif ext == ".json":
        import json

        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON dataset should be a list of objects.")
        for item in data:
            text = item.get(
                text_column,
                item.get("review_text", item.get("review", "")),
            )
            rating = item.get(label_column, item.get("rating", 3))
            try:
                rating = int(rating)
            except (ValueError, TypeError):
                rating = 3
            rows.append({"text": str(text), "rating": rating})
    else:
        raise ValueError(f"Unsupported format: {path}. Use .csv or .json.")
    return rows


def build_sft_dataset(
    data_path: Optional[str],
    tokenizer,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Build train/val HuggingFace Datasets for SFT.
    If data_path is None or missing, returns a small dummy dataset.
    """
    rng = random.Random(seed)

    if not data_path or not os.path.isfile(data_path):
        dummy = [
            {"text": "This product is great, I love it.", "rating": 5},
            {"text": "Terrible experience, would not buy again.", "rating": 1},
            {"text": "It's okay, nothing special.", "rating": 3},
        ]
        rows = dummy * 10
    else:
        rows = load_raw_data(
            data_path, data_cfg.text_column, data_cfg.label_column,
            max_rows=getattr(data_cfg, "max_csv_rows", None),
        )
        # Paper 2.1–2.2: TextBlob DQC, stratified sampling, optional VGST, oversample neutral
        rows = apply_paper_preprocessing(
            rows,
            data_cfg,
            tokenizer=tokenizer,
            max_samples=training_cfg.max_samples,
            seed=seed,
        )

    max_s = training_cfg.max_samples
    if max_s is not None and len(rows) > max_s:
        rng.shuffle(rows)
        rows = rows[:max_s]

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * training_cfg.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    one_shot_pool = create_one_shot_pool(train_rows, data_cfg, pool_size=5)
    max_len = training_cfg.max_seq_length

    def _format_and_tokenize(
        examples: list[dict], split: str
    ) -> dict[str, list]:
        prompts = []
        answers = []
        for ex in examples:
            # Use one-shot for val too (pool is train-only) so val loss
            # matches the actual eval protocol (one-shot + CoT) -- matches
            # colab/llama_sentiment_baseline_train.ipynb cell 12 exactly.
            one_shot = (
                get_one_shot_from_pool(one_shot_pool, data_cfg, rng)
                if split == "train" or (split == "val" and getattr(data_cfg, "use_one_shot", True))
                else None
            )
            formatted = prepare_conversation_format(
                ex["text"], ex["rating"], data_cfg, one_shot
            )
            prompts.append(formatted["prompt"])
            answers.append(formatted["answer"])

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for p, a in zip(prompts, answers):
            full_text = p + " " + a
            # Truncate to max_len but don't pre-pad every example up to it --
            # most reviews tokenize far shorter than 1024, and attention/FFN
            # compute scales with however long each example actually is.
            # Padded positions are masked out of the loss (labels=-100) and
            # the attention mask regardless of where padding happens, so
            # this changes zero gradients -- only wall-clock. The Trainer's
            # DataCollatorForSeq2Seq (padding=True) pads each batch to its
            # own longest example instead of a fixed 1024 for every batch.
            tok = tokenizer(
                full_text,
                truncation=True,
                max_length=max_len,
                padding=False,
                return_tensors=None,
            )
            prompt_tok = tokenizer(
                p,
                truncation=True,
                max_length=max_len,
                return_tensors=None,
            )
            prompt_len = len(prompt_tok["input_ids"])
            input_ids = list(tok["input_ids"])
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            attention_mask = [1] * len(input_ids)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    train_processed = _format_and_tokenize(train_rows, "train")
    val_processed = _format_and_tokenize(val_rows, "val")
    return (
        Dataset.from_dict(train_processed),
        Dataset.from_dict(val_processed),
    )
