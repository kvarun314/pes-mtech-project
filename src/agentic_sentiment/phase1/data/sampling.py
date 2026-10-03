"""
Paper Section 2.1–2.2: Data quality (TextBlob), stratified sampling, VGST, neutral oversampling.
"""

import random
from typing import Any, Optional

from agentic_sentiment.phase1.config import DataConfig


def filter_by_textblob_polarity(rows: list[dict]) -> list[dict]:
    """
    Paper 2.1 DQC: discard reviews where TextBlob polarity disagrees with rating.
    Discard if (polarity < 0 and rating > 3) or (polarity > 0 and rating < 3).
    Keep neutral polarity (== 0) with any rating.
    """
    try:
        from textblob import TextBlob
    except ImportError:
        return rows

    kept = []
    for r in rows:
        text = r.get("text", "")
        rating = r.get("rating", 3)
        try:
            polarity = float(TextBlob(text).sentiment.polarity)
        except (ValueError, TypeError, AttributeError):
            kept.append(r)
            continue
        # Discard misaligned: negative polarity but high rating, or positive polarity but low rating
        if polarity < 0 and rating > 3:
            continue
        if polarity > 0 and rating < 3:
            continue
        kept.append(r)
    return kept


def stratified_sample(
    rows: list[dict],
    samples_per_rating: Optional[int] = None,
    max_total: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """
    Paper 2.1: "equal number of samples for each distinct rating level".
    Groups by rating 1–5, then samples equally from each tier.
    """
    rng = rng or random.Random()
    by_rating: dict[int, list[dict]] = {i: [] for i in range(1, 6)}
    for row in rows:
        rating = row.get("rating")
        if rating in by_rating:
            by_rating[rating].append(row)

    if max_total is not None and samples_per_rating is None:
        samples_per_rating = max(1, max_total // 5)

    if samples_per_rating is None:
        # Take minimum count across classes so we have true balance
        counts = [len(by_rating[i]) for i in range(1, 6)]
        samples_per_rating = min(counts) if counts else 0

    out = []
    for rating in range(1, 6):
        pool = list(by_rating[rating])
        rng.shuffle(pool)
        out.extend(pool[:samples_per_rating])
    rng.shuffle(out)
    return out


def vgst_sample(
    rows: list[dict],
    target_size: int,
    batch_size: int,
    wishlist_len: int,
    tokenizer: Any,
    text_key: str = "text",
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """
    Paper 2.1 VGST: Variant Greedy Search Technique for diversity-aware sampling.
    1) Random batch of batch_size; 2) pick point that maximizes token diversity vs current set;
    3) add to wishlist; 4) repeat until wishlist has wishlist_len; 5) add best from wishlist to sample.
    Repeats until sample reaches target_size.
    """
    rng = rng or random.Random()
    if target_size <= 0 or not rows:
        return []

    def token_set(text: str) -> set[int]:
        tok = tokenizer.encode(text, add_special_tokens=False)
        return set(tok)

    sample: list[dict] = []
    current_tokens: set[int] = set()

    while len(sample) < target_size:
        wishlist: list[tuple[dict, int]] = []  # (row, new_tokens_count)
        for _ in range(wishlist_len):
            batch = rng.choices(rows, k=min(batch_size, len(rows)))
            best_row = None
            best_new = -1
            for row in batch:
                text = row.get(text_key, "")
                tids = token_set(text)
                new_count = len(tids - current_tokens)
                if new_count > best_new:
                    best_new = new_count
                    best_row = row
            if best_row is not None:
                wishlist.append((best_row, best_new))
        if not wishlist:
            sample.extend(rng.choices(rows, k=min(target_size - len(sample), len(rows))))
            break
        # Choose best from wishlist (max new tokens)
        chosen = max(wishlist, key=lambda x: x[1])
        sample.append(chosen[0])
        text = chosen[0].get(text_key, "")
        current_tokens |= token_set(text)
        if len(sample) >= target_size:
            break

    return sample


def oversample_neutral(
    rows: list[dict],
    neutral_rating: int = 3,
    multiplier: float = 2.0,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """
    Paper 2.2: "Due to the subtlety in distinguishing neutral reviews, this paper
    deliberately increased their representation in the dataset."
    """
    rng = rng or random.Random()
    neutral = [r for r in rows if r.get("rating") == neutral_rating]
    others = [r for r in rows if r.get("rating") != neutral_rating]
    # Add (multiplier - 1) * len(neutral) extra neutral samples (with replacement)
    n_extra = max(0, int(len(neutral) * (multiplier - 1.0)))
    extra = rng.choices(neutral, k=n_extra)
    out = others + neutral + extra
    rng.shuffle(out)
    return out


def apply_paper_preprocessing(
    rows: list[dict],
    data_cfg: DataConfig,
    tokenizer: Any = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Apply paper Section 2.1–2.2 pipeline in order:
    TextBlob DQC filter -> stratified (equal per rating) -> optional VGST -> oversample neutral -> cap.
    """
    rng = random.Random(seed)

    if getattr(data_cfg, "use_textblob_filter", True):
        rows = filter_by_textblob_polarity(rows)
    n_after_dqc = len(rows)

    if getattr(data_cfg, "use_stratified_sampling", True):
        per_rating = getattr(data_cfg, "samples_per_rating", None)
        max_tot = getattr(data_cfg, "stratified_max_total", None)
        if max_samples is not None and max_tot is None:
            max_tot = max_samples
        rows = stratified_sample(
            rows,
            samples_per_rating=per_rating,
            max_total=max_tot,
            rng=rng,
        )

    if getattr(data_cfg, "use_vgst", False) and tokenizer is not None:
        # VGST's ~1% diversity target is of the POST-DQC pool (before
        # stratification shrank it), floored at max_samples -- VGST is the
        # diversity-aware replacement for the later plain cap, not an
        # additional ~1% squeeze on top of an already-small stratified
        # pool. Matches colab/llama_sentiment_baseline_train.ipynb cell 10's
        # apply_paper_preprocessing exactly.
        ratio = getattr(data_cfg, "vgst_target_ratio", 0.01)
        one_pct_of_pool = max(1, int(ratio * n_after_dqc))
        cap = max_samples if max_samples else len(rows)
        target_size = min(len(rows), max(cap, one_pct_of_pool))
        if target_size < len(rows):
            rows = vgst_sample(
                rows,
                target_size=target_size,
                batch_size=getattr(data_cfg, "vgst_batch_size", 32),
                wishlist_len=getattr(data_cfg, "vgst_wishlist_len", 10),
                tokenizer=tokenizer,
                rng=rng,
            )

    if getattr(data_cfg, "oversample_neutral", True):
        ratio = getattr(data_cfg, "neutral_oversample_ratio", 2.0)
        rows = oversample_neutral(
            rows,
            neutral_rating=3,
            multiplier=ratio,
            rng=rng,
        )

    if max_samples is not None and len(rows) > max_samples:
        rng.shuffle(rows)
        rows = rows[:max_samples]

    return rows
