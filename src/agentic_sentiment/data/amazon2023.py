"""Amazon Reviews 2023 (Electronics) loading: balanced per-rating eval
slice + product-spec records for the RAG spec store."""

import random


def _first_image_url(meta: dict) -> str | None:
    """Product (not reviewer-uploaded) image, for the Visual Verifier --
    matches the Amazon Reviews 2023 metadata schema: images is a dict of
    lists keyed by resolution (hi_res/large/thumb)."""
    images = meta.get("images") or {}
    if isinstance(images, dict):
        for key in ("hi_res", "large", "thumb"):
            for url in images.get(key) or []:
                if url and isinstance(url, str) and url.startswith("http"):
                    return url
    return None


def load_balanced_slice(
    reviews_rows: list[dict],
    meta_by_asin: dict[str, dict],
    n_per_rating: int,
    seed: int = 42,
    min_review_chars: int = 20,
) -> list[dict]:
    """Filter to rows with metadata + long-enough text, then sample up to
    n_per_rating rows per star rating 1-5 (fewer if not enough exist)."""
    by_rating: dict[int, list[dict]] = {r: [] for r in range(1, 6)}
    for row in reviews_rows:
        asin = row.get("parent_asin")
        text = row.get("text", "") or ""
        rating = int(row.get("rating", 0))
        if asin not in meta_by_asin or len(text) < min_review_chars or rating not in by_rating:
            continue
        meta = meta_by_asin[asin]
        by_rating[rating].append({
            "review_id": f"{asin}_{rating}_{len(by_rating[rating])}",
            "asin": asin,
            "gt_rating": rating,
            "review_text": text,
            "meta_title": meta.get("title", ""),
            "meta_description": " ".join(meta.get("description", []) or []),
            "meta_details": meta.get("details", {}) or {},
            "image_url": _first_image_url(meta),
        })

    rng = random.Random(seed)
    out = []
    for rating, rows in by_rating.items():
        rng.shuffle(rows)
        out.extend(rows[:n_per_rating])
    rng.shuffle(out)
    return out


def build_spec_records(meta_by_asin: dict[str, dict]) -> list[dict]:
    """One record per spec-bearing metadata field, for SpecStore.build()."""
    records = []
    for asin, meta in meta_by_asin.items():
        if meta.get("title"):
            records.append({"asin": asin, "field": "title", "text": meta["title"]})
        description = " ".join(meta.get("description", []) or [])
        if description:
            records.append({"asin": asin, "field": "description", "text": description})
        for field, value in (meta.get("details") or {}).items():
            records.append({"asin": asin, "field": field, "text": f"{field}: {value}"})
    return records
