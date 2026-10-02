import json
from pathlib import Path

from agentic_sentiment.data.amazon2023 import build_spec_records, load_balanced_slice

FIXTURES = Path(__file__).parent / "fixtures"


def _load_jsonl(path):
    return [json.loads(line) for line in path.open()]


def test_load_balanced_slice_has_equal_counts_per_rating():
    reviews = _load_jsonl(FIXTURES / "electronics_reviews.jsonl")
    meta = {m["parent_asin"]: m for m in _load_jsonl(FIXTURES / "electronics_meta.jsonl")}

    rows = load_balanced_slice(reviews, meta, n_per_rating=2, seed=42)

    assert len(rows) == 10
    counts = {}
    for r in rows:
        counts[r["gt_rating"]] = counts.get(r["gt_rating"], 0) + 1
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert all(r["meta_title"] for r in rows)


def test_load_balanced_slice_caps_at_available_rows():
    reviews = _load_jsonl(FIXTURES / "electronics_reviews.jsonl")
    meta = {m["parent_asin"]: m for m in _load_jsonl(FIXTURES / "electronics_meta.jsonl")}

    rows = load_balanced_slice(reviews, meta, n_per_rating=5, seed=42)  # only 2 available per rating
    assert len(rows) == 10


def test_build_spec_records_one_per_field():
    meta = {m["parent_asin"]: m for m in _load_jsonl(FIXTURES / "electronics_meta.jsonl")}
    records = build_spec_records(meta)

    assert all({"asin", "field", "text"} <= r.keys() for r in records)
    usb_c = [r for r in records if "USB-C" in r["text"]]
    assert len(usb_c) == 10  # one per product's "Connector Type" detail
