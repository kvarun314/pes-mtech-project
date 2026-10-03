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
    # image_url comes from the PRODUCT's metadata (the Visual Verifier needs
    # product photos, not reviewer-uploaded images) -- half the fixture
    # products have one, half don't.
    with_image = [r for r in rows if r["image_url"]]
    assert all(r["image_url"].startswith("http://example.com/") for r in with_image)
    assert len(with_image) < len(rows)  # some products have no image in the fixture


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


def test_build_spec_records_handles_details_as_a_json_string():
    # Regression: the real Amazon Reviews 2023 metadata's `details` field
    # is a JSON *string* ('{"Manufacturer": "SIIG"}'), not a dict -- the
    # fixtures use dicts, so this was never caught locally and crashed on
    # the real dataset with AttributeError: 'str' object has no attribute
    # 'items'.
    meta = {"B1": {"title": "Widget", "description": [], "details": '{"Connector Type": "USB-C"}'}}
    records = build_spec_records(meta)
    assert any("USB-C" in r["text"] for r in records)


def test_build_spec_records_handles_malformed_details_string_without_crashing():
    meta = {"B1": {"title": "Widget", "description": [], "details": "not valid json"}}
    records = build_spec_records(meta)  # must not raise
    assert all(r["field"] != "not valid json" for r in records)


def test_load_balanced_slice_parses_details_json_string():
    reviews = [{"parent_asin": "B1", "rating": 5, "text": "a fine review with enough characters"}]
    meta = {"B1": {"title": "Widget", "details": '{"Connector Type": "USB-C"}'}}
    rows = load_balanced_slice(reviews, meta, n_per_rating=1, seed=42)
    assert rows[0]["meta_details"] == {"Connector Type": "USB-C"}
