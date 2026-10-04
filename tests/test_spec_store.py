import math

from agentic_sentiment.rag.spec_store import SpecStore

RECORDS = [
    {"asin": "B1", "field": "Connector Type", "text": "Connector Type: USB-C"},
    {"asin": "B1", "field": "Battery Life", "text": "Battery Life: 10 hours"},
    {"asin": "B2", "field": "Connector Type", "text": "Connector Type: Micro-USB"},
]


def _fake_embed(text: str) -> list[float]:
    """Deterministic 3-d embedding: [has 'usb-c', has 'micro', has 'battery'],
    so similarity is exact and the test needs no real model."""
    t = text.lower()
    return [
        1.0 if "usb-c" in t else 0.0,
        1.0 if "micro" in t else 0.0,
        1.0 if "battery" in t else 0.0,
    ]


def test_query_returns_matching_asin_records(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    results = store.query("this has a USB-C port", asin="B1", k=3)
    assert all(r["asin"] == "B1" for r in results)
    assert any("USB-C" in r["text"] for r in results)


def test_grounding_score_high_when_claim_matches_spec(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    gf = store.grounding_score(["this has a USB-C port"], asin="B1")
    assert gf == 1.0


def test_grounding_score_low_when_claim_contradicts_spec(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    gf = store.grounding_score(["this has a Micro-USB port"], asin="B1")  # B1 is USB-C
    assert gf == 0.0


def test_grounding_score_averages_across_claims(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    gf = store.grounding_score(["this has a USB-C port", "this has a Micro-USB port"], asin="B1")
    assert math.isclose(gf, 0.5)


def test_grounding_score_embeds_each_claim_exactly_once(tmp_path):
    # Regression: grounding_score previously called embed_fn twice per claim
    # (once inside query() for the search, once again for the cosine check)
    # and once more for the matched spec's own text, which query()'s result
    # already carries as a stored vector column -- this is the single most
    # repeated call in the full_graph/plus_rag_specs ablations (every row,
    # every ablation, every correction iteration), so halving it compounds.
    calls = []

    def counting_embed(text: str) -> list[float]:
        calls.append(text)
        return _fake_embed(text)

    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=counting_embed)
    store.build(RECORDS)
    calls.clear()  # build() embeds the spec records themselves; not what we're counting

    store.grounding_score(["this has a USB-C port", "this has a Micro-USB port"], asin="B1")

    assert calls == ["this has a USB-C port", "this has a Micro-USB port"]
