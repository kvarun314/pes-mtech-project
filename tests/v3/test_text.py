from agentic_sentiment.v3.text import tokens, contains, display_span, quoted, phrase_token_idx, drop_tokens


def test_tokens_match_sklearn_default_pattern():
    assert tokens("Doesn't work, at ALL!") == ["doesn", "work", "at", "all"]


def test_contains_contiguous_only():
    hay = tokens("the battery barely lasts")
    assert contains(hay, "battery barely")
    assert not contains(hay, "battery lasts")
    assert not contains(hay, "")


def test_display_span_recovers_original_text():
    text = "Honestly it doesn't work at all."
    assert display_span(text, "doesn work") == "doesn't work"
    assert display_span(text, "great value") == "great value"   # not found: phrase itself


def test_quoted_handles_straight_and_curly_quotes():
    assert quoted('It is "great value" and “battery dies”.') == ["great value", "battery dies"]


def test_phrase_token_idx_and_drop():
    text = "Great screen, but the battery dies fast."
    idx = phrase_token_idx(text, "battery dies")
    assert idx == [4, 5]
    assert drop_tokens(text, set(idx)) == "Great screen, but the   fast."
    assert phrase_token_idx(text, "screen battery") is None
