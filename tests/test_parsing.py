from agentic_sentiment.agents.parsing import parse_rating


def test_parses_full_labeled_line():
    assert parse_rating("Sentiment (1-5): 4. Positive, satisfied customer.") == 4


def test_parses_leading_digit_only():
    assert parse_rating("5. Excellent product, would buy again.") == 5


def test_parses_leading_digit_with_dash():
    assert parse_rating("2 - somewhat negative") == 2


def test_parses_digit_anywhere_as_fallback():
    assert parse_rating("I would rate this a 3 out of 5") == 3


def test_returns_none_for_out_of_range_or_missing():
    assert parse_rating("no rating here") is None
    assert parse_rating("Sentiment (1-5): 7") is None
