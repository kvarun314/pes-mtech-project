from agentic_sentiment.v3.perturb import typos, flat


def test_typos_deterministic_and_keeps_words():
    t = "This charger stopped working after three weeks of normal use"
    a, b = typos(t, rate=0.5, seed=1), typos(t, rate=0.5, seed=1)
    assert a == b and a != t and len(a.split()) == len(t.split()) and sorted(a) == sorted(t)


def test_typos_rate_zero_is_identity():
    assert typos("Great sound, works well.", rate=0.0) == "Great sound, works well."


def test_flat_lowercases_and_strips_punctuation():
    assert flat("Great!!! Doesn't work, though.") == "great doesn't work though"
