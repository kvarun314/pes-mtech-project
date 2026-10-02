import math

from agentic_sentiment.agents.critic import (
    DISSONANCE_THRESHOLD,
    dissonance_norm,
    dissonance_stdev,
    normalize_rating,
)


def test_normalize_rating_maps_1_to_5_onto_0_to_1():
    assert normalize_rating(1) == 0.0
    assert normalize_rating(5) == 1.0
    assert normalize_rating(3) == 0.5


def test_dissonance_norm_zero_when_all_agree():
    h = normalize_rating(5)
    assert dissonance_norm(h, h, h) == 0.0


def test_dissonance_norm_positive_when_disagreeing():
    h = normalize_rating(5)   # 1.0
    ev = normalize_rating(1)  # 0.0
    gf = normalize_rating(1)  # 0.0
    # Norm(H - (Ev+Gf)/2) = |1.0 - 0.0| = 1.0
    assert math.isclose(dissonance_norm(h, ev, gf), 1.0)


def test_dissonance_norm_is_clamped_to_0_1():
    assert 0.0 <= dissonance_norm(1.0, 0.0, 0.0) <= 1.0
    assert 0.0 <= dissonance_norm(0.0, 1.0, 1.0) <= 1.0


def test_dissonance_stdev_matches_run1_formula():
    # Run 1: D = stdev(votes) / 2
    assert math.isclose(dissonance_stdev([5, 5, 5]), 0.0)
    assert dissonance_stdev([1, 3, 5]) > 0


def test_dissonance_stdev_handles_fewer_than_two_votes():
    assert dissonance_stdev([]) == 0.0
    assert dissonance_stdev([3]) == 0.0


def test_threshold_is_point_four():
    assert DISSONANCE_THRESHOLD == 0.4
