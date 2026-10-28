import numpy as np
from agentic_sentiment.v3.conformal import qhat, star_range, evaluate


def test_star_range_is_contiguous_hull():
    assert star_range([.5, .0, .45, .03, .02], q=0.6) == (1, 3)


def test_star_range_falls_back_to_argmax():
    assert star_range([.1, .2, .4, .2, .1], q=-1.0) == (3, 3)


def test_coverage_close_to_target_on_fresh_data():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5) * 0.5, size=6000); y = np.array([rng.choice(5, p=p) + 1 for p in P])
    q = qhat(P[:3000], y[:3000], alpha=0.1); r = evaluate(P[3000:], y[3000:], q)
    assert 0.88 <= r["coverage"] <= 0.96 and 1 <= r["mean_size"] <= 5   # contiguous hull can only add coverage
