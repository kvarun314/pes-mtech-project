import numpy as np
from agentic_sentiment.v3.reflector import fit_reflector, choose_tau, logp


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed); y = rng.integers(1, 6, n)
    def noisy(sharp):
        P = rng.dirichlet(np.ones(5), n) * 0.3; P[np.arange(n), y - 1] += sharp; return P / P.sum(1, keepdims=True)
    return noisy(0.6), noisy(0.9), y


def test_combine_full_and_fallback():
    P, E, y = _data(); r = fit_reflector(P, E, y)
    star, probs, shares = r.combine(P[0], E[0])
    assert 1 <= star <= 5 and abs(probs.sum() - 1) < 1e-9
    assert abs(shares["probe"] + shares["evidence"] - 1) < 1e-9
    star2, _, shares2 = r.combine(None, E[0])
    assert shares2 == {"probe": 0.0, "evidence": 1.0} and 1 <= star2 <= 5


def test_choose_tau_returns_threshold_within_tolerance():
    P, E, y = _data()
    tau, base, rows = choose_tau(P, E, y)
    assert tau is None or any(t == tau and acc >= base - 0.005 for t, acc, _ in rows)
    assert all(0 <= skip <= 1 for _, _, skip in rows)


def test_logp_clips():
    assert np.isfinite(logp([0.0, 1.0])).all()
