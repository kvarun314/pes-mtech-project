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


def test_three_agent_reflector():
    P, E, y = _data(); B = _data(seed=1)[1]; r = fit_reflector(P, E, y, B)
    assert r.full.n_features_in_ == 15 and r.fallback.n_features_in_ == 10
    star, probs, sh = r.combine(P[0], E[0], B[0])
    assert 1 <= star <= 5 and set(sh) == {"probe", "evidence", "deberta"} and abs(sum(sh.values()) - 1) < 1e-9
    _, _, sh2 = r.combine(None, E[0], B[0])
    assert sh2["probe"] == 0.0 and abs(sh2["evidence"] + sh2["deberta"] - 1) < 1e-9
    tau, base, rows = choose_tau(P, E, y, B=B)
    assert tau is None or any(t == tau for t, _, _ in rows)
    import pytest
    with pytest.raises(AssertionError): r.combine(P[0], E[0])         # three-agent Reflector needs DeBERTa probabilities
