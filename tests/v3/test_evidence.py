import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from agentic_sentiment.v3.evidence import EvidenceAgent, evidence_text

DOCS = [("", "great product works great"), ("", "terrible broke after a day"), ("", "okay but battery weak"),
        ("", "love it great value"), ("", "awful terrible waste"), ("", "fine works okay")] * 5
STARS = [5, 1, 3, 5, 1, 3] * 5


def _agent():
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(evidence_text(t, x) for t, x in DOCS)
    return EvidenceAgent(vec, LogisticRegression(C=10, max_iter=1000).fit(X, STARS))


def test_evidence_text_matches_training_format():
    assert evidence_text("Nice.", "Works") == "Nice. . Works"
    assert evidence_text("", "Works") == "Works"


def test_probs_and_phrases_for_star():
    ag = _agent()
    p = ag.probs("", "great value, works great")
    assert p.shape == (3,) and abs(p.sum() - 1) < 1e-9          # toy model has 3 classes
    pos, neg = ag.phrases("", "great value but terrible battery", star=5)
    assert pos[0]["weight"] > 0 and all(d["weight"] > 0 for d in pos)
    assert any("great" in d["phrase"] for d in pos)
    assert any("terrible" in d["phrase"] for d in neg) and all(d["weight"] < 0 for d in neg)
    assert all(set(d) == {"phrase", "display", "weight"} for d in pos + neg)


def test_phrases_skip_stop_words_and_repeats():
    ag = _agent()
    pos, neg = ag.phrases("", "it is not great but it works great", star=5)
    ps = [d["phrase"] for d in pos + neg]
    assert not any(set(p.split()) <= {"it", "is", "not", "but", "didn"} for p in ps)
    assert sum(p == "great" for p in ps) <= 1 and not ("great" in ps and "works great" in ps and ps.index("works great") < ps.index("great"))
