"""Evidence agent: TF-IDF + LR (the ~497K-review Evidence model). Probabilities plus per-phrase contributions (coef[star] x tf-idf weight)."""
from dataclasses import dataclass, field
import numpy as np
from agentic_sentiment.v3.text import display_span


def evidence_text(title: str, text: str) -> str:
    return (title + " . " if title else "") + text          # same as evid_H.py txt() with title=True


@dataclass
class EvidenceAgent:
    vectorizer: object
    model: object
    names: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.names = self.vectorizer.get_feature_names_out()

    def probs(self, title: str, text: str) -> np.ndarray:
        return self.model.predict_proba(self.vectorizer.transform([evidence_text(title, text)]))[0]

    def phrases(self, title: str, text: str, star: int, k_for: int = 5, k_against: int = 3):
        full = evidence_text(title, text)
        x = self.vectorizer.transform([full]).tocsr()
        k = list(self.model.classes_).index(star)
        contrib = x.data * self.model.coef_[k, x.indices]
        order = np.argsort(contrib)
        fmt = lambda j: {"phrase": str(self.names[x.indices[j]]), "display": display_span(full, str(self.names[x.indices[j]])),
                         "weight": round(float(contrib[j]), 4)}
        pos = [fmt(j) for j in order[::-1] if contrib[j] > 0][:k_for]
        neg = [fmt(j) for j in order if contrib[j] < 0][:k_against]
        return pos, neg
