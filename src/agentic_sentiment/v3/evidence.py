"""Evidence agent: TF-IDF + LR (the ~497K-review Evidence model). Probabilities plus per-phrase contributions (coef[star] x tf-idf weight)."""
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from agentic_sentiment.v3.text import contains, display_span

# stop words plus the stubs the tokenizer leaves from contractions (didn't -> didn)
WEAK = ENGLISH_STOP_WORDS | {"don", "didn", "doesn", "isn", "wasn", "won", "couldn", "wouldn", "shouldn", "aren", "haven", "hasn", "ve", "ll", "re", "videoid"}


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
        def pick(idx, k):                                   # informative phrases only, no phrase inside an already chosen one
            out = []
            for j in idx:
                p = str(self.names[x.indices[j]])
                if any(w not in WEAK for w in p.split()) and not any(contains(o["phrase"].split(), p) for o in out):
                    out.append(fmt(j))
                if len(out) == k:
                    break
            return out
        return pick([j for j in order[::-1] if contrib[j] > 0], k_for), pick([j for j in order if contrib[j] < 0], k_against)
