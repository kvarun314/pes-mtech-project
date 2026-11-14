"""DeBERTa-v3 agent (D1, adopted LOG.md 2026-11-10): fine-tuned on 100K balanced Electronics reviews on Colab; run locally for
live use (demo, robustness check, deletion test). Same input text and truncation as colab_deberta.py."""
import numpy as np
from agentic_sentiment.v3 import paths as PT

NAME, MAXLEN = "microsoft/deberta-v3-base", 256


class DebertaAgent:
    def __init__(self, path: str = PT.DEBERTA_DIR, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch; self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).float().to(self.dev).eval()

    def probs_many(self, pairs, bs: int = 32) -> np.ndarray:
        """pairs: list of (title, text). Returns (n, 5) probabilities for stars 1-5."""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), bs):
                x = self.tok([(t + " . " if t else "") + s for t, s in pairs[i:i + bs]], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(self.dev)
                out.append(self.torch.softmax(self.model(**x).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    def probs(self, title: str, text: str) -> np.ndarray:
        return self.probs_many([(title, text)])[0]
