"""Label-preserving text changes for the robustness check (Q4, LOG.md 2026-10-28). Deterministic for a given seed."""
import random
import re


def typos(text: str, rate: float = 0.1, seed: int = 0) -> str:
    rng, out = random.Random(seed), []
    for w in text.split(" "):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            i = rng.randrange(len(w) - 1); w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def flat(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
