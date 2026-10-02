"""LanceDB-backed product-spec store for the RAG Prover. Embeds each spec
record (title/description/detail field) and, for the real pipeline, each
product image via a shared text-image embedding space (CLIP in Colab);
locally it is tested with any injected `embed_fn`. grounding_score() checks
a claim against the *correct product's* specs only, which is what catches
the "USB-C vs Micro-USB" factual-mismatch case in the mission spec."""

import math
from typing import Callable

import lancedb
import pyarrow as pa


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class SpecStore:
    def __init__(self, db_path: str, embed_fn: Callable[[str], list[float]]):
        self.embed_fn = embed_fn
        self._db = lancedb.connect(db_path)
        self._table = None

    def build(self, records: list[dict]) -> None:
        dim = len(self.embed_fn(records[0]["text"])) if records else 1
        rows = [
            {**r, "vector": self.embed_fn(r["text"])}
            for r in records
        ]
        schema = pa.schema([
            pa.field("asin", pa.string()),
            pa.field("field", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])
        self._table = self._db.create_table("specs", data=rows, schema=schema, mode="overwrite")

    def query(self, claim: str, asin: str, k: int = 3) -> list[dict]:
        vec = self.embed_fn(claim)
        results = (
            self._table.search(vec)
            .where(f"asin = '{asin}'", prefilter=True)
            .limit(k)
            .to_list()
        )
        return [{"asin": r["asin"], "field": r["field"], "text": r["text"]} for r in results]

    def grounding_score(self, claims: list[str], asin: str, threshold: float = 0.5) -> float:
        if not claims:
            return 0.0
        hits = 0
        for claim in claims:
            matches = self.query(claim, asin=asin, k=1)
            if matches and _cosine(self.embed_fn(claim), self.embed_fn(matches[0]["text"])) >= threshold:
                hits += 1
        return hits / len(claims)
