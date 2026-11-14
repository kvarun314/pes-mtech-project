"""Evidence pack: the only material the Explainer may use."""


def build_pack(star, probs, shares, agreed, pos, neg) -> dict:
    return {"star": int(star), "confidence": round(float(probs[star - 1]), 3), "shares": {k: round(float(v), 3) for k, v in shares.items()},
            "agreed": agreed, "for": list(pos[:5]), "against": list(neg[:3])}
