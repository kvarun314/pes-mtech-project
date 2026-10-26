"""Explainer: Llama-3-8B-Instruct 4-bit writes a 1-2 sentence rationale from the evidence pack only."""
from agentic_sentiment.v3 import paths as PT

EXPLAIN_SYSTEM = ("You explain the decision of a review-rating system to a non-expert. Use only the evidence you are given. "
                  "Reply with 1 or 2 sentences and nothing else.")


def explainer_user(pack, title, text, previous=None, feedback=()):
    f = lambda ps: ", ".join(f'"{p["display"]}" ({p["weight"]:+.2f})' for p in ps) or "none"
    s, sh = pack["star"], pack["shares"]
    lines = [f"Review title: {title}", f"Review: {text[:1200]}", "",
             f"Final rating: {s} stars (confidence {pack['confidence']:.0%}).",
             f"Phrases supporting {s} stars: {f(pack['for'])}", f"Phrases pointing away from {s} stars (they argue against this rating; never call them support): {f(pack['against'])}",
             f"Share of the decision: Evidence agent {sh['evidence']:.0%}, LLaMA probe agent {sh['probe']:.0%}.",
             f"Agents agreed: {'yes' if pack['agreed'] is not False else 'no'}.", "",
             ("Start with the sentence \"The two agents disagreed on this review.\" Then write" if pack["agreed"] is False else "Write")
             + f" at most 45 words that state \"{s} stars\", quote at least two supporting phrases in double quotes exactly as written above, "
             + "and add nothing that is not in the review."]
    if previous:
        lines += ["", f"Previous explanation: {previous}", "Fix these problems:"] + [f"- {x}" for x in feedback]
    return "\n".join(lines)


def make_chat_fn(model, tok, max_tokens=90):
    from mlx_lm import generate
    def chat(system, user):
        prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()
    return chat


def load_instruct():
    from mlx_lm import load
    return load(PT.INSTRUCT)
