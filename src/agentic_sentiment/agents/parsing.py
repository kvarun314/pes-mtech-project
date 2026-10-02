"""Shared rating parser for every agent's raw LLM output. Handles both the
full 'Sentiment (1-5): N' line and bare leading-digit continuations (CLAUDE.md
Section 5.1 rating-parser nuance)."""

import re

_LABELED = re.compile(r"Sentiment\s*\(1-5\)\s*:\s*([1-5])\b")
_LEADING = re.compile(r"^\s*([1-5])\b")
# _ANYWHERE second part uses negative lookbehind/lookahead to exclude digits
# that are part of the "(1-5)" scale indicator pattern
_ANYWHERE = re.compile(r"\b([1-5])\s*(?:/|out of)\s*5\b|(?<![(\-])\b([1-5])\b(?![)\-])")


def parse_rating(text: str) -> int | None:
    if not text:
        return None
    for pattern in (_LABELED, _LEADING):
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    m = _ANYWHERE.search(text)
    if m:
        return int(m.group(1) or m.group(2))
    return None
