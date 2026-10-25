"""Token helpers shared by the Evidence agent, Critic and deletion test.
Tokenisation matches sklearn's default token_pattern on lowercased text, so phrases
from the TF-IDF vocabulary can be checked against any text."""
import re

TOKEN = re.compile(r"(?u)\b\w\w+\b")
QUOTE = re.compile(r'["“”]([^"“”]+)["“”]')


def tokens(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def contains(hay: list[str], phrase: str) -> bool:
    p = tokens(phrase); n = len(p)
    return n > 0 and any(hay[i:i + n] == p for i in range(len(hay) - n + 1))


def display_span(text: str, phrase: str) -> str:
    """Original-text span for a vocabulary phrase ("doesn work" -> "doesn't work"); the phrase itself if absent."""
    p = tokens(phrase)
    if not p:
        return phrase
    m = re.search(r"(?<!\w)" + r"\W+(?:\w\W+)?".join(map(re.escape, p)) + r"(?!\w)", text, re.I)
    return m.group(0) if m else phrase


def quoted(s: str) -> list[str]:
    return QUOTE.findall(s)


def token_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in TOKEN.finditer(text)]


def phrase_token_idx(text: str, phrase: str) -> list[int] | None:
    """Indices (into token_spans(text)) of the first contiguous occurrence of phrase, or None."""
    hay, p = tokens(text), tokens(phrase); n = len(p)
    for i in range(len(hay) - n + 1):
        if n and hay[i:i + n] == p:
            return list(range(i, i + n))
    return None


def drop_tokens(text: str, idx: set[int]) -> str:
    """Remove the chosen word tokens, keep every other character (same mechanism for cited and random deletion)."""
    out, last = [], 0
    for i, (a, b) in enumerate(token_spans(text)):
        if i in idx:
            out.append(text[last:a]); last = b
    out.append(text[last:])
    return "".join(out)
