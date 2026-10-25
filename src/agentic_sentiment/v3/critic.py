"""Critic for explanations: 4 rule checks + 1 LLM check, each failure turned into specific feedback for the Explainer."""
import re
from agentic_sentiment.v3.text import tokens, contains, quoted

STAR = re.compile(r"\b([1-5])(?:\s*|-)(?:stars?|★)", re.I)
DISAGREE = ("disagree", "differ", "split", "conflict")
JUDGE_SYSTEM = ("You check an explanation against a product review. Reply exactly NO if every claim in the explanation is supported by "
                'the review. Otherwise reply YES: "<the unsupported words>".')


def judge_user(rationale: str, title: str, text: str) -> str:
    return f"Review title: {title}\nReview: {text[:1200]}\n\nExplanation: {rationale}\n\nDoes the explanation claim anything the review does not say?"


def parse_judge(answer: str):
    a = answer.strip()
    if a.upper().startswith("NO"):
        return True, None
    if a.upper().startswith("YES"):
        q = quoted(a)
        return False, (q[0] if q else a[3:].strip(" :"))
    return None, None


def rule_checks(rationale: str, pack: dict, title: str, text: str) -> list[str]:
    fb, s, rt = [], pack["star"], tokens(rationale)
    stars = STAR.findall(rationale)                                                       # 1: stated star = final star
    if not stars:
        fb.append(f"State the final rating as '{s} stars'.")
    for x in sorted({int(x) for x in stars} - {s}):
        fb.append(f"You said {x} stars; the final rating is {s} stars.")
    review, phrases = tokens((title + " . " if title else "") + text), [p["phrase"] for p in pack["for"] + pack["against"]]
    for q in quoted(rationale):                                                           # 2: quotes occur in review and pack
        if not contains(review, q):
            fb.append(f"'{q}' does not occur in the review.")
        elif not any(contains(tokens(q), p) or contains(tokens(p), q) for p in phrases):
            fb.append(f"'{q}' is not one of the evidence phrases.")
    top = pack["for"][:3]; need = min(2, len(top))                                        # 3: cover top phrases
    if sum(contains(rt, p["phrase"]) for p in top) < need:
        fb.append(f"Quote at least {need} of these phrases: " + ", ".join(f'"{p["display"]}"' for p in top))
    if pack["agreed"] is False and not any(w in rationale.lower() for w in DISAGREE):    # 4: disagreement mentioned
        fb.append("The agents disagreed; say so.")
    return fb


def _check_no(f: str) -> int:
    if f.startswith("Quote"): return 3
    if f.endswith("does not occur in the review.") or f.endswith("is not one of the evidence phrases."): return 2
    if f == "The agents disagreed; say so.": return 4
    return 1


def critique(rationale, pack, title, text, judge_fn):
    fb = rule_checks(rationale, pack, title, text); info = {"failed_checks": sorted({_check_no(f) for f in fb}), "judge_unparsed": False}
    if not fb:                                                                            # 5: LLM check, only when the rules pass
        ok, span = parse_judge(judge_fn(JUDGE_SYSTEM, judge_user(rationale, title, text)))
        if ok is None:
            info["judge_unparsed"] = True                                                 # counted and reported; treated as pass
        elif not ok:
            fb.append(f"Remove the unsupported claim: '{span}'."); info["failed_checks"] = [5]
    return fb, info
