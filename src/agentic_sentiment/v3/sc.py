"""SC2 reflective re-rating: on reviews where the main stack is unsure, an instruction-tuned LLM chooses between the stack's top-2 stars
using the review and the evidence phrases for each candidate. Asked twice with the order swapped; inconsistent answers abstain."""
import re

SC_SYSTEM = ("You are a careful reviewer of product-review ratings. Decide which of two star ratings the review deserves. "
             "Reply with the digit only.")


def sc_user(title, text, cand, phr):
    f = lambda ps: ", ".join(f'"{p["display"]}"' for p in ps) or "none"
    a, b = cand
    return (f"Review title: {title}\nReview: {text[:1200]}\n\nCandidate ratings: {a} stars or {b} stars.\n"
            f"Phrases supporting {a} stars: {f(phr[a])}\nPhrases supporting {b} stars: {f(phr[b])}\n\nWhich rating fits the review better: {a} or {b}?")


def parse_choice(answer, cand):
    m = re.search(r"[1-5]", answer)
    return int(m.group(0)) if m and int(m.group(0)) in cand else None


def reflect_pick(chat, title, text, cand, phr):
    first = parse_choice(chat(SC_SYSTEM, sc_user(title, text, cand, phr)), cand)
    second = parse_choice(chat(SC_SYSTEM, sc_user(title, text, cand[::-1], phr)), cand)
    return first if first is not None and first == second else None
