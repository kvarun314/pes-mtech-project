from agentic_sentiment.v3.sc import sc_user, parse_choice, reflect_pick

PHR = {3: [{"phrase": "okay", "display": "okay", "weight": .3}], 4: [{"phrase": "works well", "display": "works well", "weight": .4}]}


def test_prompt_lists_both_candidates_and_their_phrases():
    u = sc_user("T", "It works well, okay price.", (3, 4), PHR)
    assert "3 stars" in u and "4 stars" in u and '"works well"' in u and '"okay"' in u


def test_parse_choice():
    assert parse_choice("4", (3, 4)) == 4 and parse_choice(" 3 stars.", (3, 4)) == 3
    assert parse_choice("5", (3, 4)) is None and parse_choice("either", (3, 4)) is None


def test_reflect_pick_requires_order_consistency():
    assert reflect_pick(lambda s, u: "4", "", "x", (3, 4), PHR) == 4
    seen = []
    def flip(s, u): seen.append(u); return "3" if len(seen) == 1 else "4"   # position-biased model: answers differ
    assert reflect_pick(flip, "", "x", (3, 4), PHR) is None
