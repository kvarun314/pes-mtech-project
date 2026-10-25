from agentic_sentiment.v3.pack import build_pack
from agentic_sentiment.v3.critic import rule_checks, parse_judge, critique

TEXT = "Great screen, but the battery dies fast and it doesn't charge."
POS = [{"phrase": "great screen", "display": "Great screen", "weight": .4}, {"phrase": "screen", "display": "screen", "weight": .2},
       {"phrase": "great", "display": "Great", "weight": .1}]
NEG = [{"phrase": "battery dies", "display": "battery dies", "weight": -.5}]
PACK = build_pack(3, [.1, .1, .6, .1, .1], {"probe": .4, "evidence": .6}, False, POS, NEG)


def test_pack_shape():
    assert PACK["star"] == 3 and PACK["confidence"] == 0.6 and PACK["agreed"] is False
    assert len(PACK["for"]) == 3 and PACK["against"][0]["phrase"] == "battery dies"


def test_faithful_rationale_passes_rules():
    r = 'Rated 3 stars: "Great screen" and "screen" are positive but "battery dies" is negative, and the agents disagreed.'
    assert rule_checks(r, PACK, "", TEXT) == []


def test_each_rule_gives_specific_feedback():
    fb = rule_checks('Rated 4 stars because of "amazing sound".', PACK, "", TEXT)
    assert any("You said 4 stars; the final rating is 3 stars." == f for f in fb)
    assert any("'amazing sound' does not occur in the review." == f for f in fb)
    assert any(f.startswith("Quote at least 2 of these phrases") for f in fb)
    assert "The agents disagreed; say so." in fb
    assert rule_checks('"Great screen" and "screen" mixed.', PACK, "", TEXT)[0] == "State the final rating as '3 stars'."


def test_parse_judge():
    assert parse_judge("NO") == (True, None)
    assert parse_judge('YES: "waterproof"') == (False, "waterproof")
    assert parse_judge("maybe") == (None, None)


def test_critique_calls_judge_only_when_rules_pass():
    calls = []
    def judge(system, user): calls.append(user); return 'YES: "lasts a week"'
    fb, info = critique('Rated 4 stars.', PACK, "", TEXT, judge)
    assert calls == [] and 1 in info["failed_checks"]
    good = 'Rated 3 stars: "Great screen" and "screen" but "battery dies", and the agents disagreed.'
    fb, info = critique(good, PACK, "", TEXT, judge)
    assert fb == ["Remove the unsupported claim: 'lasts a week'."] and info["failed_checks"] == [5]
