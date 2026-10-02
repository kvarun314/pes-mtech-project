"""Critic dissonance scoring.

Primary formula (mission spec): D = Norm(H - (Ev + Gf)).
H, Ev, Gf are each normalized to [0, 1] (rating 1-5 -> 0.0-1.0, or a
grounding score already in [0, 1]). Ev and Gf are averaged before
subtracting from H so all three terms share the same [0, 1] scale, then
Norm(x) = clamp(|x|, 0, 1). This keeps D in [0, 1], comparable to the old
vote-stdev formula's [0, ~0.5] range, at the same DISSONANCE_THRESHOLD.

The Run 1 formula (D = stdev(votes) / 2) is kept as `dissonance_stdev` for
the ablation that reproduces the original run."""

import statistics

DISSONANCE_THRESHOLD = 0.4


def normalize_rating(rating: int) -> float:
    return (rating - 1) / 4


def dissonance_norm(h: float, ev: float, gf: float) -> float:
    raw = h - (ev + gf) / 2
    return min(max(abs(raw), 0.0), 1.0)


def dissonance_stdev(votes: list[int]) -> float:
    if len(votes) < 2:
        return 0.0
    return statistics.stdev(votes) / 2
