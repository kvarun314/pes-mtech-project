"""Phase 2 reflective orchestration: Analyst -> uncertainty gate -> Reflector -> trace/memory.
Analyst   = Phase 1 LLaMA-3-8B+LoRA, gives 5 class probabilities (first-digit-token probe).
Gate      = reflect only when top-2 probability margin < GATE_MARGIN.
Reflector = calibrator (multinomial LR on log-probs + log review length) fit on dev rows; it revises the Analyst's
            call using what dev showed about its systematic errors (adjacent-star confusion). One round: the
            calibrator is deterministic, so a second round would return the same answer (loop cap = 1).
Memory    = every row keeps a trace (first call, margin, reflected?, revised call, changed?)."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression

GATE_MARGIN = 1.01  # set by choose_gate() on dev CV; 1.01 = reflect on every row

def load(prob_path, row_path):
    rows = {json.loads(l)["review_id"]: json.loads(l) for l in open(row_path)}
    out = []
    for l in open(prob_path):
        r = json.loads(l); t = rows[r["review_id"]]["review_text"]
        out.append({"id": r["review_id"], "gt": r["gt_rating"], "probs": r["probs"], "text": t})
    return out

def feats(rs):
    P = np.clip(np.array([r["probs"] for r in rs]), 1e-6, 1)
    return np.c_[np.log(P), np.log1p([len(r["text"]) for r in rs])]

class Reflector:
    def __init__(self, C=10): self.m = LogisticRegression(C=C, max_iter=3000)
    def fit(self, rs): self.m.fit(feats(rs), [r["gt"] - 1 for r in rs]); return self
    def revise(self, rs): return self.m.predict(feats(rs)) + 1

def run(rs, refl, gate):
    """Returns per-row trace dicts."""
    rev = refl.revise(rs); tr = []
    for r, rv in zip(rs, rev):
        p = sorted(r["probs"], reverse=True); margin = p[0] - p[1]
        first = int(np.argmax(r["probs"])) + 1
        reflect = margin < gate
        final = int(rv) if reflect else first
        tr.append({"id": r["id"], "gt": r["gt"], "first": first, "margin": margin, "reflected": reflect, "final": final, "changed": final != first})
    return tr

def choose_gate(dev, cands=(0.2, 0.3, 0.5, 0.8, 1.01), seed=0):
    from sklearn.model_selection import KFold
    best = {}
    for g in cands:
        ok = 0
        for tri, tei in KFold(5, shuffle=True, random_state=seed).split(dev):
            refl = Reflector().fit([dev[i] for i in tri]); te = [dev[i] for i in tei]
            ok += sum(t["final"] == t["gt"] for t in run(te, refl, g))
        best[g] = ok / len(dev)
    return best
