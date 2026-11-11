"""SH3 (adopted LOG.md 2026-11-10): fit the three-agent Reflector (probe P3 + Evidence E6b + DeBERTa seed 0) on dev2 with the same recipe as
persist_agents.py and save it as agents/reflector3.joblib. Must reproduce the D1 readout: eval 0.675, natural_test 0.780.
usage: persist_sh3.py   (tee to persist_sh3.out)"""
import json, sys, joblib, numpy as np
sys.path.insert(0, "src")
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.reflector import fit_reflector, feats, load_agents
order = [json.loads(l)["review_id"] for l in open(PT.D + "p1/probs_R_dev2.jsonl")]                 # row order as persist_agents.py / freeze_tables.py
ids, P, E, B, y = load_agents("dev2"); k = [ids.index(i) for i in order]
r = fit_reflector(P[k], E[k], y[k], B[k]); joblib.dump({"full": r.full, "fallback": r.fallback}, PT.REFLECTOR_JOBLIB)
want = {"eval": 0.675, "natural_test": 0.780}
for s, w in want.items():
    _, P, E, B, y = load_agents(s); acc = float((r.full.predict(feats(P, E, B)) + 1 == y).mean())
    print(f"{s}: SH3 exact {acc:.4f} (D1 readout {w})"); assert round(acc, 3) == w, s
print("saved", PT.REFLECTOR_JOBLIB)
