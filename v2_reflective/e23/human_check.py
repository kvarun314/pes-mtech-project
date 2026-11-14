"""Human check of 50 explanations (declared LOG.md 2026-10-28). Blind: the sheet hides the Critic's verdict.
usage: human_check.py make    -> b2/human_check_50.csv (review text; gitignored) + b2/human_check_key.json
       human_check.py score   -> agreement of the human mark with the Critic (prints; tee to human_check.out)"""
import csv, json, sys, numpy as np
D = "v2_reflective/e23/"; SHEET, KEY = D + "b2/human_check_50.csv", D + "b2/human_check_key.json"
if sys.argv[1] == "make":
    T = {r["review_id"]: r for s in ("eval", "natural_test") for r in map(json.loads, open(f"{D}{s}.jsonl"))}
    B = [r for s in ("eval", "natural_test") for r in map(json.loads, open(f"{D}b2/{s}.jsonl"))]
    pick = np.random.default_rng(0).choice(len(B), 50, replace=False)
    with open(SHEET, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n", "review_id", "rating_given_by_system", "review_title", "review_text", "explanation",
                                       "faithful (Y/N): every claim is in the review and the rating is stated correctly", "note"])
        for n, i in enumerate(pick, 1):
            b = B[i]; t = T[b["review_id"]]; w.writerow([n, b["review_id"], b["star"], t["title"], t["review_text"], b["drafts"][-1], "", ""])
    json.dump({B[i]["review_id"]: bool(B[i]["faithful"]) for i in pick}, open(KEY, "w")); print("wrote", SHEET)
else:
    K = json.load(open(KEY)); rows = [r for r in csv.DictReader(open(SHEET)) if r[[k for k in r if k.startswith("faithful")][0]].strip()]
    h = np.array([r[[k for k in r if k.startswith("faithful")][0]].strip().upper().startswith("Y") for r in rows]); c = np.array([K[r["review_id"]] for r in rows])
    print(f"marked {len(rows)}/50  human faithful {h.mean():.3f}  Critic faithful {c.mean():.3f}  agreement {(h == c).mean():.3f}  "
          f"Critic pass but human fail {int((c & ~h).sum())}  Critic fail but human pass {int((~c & h).sum())}")
