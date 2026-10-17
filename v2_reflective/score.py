"""Score probe output under the paper's protocol. usage: score.py probs.jsonl"""
import json, sys
from collections import Counter
from sklearn.metrics import f1_score, precision_recall_fscore_support
R = [json.loads(l) for l in open(sys.argv[1])]
gt = [r["gt_rating"] for r in R]
pr = [max(range(5), key=lambda k: r["probs"][k]) + 1 for r in R]
n = len(R)
pol = lambda x: 0 if x <= 2 else (1 if x == 3 else 2)
acc = lambda a, b: sum(x == y for x, y in zip(a, b)) / len(a)
print(f"n={n} dist={dict(sorted(Counter(gt).items()))}")
print(f"5-class exact      {acc(pr, gt):.2%}   (always-5 baseline {acc([5]*n, gt):.2%})")
print(f"within +-1         {sum(abs(p-g)<=1 for p,g in zip(pr,gt))/n:.2%}   MAE {sum(abs(p-g) for p,g in zip(pr,gt))/n:.3f}")
P, G = [pol(x) for x in pr], [pol(x) for x in gt]
print(f"polarity 3-class   {acc(P, G):.2%}   (always-positive baseline {acc([2]*n, G):.2%})")
keep = [i for i in range(n) if gt[i] != 3]
print(f"binary (3 dropped) {sum((pr[i] >= 4) == (gt[i] >= 4) for i in keep)/len(keep):.2%}   (always-positive {sum(gt[i]>=4 for i in keep)/len(keep):.2%})")
for name, a, b in (("5-class", pr, gt), ("polarity", P, G)):
    p, r, f, _ = precision_recall_fscore_support(b, a, average="weighted", zero_division=0)
    print(f"{name:9s} weighted P/R/F1 {p:.3f}/{r:.3f}/{f:.3f}   macro-F1 {f1_score(b, a, average='macro', zero_division=0):.3f}")
