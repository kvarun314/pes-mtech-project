"""D1 (declared LOG.md 2026-10-28): DeBERTa-v3-base fine-tuned on evid_train (100K balanced Electronics reviews, title + text) as a third agent.
Runs on Colab (L4 or A100 recommended: bf16). Resumable: per-epoch checkpoint and per-split probability files in the work folder on Drive.
usage (Colab): python colab_deberta.py /content/drive/MyDrive/PES_backups/colab_deberta
writes: ckpt/ (last, best, state.json, optim.pt), train_log.txt, deb3_<split>_probs.jsonl  ({"review_id", "gt_rating", "probs"})"""
import json, os, random, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
W = sys.argv[1]; NAME = "microsoft/deberta-v3-base"; MAXLEN, BS, LR, EPOCHS, SEED = 256, 32, 2e-5, 2, 0
CK = f"{W}/ckpt"; os.makedirs(CK, exist_ok=True); LOG = open(f"{W}/train_log.txt", "a")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SMOKE = bool(os.environ.get("SMOKE"))                                              # local check: 32 train rows, 2 epochs, 8 rows per split
DEV = "cuda" if torch.cuda.is_available() else "cpu"; BF16 = DEV == "cuda" and torch.cuda.is_bf16_supported(); AMP = torch.bfloat16 if BF16 else torch.float16
if SMOKE: BS = 16


def log(*a):
    s = " ".join(map(str, a)); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


txt = lambda r: (r["title"] + " . " if r["title"] else "") + r["review_text"]       # same input as the Evidence agent
load = lambda s: [json.loads(l) for l in open(f"{W}/{s}.jsonl")][:(32 if s == "evid_train" else 8) if SMOKE else None]
tok = AutoTokenizer.from_pretrained(NAME)


def batches(rows, bs, seed=None):
    idx = list(range(len(rows)))
    if seed is not None: random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [rows[j] for j in idx[i:i + bs]]
        yield tok([txt(r) for r in b], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt"), torch.tensor([r["gt_rating"] - 1 for r in b])


def probs(m, rows):
    m.eval(); out = []
    with torch.no_grad(), torch.autocast(DEV, dtype=AMP, enabled=DEV == "cuda"):
        for x, _ in batches(rows, 128):
            out.append(torch.softmax(m(**x.to(DEV)).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


state = json.load(open(f"{CK}/state.json")) if os.path.exists(f"{CK}/state.json") else {"epoch": 0, "dev_acc": [], "best": None}
train, dev = load("evid_train"), load("dev")
log(f"start {time.ctime()} | device {torch.cuda.get_device_name(0) if DEV == 'cuda' else DEV} | smoke {SMOKE} | bf16 {BF16} | train {len(train)} dev {len(dev)} | state {state}")
if state["epoch"] < EPOCHS:
    m = AutoModelForSequenceClassification.from_pretrained(f"{CK}/last" if state["epoch"] else NAME, num_labels=5).float().to(DEV)   # fp32 master weights (newer transformers load the fp16 checkpoint as fp16 -> NaN loss)
    steps = EPOCHS * ((len(train) + BS - 1) // BS)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01); sch = get_linear_schedule_with_warmup(opt, int(.06 * steps), steps)
    scaler = torch.amp.GradScaler(DEV, enabled=DEV == "cuda" and not BF16)
    if state["epoch"]:
        o = torch.load(f"{CK}/optim.pt"); opt.load_state_dict(o["opt"]); sch.load_state_dict(o["sch"]); scaler.load_state_dict(o["scaler"])
    for ep in range(state["epoch"], EPOCHS):
        m.train(); t0 = time.time(); tl = 0.0
        for k, (x, y) in enumerate(batches(train, BS, seed=SEED + ep)):
            with torch.autocast(DEV, dtype=AMP, enabled=DEV == "cuda"):
                loss = m(**x.to(DEV), labels=y.to(DEV)).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(); tl += loss.item()
            if (k + 1) % 500 == 0: log(f"epoch {ep + 1} step {k + 1} loss {tl / (k + 1):.4f} {time.time() - t0:.0f}s")
        acc = float((probs(m, dev).argmax(1) + 1 == np.array([r["gt_rating"] for r in dev])).mean())
        m.save_pretrained(f"{CK}/last"); torch.save({"opt": opt.state_dict(), "sch": sch.state_dict(), "scaler": scaler.state_dict()}, f"{CK}/optim.pt")
        if state["best"] is None or acc > max(state["dev_acc"]): m.save_pretrained(f"{CK}/best"); state["best"] = ep + 1
        state["epoch"] = ep + 1; state["dev_acc"].append(acc); json.dump(state, open(f"{CK}/state.json", "w"))
        log(f"epoch {ep + 1} done: train loss {tl / (k + 1):.4f}, dev exact {acc:.3f}, {time.time() - t0:.0f}s; best epoch {state['best']}")
m = AutoModelForSequenceClassification.from_pretrained(f"{CK}/best").float().to(DEV)
for s in ("dev", "dev2", "natural_dev", "eval", "natural_test"):
    out = f"{W}/deb3_{s}_probs.jsonl"
    if os.path.exists(out): continue
    rows = load(s); P = probs(m, rows)
    with open(out + ".tmp", "w") as f:
        for r, p in zip(rows, P): f.write(json.dumps({"review_id": r["review_id"], "gt_rating": r["gt_rating"], "probs": [float(v) for v in p]}) + "\n")
    os.replace(out + ".tmp", out)
    log(f"{s}: {len(rows)} rows scored (DeBERTa alone exact {float((P.argmax(1) + 1 == np.array([r['gt_rating'] for r in rows])).mean()):.3f})" if s in ("dev", "dev2", "natural_dev") else f"{s}: {len(rows)} rows scored")
log(f"done {time.ctime()}")
