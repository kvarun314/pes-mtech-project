# Full Phase 2 (Agentic) + Phase 1 Retrain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable `agentic_sentiment` package implementing the full mission-spec Phase 2 (real LangGraph graph, spec-store RAG, Norm(H-(Ev+Gf)) dissonance), a full Phase 1 LoRA retrain, classical + XLNet baselines, and a balanced Electronics eval with ablations — every training/eval run checkpointed to Drive so best params survive a Colab disconnect — then thin Colab notebooks that call this package.

**Architecture:** One local-testable Python package (`src/agentic_sentiment/`) holds all logic; a `RunDir` helper standardizes Drive checkpointing/config dumps for every long-running job (LoRA train, XLNet train, Phase 2 eval). Four thin Colab notebooks import the package and call one function each. Everything that doesn't need a GPU (parsing, dissonance math, spec-store query, graph routing, checkpoint/resume, classical baselines) has a real local pytest test using a stub LLM; GPU-only paths (LoRA/XLNet training, live LLaMA calls) are validated on Colab, not locally.

**Tech Stack:** Python 3.11, pytest, langgraph, lancedb, sentence-transformers, scikit-learn, transformers/peft/datasets (Colab-only), pandas.

## Global Constraints

- Dataset category for Phase 2: **Amazon Reviews 2023, Electronics** (per spec; Run 1 used All_Beauty).
- Eval set is **balanced per star rating**, not skewed to 5-star like Run 1.
- Phase 1 retrain uses the paper-closer config already proven in Run 1: LoRA r=16, alpha=128, target q/k/v/o over 8 layers, 5 epochs, max_samples=4000 (`colab/llama_sentiment_baseline_train.ipynb` cell 16).
- Dissonance: `D = Norm(H - (Ev + Gf))` as primary; the old vote-stdev formula is kept only as an ablation variant, never the default.
- Every training run (LoRA, XLNet) must write to a Google Drive run directory: `config.json` (all hyperparameters + seed), HF `checkpoints/`, `best_adapter/` (or best model dir), `trainer_state.json`, `metrics.json`, `BEST.md` — written incrementally during training, not only at the end, so a disconnect never loses the best checkpoint.
- No new dependency when stdlib/an already-used library covers it (ponytail rung 2–5). Reuse the existing `llama_sentiment_baseline` package for Phase 1 data/LoRA code instead of rewriting it.
- `docs/PROGRESS.md` gets one line appended after every task's commit (see Task 16).
- Do not touch `paper/main.tex` or `paper/conference/ConferencePaper.tex` numbers until real `results/run2/` data exists (no invented numbers).

---

### Task 1: Package scaffold + `RunDir` (Drive checkpointing)

**Files:**
- Create: `pes-mtech-project/pyproject.toml`
- Create: `pes-mtech-project/src/agentic_sentiment/__init__.py`
- Create: `pes-mtech-project/src/agentic_sentiment/train/__init__.py`
- Create: `pes-mtech-project/src/agentic_sentiment/train/run_dir.py`
- Test: `pes-mtech-project/tests/test_run_dir.py`

**Interfaces:**
- Produces: `RunDir(base_dir: str, run_id: str | None = None)` with `.path: Path`, `.write_config(config: dict) -> None`, `.write_best(step: int, metric_name: str, value: float) -> None`, `.checkpoint_callback() -> transformers.TrainerCallback` (import deferred inside the method so this module imports fine without `transformers` installed).

- [ ] **Step 1: Create the package skeleton**

`pes-mtech-project/pyproject.toml`:
```toml
[project]
name = "agentic-sentiment"
version = "0.1.0"
description = "Phase 2 agentic sentiment pipeline (pes-mtech-project)"
requires-python = ">=3.10"
dependencies = [
    "pandas>=1.5.0",
    "scikit-learn>=1.2.0",
    "langgraph>=0.2.0",
    "lancedb>=0.13.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0"]
colab = [
    "torch>=2.0.0",
    "transformers>=4.36.0",
    "peft>=0.7.0",
    "datasets>=2.14.0",
    "accelerate>=0.25.0",
    "textblob>=0.17.0",
    "sentence-transformers>=3.0.0",
    "xgboost",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentic_sentiment"]
```

`pes-mtech-project/src/agentic_sentiment/__init__.py`: empty file.
`pes-mtech-project/src/agentic_sentiment/train/__init__.py`: empty file.

- [ ] **Step 2: Write the failing test**

`pes-mtech-project/tests/test_run_dir.py`:
```python
import json
from pathlib import Path

from agentic_sentiment.train.run_dir import RunDir


def test_write_config_creates_json(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="2026-10-02_test")
    rd.write_config({"lora_r": 16, "lora_alpha": 128, "epochs": 5, "seed": 42})

    cfg_path = rd.path / "config.json"
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text())["lora_r"] == 16


def test_write_best_updates_markdown(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="2026-10-02_test")
    rd.write_best(step=100, metric_name="eval_loss", value=0.42)
    rd.write_best(step=200, metric_name="eval_loss", value=0.31)  # improves (lower)
    rd.write_best(step=300, metric_name="eval_loss", value=0.50)  # worse, ignored

    text = (rd.path / "BEST.md").read_text()
    assert "step 200" in text
    assert "0.31" in text
    assert "step 300" not in text


def test_run_id_defaults_to_timestamp(tmp_path):
    rd1 = RunDir(base_dir=str(tmp_path))
    rd2 = RunDir(base_dir=str(tmp_path))
    assert rd1.path != rd2.path
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd pes-mtech-project && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" -q && .venv/bin/pytest tests/test_run_dir.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_sentiment.train.run_dir'`

- [ ] **Step 4: Implement `RunDir`**

`pes-mtech-project/src/agentic_sentiment/train/run_dir.py`:
```python
"""Drive-backed run directory: config dump, best-checkpoint tracking, and a
TrainerCallback that copies every HF checkpoint to Drive as it is saved, so a
Colab disconnect never loses the best adapter or its hyperparameters."""

import json
import time
from pathlib import Path


class RunDir:
    def __init__(self, base_dir: str, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H%M%S")
        self.path = Path(base_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "checkpoints").mkdir(exist_ok=True)
        self._best_value: float | None = None

    def write_config(self, config: dict) -> None:
        (self.path / "config.json").write_text(json.dumps(config, indent=2, default=str))

    def write_best(self, step: int, metric_name: str, value: float, greater_is_better: bool = False) -> None:
        is_better = (
            self._best_value is None
            or (value > self._best_value if greater_is_better else value < self._best_value)
        )
        if not is_better:
            return
        self._best_value = value
        (self.path / "BEST.md").write_text(
            f"# Best checkpoint\n\nstep {step}\n{metric_name} = {value}\n"
        )

    def checkpoint_callback(self):
        """Returns a transformers.TrainerCallback that copies each new HF
        checkpoint dir into this RunDir's checkpoints/ on Drive, and updates
        BEST.md when eval_loss improves. Import of transformers is deferred
        so this module has no hard dependency on it."""
        import shutil
        from transformers import TrainerCallback

        run_dir = self

        class DriveCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if src.is_dir():
                    dst = run_dir.path / "checkpoints" / src.name
                    shutil.copytree(src, dst, dirs_exist_ok=True)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and "eval_loss" in metrics:
                    run_dir.write_best(state.global_step, "eval_loss", metrics["eval_loss"])

        return DriveCheckpointCallback()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_run_dir.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
cd pes-mtech-project
git add pyproject.toml src/agentic_sentiment/__init__.py src/agentic_sentiment/train/__init__.py src/agentic_sentiment/train/run_dir.py tests/test_run_dir.py .venv
echo ".venv/" >> .gitignore
git rm -r --cached .venv 2>/dev/null; git add .gitignore
git commit -m "feat: agentic_sentiment package scaffold + Drive-checkpointing RunDir"
```

---

### Task 2: Vendor Phase 1 package + wire Drive checkpointing

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/phase1/` (copied from `llama_sentiment_baseline/src/llama_sentiment_baseline/`)
- Create: `pes-mtech-project/tests/phase1/` (copied from `llama_sentiment_baseline/tests/`)
- Create: `pes-mtech-project/src/agentic_sentiment/train/phase1_train.py`
- Test: existing copied tests + `pes-mtech-project/tests/test_phase1_train.py`

**Interfaces:**
- Consumes: `agentic_sentiment.train.run_dir.RunDir` (Task 1).
- Produces: `run_training(data_path: str, run_dir: RunDir, overrides: dict | None = None) -> dict` (returns the final `trainer.evaluate()` metrics dict). Also re-exports `DataConfig`, `ModelConfig`, `TrainingConfig`, `build_sft_dataset`, `load_model_and_tokenizer` from `agentic_sentiment.phase1`.

- [ ] **Step 1: Copy the proven Phase 1 code verbatim**

```bash
cd pes-mtech-project
mkdir -p src/agentic_sentiment/phase1 tests/phase1
cp -R ../llama_sentiment_baseline/src/llama_sentiment_baseline/* src/agentic_sentiment/phase1/
cp ../llama_sentiment_baseline/tests/*.py tests/phase1/
sed -i '' 's/from llama_sentiment_baseline/from agentic_sentiment.phase1/g; s/import llama_sentiment_baseline/import agentic_sentiment.phase1 as llama_sentiment_baseline/g' \
  $(grep -rl "llama_sentiment_baseline" src/agentic_sentiment/phase1 tests/phase1)
```

- [ ] **Step 2: Run the copied tests to confirm the vendor copy works untouched**

Run: `.venv/bin/pip install -e . -q && .venv/bin/pytest tests/phase1 -v`
Expected: all pass (same tests that passed in the source repo — `test_preprocessing.py`, `test_config.py`)

- [ ] **Step 3: Write the failing test for the training wrapper**

`pes-mtech-project/tests/test_phase1_train.py`:
```python
from unittest.mock import MagicMock, patch

from agentic_sentiment.train.phase1_train import build_training_config
from agentic_sentiment.train.run_dir import RunDir


def test_build_training_config_applies_paper_closer_overrides():
    cfg = build_training_config()
    assert cfg.max_samples == 4000
    assert cfg.num_train_epochs == 5


def test_run_training_writes_config_json(tmp_path):
    from agentic_sentiment.train import phase1_train

    run_dir = RunDir(base_dir=str(tmp_path), run_id="run")
    fake_trainer = MagicMock()
    fake_trainer.evaluate.return_value = {"eval_loss": 0.5}

    with patch.object(phase1_train, "_build_trainer", return_value=fake_trainer), \
         patch.object(phase1_train, "build_sft_dataset", return_value=([], [])), \
         patch.object(phase1_train, "load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
        metrics = phase1_train.run_training("dummy.csv", run_dir)

    assert metrics == {"eval_loss": 0.5}
    assert (run_dir.path / "config.json").exists()
    fake_trainer.train.assert_called_once()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_phase1_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.train.phase1_train'`

- [ ] **Step 5: Implement the training wrapper**

`pes-mtech-project/src/agentic_sentiment/train/phase1_train.py`:
```python
"""Phase 1 LoRA training, wired to Drive checkpointing via RunDir. Reuses
the Phase 1 data/model code verbatim from agentic_sentiment.phase1 (vendored
from llama_sentiment_baseline) — only the Trainer plumbing + Drive callback
is new here."""

import dataclasses

from agentic_sentiment.phase1.config import DataConfig, ModelConfig, TrainingConfig
from agentic_sentiment.phase1.data.dataset import build_sft_dataset
from agentic_sentiment.phase1.models.lora import load_model_and_tokenizer

from agentic_sentiment.train.run_dir import RunDir

# Paper-closer config proven in Run 1 (colab/llama_sentiment_baseline_train.ipynb cell 16).
PAPER_CLOSER_OVERRIDES = dict(
    lora_r=16,
    lora_alpha=128,
    trainable_layers=8,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
TRAINING_OVERRIDES = dict(
    max_samples=4000,
    num_train_epochs=5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    max_seq_length=1024,
)


def build_model_config() -> ModelConfig:
    return dataclasses.replace(ModelConfig(), **PAPER_CLOSER_OVERRIDES)


def build_training_config() -> TrainingConfig:
    return dataclasses.replace(TrainingConfig(), **TRAINING_OVERRIDES)


def _build_trainer(model, tokenizer, train_ds, eval_ds, training_cfg, run_dir: RunDir):
    from transformers import Trainer, TrainingArguments
    from transformers.data.data_collator import DataCollatorForSeq2Seq

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding="max_length",
        max_length=training_cfg.max_seq_length, label_pad_token_id=-100,
    )
    args = TrainingArguments(
        output_dir=training_cfg.output_dir,
        num_train_epochs=training_cfg.num_train_epochs,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        eval_strategy=training_cfg.evaluation_strategy,
        eval_steps=training_cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=training_cfg.fp16,
        seed=training_cfg.seed,
        report_to="none",
    )
    return Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=data_collator, callbacks=[run_dir.checkpoint_callback()],
    )


def run_training(data_path: str, run_dir: RunDir, overrides: dict | None = None) -> dict:
    model_cfg = build_model_config()
    training_cfg = build_training_config()
    data_cfg = DataConfig(stratified_max_total=10000)
    if overrides:
        training_cfg = dataclasses.replace(training_cfg, **overrides)

    run_dir.write_config({
        "model": dataclasses.asdict(model_cfg),
        "training": dataclasses.asdict(training_cfg),
        "data": dataclasses.asdict(data_cfg),
        "data_path": data_path,
    })

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    train_ds, eval_ds = build_sft_dataset(data_path, tokenizer, data_cfg, training_cfg, seed=training_cfg.seed)

    trainer = _build_trainer(model, tokenizer, train_ds, eval_ds, training_cfg, run_dir)
    trainer.train()
    metrics = trainer.evaluate()

    best_dir = run_dir.path / "best_adapter"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (run_dir.path / "metrics.json").write_text(__import__("json").dumps(metrics, indent=2))
    return metrics
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_phase1_train.py -v`
Expected: 3 passed (config test + the 2 in Step 3)

- [ ] **Step 7: Commit**

```bash
git add src/agentic_sentiment/phase1 tests/phase1 src/agentic_sentiment/train/phase1_train.py tests/test_phase1_train.py
git commit -m "feat: vendor Phase 1 package, add Drive-checkpointed training wrapper"
```

---

### Task 3: Amazon 2023 Electronics loader + balanced eval slice + spec records

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/data/__init__.py`
- Create: `pes-mtech-project/src/agentic_sentiment/data/amazon2023.py`
- Test: `pes-mtech-project/tests/test_amazon2023.py`
- Test fixtures: `pes-mtech-project/tests/fixtures/electronics_reviews.jsonl`, `pes-mtech-project/tests/fixtures/electronics_meta.jsonl`

**Interfaces:**
- Produces: `load_balanced_slice(reviews_rows: list[dict], meta_by_asin: dict[str, dict], n_per_rating: int, seed: int = 42, min_review_chars: int = 20) -> list[dict]` where each returned row has keys `review_id, asin, rating, review_text, meta_title, meta_description, meta_details, image_url`. `build_spec_records(meta_by_asin: dict[str, dict]) -> list[dict]` with keys `asin, field, text` (one record per spec-bearing metadata field, for the spec store in Task 6).

- [ ] **Step 1: Write fixtures**

`pes-mtech-project/tests/fixtures/electronics_reviews.jsonl` (10 lines, ratings 1–5 x2, varying lengths — write with a short Python one-liner, not by hand, to keep it exact):
```bash
mkdir -p pes-mtech-project/tests/fixtures
python3 - <<'EOF'
import json
rows = []
for i, rating in enumerate([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]):
    rows.append({
        "parent_asin": f"B{i:09d}",
        "rating": rating,
        "text": f"Review number {i} with enough characters to pass the min length filter.",
        "images": [{"large_image_url": f"http://example.com/{i}.jpg"}] if i % 2 == 0 else [],
    })
with open("pes-mtech-project/tests/fixtures/electronics_reviews.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
EOF
```

`pes-mtech-project/tests/fixtures/electronics_meta.jsonl`:
```bash
python3 - <<'EOF'
import json
rows = []
for i in range(10):
    rows.append({
        "parent_asin": f"B{i:09d}",
        "title": f"Widget {i} Pro",
        "description": [f"A great widget, model {i}."],
        "details": {"Connector Type": "USB-C", "Battery Life": "10 hours"},
    })
with open("pes-mtech-project/tests/fixtures/electronics_meta.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
EOF
```

- [ ] **Step 2: Write the failing test**

`pes-mtech-project/tests/test_amazon2023.py`:
```python
import json
from pathlib import Path

from agentic_sentiment.data.amazon2023 import build_spec_records, load_balanced_slice

FIXTURES = Path(__file__).parent / "fixtures"


def _load_jsonl(path):
    return [json.loads(line) for line in path.open()]


def test_load_balanced_slice_has_equal_counts_per_rating():
    reviews = _load_jsonl(FIXTURES / "electronics_reviews.jsonl")
    meta = {m["parent_asin"]: m for m in _load_jsonl(FIXTURES / "electronics_meta.jsonl")}

    rows = load_balanced_slice(reviews, meta, n_per_rating=2, seed=42)

    assert len(rows) == 10
    counts = {}
    for r in rows:
        counts[r["rating"]] = counts.get(r["rating"], 0) + 1
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert all(r["meta_title"] for r in rows)


def test_load_balanced_slice_caps_at_available_rows():
    reviews = _load_jsonl(FIXTURES / "electronics_reviews.jsonl")
    meta = {m["parent_asin"]: m for m in _load_jsonl(FIXTURES / "electronics_meta.jsonl")}

    rows = load_balanced_slice(reviews, meta, n_per_rating=5, seed=42)  # only 2 available per rating
    assert len(rows) == 10


def test_build_spec_records_one_per_field():
    meta = {m["parent_asin"]: m for m in _load_jsonl(FIXTURES / "electronics_meta.jsonl")}
    records = build_spec_records(meta)

    assert all({"asin", "field", "text"} <= r.keys() for r in records)
    usb_c = [r for r in records if "USB-C" in r["text"]]
    assert len(usb_c) == 10  # one per product's "Connector Type" detail
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_amazon2023.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.data'`

- [ ] **Step 4: Implement the loader**

`pes-mtech-project/src/agentic_sentiment/data/__init__.py`: empty file.

`pes-mtech-project/src/agentic_sentiment/data/amazon2023.py`:
```python
"""Amazon Reviews 2023 (Electronics) loading: balanced per-rating eval
slice + product-spec records for the RAG spec store."""

import random


def load_balanced_slice(
    reviews_rows: list[dict],
    meta_by_asin: dict[str, dict],
    n_per_rating: int,
    seed: int = 42,
    min_review_chars: int = 20,
) -> list[dict]:
    """Filter to rows with metadata + long-enough text, then sample up to
    n_per_rating rows per star rating 1-5 (fewer if not enough exist)."""
    by_rating: dict[int, list[dict]] = {r: [] for r in range(1, 6)}
    for row in reviews_rows:
        asin = row.get("parent_asin")
        text = row.get("text", "") or ""
        rating = int(row.get("rating", 0))
        if asin not in meta_by_asin or len(text) < min_review_chars or rating not in by_rating:
            continue
        meta = meta_by_asin[asin]
        images = row.get("images") or []
        by_rating[rating].append({
            "review_id": f"{asin}_{rating}_{len(by_rating[rating])}",
            "asin": asin,
            "rating": rating,
            "review_text": text,
            "meta_title": meta.get("title", ""),
            "meta_description": " ".join(meta.get("description", []) or []),
            "meta_details": meta.get("details", {}) or {},
            "image_url": images[0].get("large_image_url") if images else None,
        })

    rng = random.Random(seed)
    out = []
    for rating, rows in by_rating.items():
        rng.shuffle(rows)
        out.extend(rows[:n_per_rating])
    rng.shuffle(out)
    return out


def build_spec_records(meta_by_asin: dict[str, dict]) -> list[dict]:
    """One record per spec-bearing metadata field, for SpecStore.build()."""
    records = []
    for asin, meta in meta_by_asin.items():
        if meta.get("title"):
            records.append({"asin": asin, "field": "title", "text": meta["title"]})
        description = " ".join(meta.get("description", []) or [])
        if description:
            records.append({"asin": asin, "field": "description", "text": description})
        for field, value in (meta.get("details") or {}).items():
            records.append({"asin": asin, "field": field, "text": f"{field}: {value}"})
    return records
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_amazon2023.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/agentic_sentiment/data tests/test_amazon2023.py tests/fixtures
git commit -m "feat: balanced Electronics eval-slice loader + spec records"
```

---

### Task 4: Rating parser

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/agents/__init__.py`
- Create: `pes-mtech-project/src/agentic_sentiment/agents/parsing.py`
- Test: `pes-mtech-project/tests/test_parsing.py`

**Interfaces:**
- Produces: `parse_rating(text: str) -> int | None` — returns 1–5 or `None` if unparseable. Used by every agent node in Task 7.

- [ ] **Step 1: Write the failing test**

`pes-mtech-project/tests/test_parsing.py`:
```python
from agentic_sentiment.agents.parsing import parse_rating


def test_parses_full_labeled_line():
    assert parse_rating("Sentiment (1-5): 4. Positive, satisfied customer.") == 4


def test_parses_leading_digit_only():
    assert parse_rating("5. Excellent product, would buy again.") == 5


def test_parses_leading_digit_with_dash():
    assert parse_rating("2 - somewhat negative") == 2


def test_parses_digit_anywhere_as_fallback():
    assert parse_rating("I would rate this a 3 out of 5") == 3


def test_returns_none_for_out_of_range_or_missing():
    assert parse_rating("no rating here") is None
    assert parse_rating("Sentiment (1-5): 7") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_parsing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.agents'`

- [ ] **Step 3: Implement the parser**

`pes-mtech-project/src/agentic_sentiment/agents/__init__.py`: empty file.

`pes-mtech-project/src/agentic_sentiment/agents/parsing.py`:
```python
"""Shared rating parser for every agent's raw LLM output. Handles both the
full 'Sentiment (1-5): N' line and bare leading-digit continuations (CLAUDE.md
Section 5.1 rating-parser nuance)."""

import re

_LABELED = re.compile(r"Sentiment\s*\(1-5\)\s*:\s*([1-5])\b")
_LEADING = re.compile(r"^\s*([1-5])\b")
_ANYWHERE = re.compile(r"\b([1-5])\s*(?:/|out of)\s*5\b|\b([1-5])\b")


def parse_rating(text: str) -> int | None:
    if not text:
        return None
    for pattern in (_LABELED, _LEADING):
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    m = _ANYWHERE.search(text)
    if m:
        return int(m.group(1) or m.group(2))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_parsing.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentic_sentiment/agents/__init__.py src/agentic_sentiment/agents/parsing.py tests/test_parsing.py
git commit -m "feat: shared rating parser for agent LLM outputs"
```

---

### Task 5: Dissonance (Norm(H-(Ev+Gf)) + stdev ablation)

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/agents/critic.py`
- Test: `pes-mtech-project/tests/test_critic.py`

**Interfaces:**
- Consumes: nothing (pure functions).
- Produces: `normalize_rating(rating: int) -> float` (maps 1–5 to 0.0–1.0), `dissonance_norm(h: float, ev: float, gf: float) -> float`, `dissonance_stdev(votes: list[int]) -> float` (the Run 1 formula, kept as ablation), `DISSONANCE_THRESHOLD = 0.4` constant. Used by the Critic node in Task 7.

- [ ] **Step 1: Write the failing test**

`pes-mtech-project/tests/test_critic.py`:
```python
import math

from agentic_sentiment.agents.critic import (
    DISSONANCE_THRESHOLD,
    dissonance_norm,
    dissonance_stdev,
    normalize_rating,
)


def test_normalize_rating_maps_1_to_5_onto_0_to_1():
    assert normalize_rating(1) == 0.0
    assert normalize_rating(5) == 1.0
    assert normalize_rating(3) == 0.5


def test_dissonance_norm_zero_when_all_agree():
    h = normalize_rating(5)
    assert dissonance_norm(h, h, h) == 0.0


def test_dissonance_norm_positive_when_disagreeing():
    h = normalize_rating(5)   # 1.0
    ev = normalize_rating(1)  # 0.0
    gf = normalize_rating(1)  # 0.0
    # Norm(H - (Ev+Gf)/2) = |1.0 - 0.0| = 1.0
    assert math.isclose(dissonance_norm(h, ev, gf), 1.0)


def test_dissonance_norm_is_clamped_to_0_1():
    assert 0.0 <= dissonance_norm(1.0, 0.0, 0.0) <= 1.0
    assert 0.0 <= dissonance_norm(0.0, 1.0, 1.0) <= 1.0


def test_dissonance_stdev_matches_run1_formula():
    # Run 1: D = stdev(votes) / 2
    assert math.isclose(dissonance_stdev([5, 5, 5]), 0.0)
    assert dissonance_stdev([1, 3, 5]) > 0


def test_threshold_is_point_four():
    assert DISSONANCE_THRESHOLD == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_critic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.agents.critic'`

- [ ] **Step 3: Implement dissonance**

`pes-mtech-project/src/agentic_sentiment/agents/critic.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_critic.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentic_sentiment/agents/critic.py tests/test_critic.py
git commit -m "feat: Norm(H-(Ev+Gf)) dissonance, keep vote-stdev as ablation"
```

---

### Task 6: Spec store (LanceDB claim-vs-spec grounding)

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/rag/__init__.py`
- Create: `pes-mtech-project/src/agentic_sentiment/rag/spec_store.py`
- Test: `pes-mtech-project/tests/test_spec_store.py`

**Interfaces:**
- Consumes: `build_spec_records()` output from Task 3 (`{asin, field, text}` dicts).
- Produces: `SpecStore(db_path: str, embed_fn: Callable[[str], list[float]])` with `.build(records: list[dict]) -> None`, `.query(claim: str, asin: str, k: int = 3) -> list[dict]`, `.grounding_score(claims: list[str], asin: str) -> float` (returns Gf in [0, 1]: fraction of claims whose top match's embedding cosine similarity exceeds 0.5).

- [ ] **Step 1: Write the failing test**

`pes-mtech-project/tests/test_spec_store.py`:
```python
import math

from agentic_sentiment.rag.spec_store import SpecStore

RECORDS = [
    {"asin": "B1", "field": "Connector Type", "text": "Connector Type: USB-C"},
    {"asin": "B1", "field": "Battery Life", "text": "Battery Life: 10 hours"},
    {"asin": "B2", "field": "Connector Type", "text": "Connector Type: Micro-USB"},
]


def _fake_embed(text: str) -> list[float]:
    """Deterministic 3-d embedding: [has 'usb-c', has 'micro', has 'battery'],
    so similarity is exact and the test needs no real model."""
    t = text.lower()
    return [
        1.0 if "usb-c" in t else 0.0,
        1.0 if "micro" in t else 0.0,
        1.0 if "battery" in t else 0.0,
    ]


def test_query_returns_matching_asin_records(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    results = store.query("this has a USB-C port", asin="B1", k=3)
    assert all(r["asin"] == "B1" for r in results)
    assert any("USB-C" in r["text"] for r in results)


def test_grounding_score_high_when_claim_matches_spec(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    gf = store.grounding_score(["this has a USB-C port"], asin="B1")
    assert gf == 1.0


def test_grounding_score_low_when_claim_contradicts_spec(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    gf = store.grounding_score(["this has a Micro-USB port"], asin="B1")  # B1 is USB-C
    assert gf == 0.0


def test_grounding_score_averages_across_claims(tmp_path):
    store = SpecStore(db_path=str(tmp_path / "db"), embed_fn=_fake_embed)
    store.build(RECORDS)

    gf = store.grounding_score(["this has a USB-C port", "this has a Micro-USB port"], asin="B1")
    assert math.isclose(gf, 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_spec_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.rag'`

- [ ] **Step 3: Implement the spec store**

`pes-mtech-project/src/agentic_sentiment/rag/__init__.py`: empty file.

`pes-mtech-project/src/agentic_sentiment/rag/spec_store.py`:
```python
"""LanceDB-backed product-spec store for the RAG Prover. Embeds each spec
record (title/description/detail field) and, for the real pipeline, each
product image via a shared text-image embedding space (CLIP in Colab);
locally it is tested with any injected `embed_fn`. grounding_score() checks
a claim against the *correct product's* specs only, which is what catches
the "USB-C vs Micro-USB" factual-mismatch case in the mission spec."""

import math
from typing import Callable

import lancedb
import pyarrow as pa


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class SpecStore:
    def __init__(self, db_path: str, embed_fn: Callable[[str], list[float]]):
        self.embed_fn = embed_fn
        self._db = lancedb.connect(db_path)
        self._table = None

    def build(self, records: list[dict]) -> None:
        dim = len(self.embed_fn(records[0]["text"])) if records else 1
        rows = [
            {**r, "vector": self.embed_fn(r["text"])}
            for r in records
        ]
        schema = pa.schema([
            pa.field("asin", pa.string()),
            pa.field("field", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])
        self._table = self._db.create_table("specs", data=rows, schema=schema, mode="overwrite")

    def query(self, claim: str, asin: str, k: int = 3) -> list[dict]:
        vec = self.embed_fn(claim)
        results = (
            self._table.search(vec)
            .where(f"asin = '{asin}'", prefilter=True)
            .limit(k)
            .to_list()
        )
        return [{"asin": r["asin"], "field": r["field"], "text": r["text"]} for r in results]

    def grounding_score(self, claims: list[str], asin: str, threshold: float = 0.5) -> float:
        if not claims:
            return 0.0
        hits = 0
        for claim in claims:
            matches = self.query(claim, asin=asin, k=1)
            if matches and _cosine(self.embed_fn(claim), self.embed_fn(matches[0]["text"])) >= threshold:
                hits += 1
        return hits / len(claims)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pip install -e . -q && .venv/bin/pytest tests/test_spec_store.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentic_sentiment/rag tests/test_spec_store.py
git commit -m "feat: LanceDB spec store with claim-vs-spec grounding score"
```

---

### Task 7: LangGraph StateGraph (Analyst -> Visual Verifier + RAG Prover -> Critic -> loop)

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/agents/state.py`
- Create: `pes-mtech-project/src/agentic_sentiment/agents/graph.py`
- Test: `pes-mtech-project/tests/test_graph.py`

**Interfaces:**
- Consumes: `parse_rating` (Task 4), `dissonance_norm`/`normalize_rating`/`DISSONANCE_THRESHOLD` (Task 5), `SpecStore.grounding_score` (Task 6, optional).
- Produces: `AgentState` (TypedDict), `build_graph(llm_fn: Callable[[str], str], spec_store=None, max_correction_iters: int = 2) -> CompiledGraph` with `.invoke(initial_state: AgentState) -> AgentState`.

- [ ] **Step 1: Define the state**

`pes-mtech-project/src/agentic_sentiment/agents/state.py`:
```python
"""Shared state threaded through every LangGraph node."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    review_id: str
    review_text: str
    gt_rating: int | None
    meta_title: str
    meta_description: str
    image_caption: str | None
    asin: str

    use_metadata: bool
    use_image: bool
    use_rag: bool
    dissonance_method: str  # "norm" | "stdev"
    max_correction_iters: int

    analyst_rating: int | None
    visual_rating: int | None
    rag_rating: int | None
    critique: str | None
    dissonance: float
    correction_iters: int
    final_rating: int | None
    self_corrected: bool
    rationale: str
```

- [ ] **Step 2: Write the failing test**

`pes-mtech-project/tests/test_graph.py`:
```python
from agentic_sentiment.agents.graph import build_graph


def _make_stub_llm(ratings_by_prompt_substring):
    """Returns an llm_fn that looks for a substring in the prompt and
    returns a canned rating string for it."""
    def llm_fn(prompt: str) -> str:
        for substr, rating in ratings_by_prompt_substring.items():
            if substr in prompt:
                return f"Sentiment (1-5): {rating}. stub."
        return "Sentiment (1-5): 3. default."
    return llm_fn


def test_graph_agrees_no_correction_needed():
    llm = _make_stub_llm({"Analyst": 5, "Visual": 5, "RAG": 5, "Critique": 5})
    graph = build_graph(llm_fn=llm)

    result = graph.invoke({
        "review_id": "r1", "review_text": "Loved it", "gt_rating": 5,
        "meta_title": "Widget", "image_caption": "a clean widget",
        "use_metadata": True, "use_image": True, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert result["final_rating"] == 5
    assert result["self_corrected"] is False
    assert result["dissonance"] < 0.4


def test_graph_triggers_self_correction_on_high_dissonance():
    calls = {"analyst_count": 0}

    def llm_fn(prompt: str) -> str:
        if "Visual" in prompt:
            return "Sentiment (1-5): 1. stub."
        if "RAG" in prompt:
            return "Sentiment (1-5): 1. stub."
        if "Analyst" in prompt:
            calls["analyst_count"] += 1
            # First Analyst call disagrees wildly; after critique, it should
            # be called again (that's the self-correction loop).
            return "Sentiment (1-5): 5. stub."
        return "Sentiment (1-5): 3. stub."

    graph = build_graph(llm_fn=llm_fn, max_correction_iters=2)
    result = graph.invoke({
        "review_id": "r2", "review_text": "mixed feelings", "gt_rating": 1,
        "meta_title": "Widget", "image_caption": "a broken widget",
        "use_metadata": True, "use_image": True, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert calls["analyst_count"] >= 2  # Analyst was re-run after critique
    assert result["self_corrected"] is True
    assert result["correction_iters"] <= 2


def test_graph_skips_visual_when_use_image_false():
    seen_visual = {"called": False}

    def llm_fn(prompt: str) -> str:
        if "Visual" in prompt:
            seen_visual["called"] = True
        return "Sentiment (1-5): 4. stub."

    graph = build_graph(llm_fn=llm_fn)
    graph.invoke({
        "review_id": "r3", "review_text": "fine", "gt_rating": 4,
        "meta_title": "Widget", "image_caption": None,
        "use_metadata": True, "use_image": False, "use_rag": False,
        "correction_iters": 0, "max_correction_iters": 2,
    })

    assert seen_visual["called"] is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.agents.graph'`

- [ ] **Step 4: Implement the graph**

`pes-mtech-project/src/agentic_sentiment/agents/graph.py`:
```python
"""Real langgraph.graph.StateGraph: Analyst -> {Visual Verifier, RAG Prover}
-> Critic -> (loop to Analyst with critique, or END). Replaces Run 1's plain
-Python-functions pipeline (colab/phase2_agentic_full_comparison.ipynb)."""

from typing import Callable

from langgraph.graph import END, StateGraph

from agentic_sentiment.agents.critic import (
    DISSONANCE_THRESHOLD,
    dissonance_norm,
    normalize_rating,
)
from agentic_sentiment.agents.parsing import parse_rating
from agentic_sentiment.agents.state import AgentState

ANALYST_PROMPT = """Analyst: read this review and rate its sentiment 1-5.
Review: {review_text}
{critique_block}
Sentiment (1-5):"""

VISUAL_PROMPT = """Visual Verifier: the product image shows: {image_caption}
Product: {meta_title}
Based only on the image, rate expected sentiment 1-5.
Sentiment (1-5):"""

RAG_PROMPT = """RAG Prover: similar context: {context}
Review: {review_text}
Rate sentiment 1-5 given this grounding.
Sentiment (1-5):"""

CRITIQUE_PROMPT = """Critique: Analyst said {analyst}, Visual said {visual}, RAG said {rag}.
They disagree. Give one sentence of feedback for the Analyst to reconsider."""


def build_graph(llm_fn: Callable[[str], str], spec_store=None, max_correction_iters: int = 2):
    def analyst_node(state: AgentState) -> AgentState:
        critique_block = f"Critic feedback: {state['critique']}" if state.get("critique") else ""
        text = llm_fn(ANALYST_PROMPT.format(review_text=state["review_text"], critique_block=critique_block))
        rating = parse_rating(text) or 3
        return {**state, "analyst_rating": rating}

    def visual_node(state: AgentState) -> AgentState:
        if not state.get("use_image") or not state.get("image_caption"):
            return {**state, "visual_rating": None}
        text = llm_fn(VISUAL_PROMPT.format(
            image_caption=state["image_caption"], meta_title=state.get("meta_title", "")
        ))
        return {**state, "visual_rating": parse_rating(text)}

    def rag_node(state: AgentState) -> AgentState:
        if not state.get("use_rag"):
            return {**state, "rag_rating": None}
        context = state.get("meta_description", "")
        if spec_store is not None and state.get("asin"):
            context += " " + " ".join(r["text"] for r in spec_store.query(state["review_text"], asin=state["asin"]))
        text = llm_fn(RAG_PROMPT.format(context=context, review_text=state["review_text"]))
        return {**state, "rag_rating": parse_rating(text)}

    def critic_node(state: AgentState) -> AgentState:
        h = normalize_rating(state["analyst_rating"])
        ev = normalize_rating(state["visual_rating"]) if state.get("visual_rating") else h
        gf = normalize_rating(state["rag_rating"]) if state.get("rag_rating") else h

        if state.get("dissonance_method") == "stdev":
            from agentic_sentiment.agents.critic import dissonance_stdev
            votes = [v for v in (state["analyst_rating"], state.get("visual_rating"), state.get("rag_rating")) if v]
            d = dissonance_stdev(votes)
        else:
            d = dissonance_norm(h, ev, gf)

        iters = state.get("correction_iters", 0)
        max_iters = state.get("max_correction_iters", max_correction_iters)
        if d >= DISSONANCE_THRESHOLD and iters < max_iters:
            critique = llm_fn(CRITIQUE_PROMPT.format(
                analyst=state["analyst_rating"], visual=state.get("visual_rating"), rag=state.get("rag_rating"),
            ))
            return {
                **state, "dissonance": d, "critique": critique,
                "correction_iters": iters + 1, "self_corrected": True,
            }

        votes = [v for v in (state["analyst_rating"], state.get("visual_rating"), state.get("rag_rating")) if v]
        final = max(set(votes), key=votes.count) if votes else state["analyst_rating"]
        return {
            **state, "dissonance": d, "final_rating": final,
            "self_corrected": state.get("self_corrected", False),
            "rationale": f"Analyst={state['analyst_rating']}, Visual={state.get('visual_rating')}, "
                         f"RAG={state.get('rag_rating')}, Dissonance={d:.3f}",
        }

    def route_after_critic(state: AgentState) -> str:
        return "analyst" if state.get("final_rating") is None else END

    graph = StateGraph(AgentState)
    graph.add_node("analyst", analyst_node)
    graph.add_node("visual", visual_node)
    graph.add_node("rag", rag_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "visual")
    graph.add_edge("visual", "rag")
    graph.add_edge("rag", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"analyst": "analyst", END: END})

    return graph.compile()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pip install -e . -q && .venv/bin/pytest tests/test_graph.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/agentic_sentiment/agents/state.py src/agentic_sentiment/agents/graph.py tests/test_graph.py
git commit -m "feat: real LangGraph StateGraph with dissonance-triggered self-correction"
```

---

### Task 8: Checkpoint/resume for long-running eval jobs

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/eval/__init__.py`
- Create: `pes-mtech-project/src/agentic_sentiment/eval/checkpoint.py`
- Test: `pes-mtech-project/tests/test_checkpoint.py`

**Interfaces:**
- Produces: `iter_done_ids(path: str) -> set[str]`, `append_result(path: str, record: dict) -> None` (appends one JSON line, flushes immediately). Used by Task 9's eval harness and reusable by notebooks for resume-after-disconnect.

- [ ] **Step 1: Write the failing test**

`pes-mtech-project/tests/test_checkpoint.py`:
```python
import json

from agentic_sentiment.eval.checkpoint import append_result, iter_done_ids


def test_append_result_writes_jsonl_line(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    append_result(str(path), {"review_id": "r1", "final_rating": 5})
    append_result(str(path), {"review_id": "r2", "final_rating": 3})

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["review_id"] == "r1"


def test_iter_done_ids_empty_when_no_file(tmp_path):
    assert iter_done_ids(str(tmp_path / "missing.jsonl")) == set()


def test_iter_done_ids_reflects_appended_records(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    append_result(str(path), {"review_id": "r1"})
    append_result(str(path), {"review_id": "r2"})

    assert iter_done_ids(str(path)) == {"r1", "r2"}


def test_resume_skips_done_ids_simulated_crash(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    all_ids = ["r1", "r2", "r3"]

    # First "run": crashes after r1, r2.
    for rid in all_ids[:2]:
        append_result(str(path), {"review_id": rid})

    # "Resume": only r3 should be processed.
    done = iter_done_ids(str(path))
    remaining = [rid for rid in all_ids if rid not in done]
    assert remaining == ["r3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.eval'`

- [ ] **Step 3: Implement**

`pes-mtech-project/src/agentic_sentiment/eval/__init__.py`: empty file.

`pes-mtech-project/src/agentic_sentiment/eval/checkpoint.py`:
```python
"""JSONL checkpoint + resume, shared by every long Colab eval/baseline run so
a disconnect only costs the in-flight sample, not the whole run."""

import json
from pathlib import Path


def append_result(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def iter_done_ids(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["review_id"])
    return done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_checkpoint.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentic_sentiment/eval/__init__.py src/agentic_sentiment/eval/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: JSONL checkpoint/resume for long eval runs"
```

---

### Task 9: Eval harness + ablations

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/eval/run_eval.py`
- Test: `pes-mtech-project/tests/test_run_eval.py`

**Interfaces:**
- Consumes: `build_graph` (Task 7), `append_result`/`iter_done_ids` (Task 8).
- Produces: `ABLATIONS: dict[str, dict]` (name -> flag overrides), `run_ablation(name: str, rows: list[dict], llm_fn, checkpoint_path: str, spec_store=None) -> None`, `compute_metrics(records: list[dict], key: str = "final_rating") -> dict` (accuracy, macro_f1, mae, confusion matrix as `list[list[int]]`).

- [ ] **Step 1: Write the failing test**

`pes-mtech-project/tests/test_run_eval.py`:
```python
import json

from agentic_sentiment.eval.run_eval import ABLATIONS, compute_metrics, run_ablation


def _stub_llm(prompt: str) -> str:
    return "Sentiment (1-5): 5. stub."


ROWS = [
    {"review_id": f"r{i}", "review_text": "ok", "gt_rating": 5, "meta_title": "W",
     "meta_description": "", "image_caption": "a widget", "asin": "B1"}
    for i in range(3)
]


def test_ablations_includes_text_only_and_full_graph():
    assert "text_only" in ABLATIONS
    assert "full_graph" in ABLATIONS
    assert ABLATIONS["text_only"] == {"use_metadata": False, "use_image": False, "use_rag": False}


def test_run_ablation_writes_one_record_per_row(tmp_path):
    ckpt = tmp_path / "text_only.jsonl"
    run_ablation("text_only", ROWS, llm_fn=_stub_llm, checkpoint_path=str(ckpt))

    lines = ckpt.read_text().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[0])["final_rating"] == 5


def test_run_ablation_resumes_without_reprocessing(tmp_path):
    ckpt = tmp_path / "text_only.jsonl"
    run_ablation("text_only", ROWS[:2], llm_fn=_stub_llm, checkpoint_path=str(ckpt))
    run_ablation("text_only", ROWS, llm_fn=_stub_llm, checkpoint_path=str(ckpt))  # resumed run

    lines = ckpt.read_text().strip().split("\n")
    assert len(lines) == 3  # not 5 — the first 2 weren't reprocessed


def test_compute_metrics_accuracy_and_mae():
    records = [
        {"gt_rating": 5, "final_rating": 5},
        {"gt_rating": 5, "final_rating": 4},
        {"gt_rating": 1, "final_rating": 1},
    ]
    metrics = compute_metrics(records)
    assert metrics["accuracy"] == 2 / 3
    assert round(metrics["mae"], 4) == round(1 / 3, 4)
    assert len(metrics["confusion"]) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_eval.py -v`
Expected: FAIL — `ImportError` / `ModuleNotFoundError: No module named 'agentic_sentiment.eval.run_eval'`

- [ ] **Step 3: Implement**

`pes-mtech-project/src/agentic_sentiment/eval/run_eval.py`:
```python
"""Ablation runner + metrics for the Phase 2 graph eval."""

from sklearn.metrics import f1_score

from agentic_sentiment.agents.graph import build_graph
from agentic_sentiment.eval.checkpoint import append_result, iter_done_ids

ABLATIONS = {
    "text_only": {"use_metadata": False, "use_image": False, "use_rag": False},
    "plus_metadata": {"use_metadata": True, "use_image": False, "use_rag": False},
    "plus_image": {"use_metadata": True, "use_image": True, "use_rag": False},
    "plus_rag_reviews": {"use_metadata": True, "use_image": True, "use_rag": True},
    "plus_rag_specs": {"use_metadata": True, "use_image": True, "use_rag": True},  # spec_store passed in
    "full_graph": {"use_metadata": True, "use_image": True, "use_rag": True},
}


def run_ablation(name: str, rows: list[dict], llm_fn, checkpoint_path: str, spec_store=None,
                  dissonance_method: str = "norm", max_correction_iters: int = 2) -> None:
    flags = ABLATIONS[name]
    graph = build_graph(llm_fn=llm_fn, spec_store=spec_store, max_correction_iters=max_correction_iters)
    done = iter_done_ids(checkpoint_path)

    for row in rows:
        if row["review_id"] in done:
            continue
        state = {
            **row, **flags,
            "dissonance_method": dissonance_method,
            "correction_iters": 0,
            "max_correction_iters": max_correction_iters,
        }
        result = graph.invoke(state)
        append_result(checkpoint_path, {
            "review_id": row["review_id"],
            "gt_rating": row["gt_rating"],
            "final_rating": result["final_rating"],
            "analyst_rating": result.get("analyst_rating"),
            "visual_rating": result.get("visual_rating"),
            "rag_rating": result.get("rag_rating"),
            "dissonance": result.get("dissonance"),
            "self_corrected": result.get("self_corrected", False),
            "correction_iters": result.get("correction_iters", 0),
            "rationale": result.get("rationale", ""),
        })


def compute_metrics(records: list[dict], key: str = "final_rating") -> dict:
    n = len(records)
    gt = [r["gt_rating"] for r in records]
    pred = [r[key] for r in records]

    accuracy = sum(g == p for g, p in zip(gt, pred)) / n
    mae = sum(abs(g - p) for g, p in zip(gt, pred)) / n
    macro_f1 = f1_score(gt, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)

    confusion = [[0] * 5 for _ in range(5)]
    for g, p in zip(gt, pred):
        confusion[g - 1][p - 1] += 1

    return {"accuracy": accuracy, "macro_f1": macro_f1, "mae": mae, "confusion": confusion, "n": n}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_run_eval.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentic_sentiment/eval/run_eval.py tests/test_run_eval.py
git commit -m "feat: ablation runner + accuracy/macro-F1/MAE/confusion metrics"
```

---

### Task 10: Classical baselines (DT, SVM, Naive Bayes)

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/baselines/__init__.py`
- Create: `pes-mtech-project/src/agentic_sentiment/baselines/classical.py`
- Test: `pes-mtech-project/tests/test_classical.py`

**Interfaces:**
- Produces: `train_decision_tree`, `train_svm`, `train_naive_bayes` (each `(texts: list[str], labels: list[int]) -> sklearn.pipeline.Pipeline`), `evaluate(pipeline, texts, labels) -> dict` (accuracy, precision, recall, f1 — matches Wang et al. Table 3 metrics per CLAUDE.md).

- [ ] **Step 1: Write the failing test**

`pes-mtech-project/tests/test_classical.py`:
```python
from agentic_sentiment.baselines.classical import (
    evaluate,
    train_decision_tree,
    train_naive_bayes,
    train_svm,
)

TEXTS = [
    "terrible product broke immediately", "awful waste of money",
    "okay nothing special", "average performance",
    "really happy with this purchase", "excellent quality love it",
] * 5
LABELS = [1, 1, 3, 3, 5, 5] * 5


def test_decision_tree_trains_and_evaluates():
    pipeline = train_decision_tree(TEXTS, LABELS)
    metrics = evaluate(pipeline, TEXTS, LABELS)
    assert metrics["accuracy"] > 0.5
    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()


def test_svm_sigmoid_trains_and_evaluates():
    pipeline = train_svm(TEXTS, LABELS)
    metrics = evaluate(pipeline, TEXTS, LABELS)
    assert metrics["accuracy"] > 0.5


def test_naive_bayes_trains_and_evaluates():
    pipeline = train_naive_bayes(TEXTS, LABELS)
    metrics = evaluate(pipeline, TEXTS, LABELS)
    assert metrics["accuracy"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_classical.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.baselines'`

- [ ] **Step 3: Implement**

`pes-mtech-project/src/agentic_sentiment/baselines/__init__.py`: empty file.

`pes-mtech-project/src/agentic_sentiment/baselines/classical.py`:
```python
"""Classical baselines matching Wang et al. Table 3 configs (CLAUDE.md):
Decision Tree + CountVectorizer, SVM (sigmoid, gamma=1.0) + TfidfVectorizer,
Multinomial NB (alpha=0.2) + CountVectorizer."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(texts: list[str], labels: list[int]) -> Pipeline:
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def train_svm(texts: list[str], labels: list[int]) -> Pipeline:
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("clf", SVC(kernel="sigmoid", gamma=1.0)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def train_naive_bayes(texts: list[str], labels: list[int]) -> Pipeline:
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("clf", MultinomialNB(alpha=0.2)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def evaluate(pipeline: Pipeline, texts: list[str], labels: list[int]) -> dict:
    pred = pipeline.predict(texts)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, average="macro", zero_division=0),
        "recall": recall_score(labels, pred, average="macro", zero_division=0),
        "f1": f1_score(labels, pred, average="macro", zero_division=0),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_classical.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentic_sentiment/baselines/__init__.py src/agentic_sentiment/baselines/classical.py tests/test_classical.py
git commit -m "feat: classical baselines (DT, SVM-sigmoid, Multinomial NB)"
```

---

### Task 11: XLNet baseline (Drive-checkpointed)

**Files:**
- Create: `pes-mtech-project/src/agentic_sentiment/baselines/xlnet.py`
- Test: `pes-mtech-project/tests/test_xlnet.py`

**Interfaces:**
- Consumes: `RunDir` (Task 1).
- Produces: `build_training_args(run_dir: RunDir, output_dir: str) -> transformers.TrainingArguments` (pure config construction — the only part testable without a GPU/model download); `run_training(texts, labels, run_dir: RunDir) -> dict` (not unit tested locally; validated on Colab in Task 13).

- [ ] **Step 1: Write the failing test (config-only, no model download)**

`pes-mtech-project/tests/test_xlnet.py`:
```python
from agentic_sentiment.baselines.xlnet import build_training_args
from agentic_sentiment.train.run_dir import RunDir


def test_build_training_args_uses_run_dir_output(tmp_path):
    run_dir = RunDir(base_dir=str(tmp_path), run_id="xlnet_run")
    args = build_training_args(run_dir, output_dir=str(tmp_path / "out"))

    assert args.output_dir == str(tmp_path / "out")
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_xlnet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_sentiment.baselines.xlnet'`

- [ ] **Step 3: Implement (transformers import deferred — this module must import without `transformers` installed, since it's only a Colab dependency)**

`pes-mtech-project/src/agentic_sentiment/baselines/xlnet.py`:
```python
"""XLNet-large-cased baseline (Wang et al. Table 3), Drive-checkpointed via
RunDir the same way as the Phase 1 LoRA run. Training itself needs a GPU and
`transformers`/`torch`, so it is validated on Colab (notebook 02), not
locally — only the config construction is unit tested here."""

from agentic_sentiment.train.run_dir import RunDir

MODEL_NAME = "xlnet-large-cased"


def build_training_args(run_dir: RunDir, output_dir: str, num_train_epochs: int = 3,
                         per_device_train_batch_size: int = 4, learning_rate: float = 2e-5):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
    )


def run_training(texts: list[str], labels: list[int], run_dir: RunDir) -> dict:
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
    from sklearn.model_selection import train_test_split

    run_dir.write_config({
        "model": MODEL_NAME, "num_samples": len(texts), "seed": 42,
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, [l - 1 for l in labels], test_size=0.2, random_state=42
    )

    def tokenize(batch_texts):
        return tokenizer(batch_texts, truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_dict({**tokenize(train_texts), "labels": train_labels})
    eval_ds = Dataset.from_dict({**tokenize(eval_texts), "labels": eval_labels})

    args = build_training_args(run_dir, output_dir=str(run_dir.path / "output"))
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[run_dir.checkpoint_callback()],
    )
    trainer.train()
    metrics = trainer.evaluate()

    best_dir = run_dir.path / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (run_dir.path / "metrics.json").write_text(__import__("json").dumps(metrics, indent=2))
    return metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pip install -e ".[colab]" -q && .venv/bin/pytest tests/test_xlnet.py -v`

If `transformers`/`torch` are too heavy to install locally, mark it skippable:
```python
# at top of tests/test_xlnet.py
import pytest
pytest.importorskip("transformers")
```
Expected: 1 passed (or skipped if `transformers` isn't installed locally — validated for real on Colab in Task 13).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_sentiment/baselines/xlnet.py tests/test_xlnet.py
git commit -m "feat: XLNet-large baseline with Drive checkpointing"
```

---

### Task 12: Colab notebook — Phase 1 retrain

**Files:**
- Create: `pes-mtech-project/colab/01_train_phase1.ipynb`

**Interfaces:**
- Consumes: `agentic_sentiment.train.phase1_train.run_training` (Task 2), `RunDir` (Task 1).

- [ ] **Step 1: Write the notebook**

Build with `nbformat` (not by hand) so the JSON is valid:
```bash
python3 - <<'EOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# 01 — Phase 1 LoRA retrain (Drive-checkpointed)\n"
        "Clones pes-mtech-project, trains the paper-closer LoRA config (r=16, alpha=128, "
        "5 epochs, 4000 samples), and streams every checkpoint + the best adapter to "
        "Google Drive so a disconnect never loses the best params."),
    nbf.v4.new_code_cell("!nvidia-smi"),
    nbf.v4.new_code_cell(
        '!git clone https://github.com/kvarun314/pes-mtech-project.git /content/repo\n'
        '%cd /content/repo\n'
        '!pip install -q -e ".[colab]"'
    ),
    nbf.v4.new_code_cell(
        'import os\n'
        'from huggingface_hub import login\n'
        'login(os.environ.get("HF_TOKEN", ""))  # set in Colab Secrets'
    ),
    nbf.v4.new_code_cell(
        'from google.colab import drive\n'
        'drive.mount("/content/drive")\n'
        'DRIVE_RUNS_DIR = "/content/drive/MyDrive/pes-mtech-project/runs"'
    ),
    nbf.v4.new_markdown_cell("## Download the Kaggle dataset\n"
        "Upload `kaggle.json` to `/root/.kaggle/kaggle.json`, then:"),
    nbf.v4.new_code_cell(
        '!kaggle datasets download -d arhamrumi/amazon-product-reviews -p /content/data --unzip\n'
        'DATA_PATH = "/content/data/Reviews.csv"'
    ),
    nbf.v4.new_code_cell(
        'from agentic_sentiment.train.run_dir import RunDir\n'
        'from agentic_sentiment.train.phase1_train import run_training\n\n'
        'run_dir = RunDir(base_dir=DRIVE_RUNS_DIR)\n'
        'print("Run directory:", run_dir.path)\n'
        'metrics = run_training(DATA_PATH, run_dir)\n'
        'print(metrics)'
    ),
]
nbf.write(nb, "pes-mtech-project/colab/01_train_phase1.ipynb")
EOF
```

- [ ] **Step 2: Validate the notebook JSON is well-formed**

Run: `.venv/bin/python3 -c "import nbformat; nbformat.read('pes-mtech-project/colab/01_train_phase1.ipynb', as_version=4)"`
Expected: no error (silent success)

- [ ] **Step 3: Commit**

```bash
git add colab/01_train_phase1.ipynb
git commit -m "feat: Colab notebook for Drive-checkpointed Phase 1 retrain"
```

---

### Task 13: Colab notebook — baselines

**Files:**
- Create: `pes-mtech-project/colab/02_baselines.ipynb`

**Interfaces:**
- Consumes: `agentic_sentiment.baselines.classical` (Task 10), `agentic_sentiment.baselines.xlnet` (Task 11).

- [ ] **Step 1: Write the notebook**

```bash
python3 - <<'EOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# 02 — Classical + XLNet baselines (Wang et al. Table 3 configs)"),
    nbf.v4.new_code_cell(
        '!git clone https://github.com/kvarun314/pes-mtech-project.git /content/repo\n'
        '%cd /content/repo\n'
        '!pip install -q -e ".[colab]"'
    ),
    nbf.v4.new_code_cell(
        'from google.colab import drive\n'
        'drive.mount("/content/drive")\n'
        'DRIVE_RUNS_DIR = "/content/drive/MyDrive/pes-mtech-project/runs"'
    ),
    nbf.v4.new_markdown_cell("## Load the same Phase 1 Kaggle split used for the LoRA run"),
    nbf.v4.new_code_cell(
        'import pandas as pd\n'
        'df = pd.read_csv("/content/data/Reviews.csv").dropna(subset=["Text", "Score"])\n'
        'texts, labels = df["Text"].tolist(), df["Score"].astype(int).tolist()'
    ),
    nbf.v4.new_code_cell(
        'from sklearn.model_selection import train_test_split\n'
        'from agentic_sentiment.baselines.classical import (\n'
        '    train_decision_tree, train_svm, train_naive_bayes, evaluate,\n'
        ')\n\n'
        'X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)\n'
        'results = {}\n'
        'for name, trainer in [("decision_tree", train_decision_tree), ("svm_sigmoid", train_svm),\n'
        '                      ("naive_bayes", train_naive_bayes)]:\n'
        '    pipeline = trainer(X_train, y_train)\n'
        '    results[name] = evaluate(pipeline, X_test, y_test)\n'
        '    print(name, results[name])'
    ),
    nbf.v4.new_code_cell(
        'from agentic_sentiment.train.run_dir import RunDir\n'
        'from agentic_sentiment.baselines.xlnet import run_training as run_xlnet\n\n'
        'run_dir = RunDir(base_dir=DRIVE_RUNS_DIR, run_id="xlnet")\n'
        'results["xlnet"] = run_xlnet(X_train[:20000], y_train[:20000], run_dir)\n'
        'print(results["xlnet"])'
    ),
    nbf.v4.new_code_cell(
        'import json\n'
        'with open("/content/drive/MyDrive/pes-mtech-project/results_run2_baselines.json", "w") as f:\n'
        '    json.dump(results, f, indent=2)'
    ),
]
nbf.write(nb, "pes-mtech-project/colab/02_baselines.ipynb")
EOF
```

- [ ] **Step 2: Validate the notebook JSON**

Run: `.venv/bin/python3 -c "import nbformat; nbformat.read('pes-mtech-project/colab/02_baselines.ipynb', as_version=4)"`
Expected: no error

- [ ] **Step 3: Commit**

```bash
git add colab/02_baselines.ipynb
git commit -m "feat: Colab notebook for classical + XLNet baselines"
```

---

### Task 14: Colab notebook — Phase 2 eval + ablations (Electronics, balanced)

**Files:**
- Create: `pes-mtech-project/colab/03_phase2_eval_ablations.ipynb`

**Interfaces:**
- Consumes: `load_balanced_slice`/`build_spec_records` (Task 3), `SpecStore` (Task 6), `run_ablation`/`compute_metrics` (Task 9), the trained Phase 1 adapter from Task 12's `run_dir`.

- [ ] **Step 1: Write the notebook**

```bash
python3 - <<'EOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# 03 — Phase 2 eval + ablations (Amazon Reviews 2023, Electronics, balanced)\n"
        "Runs text_only / plus_metadata / plus_image / plus_rag_reviews / plus_rag_specs / "
        "full_graph, plus a norm-vs-stdev dissonance ablation, on a per-rating-balanced slice."
    ),
    nbf.v4.new_code_cell(
        '!git clone https://github.com/kvarun314/pes-mtech-project.git /content/repo\n'
        '%cd /content/repo\n'
        '!pip install -q -e ".[colab]"'
    ),
    nbf.v4.new_code_cell(
        'from google.colab import drive\n'
        'drive.mount("/content/drive")\n'
        'RESULTS_DIR = "/content/drive/MyDrive/pes-mtech-project/results/run2"\n'
        'import os; os.makedirs(RESULTS_DIR, exist_ok=True)'
    ),
    nbf.v4.new_markdown_cell("## Load Amazon Reviews 2023 — Electronics"),
    nbf.v4.new_code_cell(
        'from datasets import load_dataset\n\n'
        'reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics",\n'
        '                       split="full", trust_remote_code=False)\n'
        'meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Electronics",\n'
        '                    split="full", trust_remote_code=False)\n'
        'meta_by_asin = {m["parent_asin"]: m for m in meta.select(range(min(len(meta), 200000)))}'
    ),
    nbf.v4.new_code_cell(
        'from agentic_sentiment.data.amazon2023 import load_balanced_slice, build_spec_records\n\n'
        'rows = load_balanced_slice(list(reviews.select(range(min(len(reviews), 500000)))),\n'
        '                            meta_by_asin, n_per_rating=400, seed=42)\n'
        'print(len(rows), "balanced eval rows")\n'
        'spec_records = build_spec_records(meta_by_asin)'
    ),
    nbf.v4.new_markdown_cell("## BLIP captions (before loading LLaMA, to save VRAM)"),
    nbf.v4.new_code_cell(
        'import torch\n'
        'from transformers import BlipProcessor, BlipForConditionalGeneration\n'
        'import requests\n'
        'from PIL import Image\n\n'
        'processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")\n'
        'blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")\n'
        'for row in rows:\n'
        '    row["image_caption"] = None\n'
        '    if row.get("image_url"):\n'
        '        try:\n'
        '            img = Image.open(requests.get(row["image_url"], timeout=5, stream=True).raw).convert("RGB")\n'
        '            inputs = processor(img, return_tensors="pt").to("cuda")\n'
        '            row["image_caption"] = processor.decode(blip.generate(**inputs)[0], skip_special_tokens=True)\n'
        '        except Exception:\n'
        '            pass\n'
        'del blip\n'
        'torch.cuda.empty_cache()'
    ),
    nbf.v4.new_markdown_cell("## Build spec store (CLIP text embeddings)"),
    nbf.v4.new_code_cell(
        'from sentence_transformers import SentenceTransformer\n'
        'from agentic_sentiment.rag.spec_store import SpecStore\n\n'
        'embedder = SentenceTransformer("all-MiniLM-L6-v2")\n'
        'embed_fn = lambda text: embedder.encode(text).tolist()\n\n'
        'spec_store = SpecStore(db_path="/content/spec_lancedb", embed_fn=embed_fn)\n'
        'spec_store.build(spec_records)'
    ),
    nbf.v4.new_markdown_cell("## Load Phase 1 LoRA adapter from Task 12's run_dir"),
    nbf.v4.new_code_cell(
        'import glob\n'
        'from transformers import AutoModelForCausalLM, AutoTokenizer\n'
        'from peft import PeftModel\n\n'
        'adapter_dir = sorted(glob.glob("/content/drive/MyDrive/pes-mtech-project/runs/*/best_adapter"))[-1]\n'
        'print("Using adapter:", adapter_dir)\n'
        'tokenizer = AutoTokenizer.from_pretrained(adapter_dir)\n'
        'base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", load_in_4bit=True, device_map="auto")\n'
        'model = PeftModel.from_pretrained(base, adapter_dir)\n'
        'model.eval()\n\n'
        'def llm_fn(prompt: str) -> str:\n'
        '    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)\n'
        '    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)\n'
        '    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)'
    ),
    nbf.v4.new_markdown_cell("## Run every ablation (each resumable via its own checkpoint)"),
    nbf.v4.new_code_cell(
        'from agentic_sentiment.eval.run_eval import ABLATIONS, run_ablation, compute_metrics\n'
        'import json\n\n'
        'all_metrics = {}\n'
        'for name in ABLATIONS:\n'
        '    store = spec_store if name == "plus_rag_specs" or name == "full_graph" else None\n'
        '    ckpt = f"{RESULTS_DIR}/{name}_checkpoint.jsonl"\n'
        '    run_ablation(name, rows, llm_fn=llm_fn, checkpoint_path=ckpt, spec_store=store)\n'
        '    records = [json.loads(l) for l in open(ckpt)]\n'
        '    all_metrics[name] = compute_metrics(records)\n'
        '    print(name, all_metrics[name])\n\n'
        'with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:\n'
        '    json.dump(all_metrics, f, indent=2)'
    ),
    nbf.v4.new_markdown_cell("## Dissonance-formula ablation (norm vs. Run 1 stdev)"),
    nbf.v4.new_code_cell(
        'for method in ("norm", "stdev"):\n'
        '    ckpt = f"{RESULTS_DIR}/dissonance_{method}_checkpoint.jsonl"\n'
        '    run_ablation("full_graph", rows, llm_fn=llm_fn, checkpoint_path=ckpt,\n'
        '                 spec_store=spec_store, dissonance_method=method)\n'
        '    records = [json.loads(l) for l in open(ckpt)]\n'
        '    print(method, compute_metrics(records))'
    ),
]
nbf.write(nb, "pes-mtech-project/colab/03_phase2_eval_ablations.ipynb")
EOF
```

- [ ] **Step 2: Validate the notebook JSON**

Run: `.venv/bin/python3 -c "import nbformat; nbformat.read('pes-mtech-project/colab/03_phase2_eval_ablations.ipynb', as_version=4)"`
Expected: no error

- [ ] **Step 3: Commit**

```bash
git add colab/03_phase2_eval_ablations.ipynb
git commit -m "feat: Colab notebook for balanced Electronics eval + full ablation suite"
```

---

### Task 15: Colab notebook — report (collects results/run2/ + charts)

**Files:**
- Create: `pes-mtech-project/colab/04_report.ipynb`

**Interfaces:**
- Consumes: `results/run2/metrics_summary.json` and the ablation checkpoint JSONLs written by Task 14; `compute_metrics` (Task 9).

- [ ] **Step 1: Write the notebook**

```bash
python3 - <<'EOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# 04 — Report: collect results/run2/ into charts for the paper"),
    nbf.v4.new_code_cell(
        'from google.colab import drive\n'
        'drive.mount("/content/drive")\n'
        'RESULTS_DIR = "/content/drive/MyDrive/pes-mtech-project/results/run2"'
    ),
    nbf.v4.new_code_cell(
        'import json, matplotlib.pyplot as plt\n\n'
        'with open(f"{RESULTS_DIR}/metrics_summary.json") as f:\n'
        '    metrics = json.load(f)\n\n'
        'names = list(metrics.keys())\n'
        'accs = [metrics[n]["accuracy"] for n in names]\n'
        'plt.figure(figsize=(10, 5))\n'
        'plt.bar(names, accs)\n'
        'plt.ylabel("Accuracy")\n'
        'plt.title("Phase 2 ablations — Amazon Reviews 2023 Electronics (balanced, Run 2)")\n'
        'plt.xticks(rotation=30, ha="right")\n'
        'plt.tight_layout()\n'
        'plt.savefig(f"{RESULTS_DIR}/ablation_accuracy.png", dpi=150)\n'
        'plt.show()'
    ),
    nbf.v4.new_code_cell(
        '# Copy local to repo for commit (download this file from the Colab file panel,\n'
        '# or mount Drive locally and cp into pes-mtech-project/results/run2/).\n'
        'print("Download", f"{RESULTS_DIR}/metrics_summary.json", "and",\n'
        '      f"{RESULTS_DIR}/ablation_accuracy.png", "into pes-mtech-project/results/run2/ locally.")'
    ),
]
nbf.write(nb, "pes-mtech-project/colab/04_report.ipynb")
EOF
```

- [ ] **Step 2: Validate the notebook JSON**

Run: `.venv/bin/python3 -c "import nbformat; nbformat.read('pes-mtech-project/colab/04_report.ipynb', as_version=4)"`
Expected: no error

- [ ] **Step 3: Commit**

```bash
git add colab/04_report.ipynb
git commit -m "feat: Colab notebook to chart Run 2 ablation results"
```

---

### Task 16: Documentation — CLAUDE.md status table + PROGRESS.md

**Files:**
- Modify: `pes-mtech-project/CLAUDE.md`
- Modify: `pes-mtech-project/docs/PROGRESS.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `CLAUDE.md`'s "Current repository state" status table**

Add a row (after the existing 4-row table) noting the new package and that Run 2 numbers are pending the user's Colab execution:
```markdown
| Full Phase 2 rebuild (LangGraph + spec RAG + Norm dissonance) | `src/agentic_sentiment/` + `colab/01_train_phase1.ipynb`..`04_report.ipynb` | Code complete, unit-tested locally with a stub LLM; **Run 2 numbers pending** — GPU steps (LoRA retrain, XLNet, live Phase 2 eval on Electronics) run on the user's Colab, not here. |
```

- [ ] **Step 2: Append a final `docs/PROGRESS.md` entry**

```markdown
## 2026-10-02 (cont.)
- Implemented `agentic_sentiment` package: Drive-checkpointed RunDir, vendored
  Phase 1 LoRA training, balanced Electronics loader + spec records, rating
  parser, Norm(H-(Ev+Gf)) dissonance (+ stdev ablation), LanceDB spec store,
  real LangGraph StateGraph with self-correction loop, JSONL checkpoint/resume,
  ablation eval harness, classical + XLNet baselines. All unit-tested locally
  (stub LLM, fake embeddings) except GPU-only training paths.
- Added 4 Colab notebooks (`01_train_phase1` .. `04_report`) that call the
  package; each writes to a Drive run directory so no training run can lose
  its best checkpoint to a disconnect.
- Next: user runs the 4 notebooks on Colab in order, downloads
  `results/run2/` back into the repo, then the paper gets updated with real
  Run 2 numbers (not before).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/PROGRESS.md
git commit -m "docs: record full Phase 2 rebuild status, next steps for Run 2"
```

---

## Self-Review

**Spec coverage:**
- LangGraph StateGraph → Task 7. ✓
- Spec-store RAG (claim vs. product spec) → Task 6, wired into Task 7's `rag_node` and Task 14's `plus_rag_specs`/`full_graph` ablations. ✓
- Norm(H-(Ev+Gf)) dissonance, stdev kept as ablation → Task 5, used by Task 7, exercised by Task 14. ✓
- Full Phase 1 retrain, paper-closer config → Task 2 + Task 12. ✓
- Balanced Electronics eval, larger than 2,000 skewed samples → Task 3 (`n_per_rating=400` = 2,000 balanced, set in Task 14; raise if the user wants more). ✓
- Ablations (text-only, +metadata, +image, +RAG-reviews, +RAG-specs, full) → Task 9 (`ABLATIONS`), run in Task 14. ✓
- Classical baselines (DT, SVM-sigmoid, NB) + XLNet → Tasks 10–11, run in Task 13. ✓
- Every training run checkpointed + best params saved to Drive → `RunDir` (Task 1), used by Task 2's LoRA training and Task 11's XLNet training. ✓
- Documentation kept current → Task 16, plus each task's commit message.

**Placeholder scan:** no TBD/TODO; every step has real code or an exact shell command.

**Type consistency:** `AgentState` keys used in Task 7 match what Task 9's `run_ablation` constructs and what Task 14's notebook reads back (`final_rating`, `analyst_rating`, `visual_rating`, `rag_rating`, `dissonance`, `self_corrected`, `correction_iters`, `rationale`). `RunDir` methods (`write_config`, `write_best`, `checkpoint_callback`) are identical across Tasks 1, 2, 11.

**Known open call for the user:** Task 14 sets `n_per_rating=400` (2,000 balanced rows). Say if a different eval size is wanted before running it on Colab.
