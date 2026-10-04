import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_resume_preserves_best_value_across_reconnect(tmp_path):
    """Reproduces bug: disconnecting and resuming with same run_id should not
    overwrite BEST.md with a worse value."""
    base_dir = str(tmp_path)
    run_id = "test-resume-run"

    # First session: write a good result
    rd1 = RunDir(base_dir=base_dir, run_id=run_id)
    rd1.write_best(step=100, metric_name="eval_loss", value=0.30)

    # Check the file was written correctly
    best_md_1 = (rd1.path / "BEST.md").read_text()
    assert "step 100" in best_md_1
    assert "0.3" in best_md_1

    # Simulate disconnect and reconnect: create a new RunDir with same run_id
    # (representing resumed training after Colab disconnect)
    rd2 = RunDir(base_dir=base_dir, run_id=run_id)

    # Try to write a worse result (should be ignored)
    rd2.write_best(step=200, metric_name="eval_loss", value=0.50)

    # BEST.md should still show the better value (step 100 / 0.3)
    # not the worse value (step 200 / 0.5)
    best_md_2 = (rd2.path / "BEST.md").read_text()
    assert "step 100" in best_md_2, f"Expected step 100 in BEST.md, got: {best_md_2}"
    assert "0.3" in best_md_2, f"Expected 0.3 in BEST.md, got: {best_md_2}"
    assert "step 200" not in best_md_2, f"Should not have step 200, got: {best_md_2}"


def test_history_row_includes_run_id_type_best_and_path(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="2026-10-02_test")
    rd.write_config({"lora_r": 16, "lora_alpha": 128})
    rd.write_best(step=100, metric_name="eval_loss", value=0.42)

    row = rd.history_row("phase1")

    assert "2026-10-02_test" in row
    assert "phase1" in row
    assert "step 100" in row
    assert "0.42" in row
    assert "lora_r=16" in row
    assert str(rd.path) in row


def test_history_row_handles_missing_config_and_best(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="fresh_run")
    row = rd.history_row("xlnet")
    assert "fresh_run" in row
    assert "xlnet" in row


def test_history_row_flattens_nested_config_one_level(tmp_path):
    # phase1_train.py's actual config.json shape: {"model": {...}, "training": {...}}
    rd = RunDir(base_dir=str(tmp_path), run_id="run")
    rd.write_config({
        "model": {"lora_r": 16, "lora_alpha": 128},
        "training": {"num_train_epochs": 5},
        "data_path": "/content/data/Reviews.csv",
    })

    row = rd.history_row("phase1")

    assert "model.lora_r=16" in row
    assert "model.lora_alpha=128" in row
    assert "training.num_train_epochs=5" in row
    assert "data_path=/content/data/Reviews.csv" in row


def test_history_row_shows_the_actual_best_metric_name(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="run")
    rd.write_best(step=100, metric_name="eval_accuracy", value=0.9, greater_is_better=True)
    row = rd.history_row("xlnet")
    assert "eval_accuracy=0.9" in row
    assert "eval_loss" not in row  # not hardcoded to the wrong metric name


def test_history_row_escapes_pipe_and_newline_in_config_values(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="run")
    rd.write_config({"note": "a | pipe\nand a newline"})
    row = rd.history_row("phase1")

    assert "note=a \\| pipe and a newline" in row  # pipe escaped, newline collapsed to a space
    assert "\n" not in row


def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="run")
    assert rd.latest_checkpoint() is None


def _write_fake_hf_checkpoint(output_dir: Path, step: int) -> None:
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step}))


def test_latest_checkpoint_skips_incomplete_checkpoint_missing_trainer_state(tmp_path):
    # Regression: a disconnect mid-copytree can leave the newest checkpoint
    # directory without trainer_state.json. Resuming from it either
    # crashes or silently restarts from step 0 -- fall back to the next-
    # newest COMPLETE checkpoint instead.
    rd = RunDir(base_dir=str(tmp_path), run_id="run")
    complete = rd.path / "checkpoints" / "checkpoint-100"
    complete.mkdir(parents=True)
    (complete / "trainer_state.json").write_text("{}")

    incomplete = rd.path / "checkpoints" / "checkpoint-200"
    incomplete.mkdir(parents=True)  # no trainer_state.json -- half-copied

    assert rd.latest_checkpoint() == str(complete)


def test_checkpoint_callback_copies_and_reports_latest(tmp_path):
    pytest.importorskip("transformers")
    rd = RunDir(base_dir=str(tmp_path / "runs"), run_id="run")
    callback = rd.checkpoint_callback()

    hf_output = tmp_path / "hf_output"
    for step in (100, 200):
        _write_fake_hf_checkpoint(hf_output, step)
        callback.on_save(SimpleNamespace(output_dir=str(hf_output)), SimpleNamespace(global_step=step), None)

    assert rd.latest_checkpoint() == str(rd.path / "checkpoints" / "checkpoint-200")
    assert (rd.path / "checkpoints" / "checkpoint-100").exists()
    assert (rd.path / "checkpoints" / "checkpoint-200").exists()


def test_checkpoint_callback_prunes_beyond_keep_limit(tmp_path):
    pytest.importorskip("transformers")
    rd = RunDir(base_dir=str(tmp_path / "runs"), run_id="run")
    callback = rd.checkpoint_callback(keep=2)

    hf_output = tmp_path / "hf_output"
    for step in (100, 200, 300):
        _write_fake_hf_checkpoint(hf_output, step)
        callback.on_save(SimpleNamespace(output_dir=str(hf_output)), SimpleNamespace(global_step=step), None)

    kept = sorted(p.name for p in (rd.path / "checkpoints").glob("checkpoint-*"))
    assert kept == ["checkpoint-200", "checkpoint-300"]  # oldest (100) pruned
    assert rd.latest_checkpoint() == str(rd.path / "checkpoints" / "checkpoint-300")


def test_checkpoint_callback_never_prunes_the_best_checkpoint(tmp_path):
    # Regression: keep=2 pruning used to delete the best-eval checkpoint
    # once 2 newer (but worse) checkpoints were saved after it, leaving
    # BEST.md pointing at a step whose files no longer exist on Drive.
    pytest.importorskip("transformers")
    rd = RunDir(base_dir=str(tmp_path / "runs"), run_id="run")
    callback = rd.checkpoint_callback(keep=2)

    hf_output = tmp_path / "hf_output"
    for step in (100, 200, 300, 400):
        _write_fake_hf_checkpoint(hf_output, step)
        callback.on_save(SimpleNamespace(output_dir=str(hf_output)), SimpleNamespace(global_step=step), None)
        if step == 100:  # step 100 is the (only) eval improvement -- the best
            rd.write_best(step=100, metric_name="eval_loss", value=0.1)

    kept = sorted(p.name for p in (rd.path / "checkpoints").glob("checkpoint-*"))
    assert "checkpoint-100" in kept  # best, protected despite being oldest
    assert rd.best_checkpoint_dir() == str(rd.path / "checkpoints" / "checkpoint-100")


def test_save_best_prefers_protected_checkpoint_over_trainer_model(tmp_path):
    # Simulates the post-resume scenario: trainer.save_model() would write
    # whatever the (possibly non-best, e.g. after a failed reload) in-memory
    # model currently is. save_best() must prefer the protected Drive copy.
    rd = RunDir(base_dir=str(tmp_path / "runs"), run_id="run")
    best_src = rd.path / "checkpoints" / "checkpoint-100"
    best_src.mkdir(parents=True)
    (best_src / "model.safetensors").write_text("BEST WEIGHTS")
    rd.write_best(step=100, metric_name="eval_loss", value=0.1)

    class _FakeTrainer:
        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_text("WRONG: final, not best")

    class _FakeTokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    dest = rd.save_best(_FakeTrainer(), _FakeTokenizer(), "best_adapter")

    assert (Path(dest) / "model.safetensors").read_text() == "BEST WEIGHTS"
    assert (Path(dest) / "tokenizer.json").exists()


def test_save_best_falls_back_to_trainer_when_no_best_recorded_yet(tmp_path):
    rd = RunDir(base_dir=str(tmp_path / "runs"), run_id="run")

    class _FakeTrainer:
        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_text("only model available")

    class _FakeTokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    dest = rd.save_best(_FakeTrainer(), _FakeTokenizer(), "best_adapter")
    assert (Path(dest) / "model.safetensors").read_text() == "only model available"
