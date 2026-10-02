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


def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    rd = RunDir(base_dir=str(tmp_path), run_id="run")
    assert rd.latest_checkpoint() is None


def _write_fake_hf_checkpoint(output_dir: Path, step: int) -> None:
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step}))


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
