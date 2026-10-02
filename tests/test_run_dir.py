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
