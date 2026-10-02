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
