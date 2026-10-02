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
