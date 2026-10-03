from unittest.mock import MagicMock, patch

from agentic_sentiment.train.phase1_train import build_training_config
from agentic_sentiment.train.run_dir import RunDir


def test_build_training_config_is_paper_closer():
    # No overrides needed anymore -- TrainingConfig's own defaults are the
    # paper-closer config (matches colab/llama_sentiment_baseline_train.ipynb
    # cell 9 exactly), not Table 1's literal baseline.
    cfg = build_training_config()
    assert cfg.max_samples == 4000
    assert cfg.num_train_epochs == 5
    assert cfg.gradient_checkpointing is True


def test_build_model_config_is_paper_closer():
    from agentic_sentiment.train.phase1_train import build_model_config

    cfg = build_model_config()
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 128
    assert cfg.trainable_layers == 8
    # None here is correct: get_lora_config()'s own fallback supplies
    # ["q_proj", "k_proj", "v_proj", "o_proj"] when this is unset (tested
    # directly in tests/phase1/test_lora.py), matching how the notebook's
    # ModelConfig leaves this field unset too.
    assert cfg.lora_target_modules is None


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
