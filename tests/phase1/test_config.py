"""Tests for config (paper Table 1 defaults)."""


from agentic_sentiment.phase1.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
)


def test_get_default_config_returns_dict_with_three_keys():
    cfg = get_default_config()
    assert set(cfg) == {"model", "training", "data"}
    assert isinstance(cfg["model"], ModelConfig)
    assert isinstance(cfg["training"], TrainingConfig)
    assert isinstance(cfg["data"], DataConfig)


def test_model_config_paper_defaults():
    m = ModelConfig()
    assert m.lora_r == 4
    assert m.lora_alpha == 64
    assert m.lora_dropout == 0.0
    assert m.trainable_layers == 3
    assert m.model_name_or_path == "meta-llama/Meta-Llama-3-8B"


def test_training_config_paper_defaults():
    t = TrainingConfig()
    assert t.num_train_epochs == 3
    assert t.learning_rate == 5e-5
    assert t.max_grad_norm == 1.0
    assert t.per_device_train_batch_size == 3
    assert t.gradient_accumulation_steps == 4
    assert t.max_seq_length == 1024
    assert t.max_samples == 2000
    assert t.val_ratio == 0.2
    assert t.neftune_alpha == 0.02


def test_data_config_sentiment_labels():
    d = DataConfig()
    assert d.negative_labels == (1, 2)
    assert d.neutral_labels == (3,)
    assert d.positive_labels == (4, 5)
    assert d.use_one_shot is True
    assert d.use_cot is True
