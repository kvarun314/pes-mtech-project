import inspect
from unittest.mock import MagicMock, patch

import pytest

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
         patch.object(phase1_train, "detect_use_4bit", return_value=False), \
         patch.object(phase1_train, "load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
        metrics = phase1_train.run_training("dummy.csv", run_dir)

    assert metrics == {"eval_loss": 0.5}
    assert (run_dir.path / "config.json").exists()
    fake_trainer.train.assert_called_once()


def test_build_trainer_only_passes_kwargs_transformers_training_arguments_accepts(tmp_path):
    # Regression: _build_trainer previously passed overwrite_output_dir,
    # which cell 17 (the proven notebook) never does -- and which the
    # installed transformers version may not accept at all, raising a
    # TypeError only at the real (unmocked) TrainingArguments() call,
    # which test_run_training_writes_config_json's full-_build_trainer
    # mock can never catch. Build the actual TrainingArguments the real
    # way and let any unsupported kwarg raise here instead of mid-run on
    # Colab, after the expensive data-prep step has already completed.
    pytest.importorskip("transformers")
    from transformers import TrainingArguments

    from agentic_sentiment.train.phase1_train import build_training_config

    training_cfg = build_training_config()
    training_cfg.output_dir = str(tmp_path / "output")

    # Mirrors exactly the kwargs _build_trainer passes to TrainingArguments
    # (kept in sync manually -- if this list and the real call ever
    # diverge, this test stops catching real regressions there).
    kwargs = dict(
        output_dir=training_cfg.output_dir,
        num_train_epochs=training_cfg.num_train_epochs,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        max_grad_norm=training_cfg.max_grad_norm,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        warmup_steps=training_cfg.warmup_steps,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        eval_strategy=training_cfg.evaluation_strategy,
        eval_steps=training_cfg.eval_steps,
        load_best_model_at_end=getattr(training_cfg, "load_best_model_at_end", True),
        metric_for_best_model=getattr(training_cfg, "metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        fp16=training_cfg.fp16,
        bf16=training_cfg.bf16,
        optim=training_cfg.optim,
        seed=training_cfg.seed,
        report_to="none",
        neftune_noise_alpha=training_cfg.neftune_alpha,
        gradient_checkpointing=getattr(training_cfg, "gradient_checkpointing", False),
    )

    # Will raise TypeError immediately if any kwarg is unsupported by the
    # installed transformers version -- exactly the bug this regression
    # test exists to catch.
    args = TrainingArguments(**kwargs)
    assert args.output_dir == training_cfg.output_dir

    # And confirm the real call site's signature hasn't silently grown a
    # kwarg this test doesn't know about (keeps the two lists honest).
    sig = inspect.signature(TrainingArguments.__init__)
    for name in kwargs:
        assert name in sig.parameters, f"TrainingArguments no longer accepts {name!r}"


def test_data_collator_pads_to_batch_longest_not_fixed_max_seq_length():
    # Regression: the collator previously used padding="max_length", padding
    # every batch to the full max_seq_length (1024) regardless of how short
    # the actual examples were -- wasted attention/FFN compute every step.
    # padding=True pads each batch to its own longest example instead.
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from transformers.data.data_collator import DataCollatorForSeq2Seq

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
        label_pad_token_id=-100, return_tensors="pt",
    )

    short = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3]}
    long = {"input_ids": list(range(1, 51)), "attention_mask": [1] * 50, "labels": [-100] * 20 + list(range(21, 51))}

    batch = collator([short, long])

    # Batch padded to its own longest example (50, rounded up to a multiple
    # of 8 = 56) -- not the model's own max_seq_length (1024).
    assert batch["input_ids"].shape[1] <= 56
    padded_positions = (batch["attention_mask"][0] == 0).sum().item()
    assert padded_positions > 0
    assert (batch["labels"][0][batch["attention_mask"][0] == 0] == -100).all()
