"""Phase 1 LoRA training, wired to Drive checkpointing via RunDir.

The data/model logic (agentic_sentiment.phase1.*) is a faithful port of
colab/llama_sentiment_baseline_train.ipynb's own cells -- that notebook is
the proven source of the Run 1 Phase 1 numbers, so its ModelConfig/
TrainingConfig/DataConfig dataclass DEFAULTS already are the paper-closer
config (not Table 1's literal baseline). run_training() auto-detects a
GPU with real headroom beyond the T4 this was tuned for (see
detect_fast_training_config) and skips gradient checkpointing / widens
the micro-batch on it, holding the effective batch size (and every other
hyperparameter) fixed -- not a config override, an engineering-only
speedup. Only the Trainer plumbing + Drive checkpoint callback is new in
this module."""

import json

from agentic_sentiment.phase1.config import DataConfig, ModelConfig, TrainingConfig
from agentic_sentiment.phase1.data.dataset import build_sft_dataset
from agentic_sentiment.phase1.models.lora import (
    detect_fast_training_config,
    detect_use_4bit,
    load_model_and_tokenizer,
)

from agentic_sentiment.train.run_dir import RunDir


def build_model_config() -> ModelConfig:
    return ModelConfig()


def build_training_config() -> TrainingConfig:
    return TrainingConfig()


def _build_trainer(model, tokenizer, train_ds, eval_ds, training_cfg, run_dir: RunDir):
    """Matches colab/llama_sentiment_baseline_train.ipynb cell 17's
    TrainingArguments/data collator/gradient-checkpointing wiring exactly
    (report_to is the one deliberate difference: "none" here, since this
    runs non-interactively and doesn't need a tensorboard setup)."""
    from transformers import Trainer, TrainingArguments
    from transformers.data.data_collator import DataCollatorForSeq2Seq

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding="max_length",
        max_length=training_cfg.max_seq_length,
        pad_to_multiple_of=8 if training_cfg.fp16 else None,
        label_pad_token_id=-100, return_tensors="pt",
    )
    args = TrainingArguments(
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
    if getattr(training_cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=data_collator, callbacks=[run_dir.checkpoint_callback()],
    )


def run_training(data_path: str, run_dir: RunDir, overrides: dict | None = None) -> dict:
    import dataclasses

    model_cfg = build_model_config()
    training_cfg = build_training_config()
    data_cfg = DataConfig()
    # Same effective batch (3x4=12) and same gradient math either way --
    # only skips the T4-only checkpointing tax on a GPU that doesn't need
    # it. Explicit overrides (if any) still win over the auto-detected ones.
    merged_overrides = {**detect_fast_training_config(), **(overrides or {})}
    if merged_overrides:
        training_cfg = dataclasses.replace(training_cfg, **merged_overrides)

    run_dir.write_config({
        "model": dataclasses.asdict(model_cfg),
        "training": dataclasses.asdict(training_cfg),
        "data": dataclasses.asdict(data_cfg),
        "data_path": data_path,
    })

    model, tokenizer = load_model_and_tokenizer(model_cfg, use_4bit=detect_use_4bit())
    train_ds, eval_ds = build_sft_dataset(data_path, tokenizer, data_cfg, training_cfg, seed=training_cfg.seed)

    trainer = _build_trainer(model, tokenizer, train_ds, eval_ds, training_cfg, run_dir)
    # Resume from the latest Drive-side checkpoint if this run_id already has
    # one (a reconnect under the same run_id), instead of restarting at step 0.
    trainer.train(resume_from_checkpoint=run_dir.latest_checkpoint())
    metrics = trainer.evaluate()

    run_dir.save_best(trainer, tokenizer, "best_adapter")
    (run_dir.path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
