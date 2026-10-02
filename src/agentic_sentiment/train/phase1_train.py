"""Phase 1 LoRA training, wired to Drive checkpointing via RunDir. Reuses
the Phase 1 data/model code verbatim from agentic_sentiment.phase1 (vendored
from llama_sentiment_baseline) — only the Trainer plumbing + Drive callback
is new here."""

import dataclasses
import json

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
        overwrite_output_dir=training_cfg.overwrite_output_dir,
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
    (run_dir.path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
