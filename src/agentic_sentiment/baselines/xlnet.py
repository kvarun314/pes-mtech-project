"""XLNet-large-cased baseline (Wang et al. Table 3), Drive-checkpointed via
RunDir the same way as the Phase 1 LoRA run. Training itself needs a GPU and
`transformers`/`torch`, so it is validated on Colab (notebook 02), not
locally — only the config construction is unit tested here."""

import json

from agentic_sentiment.train.run_dir import RunDir

MODEL_NAME = "xlnet-large-cased"


def build_training_args(run_dir: RunDir, output_dir: str, num_train_epochs: int = 3,
                         per_device_train_batch_size: int = 4, learning_rate: float = 2e-5):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
    )


def run_training(texts: list[str], labels: list[int], run_dir: RunDir, local_output_dir: str = "/tmp/xlnet_output") -> dict:
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
    from sklearn.model_selection import train_test_split

    run_dir.write_config({
        "model": MODEL_NAME, "num_samples": len(texts), "seed": 42,
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, [l - 1 for l in labels], test_size=0.2, random_state=42
    )

    def tokenize(batch_texts):
        return tokenizer(batch_texts, truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_dict({**tokenize(train_texts), "labels": train_labels})
    eval_ds = Dataset.from_dict({**tokenize(eval_texts), "labels": eval_labels})

    # local_output_dir must NOT live under run_dir.path -- that's on Drive,
    # and the checkpoint_callback already copies the checkpoints Drive-side;
    # staging HF's own (unpruned) checkpoint-* dirs there too would double
    # Drive usage for no benefit.
    args = build_training_args(run_dir, output_dir=local_output_dir)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[run_dir.checkpoint_callback()],
    )
    # Resume from the latest Drive-side checkpoint if this run_id already
    # has one (a reconnect under the same run_id), instead of silently
    # restarting at step 0 while BEST.md/checkpoints/ still reflect the
    # prior attempt.
    trainer.train(resume_from_checkpoint=run_dir.latest_checkpoint())
    metrics = trainer.evaluate()

    best_dir = run_dir.path / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (run_dir.path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
