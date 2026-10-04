"""XLNet-large-cased baseline (Wang et al. Table 3), Drive-checkpointed via
RunDir the same way as the Phase 1 LoRA run. Training itself needs a GPU and
`transformers`/`torch`, so it is validated on Colab (notebook 02), not
locally — only the config construction is unit tested here."""

import json

from agentic_sentiment.train.run_dir import RunDir

MODEL_NAME = "xlnet-large-cased"


def build_training_args(run_dir: RunDir, output_dir: str, num_train_epochs: int = 3,
                         per_device_train_batch_size: int = 4, learning_rate: float = 2e-5,
                         num_train_samples: int | None = None):
    from transformers import TrainingArguments

    # Without warmup, AdamW hits full LR against a randomly-initialized
    # classification head (sequence_summary/logits_proj, not in the
    # xlnet-large checkpoint -- see the LOAD REPORT's MISSING rows) from
    # step 1 on a tiny batch_size=4 -- observed in practice as a fast,
    # permanent collapse to predicting one majority class (identical
    # accuracy/precision/recall/F1 at every eval step from step 1000
    # onward, the textbook always-predict-one-class signature: macro
    # precision == accuracy/5, macro recall == 1/5). Wang et al.'s own
    # XLNet baseline (Table 3) is a real, non-degenerate classifier
    # (acc 0.916, recall 0.973) -- that paper's Table 1 hyperparameters
    # are LoRA/LLaMA3-only and say nothing about XLNet's own training, so
    # this isn't a paper-mandated value to preserve, just a missing
    # standard safeguard against this exact failure mode. Installed
    # transformers (5.x) dropped warmup_ratio -- only warmup_steps remains,
    # so compute the equivalent 10% directly from the actual train size
    # instead of guessing a fixed step count that may be wildly off for a
    # different dataset size.
    if num_train_samples:
        total_steps = -(-num_train_samples // per_device_train_batch_size) * num_train_epochs
        warmup_steps = max(1, int(0.1 * total_steps))
    else:
        warmup_steps = 100

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        save_strategy="steps",
        # xlnet-large is ~1.4GB of weights + ~2.9GB of optimizer state per
        # checkpoint (~4.3GB); at save_steps=100 over a multi-thousand-step
        # run that's dozens of Drive copies and can exhaust a free-tier
        # Drive quota mid-training. 1000 keeps total Drive traffic bounded
        # while still checkpointing often enough to resume cheaply.
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
    )


def _compute_metrics(eval_pred):
    """accuracy + macro P/R/F1, matching classical.evaluate()'s metric
    shape -- without this, trainer.evaluate() reports only eval_loss and
    there is nothing comparable to the classical baselines' table."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def run_training(texts: list[str], labels: list[int], run_dir: RunDir,
                  eval_texts: list[str] | None = None, eval_labels: list[int] | None = None,
                  local_output_dir: str = "/tmp/xlnet_output") -> dict:
    """Trains on (texts, labels). If (eval_texts, eval_labels) are given,
    evaluates on exactly that held-out set -- pass the SAME X_test/y_test
    the classical baselines use, so XLNet's reported accuracy/F1 are
    comparable to theirs in the results table, not computed on a
    different internal split. Falls back to an internal 80/20 split of
    (texts, labels) if no explicit eval set is given."""
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
    from sklearn.model_selection import train_test_split

    run_dir.write_config({
        "model": MODEL_NAME, "num_samples": len(texts), "seed": 42,
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

    if eval_texts is not None and eval_labels is not None:
        train_texts, train_labels = texts, [l - 1 for l in labels]
        eval_texts_, eval_labels_ = eval_texts, [l - 1 for l in eval_labels]
    else:
        train_texts, eval_texts_, train_labels, eval_labels_ = train_test_split(
            texts, [l - 1 for l in labels], test_size=0.2, random_state=42
        )

    def tokenize(batch_texts):
        return tokenizer(batch_texts, truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_dict({**tokenize(train_texts), "labels": train_labels})
    eval_ds = Dataset.from_dict({**tokenize(eval_texts_), "labels": eval_labels_})

    # local_output_dir must NOT live under run_dir.path -- that's on Drive,
    # and the checkpoint_callback already copies the checkpoints Drive-side;
    # staging HF's own (unpruned) checkpoint-* dirs there too would double
    # Drive usage for no benefit.
    args = build_training_args(run_dir, output_dir=local_output_dir, num_train_samples=len(train_texts))
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=_compute_metrics, callbacks=[run_dir.checkpoint_callback()],
    )
    # Resume from the latest Drive-side checkpoint if this run_id already
    # has one (a reconnect under the same run_id), instead of silently
    # restarting at step 0 while BEST.md/checkpoints/ still reflect the
    # prior attempt.
    trainer.train(resume_from_checkpoint=run_dir.latest_checkpoint())
    metrics = trainer.evaluate()

    run_dir.save_best(trainer, tokenizer, "best_model")
    (run_dir.path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
