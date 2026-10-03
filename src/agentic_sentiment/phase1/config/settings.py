"""
Configuration for LLaMA-3-8B sentiment baseline (itmconf_dai2024_04021).
All Table 1 hyperparameters are set below.

Table 1 checklist (paper):
  learning rate 0.00005, epochs 3, max grad norm 1, max samples 2000, compute fp16,
  cutoff 1024, batch 3, grad accum 4, val 0.2, LR scheduler cosine, logging 5, save 100,
  warmup 0, NEFTune 0.02, optimizer adamw_torch, resize tok emb 0, upcast layernorm 1,
  enable external logger 1, trainable layers 3, LoRA r 4, alpha 64, dropout 0,
  LoRA+ LR ratio 8, use rsLoRA 0, use DoRA 0, Beta 0.5, Ftx gamma 2, loss sigmoid,
  use Galore 0, Galore rank/scale/update 0, use BAdam 0, BAdam mode 0,
  enable S2 attention 0, switch mode 0, pack sequences 0, update ratio 0, LLaMA Pro 0.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Base model and LoRA settings. Defaults match the proven, paper-closer
    config in colab/llama_sentiment_baseline_train.ipynb cell 9/13 (not the
    Table 1 literal baseline r=4/alpha=64/3-layers below it in that paper's
    own Table 1 text) -- this is the config that actually produced Run 1's
    Phase 1 numbers."""

    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True

    lora_r: int = 16
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: list | None = None
    trainable_layers: int = 8

    use_rslora: bool = False
    use_dora: bool = False
    lora_lr_ratio: float = 8.0
    beta: float = 0.5
    ftx_gamma: float = 2.0

    enable_s2_attention: bool = False
    pack_sequences: bool = False
    enable_llama_pro: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters. Defaults match
    colab/llama_sentiment_baseline_train.ipynb cell 9/16 exactly (the proven
    config), not Table 1's literal baseline values."""

    output_dir: str = "./output"
    overwrite_output_dir: bool = True

    num_train_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 3
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    max_seq_length: int = 1024
    fp16: bool = True
    bf16: bool = False

    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 200

    logging_steps: int = 5
    save_steps: int = 50
    save_total_limit: int | None = 3
    evaluation_strategy: str = "steps"
    eval_steps: int | None = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    val_ratio: float = 0.2

    max_samples: int | None = 4000
    max_eval_samples: int | None = None

    # Needed at this LoRA size (r=16, 8 layers) to avoid OOM on a T4 -- the
    # notebook enables it by default and wires it into both
    # TrainingArguments and model.gradient_checkpointing_enable().
    gradient_checkpointing: bool = True

    neftune_alpha: float = 0.02
    upcast_layernorm: bool = True
    resize_token_embeddings: bool = False
    loss_type: str = "sigmoid"
    enable_external_logger: bool = True
    seed: int = 42

    use_galore: bool = False
    galore_rank: int = 0
    galore_scale: float = 0.0
    update_interval: int = 0
    use_badam: bool = False
    badam_mode: int = 0
    switch_mode: bool = False
    update_ratio: float = 0.0


@dataclass
class DataConfig:
    """Data and prompt settings. Defaults match
    colab/llama_sentiment_baseline_train.ipynb cell 9/16 exactly."""

    dataset_path: str | None = None
    text_column: str = "review_text"
    label_column: str = "rating"

    negative_labels: tuple = (1, 2)
    neutral_labels: tuple = (3,)
    positive_labels: tuple = (4, 5)

    # Paper 500k-row pool; 250k balances Colab RAM vs diversity (set None for the full file).
    max_csv_rows: Optional[int] = 250_000

    # Paper 2.1 Dataset sampling
    use_textblob_filter: bool = True  # DQC: discard polarity–rating misaligned reviews
    use_stratified_sampling: bool = True  # equal number per rating 1–5
    samples_per_rating: Optional[int] = None  # if set, use this per class; else from max_total
    stratified_max_total: Optional[int] = 10_000  # larger pool before VGST/cap (paper-aligned diversity)
    use_vgst: bool = True  # VGST ~1% (of post-DQC pool, floored at max_samples) diversity sampling
    vgst_batch_size: int = 32
    vgst_wishlist_len: int = 10
    vgst_target_ratio: float = 0.01

    # Paper 2.2 "deliberately increased their representation" for neutral
    oversample_neutral: bool = True
    neutral_oversample_ratio: float = 2.0

    system_prompt: str = (
        "Evaluate the sentiment expressed in user reviews and classify each one "
        "according to its sentiment rating. Use a five-point scale: "
        "1–2 negative, 3 neutral, 4–5 positive."
    )
    use_one_shot: bool = True
    use_cot: bool = True
    cot_phrase: str = "Let's take it one step at a time."


def get_default_config() -> dict:
    """Return default model, training, and data configs as in the paper."""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "data": DataConfig(),
    }
