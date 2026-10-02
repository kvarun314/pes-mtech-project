"""
LLaMA-3-8B + LoRA loading (itmconf_dai2024_04021, Table 1).
Freeze base model, train only LoRA on last 3 layers.

Peft and transformers are imported only when load_model_and_tokenizer runs
(not at module import), so the rest of the app can run without loading peft.
"""

import torch
from agentic_sentiment.phase1.config import ModelConfig


def get_lora_config(cfg: ModelConfig):
    """Build PEFT LoRA config. Only last trainable_layers get LoRA."""
    from peft import LoraConfig, TaskType
    num_layers = 32
    trainable_layers = min(cfg.trainable_layers, num_layers)
    layers_to_transform = (
        list(range(num_layers - trainable_layers, num_layers))
        if trainable_layers < num_layers
        else None
    )

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules or ["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
        layers_to_transform=layers_to_transform,
        layers_pattern="layers",
    )


def freeze_base_model(model) -> None:
    """Freeze all base (non-LoRA) parameters (paper: Freeze technique)."""
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False


def print_frozen_trainable_summary(model) -> None:
    """Print frozen vs trainable layer summary."""
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_layers: set[int] = set()
    for n in trainable_names:
        if "model.layers." in n:
            idx = n.split("model.layers.")[1].split(".")[0]
            if idx.isdigit():
                trainable_layers.add(int(idx))

    n_trainable = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad
    )
    n_frozen = sum(
        p.numel() for n, p in model.named_parameters() if not p.requires_grad
    )
    print("--- Frozen vs trainable (paper: Freeze technique, trainable layers: 3) ---", flush=True)
    print(
        "  Frozen: all base model params (embedding + layers 0–28) + base in last 3.",
        flush=True,
    )
    print(f"  Trainable: LoRA adapters on decoder layers {sorted(trainable_layers)}.", flush=True)
    print(f"  Trainable params: {n_trainable:,}  |  Frozen params: {n_frozen:,}", flush=True)


def load_model_and_tokenizer(
    model_cfg: ModelConfig,
    use_4bit: bool = False,
    device_map: str | None = "auto",
):
    """Load LLaMA-3-8B and tokenizer, apply LoRA, freeze base."""
    print("  [1/4] Loading transformers...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("  [1/4] Loading tokenizer (downloads if not cached)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        use_fast=model_cfg.use_fast_tokenizer,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  [1/4] Tokenizer ready.", flush=True)

    print("  [2/4] Loading model weights (downloads if not cached; may take several minutes)...", flush=True)
    model_kwargs: dict = {
        "trust_remote_code": model_cfg.trust_remote_code,
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        **model_kwargs,
    )
    print("  [2/4] Model weights loaded.", flush=True)

    print("  [3/4] Applying LoRA adapters (loading peft, then last 3 layers)...", flush=True)
    from peft import get_peft_model
    lora_config = get_lora_config(model_cfg)
    model = get_peft_model(model, lora_config)
    print("  [3/4] LoRA applied.", flush=True)

    print("  [4/4] Freezing base model (only LoRA will train)...", flush=True)
    freeze_base_model(model)
    model.print_trainable_parameters()
    print_frozen_trainable_summary(model)
    print("  [4/4] Done. Model and tokenizer ready.", flush=True)
    return model, tokenizer
