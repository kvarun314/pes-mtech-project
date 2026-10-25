"""One LLaMA-3-8B Q4 model in memory for both phases: Phase 1 = adapter R on (first-digit probabilities, as eval_p1.py);
Phase 2 probe = LoRA scale 0 (base model, L20 last-token state, as probe_agent4.py)."""
import math
import joblib
import numpy as np
from agentic_sentiment.v3 import paths as PT

PROBE_PROMPT = "Review: {text}\nRating (1-5):"


class SharedLlama:
    def __init__(self, base: str = PT.MLX_BASE, adapter: str | None = PT.ADAPTER_R):
        from mlx_lm import load
        from mlx_lm.tuner.lora import LoRALinear
        self.model, self.tok = load(base, adapter_path=adapter)
        self.loras = [m for _, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        self.scale_on = self.loras[0].scale if self.loras else 0.0
        self.sp = self.tok.encode(" ", add_special_tokens=False)

    def lora(self, on: bool):
        for m in self.loras:
            m.scale = self.scale_on if on else 0.0


class ProbeAgent:
    def __init__(self, llama: SharedLlama, scaler, clf):
        self.llama, self.scaler, self.clf = llama, scaler, clf

    def hidden(self, text: str) -> np.ndarray:
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        self.llama.lora(False); m = self.llama.model
        ids = self.llama.tok.encode(PROBE_PROMPT.format(text=text[:1200]), add_special_tokens=False) + self.llama.sp
        h = m.model.embed_tokens(mx.array(ids)[None]); mask = create_attention_mask(h, None)
        for layer in m.model.layers[:20]:
            h = layer(h, mask, None)
        v = h[0, -1]; mx.eval(v)
        return np.array(v.astype(mx.float16)).astype(np.float32)

    def probs(self, text: str) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(self.hidden(text)[None]))[0]


def load_probe(llama: SharedLlama) -> ProbeAgent:
    d = joblib.load(PT.PROBE_JOBLIB)
    return ProbeAgent(llama, d["scaler"], d["clf"])


def phase1_probs(llama: SharedLlama, text: str) -> np.ndarray:
    """Phase 1 Analyst (LLaMA + LoRA arm R): softmax over first-digit logits 1-5, same prompt as eval_p1.py."""
    import mlx.core as mx
    from agentic_sentiment.phase1.config import DataConfig
    from agentic_sentiment.phase1.data.preprocessing import build_prompt
    llama.lora(True); tok = llama.tok
    dig = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]
    ids = tok.encode(build_prompt(text[:1200], DataConfig(use_one_shot=False)), add_special_tokens=False) + llama.sp
    lg = llama.model(mx.array(ids)[None])[0, -1]; v = [float(lg[d]) for d in dig]; mx_ = max(v); e = [math.exp(x - mx_) for x in v]
    return np.array(e) / sum(e)
