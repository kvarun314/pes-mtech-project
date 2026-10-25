import json, os, glob
import numpy as np
import pytest
from agentic_sentiment.v3 import paths as PT
from agentic_sentiment.v3.llama import PROBE_PROMPT

HAVE_MODEL = os.path.exists(PT.MLX_BASE + "/config.json") and os.path.exists(PT.ADAPTER_R)


def test_probe_prompt_matches_extraction_scripts():
    assert PROBE_PROMPT.format(text="ok") == "Review: ok\nRating (1-5):"


@pytest.mark.skipif(not HAVE_MODEL, reason="local MLX model not present")
def test_lora_off_reproduces_stored_hidden_state():
    """With adapter R loaded but LoRA scale 0, the L20 state must equal the stored base-model state (hid3 dev2 row 0)."""
    from agentic_sentiment.v3.llama import SharedLlama, ProbeAgent
    llama = SharedLlama()
    ids = json.load(open(PT.D + "hid3/ids.json"))["dev2"]
    row = {json.loads(l)["review_id"]: json.loads(l) for l in open(PT.D + "dev2.jsonl")}[ids[0]]
    stored = np.load(sorted(glob.glob(PT.D + "hid3/dev2_*.npy"))[0])[0, :4096].astype(np.float32)
    live = ProbeAgent(llama, None, None).hidden(row["review_text"])
    assert np.abs(live - stored).max() < 0.05          # fp16 storage tolerance
    llama.lora(True)
    assert np.abs(ProbeAgent(llama, None, None).hidden(row["review_text"]) - stored).max() < 0.05  # hidden() forces LoRA off
