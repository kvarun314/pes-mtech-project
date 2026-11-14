# Phase 1 run card — the LLaMA row (70.7% exact / 85.0% polarity on the paper-protocol set)

## Model (saved: `output/final/adapter_config.json`, `training_args.bin`)
- Base: `meta-llama/Meta-Llama-3-8B`. LoRA r=4, alpha=64, dropout 0, layers 29-31 only, target q_proj + v_proj. Paper Table 1 config.
- Training: lr 5e-5, 3 epochs, batch 3 x grad-accum 4, warmup 0, cosine, adamw_torch, fp16, max grad norm 1.0, seed 42, save every 100.
- NOT saved: which rows it trained on, how many (paper Table 1 says 2,000; our pipeline default is 4,000), the Colab training log/loss curve.
  Consequence: test-row disjointness from training is statistical only (4K of ~568K rows).

## Inference (saved in code/logs; no single config file existed before this card)
- Base weights actually used: `.scratch/gguf/Meta-Llama-3-8B.Q4_K_M.gguf` (community 4-bit requant `QuantFactory/Meta-Llama-3-8B-GGUF`),
  NOT the fp16 weights the adapter was trained on. Adapter: `.scratch/phase1_lora.gguf` (converted with llama.cpp `convert_lora_to_gguf.py`).
- Server: `llama-server -m <gguf> --lora <lora.gguf> -c 8192 --parallel 1 --port 8090` (log: `v2_reflective/llama-server.log`).
- Request: temperature 0, n_predict 2, n_probs 50. Prompt: `agents/graph.py` ANALYST_PROMPT (instruction + CoT phrase + one-shot example).
  The one-shot example comes from `eval/one_shot.py`, chosen with a rating different from the row's TRUE rating (mild label leak, inherited).
- Prediction = argmax over P(1..5) at the first digit token (script `.scratch/probe_rating_probs.py`). Matched stored generated predictions 999/1000 earlier.

## Test set and scoring
- `v2_reflective/paper_protocol_rows.jsonl`: 4,000 random Kaggle rows, seed 2024, natural distribution
  (1: 8.8%, 2: 5.6%, 3: 7.1%, 4: 14.8%, 5: 63.8%), disjoint from the balanced eval and dev sets. Builder: `build_paper_protocol_test.py`.
- Scorer `v2_reflective/score.py`; raw output `paper_protocol_probs.jsonl`. Polarity = 1-2 negative / 3 neutral / 4-5 positive (paper §2.2).

## Accuracy caveats for the table
- 70.7% and 85.0% are two views of the same predictions (5-class exact vs polarity). The paper's 92.6% is one unlabelled "accuracy";
  our evidence says it is polarity-like, so the fair comparison is 85.0% vs 92.6%, and the paper protocol is not fully known.
- Q4 base vs fp16 training weights, and the one-shot label leak, could each move results by about a point in either direction; not measured.
