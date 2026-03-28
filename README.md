# pes-mtech-project

Final semester project — **LLaMA-3-8B + LoRA** sentiment (1–5) on Amazon-style reviews, aligned with **itmconf_dai2024_04021**.

## Goal

**Get close to the paper** (pipeline, prompts, metrics), not an exact match to every table cell. Colab / data / seeds will differ from the authors’ run.

## Contents

- **`colab/llama_sentiment_baseline_train.ipynb`** — Phase 1 training, eval, adapter export.
- **`colab/phase1_vs_phase2_amazon2023_presentation.ipynb`** — **Amazon Reviews 2023** demo: **Phase 1** (LLaMA+LoRA, text-only) vs **context-enriched** prompt (metadata + BLIP caption), same adapter.
- **`colab/phase2_agentic_vs_phase1_amazon2023.ipynb`** — **LangGraph** demo: Analyst → RAG-style grounding → visual (BLIP) → Critic vs **Phase 1** single pass; Colab-oriented (HF token, `/content` paths).
- **`assets/`** — e.g. Phase 2 architecture image.

See repo root **`CODE_EXPLANATION.md`** and **`IMPLEMENTATION_STATUS.md`** (if present in your checkout) for details.
