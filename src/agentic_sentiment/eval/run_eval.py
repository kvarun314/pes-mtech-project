"""Ablation runner + metrics for the Phase 2 graph eval."""

import random

from sklearn.metrics import f1_score

from agentic_sentiment.agents.graph import build_graph
from agentic_sentiment.eval.checkpoint import append_result, iter_done_ids
from agentic_sentiment.eval.one_shot import build_one_shot_pool, pick_one_shot_text

ABLATIONS = {
    "text_only": {"use_metadata": False, "use_image": False, "use_rag": False},
    "plus_metadata": {"use_metadata": True, "use_image": False, "use_rag": False},
    "plus_image": {"use_metadata": True, "use_image": True, "use_rag": False},
    "plus_rag_reviews": {"use_metadata": True, "use_image": True, "use_rag": True},
    "plus_rag_specs": {"use_metadata": True, "use_image": True, "use_rag": True},  # spec_store passed in
    "full_graph": {"use_metadata": True, "use_image": True, "use_rag": True},
}


def run_ablation(name: str, rows: list[dict], llm_fn, checkpoint_path: str, spec_store=None,
                  dissonance_method: str = "norm", max_correction_iters: int = 2) -> None:
    flags = ABLATIONS[name]
    graph = build_graph(llm_fn=llm_fn, spec_store=spec_store, max_correction_iters=max_correction_iters)
    done = iter_done_ids(checkpoint_path)
    # One-shot + CoT framing is task framing, not multimodal context -- it's
    # given to the Analyst for every ablation (matching both proven Phase 2
    # notebooks and how the adapter was trained), never toggled by the
    # text_only/+metadata/+image/+rag ladder above.
    one_shot_pool = build_one_shot_pool(rows)

    for row in rows:
        if row["review_id"] in done:
            continue
        # Seeded by review_id (not a shared, loop-advancing rng) so the same
        # row gets the same one-shot example whether this call is a fresh
        # run or a resume, and across every ablation.
        rng = random.Random(row["review_id"])
        state = {
            **row, **flags,
            "one_shot_example": pick_one_shot_text(row, one_shot_pool, rng),
            "use_cot": True,
            "dissonance_method": dissonance_method,
            "correction_iters": 0,
            "max_correction_iters": max_correction_iters,
        }
        result = graph.invoke(state)
        append_result(checkpoint_path, {
            "review_id": row["review_id"],
            "gt_rating": row["gt_rating"],
            "final_rating": result["final_rating"],
            "analyst_rating": result.get("analyst_rating"),
            "visual_rating": result.get("visual_rating"),
            "rag_rating": result.get("rag_rating"),
            "rag_grounding": result.get("rag_grounding"),
            "dissonance": result.get("dissonance"),
            "self_corrected": result.get("self_corrected", False),
            "correction_iters": result.get("correction_iters", 0),
            "rationale": result.get("rationale", ""),
        })


def compute_metrics(records: list[dict], key: str = "final_rating") -> dict:
    n = len(records)
    gt = [r["gt_rating"] for r in records]
    pred = [r[key] for r in records]

    accuracy = sum(g == p for g, p in zip(gt, pred)) / n
    mae = sum(abs(g - p) for g, p in zip(gt, pred)) / n
    macro_f1 = f1_score(gt, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)

    confusion = [[0] * 5 for _ in range(5)]
    for g, p in zip(gt, pred):
        confusion[g - 1][p - 1] += 1

    return {"accuracy": accuracy, "macro_f1": macro_f1, "mae": mae, "confusion": confusion, "n": n}
