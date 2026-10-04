"""Ablation runner + metrics for the Phase 2 graph eval."""

import random
import time

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
        start = time.perf_counter()
        result = graph.invoke(state)
        latency_s = time.perf_counter() - start
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
            "latency_s": latency_s,
        })


def average_grounding(records: list[dict]) -> float | None:
    """Faithfulness (mission spec §14): mean rag_grounding across rows
    where RAG ran (None where it didn't, e.g. text_only/plus_metadata/
    plus_image). None if no row has a grounding score."""
    scores = [r["rag_grounding"] for r in records if r.get("rag_grounding") is not None]
    return sum(scores) / len(scores) if scores else None


def average_correction_iters(records: list[dict]) -> float:
    """Average reflection edges per input (mission spec §14)."""
    if not records:
        return 0.0
    return sum(r.get("correction_iters", 0) for r in records) / len(records)


def average_latency(records: list[dict]) -> float | None:
    values = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    return sum(values) / len(values) if values else None


def error_recovery(baseline_records: list[dict], improved_records: list[dict]) -> dict:
    """Net error recovery between two ablations on the same rows (e.g.
    text_only -> full_graph), matched by review_id so misaligned row order
    between the two checkpoints can't silently produce a wrong count."""
    baseline_by_id = {r["review_id"]: r for r in baseline_records}
    fixed = regressed = 0
    for row in improved_records:
        rid = row["review_id"]
        if rid not in baseline_by_id:
            continue
        base = baseline_by_id[rid]
        base_correct = base["final_rating"] == base["gt_rating"]
        improved_correct = row["final_rating"] == row["gt_rating"]
        if not base_correct and improved_correct:
            fixed += 1
        elif base_correct and not improved_correct:
            regressed += 1
    return {"fixed": fixed, "regressed": regressed, "net": fixed - regressed}


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
