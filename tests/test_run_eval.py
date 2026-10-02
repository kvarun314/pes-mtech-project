import json

from agentic_sentiment.eval.run_eval import ABLATIONS, compute_metrics, run_ablation


def _stub_llm(prompt: str) -> str:
    return "Sentiment (1-5): 5. stub."


ROWS = [
    {"review_id": f"r{i}", "review_text": "ok", "gt_rating": 5, "meta_title": "W",
     "meta_description": "", "image_caption": "a widget", "asin": "B1"}
    for i in range(3)
]


def test_ablations_includes_text_only_and_full_graph():
    assert "text_only" in ABLATIONS
    assert "full_graph" in ABLATIONS
    assert ABLATIONS["text_only"] == {"use_metadata": False, "use_image": False, "use_rag": False}


def test_run_ablation_writes_one_record_per_row(tmp_path):
    ckpt = tmp_path / "text_only.jsonl"
    run_ablation("text_only", ROWS, llm_fn=_stub_llm, checkpoint_path=str(ckpt))

    lines = ckpt.read_text().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[0])["final_rating"] == 5


def test_run_ablation_resumes_without_reprocessing(tmp_path):
    ckpt = tmp_path / "text_only.jsonl"
    run_ablation("text_only", ROWS[:2], llm_fn=_stub_llm, checkpoint_path=str(ckpt))
    run_ablation("text_only", ROWS, llm_fn=_stub_llm, checkpoint_path=str(ckpt))  # resumed run

    lines = ckpt.read_text().strip().split("\n")
    assert len(lines) == 3  # not 5 — the first 2 weren't reprocessed


def test_compute_metrics_accuracy_and_mae():
    records = [
        {"gt_rating": 5, "final_rating": 5},
        {"gt_rating": 5, "final_rating": 4},
        {"gt_rating": 1, "final_rating": 1},
    ]
    metrics = compute_metrics(records)
    assert metrics["accuracy"] == 2 / 3
    assert round(metrics["mae"], 4) == round(1 / 3, 4)
    assert len(metrics["confusion"]) == 5
