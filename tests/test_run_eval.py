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


def test_run_ablation_gives_the_analyst_a_one_shot_example(tmp_path):
    # Regression: both proven Phase 2 notebooks always give the Analyst a
    # one-shot example; the Analyst prompt must actually contain one.
    seen_one_shot = {"present": False}

    def llm_fn(prompt: str) -> str:
        if "Example:\nReview:" in prompt:
            seen_one_shot["present"] = True
        return "Sentiment (1-5): 5. stub."

    varied_rows = [
        {"review_id": f"v{i}", "review_text": f"review {i}", "gt_rating": (i % 5) + 1,
         "meta_title": "W", "meta_description": "", "image_caption": "a widget", "asin": "B1"}
        for i in range(10)
    ]
    ckpt = tmp_path / "text_only.jsonl"
    run_ablation("text_only", varied_rows, llm_fn=llm_fn, checkpoint_path=str(ckpt))

    assert seen_one_shot["present"] is True


def test_run_ablation_one_shot_choice_is_stable_across_resume(tmp_path):
    # The same row must get the same one-shot example whether this is a
    # fresh run or a resume -- seeded by review_id, not loop position.
    captured = []

    def llm_fn(prompt: str) -> str:
        if "Example:" in prompt and "review 9" in prompt:
            captured.append(prompt)
        return "Sentiment (1-5): 5. stub."

    varied_rows = [
        {"review_id": f"v{i}", "review_text": f"review {i}", "gt_rating": (i % 5) + 1,
         "meta_title": "W", "meta_description": "", "image_caption": "a widget", "asin": "B1"}
        for i in range(10)
    ]

    run_ablation("text_only", varied_rows, llm_fn=llm_fn, checkpoint_path=str(tmp_path / "a.jsonl"))
    fresh_prompt_for_v9 = captured[0]

    captured.clear()
    ckpt_b = str(tmp_path / "b.jsonl")
    run_ablation("text_only", varied_rows[:5], llm_fn=llm_fn, checkpoint_path=ckpt_b)  # v9 not reached yet
    run_ablation("text_only", varied_rows, llm_fn=llm_fn, checkpoint_path=ckpt_b)  # resume, reaches v9
    resumed_prompt_for_v9 = captured[0]

    assert fresh_prompt_for_v9 == resumed_prompt_for_v9


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
