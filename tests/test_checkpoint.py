import json

from agentic_sentiment.eval.checkpoint import append_result, iter_done_ids


def test_append_result_writes_jsonl_line(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    append_result(str(path), {"review_id": "r1", "final_rating": 5})
    append_result(str(path), {"review_id": "r2", "final_rating": 3})

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["review_id"] == "r1"


def test_iter_done_ids_empty_when_no_file(tmp_path):
    assert iter_done_ids(str(tmp_path / "missing.jsonl")) == set()


def test_iter_done_ids_reflects_appended_records(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    append_result(str(path), {"review_id": "r1"})
    append_result(str(path), {"review_id": "r2"})

    assert iter_done_ids(str(path)) == {"r1", "r2"}


def test_resume_skips_done_ids_simulated_crash(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    all_ids = ["r1", "r2", "r3"]

    # First "run": crashes after r1, r2.
    for rid in all_ids[:2]:
        append_result(str(path), {"review_id": rid})

    # "Resume": only r3 should be processed.
    done = iter_done_ids(str(path))
    remaining = [rid for rid in all_ids if rid not in done]
    assert remaining == ["r3"]
