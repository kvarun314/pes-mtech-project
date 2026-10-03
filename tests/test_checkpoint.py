import json

from agentic_sentiment.eval.checkpoint import append_result, iter_done_ids, read_records


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


def test_iter_done_ids_skips_truncated_last_line(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    append_result(str(path), {"review_id": "r1"})
    # Simulate a crash mid-write: a truncated, invalid JSON trailing line.
    with open(path, "a") as f:
        f.write('{"review_id": "r2", "final_rat')

    assert iter_done_ids(str(path)) == {"r1"}


def test_read_records_skips_truncated_trailing_line(tmp_path):
    # read_records (not a bare [json.loads(l) for l in open(path)]) is
    # what the Colab notebooks use to read checkpoints back for metrics
    # -- the ablation's results are already safely on disk at that point;
    # a truncated last line shouldn't throw away the whole run's output.
    path = tmp_path / "ckpt.jsonl"
    append_result(str(path), {"review_id": "r1", "final_rating": 5})
    with open(path, "a") as f:
        f.write('{"review_id": "r2", "final_rat')  # truncated

    records = read_records(str(path))
    assert records == [{"review_id": "r1", "final_rating": 5}]


def test_read_records_empty_when_no_file(tmp_path):
    assert read_records(str(tmp_path / "missing.jsonl")) == []


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
