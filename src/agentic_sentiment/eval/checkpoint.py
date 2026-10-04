"""JSONL checkpoint + resume, shared by every long Colab eval/baseline run so
a disconnect only costs the in-flight sample, not the whole run."""

import json
from pathlib import Path


def append_result(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def iter_done_ids(path: str) -> set[str]:
    return {rid for record in read_records(path) if (rid := record.get("review_id")) is not None}


def read_records(path: str) -> list[dict]:
    """Every record in the checkpoint, skipping a truncated/malformed
    trailing line (a crash can interrupt the last write) instead of
    raising -- read the checkpoint back with this, not a bare
    `[json.loads(l) for l in open(path)]`, once a long ablation run has
    already finished and only the metrics step is left; losing the whole
    run's results to one bad line at the very end is the worst time for
    a strict parser to raise."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
