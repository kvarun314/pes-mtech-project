"""JSONL checkpoint + resume, shared by every long Colab eval/baseline run so
a disconnect only costs the in-flight sample, not the whole run."""

import json
from pathlib import Path


def append_result(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def iter_done_ids(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["review_id"])
    return done
