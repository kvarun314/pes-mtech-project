import json, os
from agentic_sentiment.v3 import demo


def test_cached_records_then_replays(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    monkeypatch.setenv("DEMO_RECORD", "1"); assert demo.cached("k", lambda: [1, 2], str(p)) == [1, 2]
    monkeypatch.delenv("DEMO_RECORD")
    assert demo.cached("k", lambda: 1 / 0, str(p)) == [1, 2]                     # live failure -> cached value
    monkeypatch.setenv("DEMO_OFFLINE", "1"); assert demo.cached("k", lambda: [9], str(p)) == [1, 2]


def test_paper_line_format(capsys):
    demo.paper_line("SH eval exact", "67.5"); assert capsys.readouterr().out == "PAPER: SH eval exact 67.5\n"
