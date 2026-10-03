from agentic_sentiment.eval.results_log import format_results_entry


def test_format_results_entry_includes_heading_context_and_table():
    entry = format_results_entry(
        run_label="Phase 2 ablations",
        context={"dataset": "Amazon Reviews 2023, Electronics", "n_per_rating": 400, "seed": 42},
        metrics_by_name={
            "text_only": {"accuracy": 0.161, "macro_f1": 0.0945, "mae": 1.52},
            "full_graph": {"accuracy": 0.387, "macro_f1": 0.2725, "mae": 0.96},
        },
        date="2026-10-05",
    )

    assert "## 2026-10-05 — Phase 2 ablations" in entry
    assert "**dataset:** Amazon Reviews 2023, Electronics" in entry
    assert "**seed:** 42" in entry
    assert "| text_only |" in entry
    assert "| full_graph |" in entry
    assert "0.1610" in entry  # accuracy formatted to 4 decimals
    assert "0.3870" in entry


def test_format_results_entry_handles_mismatched_metric_keys_across_rows():
    entry = format_results_entry(
        run_label="baselines",
        context={"dataset": "Kaggle amazon-product-reviews"},
        metrics_by_name={
            "decision_tree": {"accuracy": 0.5, "precision": 0.4, "recall": 0.4, "f1": 0.4},
            "xlnet": {"accuracy": 0.6, "eval_loss": 0.9},  # different keys than decision_tree
        },
    )

    assert "| decision_tree |" in entry
    assert "| xlnet |" in entry
    assert "—" in entry  # xlnet's missing precision/recall/f1 cells


def test_format_results_entry_defaults_to_today_when_date_omitted():
    import datetime

    entry = format_results_entry(run_label="x", context={}, metrics_by_name={"a": {"accuracy": 1.0}})
    assert datetime.date.today().isoformat() in entry
