"""Formats a Markdown entry for docs/RESULTS.md, per the mission spec's own
§7 convention: date, config, dataset slice, metrics, deviations -- for
every real training/eval run, not held only in a Colab session's memory."""

import datetime


def format_results_entry(
    run_label: str,
    context: dict,
    metrics_by_name: dict[str, dict],
    date: str | None = None,
) -> str:
    """One Markdown block: a heading, a context line, and a metrics table.

    `context` is a flat dict of run metadata (dataset, sample size, seed,
    model, etc.) rendered as "key: value" lines. `metrics_by_name` maps a
    run/ablation name to its metrics dict (as `compute_metrics()` or
    `evaluate()` return) -- rendered as one table row per name, one column
    per metric key that appears in ANY of the entries (so callers can mix
    classical-baseline metrics with ablation metrics in one table without
    every row needing every column).
    """
    date = date or datetime.date.today().isoformat()

    lines = [f"## {date} — {run_label}", ""]
    for key, value in context.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")

    columns: list[str] = []
    for metrics in metrics_by_name.values():
        for key in metrics:
            if key not in columns:
                columns.append(key)

    lines.append("| name | " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * (len(columns) + 1))
    for name, metrics in metrics_by_name.items():
        row = [name] + [_fmt(metrics.get(col)) for col in columns]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return "(see confusion matrix in checkpoint)" if len(value) > 3 else str(value)
    return str(value)
