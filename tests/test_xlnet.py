import pytest

from agentic_sentiment.baselines.xlnet import build_training_args
from agentic_sentiment.train.run_dir import RunDir


def test_build_training_args_uses_run_dir_output(tmp_path):
    run_dir = RunDir(base_dir=str(tmp_path), run_id="xlnet_run")
    args = build_training_args(run_dir, output_dir=str(tmp_path / "out"))

    assert args.output_dir == str(tmp_path / "out")
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"


def test_build_training_args_checkpoints_sparsely_enough_for_drive_quota(tmp_path):
    # xlnet-large checkpoints are ~4.3GB each; too-frequent saves can
    # exhaust a free-tier Drive quota mid-training.
    run_dir = RunDir(base_dir=str(tmp_path), run_id="xlnet_run")
    args = build_training_args(run_dir, output_dir=str(tmp_path / "out"))
    assert args.save_steps >= 1000
    assert args.eval_steps >= 1000


def test_compute_metrics_matches_classical_evaluate_shape():
    pytest.importorskip("transformers")
    import numpy as np

    from agentic_sentiment.baselines.xlnet import _compute_metrics

    logits = np.array([[0, 0, 0, 0, 5], [5, 0, 0, 0, 0], [0, 5, 0, 0, 0]])  # argmax -> 4, 0, 1
    labels = np.array([4, 0, 1])

    metrics = _compute_metrics((logits, labels))

    assert metrics["accuracy"] == 1.0
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
