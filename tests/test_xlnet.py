from agentic_sentiment.baselines.xlnet import build_training_args
from agentic_sentiment.train.run_dir import RunDir


def test_build_training_args_uses_run_dir_output(tmp_path):
    run_dir = RunDir(base_dir=str(tmp_path), run_id="xlnet_run")
    args = build_training_args(run_dir, output_dir=str(tmp_path / "out"))

    assert args.output_dir == str(tmp_path / "out")
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
