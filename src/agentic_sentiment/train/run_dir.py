"""Drive-backed run directory: config dump, best-checkpoint tracking, and a
TrainerCallback that copies every HF checkpoint to Drive as it is saved, so a
Colab disconnect never loses the best adapter or its hyperparameters."""

import json
import time
from pathlib import Path


class RunDir:
    def __init__(self, base_dir: str, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H%M%S") + f"_{int(time.time() * 1e6) % 1_000_000}"
        self.path = Path(base_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "checkpoints").mkdir(exist_ok=True)
        self._best_value: float | None = None

    def write_config(self, config: dict) -> None:
        (self.path / "config.json").write_text(json.dumps(config, indent=2, default=str))

    def write_best(self, step: int, metric_name: str, value: float, greater_is_better: bool = False) -> None:
        is_better = (
            self._best_value is None
            or (value > self._best_value if greater_is_better else value < self._best_value)
        )
        if not is_better:
            return
        self._best_value = value
        (self.path / "BEST.md").write_text(
            f"# Best checkpoint\n\nstep {step}\n{metric_name} = {value}\n"
        )

    def checkpoint_callback(self):
        """Returns a transformers.TrainerCallback that copies each new HF
        checkpoint dir into this RunDir's checkpoints/ on Drive, and updates
        BEST.md when eval_loss improves. Import of transformers is deferred
        so this module has no hard dependency on it."""
        import shutil
        from transformers import TrainerCallback

        run_dir = self

        class DriveCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if src.is_dir():
                    dst = run_dir.path / "checkpoints" / src.name
                    shutil.copytree(src, dst, dirs_exist_ok=True)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and "eval_loss" in metrics:
                    run_dir.write_best(state.global_step, "eval_loss", metrics["eval_loss"])

        return DriveCheckpointCallback()
