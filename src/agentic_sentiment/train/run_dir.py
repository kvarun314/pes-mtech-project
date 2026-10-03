"""Drive-backed run directory: config dump, best-checkpoint tracking, and a
TrainerCallback that copies every HF checkpoint to Drive as it is saved, so a
Colab disconnect never loses the best adapter or its hyperparameters."""

import json
import time
from pathlib import Path


def _escape_md(text: str) -> str:
    """A `|` or newline in a value would otherwise break a Markdown table
    row's column structure."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _flatten_scalars(config: dict, prefix: str = "") -> list[str]:
    """`key=value` strings for every scalar in `config`, recursing into
    nested dicts (e.g. phase1_train's {"model": {...}, "training": {...}}
    config.json) with a dotted prefix, so history_row() doesn't silently
    show nothing but a few top-level fields for a nested config. JSON has
    no cycles, so this always terminates; the caller truncates the overall
    string to 120 chars regardless of how deep a config happens to nest."""
    parts = []
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            parts.extend(_flatten_scalars(value, prefix=f"{name}."))
        elif not isinstance(value, list):
            parts.append(f"{name}={value}")
    return parts


class RunDir:
    def __init__(self, base_dir: str, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H%M%S") + f"_{int(time.time() * 1e6) % 1_000_000}"
        self.path = Path(base_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "checkpoints").mkdir(exist_ok=True)

        # Seed best_value/best_step/best_metric_name from existing BEST.md if
        # it exists (for resume after disconnect)
        self._best_value, self._best_step, self._best_metric_name = self._load_best_from_file()

    def _load_best_from_file(self) -> tuple[float | None, int | None, str | None]:
        """Parse (value, step, metric_name) from existing BEST.md if it exists.

        Format: "# Best checkpoint\n\nstep N\nmetric_name = value\n"
        Returns (None, None, None) if not found/unparseable.
        """
        best_md_path = self.path / "BEST.md"
        if not best_md_path.exists():
            return None, None, None

        try:
            lines = [line.strip() for line in best_md_path.read_text().split("\n") if line.strip()]
            step = None
            for line in lines:
                if line.startswith("step "):
                    step = int(line.removeprefix("step ").strip())
            if not lines or " = " not in lines[-1]:
                return None, None, None
            metric_name, _, raw_value = lines[-1].partition(" = ")
            return float(raw_value), step, metric_name
        except (ValueError, IndexError):
            # Malformed file; fall back to None
            return None, None, None

    def write_config(self, config: dict) -> None:
        (self.path / "config.json").write_text(json.dumps(config, indent=2, default=str))

    def history_row(self, run_type: str) -> str:
        """One markdown table row summarizing this run, for a
        CHECKPOINT_HISTORY.md log: run_id, type, best step/metric, and the
        Drive path -- so every training run is documented even if nobody
        downloads the checkpoint itself. Call after training completes
        (reads config.json/BEST.md as they currently stand)."""
        config_path = self.path / "config.json"
        config_summary = "—"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                config_summary = ", ".join(_flatten_scalars(config))
            except (json.JSONDecodeError, OSError):
                pass
        if self._best_step is not None:
            best = f"step {self._best_step}, {self._best_metric_name}={self._best_value}"
        else:
            best = "—"
        return (
            f"| {_escape_md(self.run_id)} | {_escape_md(run_type)} | {_escape_md(best)} | "
            f"{_escape_md(config_summary[:120])} | `{self.path}` |"
        )

    def save_best(self, trainer, tokenizer, dir_name: str) -> str:
        """Save the best model into self.path/dir_name. Prefers the
        protected Drive-side best checkpoint over the Trainer's in-memory
        model: after a resume, HF's own `load_best_model_at_end` can try to
        reload from a local checkpoint path that no longer exists (the
        runtime was wiped by the disconnect), silently falling back to the
        final (not best) weights instead of raising -- copying the Drive
        checkpoint directly sidesteps that failure mode entirely. Falls
        back to `trainer.save_model()` if no eval has produced a best
        checkpoint yet (e.g. training ended before the first eval_steps)."""
        import shutil

        dest = self.path / dir_name
        best_dir = self.best_checkpoint_dir()
        if best_dir is not None:
            shutil.copytree(best_dir, dest, dirs_exist_ok=True)
        else:
            trainer.save_model(str(dest))
        tokenizer.save_pretrained(str(dest))
        return str(dest)

    def latest_checkpoint(self) -> str | None:
        """Path to the highest-step checkpoint already copied to Drive, or
        None if this run dir has none yet. Pass this to
        `Trainer.train(resume_from_checkpoint=...)` to resume a reconnected
        run instead of restarting at step 0 (which would also corrupt
        BEST.md/checkpoints/ if the run reuses the same run_id, since a
        fresh restart's early, worse evals would otherwise get copied over
        the prior attempt's better checkpoints)."""
        checkpoints = sorted(
            (self.path / "checkpoints").glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        return str(checkpoints[-1]) if checkpoints else None

    def best_checkpoint_dir(self) -> str | None:
        """Path to the Drive-side checkpoint directory for the best step
        recorded in BEST.md, if that checkpoint still exists -- this is the
        protected copy `checkpoint_callback()`'s pruning never deletes, so
        prefer it over trusting an in-memory `trainer.model` after a resume
        (where HF's own best-checkpoint reload can point at a local path
        that no longer exists post-disconnect)."""
        if self._best_step is None:
            return None
        candidate = self.path / "checkpoints" / f"checkpoint-{self._best_step}"
        return str(candidate) if candidate.is_dir() else None

    def write_best(self, step: int, metric_name: str, value: float, greater_is_better: bool = False) -> None:
        is_better = (
            self._best_value is None
            or (value > self._best_value if greater_is_better else value < self._best_value)
        )
        if not is_better:
            return
        self._best_value = value
        self._best_step = step
        self._best_metric_name = metric_name
        (self.path / "BEST.md").write_text(
            f"# Best checkpoint\n\nstep {step}\n{metric_name} = {value}\n"
        )

    def checkpoint_callback(self, keep: int = 2):
        """Returns a transformers.TrainerCallback that copies each new HF
        checkpoint dir into this RunDir's checkpoints/ on Drive (pruning all
        but the `keep` most recent AND the current best, so an unattended
        run with many save steps can't silently fill the Drive quota --
        and crucially can't prune away the one checkpoint `best_checkpoint_dir()`
        depends on, which would otherwise make a post-resume "best" save
        silently fall back to the final, not actually-best, weights), and
        updates BEST.md when eval_loss improves. Import of transformers is
        deferred so this module has no hard dependency on it."""
        import shutil
        from transformers import TrainerCallback

        run_dir = self

        class DriveCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if not src.is_dir():
                    return
                dst_dir = run_dir.path / "checkpoints"
                dst = dst_dir / src.name
                shutil.copytree(src, dst, dirs_exist_ok=True)

                existing = sorted(
                    dst_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]),
                )
                best_name = f"checkpoint-{run_dir._best_step}" if run_dir._best_step is not None else None
                prunable = [p for p in existing if p.name != best_name]
                for stale in prunable[:-keep] if keep > 0 else prunable:
                    shutil.rmtree(stale, ignore_errors=True)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and "eval_loss" in metrics:
                    run_dir.write_best(state.global_step, "eval_loss", metrics["eval_loss"])

        return DriveCheckpointCallback()
