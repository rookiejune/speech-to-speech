from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...datamodule import ModelBatch
from ...generation.evaluation import evaluate
from ...model.acoustic import FlowModel, RVQModel
from ...runtime.types import AcousticCodec
from ..interval import TrainInterval


class AcousticEvaluation(Callback):
    def __init__(
        self,
        model: FlowModel | RVQModel,
        batch: ModelBatch,
        codec: AcousticCodec,
        output_dir: Path,
        *,
        every_n_steps: int | None,
        every_audio_seconds: float | None = None,
        seeds: Sequence[int],
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.codec = codec
        self.path = output_dir / "evaluation.json"
        self.every_n_steps = every_n_steps
        self.interval = TrainInterval(
            every_n_steps=every_n_steps,
            every_audio_seconds=every_audio_seconds,
        )
        self.seeds = tuple(seeds)
        self.values: dict[int, dict[str, float]] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.evaluate(trainer, 0)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        should_run = self.interval.should_run(trainer, pl_module, batch)
        if trainer.is_global_zero and should_run:
            self.evaluate(trainer, trainer.global_step)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.evaluate(trainer, trainer.global_step)

    def evaluate(self, trainer: Trainer, step: int) -> None:
        if step in self.values:
            return
        metrics = evaluate(self.model, self.batch, self.codec, seeds=self.seeds)
        self.values[step] = metrics
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {f"evaluation/{name}": value for name, value in metrics.items()},
                step=step,
            )
        self.path.write_text(
            json.dumps(
                {str(key): value for key, value in self.values.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "interval": self.interval.state_dict(),
            "values": {str(key): value for key, value in self.values.items()},
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        interval = state_dict.get("interval", {})
        if isinstance(interval, dict):
            self.interval.load_state_dict(cast(dict[str, float], interval))
        values = state_dict.get("values", {})
        if isinstance(values, dict):
            self.values = {
                int(key): cast(dict[str, float], value)
                for key, value in values.items()
            }
