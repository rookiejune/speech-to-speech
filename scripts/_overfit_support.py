from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from anydataset.types import Sample as RawSample
from lightning import pytorch as pl
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from speech_to_speech.datamodule import (
    Collator,
    DatasetConfig,
    DatasetRuntime,
    ModelBatch,
    load_dataset,
)
from speech_to_speech.loss import Outputs, loss_items
from speech_to_speech.model import SpeechToSpeechFlowModel, SpeechToSpeechRVQModel
from speech_to_speech.reporting import window_summary
from speech_to_speech.runtime.types import Codec
from speech_to_speech.task import Task

if __package__:
    from ._acoustic_evaluation import evaluate
else:
    from _acoustic_evaluation import evaluate


class LossSummary(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, list[float]] = {}

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if not isinstance(outputs, Mapping):
            return
        typed_outputs = cast(Outputs, outputs)
        self._append("loss", typed_outputs["loss"])
        for name, item in loss_items(typed_outputs):
            self._append(name, item.loss)

    def report(self, window: int = 20) -> dict[str, dict[str, float | int | None]]:
        return {
            name: window_summary(values, window)
            for name, values in self.values.items()
        }

    def _append(self, name: str, value: Tensor) -> None:
        self.values.setdefault(name, []).append(float(value.detach().float().mean()))


class AcousticEvaluation(Callback):
    def __init__(
        self,
        model: SpeechToSpeechFlowModel | SpeechToSpeechRVQModel,
        batch: ModelBatch,
        codec: Codec,
        output_dir: Path,
        *,
        every_n_steps: int,
        seeds: Sequence[int],
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.codec = codec
        self.path = output_dir / "evaluation.json"
        self.every_n_steps = every_n_steps
        self.seeds = tuple(seeds)
        self.values: dict[int, dict[str, float]] = {}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.evaluate(trainer, 0)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if trainer.is_global_zero and trainer.global_step % self.every_n_steps == 0:
            self.evaluate(trainer, trainer.global_step)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.evaluate(trainer, trainer.global_step)

    def evaluate(self, trainer: pl.Trainer, step: int) -> None:
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


class FixedDataModule(LightningDataModule):
    def __init__(
        self,
        codec: str,
        runtime: DatasetRuntime,
        task_weights: Mapping[Task, float],
        sample_index: int,
        *,
        dataset: DatasetConfig | None = None,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.runtime = runtime
        self.collator = Collator(runtime, task_weights)
        self.sample_index = sample_index
        self.dataset_config = dataset or DatasetConfig()
        self._dataset: Dataset[RawSample] | None = None
        self._training: Subset[RawSample] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._dataset is not None:
            return
        if self.codec != self.runtime.codec_name:
            raise ValueError(
                "fixed datamodule and runtime must use the same codec: "
                f"{self.codec!r} != {self.runtime.codec_name!r}."
            )
        self._dataset = cast(
            Dataset[RawSample],
            cast(object, load_dataset(self.dataset_config, self.runtime)),
        )
        self._training = Subset(self._dataset, [self.sample_index])

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self.collator.set_task_weights(task_weights)

    def train_samples(self, indices: Sequence[int]) -> list[RawSample]:
        if self._dataset is None:
            raise RuntimeError("FixedDataModule.setup() must run before reading samples.")
        return [self._dataset[index] for index in indices]

    def train_dataloader(self) -> Iterable[ModelBatch]:
        if self._training is None:
            raise RuntimeError("FixedDataModule.setup() must run before training.")
        return DataLoader(
            self._training,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collator,
        )
