from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict

from anydataset.types import Sample as RawSample
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .collator import Collator
from .protocol import DataRuntime
from ..task import Task
from .types import ModelBatch


class DataLoaderConfig(TypedDict):
    batch_size: int
    num_workers: int


@dataclass
class Config:
    codec: str
    dataloader: DataLoaderConfig


class DataModule(LightningDataModule):
    def __init__(
        self,
        config: Config,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
    ) -> None:
        super().__init__()

        self.config = config
        self.runtime = runtime
        self.collator = Collator(runtime, task_weights)
        self._train_dataset = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._train_dataset is not None:
            return
        runtime_codec = self.runtime.codec_name
        if self.config.codec != runtime_codec:
            raise ValueError(
                "datamodule and runtime must use the same codec: "
                f"{self.config.codec!r} != {runtime_codec!r}."
            )
        from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

        self._train_dataset = wmt19_tts_codec(codec=self.config.codec)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self.collator.set_task_weights(task_weights)

    def train_samples(self, indices: Sequence[int]) -> list[RawSample]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before reading samples.")
        return [self._train_dataset[index] for index in indices]

    def train_dataloader(self) -> Iterable[ModelBatch]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before train_dataloader().")
        return DataLoader(
            self._train_dataset,
            **self.config.dataloader,
            persistent_workers=False,
            collate_fn=self.collator,
        )
