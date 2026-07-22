from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from anydataset.types import Sample as RawSample
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from ..task import Task
from .collator import Collator
from .dataset import DatasetConfig, load_dataset
from .protocol import DatasetRuntime
from .types import ModelBatch


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
