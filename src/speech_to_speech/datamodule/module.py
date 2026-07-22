from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypedDict, cast

from anydataset.dataset import AnyDataset, MergedDataset
from anydataset.store import StoreLocalBatchSampler
from anydataset.store.reader import StoreDataset
from anydataset.types import AudioView, Modality, Role, Sample as RawSample
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from typing_extensions import NotRequired

from ..task import Task
from .collator import Collator
from .dataset import DatasetConfig, load_dataset
from .protocol import DataRuntime, DataRuntimeSnapshot, DatasetRuntime
from .types import ModelBatch


class DataLoaderConfig(TypedDict):
    batch_size: int
    num_workers: int
    pin_memory: NotRequired[bool]
    persistent_workers: NotRequired[bool]


@dataclass
class Config:
    codec: str
    dataloader: DataLoaderConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self) -> None:
        batch_size = self.dataloader["batch_size"]
        num_workers = self.dataloader["num_workers"]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("dataloader batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("dataloader batch_size must be positive.")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int):
            raise TypeError("dataloader num_workers must be an integer.")
        if num_workers < 0:
            raise ValueError("dataloader num_workers must be non-negative.")
        for name in ("pin_memory", "persistent_workers"):
            value = self.dataloader.get(name, False)
            if not isinstance(value, bool):
                raise TypeError(f"dataloader {name} must be a boolean.")


class DataModule(LightningDataModule):
    def __init__(
        self,
        config: Config,
        runtime: DatasetRuntime,
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
        self._train_dataset = load_dataset(self.config.dataset, self.runtime)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self.collator.set_task_weights(task_weights)

    def train_samples(self, indices: Sequence[int]) -> list[RawSample]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before reading samples.")
        return [self._train_dataset[index] for index in indices]

    def train_dataloader(self) -> Iterable[ModelBatch]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before train_dataloader().")
        loader = self.config.dataloader
        num_workers = loader["num_workers"]
        if not isinstance(self.collator.runtime, DataRuntimeSnapshot):
            snapshot = DataRuntimeSnapshot.from_runtime(self.runtime)
            self.collator.runtime = cast(DataRuntime, cast(object, snapshot))
        store_dataset = _store_group_dataset(self._train_dataset)
        if store_dataset is not None:
            return DataLoader(
                self._train_dataset,
                batch_sampler=StoreLocalBatchSampler(
                    store_dataset,
                    batch_size=loader["batch_size"],
                    views=_audio_views(self.runtime.audio_view),
                    shuffle=True,
                ),
                num_workers=num_workers,
                pin_memory=loader.get("pin_memory", False),
                persistent_workers=(
                    loader.get("persistent_workers", False) and num_workers > 0
                ),
                collate_fn=self.collator,
            )
        return DataLoader(
            self._train_dataset,
            batch_size=loader["batch_size"],
            num_workers=num_workers,
            pin_memory=loader.get("pin_memory", False),
            persistent_workers=(
                loader.get("persistent_workers", False) and num_workers > 0
            ),
            collate_fn=self.collator,
        )


def _store_group_dataset(dataset: object) -> StoreDataset | None:
    if isinstance(dataset, StoreDataset):
        return dataset
    if isinstance(dataset, AnyDataset):
        source = dataset.dataset
        return source if isinstance(source, StoreDataset) else None
    if isinstance(dataset, MergedDataset):
        left = _store_group_dataset(dataset.left)
        if left is not None:
            return left
        return _store_group_dataset(dataset.right)
    return None


def _audio_views(audio_view: AudioView) -> tuple[tuple[Role, Modality, AudioView], ...]:
    return (
        (Role.SOURCE, Modality.AUDIO, audio_view),
        (Role.TARGET, Modality.AUDIO, audio_view),
    )
