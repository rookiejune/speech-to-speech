from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from anydataset.dataset import MapStyleABC
from anydataset.types import Sample as RawSample
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from typing_extensions import NotRequired

from ..task import Task
from .collator import Collator
from .dataset import DatasetConfig, load_dataset
from .protocol import DataRuntime, DataRuntimeSnapshot, DatasetRuntime
from .single import SingleCollator
from .types import ConcreteTrainInput, DataShape


class DataLoaderConfig(TypedDict):
    batch_size: int
    num_workers: int
    pin_memory: NotRequired[bool]
    persistent_workers: NotRequired[bool]


@dataclass
class Config:
    codec: str
    dataloader: DataLoaderConfig
    shape: DataShape = DataShape.PAIR
    encode_missing_codes: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.shape, DataShape):
            raise TypeError("data shape must be a DataShape.")
        if not isinstance(self.encode_missing_codes, bool):
            raise TypeError("encode_missing_codes must be a boolean.")
        if self.encode_missing_codes and self.shape is not DataShape.SINGLE:
            raise ValueError("encode_missing_codes requires data shape single.")
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
        self.collator = _collator(
            config.shape,
            runtime,
            task_weights,
            encode_missing_codes=config.encode_missing_codes,
        )
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

    def train_dataloader(self) -> Iterable[ConcreteTrainInput]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before train_dataloader().")
        loader = self.config.dataloader
        num_workers = loader["num_workers"]
        if not isinstance(self.collator.runtime, DataRuntimeSnapshot):
            snapshot = DataRuntimeSnapshot.from_runtime(self.runtime)
            self.collator.runtime = cast(DataRuntime, cast(object, snapshot))
        source_loader = _source_loader(
            self._train_dataset,
            loader=loader,
            collate_fn=self.collator,
        )
        if source_loader is not None:
            if source_loader.dataset is self._train_dataset:
                return cast(Iterable[ConcreteTrainInput], source_loader)
            return DataLoader(
                self._train_dataset,
                batch_sampler=source_loader.batch_sampler,
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


class FixedDataModule(LightningDataModule):
    def __init__(
        self,
        codec: str,
        runtime: DatasetRuntime,
        task_weights: Mapping[Task, float],
        sample_index: int,
        *,
        shape: DataShape = DataShape.PAIR,
        encode_missing_codes: bool = False,
        dataset: DatasetConfig | None = None,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.runtime = runtime
        self.shape = shape
        self.encode_missing_codes = encode_missing_codes
        self.collator = _collator(
            shape,
            runtime,
            task_weights,
            encode_missing_codes=encode_missing_codes,
        )
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

    def train_dataloader(self) -> Iterable[ConcreteTrainInput]:
        if self._training is None:
            raise RuntimeError("FixedDataModule.setup() must run before training.")
        return DataLoader(
            self._training,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collator,
        )


def _source_loader(
    dataset: object,
    *,
    loader: DataLoaderConfig,
    collate_fn: Any,
) -> DataLoader[Any] | None:
    source = _source_dataset(dataset)
    if source is None:
        return None
    batch_size = loader["batch_size"]
    return source.dataloader(
        cost_fn=_unit_cost,
        max_batch_memory=batch_size,
        max_batch_samples=batch_size,
        shuffle=True,
        num_workers=loader["num_workers"],
        pin_memory=loader.get("pin_memory", False),
        persistent_workers=(
            loader.get("persistent_workers", False) and loader["num_workers"] > 0
        ),
        collate_fn=collate_fn,
    )


def _source_dataset(dataset: object) -> MapStyleABC | None:
    return dataset if isinstance(dataset, MapStyleABC) else None


def _collator(
    shape: DataShape,
    runtime: DatasetRuntime,
    task_weights: Mapping[Task, float],
    *,
    encode_missing_codes: bool = False,
):
    if encode_missing_codes and shape is not DataShape.SINGLE:
        raise ValueError("encode_missing_codes requires data shape single.")
    if shape is DataShape.PAIR:
        return Collator(runtime, task_weights)
    if shape is DataShape.SINGLE:
        return SingleCollator(
            runtime,
            task_weights,
            encode_missing_codes=encode_missing_codes,
        )
    raise AssertionError(f"unsupported data shape: {shape}")


def _unit_cost(_index: int) -> int:
    return 1
