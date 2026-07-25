from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TypedDict, cast

from anydataset.dataset import MapStyleABC
from anydataset.types import AudioView, Sample as RawSample
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from typing_extensions import NotRequired

from ..task import Task
from .collator import Collator
from .dataset import DatasetConfig, load_dataset
from .lba import LBA, LBAConfig, PlannerMode, metadata_speech_length, speech_length
from .protocol import DataRuntime, DataRuntimeSnapshot, DatasetRuntime
from .single import SingleCollator
from .types import ConcreteTrainInput, DataShape


class DataLoaderConfig(TypedDict):
    batch_size: int
    num_workers: int
    pin_memory: NotRequired[bool]
    persistent_workers: NotRequired[bool]
    lba: NotRequired[LBAConfig]


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
        lba = self.dataloader.get("lba")
        if lba is not None and not isinstance(lba, LBAConfig):
            raise TypeError("dataloader lba must be an LBAConfig.")


class DataModule(LightningDataModule):
    def __init__(
        self,
        config: Config,
        runtime: DatasetRuntime,
        task_weights: Mapping[Task, float],
        *,
        output_dir: Path | None = None,
        loader_name: str = "speech",
    ) -> None:
        super().__init__()

        self.config = config
        self.runtime = runtime
        self.collator = _collator(config, runtime, task_weights)
        self.output_dir = output_dir
        self.loader_name = loader_name
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
        lba = loader.get("lba")
        if lba is not None and lba.enabled:
            if self.config.shape is DataShape.SINGLE:
                raise ValueError("single data shape does not support LBA yet.")
            source_loader = _source_loader(
                self._train_dataset,
                loader=loader,
                collate_fn=self.collator,
            )
            if source_loader is not None:
                return LBA(
                    self._train_dataset,
                    batch_sampler=source_loader.batch_sampler,
                    num_workers=num_workers,
                    pin_memory=loader.get("pin_memory", False),
                    persistent_workers=(
                        loader.get("persistent_workers", False) and num_workers > 0
                    ),
                    collate_fn=self.collator,
                    len_fn=partial(
                        metadata_speech_length,
                        audio_view=self.runtime.audio_view,
                        frame_rate=self.runtime.codec_frame_rate,
                        tasks=tuple(self.collator.tasks),
                        config=lba,
                    ),
                    max_padded_length=lba.max_batch_cost,
                    max_padding_ratio=lba.max_padding_ratio,
                    prefetch_batches=lba.prefetch_batches,
                    planner_mode=cast(PlannerMode, lba.planner_mode),
                    drop_last_flush=lba.drop_last_flush,
                    log_dir=_lba_log_dir(self.output_dir, self.loader_name),
                )
            return LBA(
                self._train_dataset,
                batch_size=loader["batch_size"],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=loader.get("pin_memory", False),
                persistent_workers=(
                    loader.get("persistent_workers", False) and num_workers > 0
                ),
                collate_fn=self.collator,
                len_fn=partial(
                    speech_length,
                    runtime=cast(DataRuntime, self.collator.runtime),
                    tasks=tuple(self.collator.tasks),
                    config=lba,
                ),
                max_padded_length=lba.max_batch_cost,
                max_padding_ratio=lba.max_padding_ratio,
                prefetch_batches=lba.prefetch_batches,
                planner_mode=cast(PlannerMode, lba.planner_mode),
                drop_last_flush=lba.drop_last_flush,
                log_dir=_lba_log_dir(self.output_dir, self.loader_name),
            )
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
    config: Config,
    runtime: DatasetRuntime,
    task_weights: Mapping[Task, float],
):
    if config.shape is DataShape.PAIR:
        return Collator(runtime, task_weights)
    if config.shape is DataShape.SINGLE:
        return SingleCollator(
            runtime,
            task_weights,
            encode_missing_codes=config.encode_missing_codes,
        )
    raise AssertionError(f"unsupported data shape: {config.shape}")


def _unit_cost(_index: int) -> int:
    return 1


def _lba_log_dir(output_dir: Path | None, loader_name: str) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / "lba" / loader_name
