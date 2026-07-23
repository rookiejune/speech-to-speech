from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
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
from .lba import LBA, LBAConfig, PlannerMode, speech_length
from .protocol import DataRuntime, DataRuntimeSnapshot, DatasetRuntime
from .types import ModelBatch


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
        self.collator = Collator(runtime, task_weights)
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

    def train_dataloader(self) -> Iterable[ModelBatch]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before train_dataloader().")
        loader = self.config.dataloader
        num_workers = loader["num_workers"]
        if not isinstance(self.collator.runtime, DataRuntimeSnapshot):
            snapshot = DataRuntimeSnapshot.from_runtime(self.runtime)
            self.collator.runtime = cast(DataRuntime, cast(object, snapshot))
        store_dataset = _store_group_dataset(self._train_dataset)
        lba = loader.get("lba")
        if lba is not None and lba.enabled:
            if store_dataset is not None:
                return LBA(
                    self._train_dataset,
                    batch_sampler=_store_batch_sampler(
                        store_dataset,
                        batch_size=loader["batch_size"],
                        audio_view=self.runtime.audio_view,
                    ),
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
        if store_dataset is not None:
            return DataLoader(
                self._train_dataset,
                batch_sampler=_store_batch_sampler(
                    store_dataset,
                    batch_size=loader["batch_size"],
                    audio_view=self.runtime.audio_view,
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


def _store_batch_sampler(
    dataset: StoreDataset,
    *,
    batch_size: int,
    audio_view: AudioView,
) -> StoreLocalBatchSampler:
    return StoreLocalBatchSampler(
        dataset,
        batch_size=batch_size,
        views=_audio_views(audio_view),
        shuffle=True,
    )


def _lba_log_dir(output_dir: Path | None, loader_name: str) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / "lba" / loader_name
