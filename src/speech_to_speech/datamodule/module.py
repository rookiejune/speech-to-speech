from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from anydataset.dataset import MapStyleABC
from anydataset.types import Sample as RawSample
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset

from .._compat import StrEnum, auto
from ..task import Task
from ._text import TextLoader
from .collator import Collator
from .config import DataLoaderConfig, SpeechConfig
from .dataset import load_dataset
from .diagnostic import SampleSplit
from .joint import LoaderSchedule, ScheduledDataLoader
from .protocol import (
    DataRuntime,
    DataRuntimeSnapshot,
    DatasetRuntime,
    TextRuntime,
)
from .single import SingleCollator
from .types import ConcreteTrainInput, DataShape, TrainInputBatch

if TYPE_CHECKING:
    from .text import TextConfig


class LoaderKind(StrEnum):
    SPEECH = auto()
    TEXT = auto()


@dataclass(frozen=True)
class LoaderSpec:
    kind: LoaderKind
    task_weights: Mapping[Task, float]
    speech_config: SpeechConfig | None = None
    text_config: TextConfig | None = None
    sample_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LoaderKind):
            raise TypeError("loader kind must be a LoaderKind.")
        if not isinstance(self.task_weights, Mapping):
            raise TypeError("loader task_weights must be a mapping.")
        if self.kind is LoaderKind.SPEECH:
            if self.speech_config is None or self.text_config is not None:
                raise ValueError(
                    "speech loaders require speech_config and must not set text_config."
                )
            _validate_sample_index(self.sample_index)
        else:
            if self.text_config is None or self.speech_config is not None:
                raise ValueError(
                    "text loaders require text_config and must not set speech_config."
                )
            if self.sample_index is not None:
                raise ValueError("text loaders do not support sample_index.")

    @classmethod
    def speech(
        cls,
        config: SpeechConfig,
        task_weights: Mapping[Task, float],
        *,
        sample_index: int | None = None,
    ) -> LoaderSpec:
        return cls(
            kind=LoaderKind.SPEECH,
            task_weights=task_weights,
            speech_config=config,
            sample_index=sample_index,
        )

    @classmethod
    def text(
        cls,
        config: TextConfig,
        task_weights: Mapping[Task, float],
    ) -> LoaderSpec:
        return cls(
            kind=LoaderKind.TEXT,
            task_weights=task_weights,
            text_config=config,
        )


class _Loader(Protocol):
    def setup(self, stage: str | None = None) -> None: ...


class _DiagnosticLoader(_Loader, Protocol):
    def train_samples(self, indices: Sequence[int]) -> list[RawSample]: ...

    def diagnostic_collator(
        self,
        task: Task,
    ) -> Callable[[list[RawSample]], ConcreteTrainInput]: ...


class _TrainLoader(_DiagnosticLoader, Protocol):
    @property
    def collator(self) -> Callable[[list[RawSample]], ConcreteTrainInput]: ...

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None: ...

    def train_dataloader(self) -> Iterable[ConcreteTrainInput]: ...


class _ValidationLoader(_DiagnosticLoader, Protocol):
    def validation_dataloader(self) -> Iterable[ConcreteTrainInput]: ...


class _SpeechLoader:
    def __init__(
        self,
        config: SpeechConfig,
        runtime: DatasetRuntime,
        task_weights: Mapping[Task, float],
        sample_index: int | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.collator = _collator(
            config.shape,
            runtime,
            task_weights,
            encode_missing_codes=config.encode_missing_codes,
        )
        self.sample_index = sample_index
        self._dataset = None
        self._subset: Subset[RawSample] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._dataset is not None:
            return
        try:
            runtime_codec = self.runtime.codec_name
        except AttributeError as error:
            raise TypeError("speech loaders require a DatasetRuntime.") from error
        if self.config.codec != runtime_codec:
            raise ValueError(
                "datamodule and runtime must use the same codec: "
                f"{self.config.codec!r} != {runtime_codec!r}."
            )
        self._dataset = load_dataset(self.config.dataset, self.runtime)
        if self.sample_index is not None:
            if self.sample_index >= len(cast(Sized, self._dataset)):
                raise IndexError(
                    f"sample_index {self.sample_index} is outside the training dataset."
                )
            self._subset = Subset(self._dataset, [self.sample_index])

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self.collator.set_task_weights(task_weights)

    def train_samples(self, indices: Sequence[int]) -> list[RawSample]:
        if self._dataset is None:
            raise RuntimeError("DataModule.setup() must run before reading samples.")
        return [self._dataset[index] for index in indices]

    def diagnostic_collator(
        self,
        task: Task,
    ) -> Callable[[list[RawSample]], ConcreteTrainInput]:
        return _collator(
            self.config.shape,
            self.runtime,
            {task: 1.0},
            encode_missing_codes=self.config.encode_missing_codes,
        )

    def train_dataloader(self) -> Iterable[ConcreteTrainInput]:
        return self._dataloader(shuffle=True)

    def validation_dataloader(self) -> Iterable[ConcreteTrainInput]:
        return self._dataloader(shuffle=False)

    def _dataloader(self, *, shuffle: bool) -> Iterable[ConcreteTrainInput]:
        if self._dataset is None:
            raise RuntimeError(
                "speech loader setup() must run before building a loader."
            )
        if self._subset is not None:
            return DataLoader(
                self._subset,
                batch_size=1,
                num_workers=0,
                collate_fn=self.collator,
            )
        loader = self.config.dataloader
        num_workers = loader.num_workers
        if not isinstance(self.collator.runtime, DataRuntimeSnapshot):
            snapshot = DataRuntimeSnapshot.from_runtime(self.runtime)
            self.collator.runtime = cast(DataRuntime, cast(object, snapshot))
        source_loader = _source_loader(
            self._dataset,
            loader=loader,
            collate_fn=self.collator,
            shuffle=shuffle,
        )
        if source_loader is not None:
            if source_loader.dataset is self._dataset:
                return cast(Iterable[ConcreteTrainInput], source_loader)
            return DataLoader(
                self._dataset,
                batch_sampler=source_loader.batch_sampler,
                num_workers=num_workers,
                pin_memory=loader.pin_memory,
                persistent_workers=(
                    loader.persistent_workers and num_workers > 0
                ),
                collate_fn=self.collator,
            )
        return DataLoader(
            self._dataset,
            batch_size=loader.batch_size,
            num_workers=num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=(
                loader.persistent_workers and num_workers > 0
            ),
            collate_fn=self.collator,
        )


class DataModule(LightningDataModule):
    def __init__(
        self,
        runtime: DatasetRuntime | TextRuntime,
        loaders: Mapping[str, LoaderSpec],
        schedule: LoaderSchedule | None = None,
        *,
        validation: LoaderSpec | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.loader_specs = dict(loaders)
        if not self.loader_specs:
            raise ValueError("DataModule requires at least one loader.")
        self._loaders = {
            name: _build_loader(spec, runtime)
            for name, spec in self.loader_specs.items()
        }
        self.validation_spec = validation
        self._validation_loader = (
            None
            if validation is None
            else _build_validation_loader(validation, runtime)
        )
        if any(not name for name in self.loader_specs):
            raise ValueError("loader names must not be empty.")
        self.schedule = schedule or LoaderSchedule(
            {name: 1.0 for name in self.loader_specs}
        )
        _validate_loader_names(self.loader_specs, self.schedule.weights)

    def setup(self, stage: str | None = None) -> None:
        for loader in self._loaders.values():
            loader.setup(stage)
        if self._validation_loader is not None:
            self._validation_loader.setup(stage)

    @property
    def loader_names(self) -> tuple[str, ...]:
        return tuple(self.loader_specs)

    def set_loader_weights(self, weights: Mapping[str, float]) -> None:
        schedule = LoaderSchedule(
            dict(weights),
            batches_per_step=self.schedule.batches_per_step,
        )
        _validate_loader_names(self.loader_specs, schedule.weights)
        self.schedule = schedule

    def set_task_weights(
        self,
        task_weights: Mapping[Task, float],
        *,
        loader_name: str | None = None,
    ) -> None:
        loader = self._single_loader(loader_name, "set task weights")
        loader.set_task_weights(task_weights)

    def train_samples(
        self,
        indices: Sequence[int],
        *,
        loader_name: str | None = None,
    ) -> list[RawSample]:
        loader = self._single_loader(loader_name, "read samples")
        return loader.train_samples(indices)

    def collator_for(
        self,
        loader_name: str | None = None,
    ) -> Callable[[list[RawSample]], ConcreteTrainInput]:
        loader = self._single_loader(loader_name, "read collator")
        return loader.collator

    def diagnostic_samples(
        self,
        indices: Sequence[int],
        *,
        split: SampleSplit,
        loader_name: str,
    ) -> list[RawSample]:
        return self._diagnostic_loader(split, loader_name).train_samples(indices)

    def diagnostic_collator(
        self,
        task: Task,
        *,
        split: SampleSplit,
        loader_name: str,
    ) -> Callable[[list[RawSample]], ConcreteTrainInput]:
        return self._diagnostic_loader(split, loader_name).diagnostic_collator(task)

    def train_dataloader(self) -> Iterable[TrainInputBatch]:
        loaders = {
            name: loader.train_dataloader() for name, loader in self._loaders.items()
        }
        if len(loaders) == 1 and self.schedule.batches_per_step == 1:
            return cast(Iterable[TrainInputBatch], next(iter(loaders.values())))
        return ScheduledDataLoader(loaders, self.schedule)

    def val_dataloader(self) -> Iterable[TrainInputBatch]:
        if self._validation_loader is None:
            return ()
        return cast(
            Iterable[TrainInputBatch],
            self._validation_loader.validation_dataloader(),
        )

    def _single_loader(self, loader_name: str | None, operation: str) -> _TrainLoader:
        if loader_name is None:
            if len(self._loaders) != 1:
                raise ValueError(
                    f"{operation} requires loader_name when DataModule has multiple loaders."
                )
            loader_name = next(iter(self._loaders))
        try:
            return self._loaders[loader_name]
        except KeyError as error:
            raise ValueError(f"unknown loader {loader_name!r}.") from error

    def _diagnostic_loader(
        self,
        split: SampleSplit,
        loader_name: str,
    ) -> _DiagnosticLoader:
        if not isinstance(split, SampleSplit):
            raise TypeError("diagnostic split must be a SampleSplit.")
        try:
            spec = self.loader_specs[loader_name]
        except KeyError as error:
            raise ValueError(f"unknown loader {loader_name!r}.") from error
        if split is SampleSplit.TRAIN:
            return self._loaders[loader_name]
        if spec.kind is not LoaderKind.SPEECH:
            raise ValueError("validation diagnostic samples require a speech loader.")
        loader = self._validation_loader
        if loader is None:
            raise RuntimeError(
                "validation diagnostic samples require a validation dataset."
            )
        return loader


def _build_loader(
    spec: LoaderSpec,
    runtime: DatasetRuntime | TextRuntime,
) -> _TrainLoader:
    if spec.kind is LoaderKind.SPEECH:
        assert spec.speech_config is not None
        return _SpeechLoader(
            spec.speech_config,
            cast(DatasetRuntime, runtime),
            spec.task_weights,
            sample_index=spec.sample_index,
        )
    assert spec.text_config is not None
    return TextLoader(
        spec.text_config,
        cast(TextRuntime, runtime),
        spec.task_weights,
    )


def _build_validation_loader(
    spec: LoaderSpec,
    runtime: DatasetRuntime | TextRuntime,
) -> _ValidationLoader:
    if spec.kind is not LoaderKind.SPEECH:
        raise ValueError("validation requires a speech loader.")
    assert spec.speech_config is not None
    return _SpeechLoader(
        spec.speech_config,
        cast(DatasetRuntime, runtime),
        spec.task_weights,
        sample_index=spec.sample_index,
    )


def _validate_sample_index(sample_index: int | None) -> None:
    if sample_index is None:
        return
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError("sample_index must be an integer or None.")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")


def _validate_loader_names(
    available: Mapping[str, object],
    scheduled: Mapping[str, object],
) -> None:
    missing = set(scheduled) - set(available)
    if missing:
        raise ValueError("scheduled loaders are missing: " + ", ".join(sorted(missing)))
    extra = set(available) - set(scheduled)
    if extra:
        raise ValueError("loader weights are missing: " + ", ".join(sorted(extra)))


def _source_loader(
    dataset: object,
    *,
    loader: DataLoaderConfig,
    collate_fn: Any,
    shuffle: bool,
) -> DataLoader[Any] | None:
    source = _source_dataset(dataset)
    if source is None:
        return None
    batch_size = loader.batch_size
    return source.dataloader(
        cost_fn=_unit_cost,
        max_batch_memory=batch_size,
        max_batch_samples=batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=(
            loader.persistent_workers and loader.num_workers > 0
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
    if shape is DataShape.PAIR:
        return Collator(
            runtime,
            task_weights,
            encode_missing_codes=encode_missing_codes,
        )
    if shape is DataShape.SINGLE:
        return SingleCollator(
            runtime,
            task_weights,
            encode_missing_codes=encode_missing_codes,
        )
    raise AssertionError(f"unsupported data shape: {shape}")


def _unit_cost(_index: int) -> int:
    return 1
