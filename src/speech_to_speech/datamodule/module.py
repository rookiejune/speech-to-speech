from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast

from anydataset.dataset import MapStyleABC
from anydataset.types import AudioMeta, Modality, Sample as RawSample
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from .._compat import StrEnum, auto
from .asset import AssetJob, resolve_workspace_asset
from .loader.contract import ARFraming, validate_ar_framing
from ..task import Task, resolve_response
from .collate import Collator
from .config import TaskConfig
from .config import DataLoaderConfig, SpeechConfig
from .dataset.speech import _apply_split_manifest, load_dataset
from .dataset.text import TextLoader
from .diagnostic import SampleSplit
from .loader.schedule import LoaderSchedule, ScheduledDataLoader
from .contract import (
    DataRuntime,
    DataRuntimeSnapshot,
    DatasetRuntime,
    TextRuntime,
)
from .single import SingleCollator
from .batch import (
    TrainBatch,
    TrainInput,
)
from .sample import (
    AudioContextCostRow,
    DataShape,
)
from .streaming import (
    PublishedSample,
    SnapshotFeed,
    StreamingDataLoader,
    StreamingSnapshotDataset,
    StreamingTelemetry,
    SynthesisController,
    SynthesisRequest,
    WorkspaceSnapshotLoader,
    synthesis_controller,
    workspace_stream_root,
)

if TYPE_CHECKING:
    from .dataset.text import TextConfig


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
    trace: str | None = None
    ar_framing: ARFraming = ARFraming.INSTRUCTION
    max_samples: int | None = None
    task_configs: Mapping[Task, TaskConfig] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LoaderKind):
            raise TypeError("loader kind must be a LoaderKind.")
        if not isinstance(self.task_weights, Mapping):
            raise TypeError("loader task_weights must be a mapping.")
        if self.trace is not None and (
            not isinstance(self.trace, str) or not self.trace
        ):
            raise TypeError("loader trace must be a non-empty string or None.")
        validate_ar_framing(
            self.ar_framing,
            [task for task, weight in self.task_weights.items() if weight > 0],
        )
        for task, weight in self.task_weights.items():
            if weight > 0:
                resolve_response(
                    task,
                    trace=self.trace,
                )
        _validate_max_samples(self.max_samples)
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
        trace: str | None = None,
        ar_framing: ARFraming = ARFraming.INSTRUCTION,
        max_samples: int | None = None,
    ) -> LoaderSpec:
        return cls(
            kind=LoaderKind.SPEECH,
            task_weights=task_weights,
            speech_config=config,
            sample_index=sample_index,
            trace=trace,
            ar_framing=ar_framing,
            max_samples=max_samples,
        )

    @classmethod
    def text(
        cls,
        config: TextConfig,
        task_weights: Mapping[Task, float],
        *,
        trace: str | None = None,
        ar_framing: ARFraming = ARFraming.INSTRUCTION,
        max_samples: int | None = None,
        tasks: Mapping[Task, TaskConfig] | None = None,
    ) -> LoaderSpec:
        return cls(
            kind=LoaderKind.TEXT,
            task_weights=task_weights,
            text_config=config,
            max_samples=max_samples,
            trace=trace,
            ar_framing=ar_framing,
            task_configs=tasks,
        )


class _Loader(Protocol):
    def setup(self, stage: str | None = None) -> None: ...


class _DiagnosticLoader(_Loader, Protocol):
    def train_samples(self, indices: Sequence[int]) -> list[RawSample]: ...

    def diagnostic_collator(
        self,
        task: Task,
    ) -> Callable[[list[RawSample]], TrainInput]: ...


class _TrainLoader(_DiagnosticLoader, Protocol):
    def train_dataloader(self) -> Iterable[TrainInput]: ...


class _ValidationLoader(_DiagnosticLoader, Protocol):
    def validation_dataloader(self) -> Iterable[TrainInput]: ...


class _SpeechLoader:
    def __init__(
        self,
        config: SpeechConfig,
        runtime: DatasetRuntime,
        task_weights: Mapping[Task, float],
        sample_index: int | None = None,
        *,
        trace: str | None = None,
        ar_framing: ARFraming = ARFraming.INSTRUCTION,
        max_samples: int | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.collator = _collator(
            config.shape,
            runtime,
            task_weights,
            encode_missing_codes=config.encode_missing_codes,
            interleave_audio_frames=config.interleave_audio_frames,
            mask_text_ratio=config.mask_text_ratio,
            mask_audio_ratio=config.mask_audio_ratio,
            trace=trace,
            ar_framing=ar_framing,
            tasks=config.tasks,
        )
        self.sample_index = sample_index
        self.max_samples = max_samples
        self.num_workers = config.dataloader.num_workers
        self._dataset: Dataset[RawSample] | None = None
        self._subset: Subset[RawSample] | None = None
        self._asset_job: AssetJob | None = None
        self._streaming_dataset: StreamingSnapshotDataset | None = None
        self._streaming_loader: StreamingDataLoader | None = None
        self._pending_streaming_state: Mapping[str, object] | None = None
        self._synthesis_controller: SynthesisController | None = None
        self._streaming_stop_requested: Callable[[], bool] | None = None

    def setup(
        self,
        stage: str | None = None,
        *,
        dataset: Dataset[RawSample] | None = None,
    ) -> None:
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
        if dataset is not None:
            self._set_dataset(dataset)
            return
        streaming = self.config.streaming
        if streaming.enabled:
            root = workspace_stream_root(streaming.root or self.config.dataset.root)
            if not isinstance(self.collator.runtime, DataRuntimeSnapshot):
                snapshot = DataRuntimeSnapshot.from_runtime(self.runtime)
                self.collator.runtime = cast(DataRuntime, cast(object, snapshot))
            feed = SnapshotFeed(
                root,
                stream_id=cast(str, streaming.stream_id),
                expected_samples=cast(int, streaming.expected_samples),
                codec=self.runtime.codec_name,
                input_codec=self.runtime.input_codec_name,
                loader=WorkspaceSnapshotLoader(
                    codec=self.runtime.codec_name,
                    split=self.config.dataset.split,
                    input_codec=self.runtime.input_codec_name,
                ),
            )
            self._streaming_dataset = StreamingSnapshotDataset(
                feed,
                batch_size=self.config.dataloader.batch_size,
                poll_seconds=streaming.poll_seconds,
                status_seconds=streaming.status_seconds,
            )
            if self._streaming_stop_requested is not None:
                self._streaming_dataset.set_stop_requested(
                    self._streaming_stop_requested
                )
            self._streaming_loader = StreamingDataLoader(
                self._streaming_dataset,
                collate_fn=self.collator,
                pin_memory=self.config.dataloader.pin_memory,
            )
            pending = self._pending_streaming_state
            if pending is not None:
                self._streaming_loader.load_state_dict(pending)
                self._pending_streaming_state = None
            self._set_dataset(self._streaming_dataset)
            return
        materialization = self.config.materialization
        if materialization.enabled:
            resolution = resolve_workspace_asset(
                self.config.dataset,
                self.runtime,
                materialization,
            )
            self._asset_job = resolution.job
            self._set_dataset(
                _apply_split_manifest(resolution.dataset, self.config.dataset)
            )
            return
        self._set_dataset(load_dataset(self.config.dataset, self.runtime))

    @property
    def asset_job(self) -> AssetJob | None:
        return self._asset_job

    def refresh_materialized_asset(self) -> None:
        job = self._asset_job
        if job is None:
            return
        ready = job.load_ready()
        self._set_dataset(_apply_split_manifest(ready, self.config.dataset))
        self._asset_job = None

    def _set_dataset(self, dataset: Dataset[RawSample]) -> None:
        self._dataset = dataset
        self._subset = None
        if self.sample_index is not None:
            if self.sample_index >= len(cast(Sized, self._dataset)):
                raise IndexError(
                    f"sample_index {self.sample_index} is outside the training dataset."
                )
            self._subset = Subset(self._dataset, [self.sample_index])
        elif self.max_samples is not None:
            self._subset = Subset(
                self._dataset,
                range(min(self.max_samples, len(cast(Sized, self._dataset)))),
            )

    def train_samples(self, indices: Sequence[int]) -> list[RawSample]:
        if self._streaming_dataset is not None:
            published = self._streaming_dataset.feed.published(indices)
            if len(published) != len(indices):
                ready = {sample.index for sample in published}
                missing = [index for index in indices if index not in ready]
                raise RuntimeError(
                    "streaming diagnostic samples are not published yet: "
                    + ", ".join(str(index) for index in missing)
                )
            by_index = {sample.index: sample.sample for sample in published}
            return [by_index[index] for index in indices]
        if self._dataset is None:
            raise RuntimeError("DataModule.setup() must run before reading samples.")
        return [self._dataset[index] for index in indices]

    def published_samples(self, indices: Sequence[int]) -> list[PublishedSample]:
        dataset = self._streaming_dataset
        if dataset is None:
            return []
        return dataset.feed.published(indices)

    def streaming_telemetry(self) -> StreamingTelemetry | None:
        loader = self._streaming_loader
        if loader is None:
            return None
        return loader.telemetry()

    def set_streaming_stop_requested(self, requested: Callable[[], bool]) -> None:
        if not callable(requested):
            raise TypeError("streaming stop request must be callable.")
        self._streaming_stop_requested = requested
        dataset = self._streaming_dataset
        if dataset is not None:
            dataset.set_stop_requested(requested)

    def streaming_state_dict(self) -> dict[str, object] | None:
        loader = self._streaming_loader
        if loader is not None:
            return loader.state_dict()
        if self._pending_streaming_state is not None:
            return dict(self._pending_streaming_state)
        return None

    def load_streaming_state_dict(self, state: Mapping[str, object]) -> None:
        loader = self._streaming_loader
        if loader is None:
            self._pending_streaming_state = dict(state)
            return
        loader.load_state_dict(state)

    def set_streaming_global_step(self, step: int) -> None:
        if self._streaming_loader is not None:
            self._streaming_loader.set_global_step(step)

    def acknowledge_streaming_batch(self, global_step: int) -> None:
        if self._streaming_loader is None:
            raise RuntimeError("streaming batch acknowledgement requires its loader.")
        self._streaming_loader.acknowledge(global_step)

    def start_synthesis(self, *, owner: bool) -> None:
        streaming = self.config.streaming
        if not streaming.enabled or not owner or streaming.producer_factory is None:
            return
        if self._synthesis_controller is not None:
            return
        root = workspace_stream_root(streaming.root or self.config.dataset.root)
        controller = synthesis_controller(
            streaming.producer_factory,
            SynthesisRequest(
                root=root,
                stream_id=cast(str, streaming.stream_id),
                expected_samples=cast(int, streaming.expected_samples),
                codec=self.runtime.codec_name,
                split=self.config.dataset.split,
                options=dict(streaming.producer_options),
                input_codec=self.runtime.input_codec_name,
            ),
        )
        try:
            controller.start()
        except Exception:
            with suppress(Exception):
                controller.close()
            raise
        self._synthesis_controller = controller

    def check_synthesis(self, *, owner: bool) -> None:
        if not owner or self._synthesis_controller is None:
            return
        self._synthesis_controller.check()

    def close_synthesis(self, *, owner: bool) -> None:
        if not owner:
            return
        controller = self._synthesis_controller
        self._synthesis_controller = None
        if controller is not None:
            controller.close()

    def diagnostic_collator(
        self,
        task: Task,
    ) -> Callable[[list[RawSample]], TrainInput]:
        return _collator(
            self.config.shape,
            self.runtime,
            {task: 1.0},
            encode_missing_codes=self.config.encode_missing_codes,
            interleave_audio_frames=self.config.interleave_audio_frames,
            mask_text_ratio=self.config.mask_text_ratio,
            mask_audio_ratio=self.config.mask_audio_ratio,
            trace=self.collator.trace,
            ar_framing=self.collator.ar_framing,
            tasks=self.config.tasks,
        )

    def train_dataloader(self) -> Iterable[TrainInput]:
        return self._dataloader(shuffle=True)

    def validation_dataloader(self) -> Iterable[TrainInput]:
        return self._dataloader(shuffle=False)

    def _dataloader(self, *, shuffle: bool) -> Iterable[TrainInput]:
        if self._dataset is None:
            raise RuntimeError(
                "speech loader setup() must run before building a loader."
            )
        if self._subset is not None and self.sample_index is not None:
            _reject_enabled_costs(
                self.config.dataloader,
                "fixed-sample speech loaders",
            )
            return DataLoader(
                self._subset,
                batch_size=1,
                num_workers=0,
                collate_fn=self.collator,
            )
        if self._streaming_dataset is not None:
            if self._streaming_loader is None:
                raise RuntimeError("streaming dataset is missing its stateful loader.")
            return cast(Iterable[TrainInput], self._streaming_loader)
        loader = self.config.dataloader
        num_workers = self.num_workers
        if not isinstance(self.collator.runtime, DataRuntimeSnapshot):
            snapshot = DataRuntimeSnapshot.from_runtime(self.runtime)
            self.collator.runtime = cast(DataRuntime, cast(object, snapshot))
        dataset = self._dataset if self._subset is None else self._subset
        source_loader = _source_loader(
            dataset,
            loader=loader,
            num_workers=num_workers,
            collate_fn=self.collator,
            shuffle=shuffle,
            frame_rate=self.runtime.codec_frame_rate,
        )
        if source_loader is not None:
            if source_loader.dataset is dataset:
                return cast(Iterable[TrainInput], source_loader)
            return DataLoader(
                dataset,
                batch_sampler=source_loader.batch_sampler,
                num_workers=num_workers,
                pin_memory=loader.pin_memory,
                persistent_workers=(
                    loader.persistent_workers and num_workers > 0
                ),
                collate_fn=self.collator,
            )
        _reject_enabled_costs(loader, "non-MapStyle speech datasets")
        return DataLoader(
            dataset,
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
        validation: LoaderSpec | Mapping[str, LoaderSpec] | None = None,
        training_datasets: Mapping[str, Dataset[RawSample]] | None = None,
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
        self._training_datasets = dict(training_datasets or {})
        unknown_datasets = set(self._training_datasets) - set(self._loaders)
        if unknown_datasets:
            raise ValueError(
                "training dataset injection references unknown loaders: "
                + ", ".join(sorted(unknown_datasets))
            )
        non_speech_datasets = [
            name
            for name in self._training_datasets
            if not isinstance(self._loaders[name], _SpeechLoader)
        ]
        if non_speech_datasets:
            raise ValueError(
                "training datasets can only be injected into speech loaders: "
                + ", ".join(sorted(non_speech_datasets))
            )
        self.validation_specs = _validation_specs(validation)
        self.validation_spec = (
            next(iter(self.validation_specs.values()))
            if len(self.validation_specs) == 1
            else None
        )
        self._validation_loaders = {
            name: _build_validation_loader(spec, runtime)
            for name, spec in self.validation_specs.items()
        }
        self._validation_loader = (
            next(iter(self._validation_loaders.values()))
            if len(self._validation_loaders) == 1
            else None
        )
        if any(not name for name in self.loader_specs):
            raise ValueError("loader names must not be empty.")
        _validate_materialization_plan(self.loader_specs, self.validation_specs)
        _validate_streaming_plan(self.loader_specs, self.validation_specs)
        self.schedule = schedule or LoaderSchedule(
            {name: 1.0 for name in self.loader_specs}
        )
        _validate_loader_names(self.loader_specs, self.schedule.weights)
        _assign_worker_budgets(self._loaders, self.loader_specs, self.schedule.weights)

    def setup(self, stage: str | None = None) -> None:
        speech_datasets: list[tuple[object, Dataset[RawSample]]] = []
        for name, loader in self._loaders.items():
            if not isinstance(loader, _SpeechLoader):
                loader.setup(stage)
                continue
            injected = self._training_datasets.get(name)
            if injected is not None:
                loader.setup(stage, dataset=injected)
                continue
            shared = next(
                (
                    dataset
                    for config, dataset in speech_datasets
                    if config == loader.config.dataset
                ),
                None,
            )
            loader.setup(stage, dataset=shared)
            if shared is None:
                if loader._dataset is None:
                    raise RuntimeError("speech loader setup did not load a dataset.")
                speech_datasets.append((loader.config.dataset, loader._dataset))
        for loader in self._validation_loaders.values():
            loader.setup(stage)

    @property
    def loader_names(self) -> tuple[str, ...]:
        return tuple(self.loader_specs)

    @property
    def validation_names(self) -> tuple[str, ...]:
        return tuple(self.validation_specs)

    @property
    def materialization_enabled(self) -> bool:
        return any(
            spec.kind is LoaderKind.SPEECH
            and spec.speech_config is not None
            and spec.speech_config.materialization.enabled
            for spec in self.loader_specs.values()
        )

    @property
    def streaming_enabled(self) -> bool:
        return any(
            spec.kind is LoaderKind.SPEECH
            and spec.speech_config is not None
            and spec.speech_config.streaming.enabled
            for spec in self.loader_specs.values()
        )

    @property
    def has_pending_assets(self) -> bool:
        return bool(self._asset_jobs())

    def start_asset_materialization(self, *, owner: bool) -> None:
        for job in self._asset_jobs():
            job.start(owner=owner)

    def finish_asset_materialization(self, *, owner: bool) -> None:
        for job in self._asset_jobs():
            job.finish(owner=owner)

    def refresh_materialized_assets(self) -> None:
        for loader in self._speech_loaders(include_validation=True):
            loader.refresh_materialized_asset()

    def close_asset_materialization(self) -> None:
        for job in self._asset_jobs():
            job.close()

    def start_streaming_synthesis(self, *, owner: bool) -> None:
        for loader in self._speech_loaders(include_validation=False):
            loader.start_synthesis(owner=owner)

    def set_streaming_stop_requested(self, requested: Callable[[], bool]) -> None:
        for loader in self._speech_loaders(include_validation=False):
            loader.set_streaming_stop_requested(requested)

    def check_streaming_synthesis(self, *, owner: bool) -> None:
        for loader in self._speech_loaders(include_validation=False):
            loader.check_synthesis(owner=owner)

    def close_streaming_synthesis(self, *, owner: bool) -> None:
        for loader in self._speech_loaders(include_validation=False):
            loader.close_synthesis(owner=owner)

    def set_streaming_global_step(self, step: int) -> None:
        for loader in self._speech_loaders(include_validation=False):
            loader.set_streaming_global_step(step)

    def acknowledge_streaming_batch(self, global_step: int) -> None:
        loaders = self._speech_loaders(include_validation=False)
        streaming = [loader for loader in loaders if loader.config.streaming.enabled]
        if len(streaming) != 1:
            raise RuntimeError(
                "streaming batch acknowledgement requires exactly one loader."
            )
        streaming[0].acknowledge_streaming_batch(global_step)

    def teardown(self, stage: str | None = None) -> None:
        del stage
        self.close_asset_materialization()
        self.close_streaming_synthesis(owner=True)

    def state_dict(self) -> dict[str, object]:
        streaming = {
            name: state
            for name, loader in self._loaders.items()
            if isinstance(loader, _SpeechLoader)
            and (state := loader.streaming_state_dict()) is not None
        }
        if not streaming:
            return {}
        return {
            "schema": "speech-to-speech-datamodule-v1",
            "streaming": streaming,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DataModule checkpoint state must be a mapping.")
        if state_dict.get("schema") != "speech-to-speech-datamodule-v1":
            raise ValueError("DataModule checkpoint schema is incompatible.")
        streaming = state_dict.get("streaming")
        if not isinstance(streaming, Mapping):
            raise TypeError("DataModule streaming checkpoint state must be a mapping.")
        unknown = set(streaming) - set(self._loaders)
        if unknown:
            raise ValueError(
                "DataModule checkpoint references unknown loaders: "
                + ", ".join(sorted(cast(set[str], unknown)))
            )
        for name, state in streaming.items():
            loader = self._loaders[name]
            if not isinstance(loader, _SpeechLoader):
                raise ValueError(
                    f"DataModule checkpoint loader {name!r} is not a speech loader."
                )
            if not isinstance(state, Mapping):
                raise TypeError(
                    f"DataModule checkpoint loader {name!r} state must be a mapping."
                )
            loader.load_streaming_state_dict(cast(Mapping[str, object], state))

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
    ) -> Callable[[list[RawSample]], TrainInput]:
        return self._diagnostic_loader(split, loader_name).diagnostic_collator(task)

    def published_streaming_samples(
        self,
        indices: Sequence[int],
        *,
        loader_name: str,
    ) -> list[PublishedSample]:
        try:
            loader = self._loaders[loader_name]
        except KeyError as error:
            raise ValueError(f"unknown loader {loader_name!r}.") from error
        if not isinstance(loader, _SpeechLoader):
            raise ValueError("streaming samples require a speech loader.")
        return loader.published_samples(indices)

    def streaming_telemetry(self, *, loader_name: str | None = None) -> StreamingTelemetry | None:
        if loader_name is not None:
            try:
                loader = self._loaders[loader_name]
            except KeyError as error:
                raise ValueError(f"unknown loader {loader_name!r}.") from error
            if not isinstance(loader, _SpeechLoader):
                raise ValueError("streaming telemetry requires a speech loader.")
            return loader.streaming_telemetry()

        streaming = [
            loader
            for loader in self._speech_loaders(include_validation=False)
            if loader.config.streaming.enabled
        ]
        if not streaming:
            return None
        if len(streaming) != 1:
            raise RuntimeError(
                "streaming telemetry requires exactly one streaming loader when "
                "loader_name is omitted."
            )
        return streaming[0].streaming_telemetry()

    def train_dataloader(self) -> Iterable[TrainBatch]:
        loaders = {
            name: loader.train_dataloader() for name, loader in self._loaders.items()
        }
        if len(loaders) == 1:
            return cast(Iterable[TrainBatch], next(iter(loaders.values())))
        return ScheduledDataLoader(loaders, self.schedule)

    def val_dataloader(self) -> Iterable[TrainInput]:
        if not self._validation_loaders:
            return ()
        loaders = [
            loader.validation_dataloader()
            for loader in self._validation_loaders.values()
        ]
        if len(loaders) == 1:
            return cast(Iterable[TrainInput], loaders[0])
        return cast(Iterable[TrainInput], loaders)

    def _asset_jobs(self) -> tuple[AssetJob, ...]:
        jobs: dict[str, AssetJob] = {}
        for loader in self._speech_loaders(include_validation=True):
            job = loader.asset_job
            if job is not None:
                jobs.setdefault(job.request_id, job)
        return tuple(jobs.values())

    def _speech_loaders(
        self,
        *,
        include_validation: bool,
    ) -> tuple[_SpeechLoader, ...]:
        loaders = tuple(
            loader
            for loader in self._loaders.values()
            if isinstance(loader, _SpeechLoader)
        )
        if include_validation:
            validations = tuple(
                loader
                for loader in self._validation_loaders.values()
                if isinstance(loader, _SpeechLoader)
            )
            return (*loaders, *validations)
        return loaders

    def _diagnostic_loader(
        self,
        split: SampleSplit,
        loader_name: str,
    ) -> _DiagnosticLoader:
        if not isinstance(split, SampleSplit):
            raise TypeError("diagnostic split must be a SampleSplit.")
        if split is SampleSplit.TRAIN:
            try:
                return self._loaders[loader_name]
            except KeyError as error:
                raise ValueError(f"unknown loader {loader_name!r}.") from error
        if not self._validation_loaders:
            raise ValueError("validation diagnostic samples require a speech loader.")
        loader = self._validation_loaders.get(loader_name)
        if loader is None and len(self._validation_loaders) == 1:
            loader = next(iter(self._validation_loaders.values()))
        if loader is None:
            raise ValueError(f"unknown validation loader {loader_name!r}.")
        if not isinstance(loader, _SpeechLoader):
            raise ValueError("validation diagnostic samples require a speech loader.")
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
            trace=spec.trace,
            ar_framing=spec.ar_framing,
            max_samples=spec.max_samples,
        )
    assert spec.text_config is not None
    return TextLoader(
        spec.text_config,
        cast(TextRuntime, runtime),
        spec.task_weights,
        trace=spec.trace,
        ar_framing=spec.ar_framing,
        max_samples=spec.max_samples,
        tasks=spec.task_configs,
    )


def _build_validation_loader(
    spec: LoaderSpec,
    runtime: DatasetRuntime | TextRuntime,
) -> _ValidationLoader:
    if spec.kind is LoaderKind.SPEECH:
        assert spec.speech_config is not None
        return _SpeechLoader(
            spec.speech_config,
            cast(DatasetRuntime, runtime),
            spec.task_weights,
            sample_index=spec.sample_index,
            trace=spec.trace,
            ar_framing=spec.ar_framing,
            max_samples=spec.max_samples,
        )
    assert spec.text_config is not None
    return TextLoader(
        spec.text_config,
        cast(TextRuntime, runtime),
        spec.task_weights,
        trace=spec.trace,
        ar_framing=spec.ar_framing,
        max_samples=spec.max_samples,
        tasks=spec.task_configs,
    )


def _validation_specs(
    value: LoaderSpec | Mapping[str, LoaderSpec] | None,
) -> dict[str, LoaderSpec]:
    if value is None:
        return {}
    if isinstance(value, LoaderSpec):
        return {"validation": value}
    if not isinstance(value, Mapping):
        raise TypeError("validation must be a LoaderSpec, mapping, or None.")
    result = dict(value)
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError("validation loader names must be non-empty strings.")
    if any(not isinstance(spec, LoaderSpec) for spec in result.values()):
        raise TypeError("validation loader mappings must contain LoaderSpec values.")
    return result


def _validate_materialization_plan(
    loaders: Mapping[str, LoaderSpec],
    validations: Mapping[str, LoaderSpec],
) -> None:
    enabled = [
        spec
        for spec in loaders.values()
        if spec.kind is LoaderKind.SPEECH
        and spec.speech_config is not None
        and spec.speech_config.materialization.enabled
    ]
    validation_configs = [
        spec.speech_config
        for spec in validations.values()
        if spec.kind is LoaderKind.SPEECH and spec.speech_config is not None
    ]
    if not enabled:
        if any(config.materialization.enabled for config in validation_configs):
            raise ValueError(
                "validation asset materialization requires an enabled training "
                "speech loader."
            )
        return
    if len(loaders) != 1 or len(enabled) != 1:
        raise ValueError(
            "asset materialization currently requires exactly one training "
            "speech loader so the epoch has a finite reload boundary."
        )
    train = enabled[0].speech_config
    if train is None:
        return
    for validation_config in validation_configs:
        if not validation_config.materialization.enabled:
            continue
        if (
            train.codec != validation_config.codec
            or train.materialization != validation_config.materialization
            or _asset_source_key(train) != _asset_source_key(validation_config)
        ):
            raise ValueError(
                "training and validation asset materialization must resolve the same "
                "codec source request."
            )


def _validate_streaming_plan(
    loaders: Mapping[str, LoaderSpec],
    validations: Mapping[str, LoaderSpec],
) -> None:
    enabled = [
        spec
        for spec in loaders.values()
        if spec.kind is LoaderKind.SPEECH
        and spec.speech_config is not None
        and spec.speech_config.streaming.enabled
    ]
    if not enabled:
        return
    if len(loaders) != 1 or len(enabled) != 1:
        raise ValueError(
            "streaming synthesis currently requires exactly one training speech "
            "loader so one global checkpoint cursor owns the logical epoch."
        )
    if enabled[0].sample_index is not None:
        raise ValueError(
            "streaming synthesis does not support sample_index; the backbone must "
            "consume the complete logical epoch."
        )
    if validations:
        raise ValueError(
            "streaming synthesis does not accept a validation loader; run validation "
            "from an immutable sealed dataset."
        )


def _asset_source_key(config: SpeechConfig) -> tuple[object, ...]:
    dataset = config.dataset
    return (
        dataset.name,
        dataset.root,
        dataset.split,
        dataset.filter,
    )


def _validate_sample_index(sample_index: int | None) -> None:
    if sample_index is None:
        return
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError("sample_index must be an integer or None.")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")


def _validate_max_samples(max_samples: int | None) -> None:
    if max_samples is None:
        return
    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise TypeError("max_samples must be an integer or None.")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")


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


def _assign_worker_budgets(
    loaders: Mapping[str, _TrainLoader],
    specs: Mapping[str, LoaderSpec],
    weights: Mapping[str, float],
) -> None:
    groups: dict[int, list[str]] = {}
    configs: dict[int, DataLoaderConfig] = {}
    for name, spec in specs.items():
        config = (
            spec.speech_config.dataloader
            if spec.speech_config is not None
            else spec.text_config.dataloader
            if spec.text_config is not None
            else None
        )
        if config is None:
            continue
        key = id(config)
        groups.setdefault(key, []).append(name)
        configs[key] = config

    for key, names in groups.items():
        allocations = _worker_allocations(
            {name: weights[name] for name in names},
            configs[key].num_workers,
        )
        for name, count in allocations.items():
            loader = loaders[name]
            if isinstance(loader, (_SpeechLoader, TextLoader)):
                loader.num_workers = count


def _worker_allocations(
    weights: Mapping[str, float],
    budget: int,
) -> dict[str, int]:
    if budget == 0:
        return {name: 0 for name in weights}
    total = sum(weights.values())
    quotas = {name: budget * weight / total for name, weight in weights.items()}
    result = {name: math.floor(quota) for name, quota in quotas.items()}
    remaining = budget - sum(result.values())
    ranked = sorted(
        weights,
        key=lambda name: (quotas[name] - result[name], weights[name]),
        reverse=True,
    )
    for name in ranked[:remaining]:
        result[name] += 1
    return result


def _source_loader(
    dataset: object,
    *,
    loader: DataLoaderConfig,
    num_workers: int,
    collate_fn: Any,
    shuffle: bool,
    frame_rate: float,
) -> DataLoader[Any] | None:
    source = _source_dataset(dataset)
    if source is None:
        return None
    batch_size = loader.batch_size
    if not loader.costs.enabled:
        return source.dataloader(
            costs=None,
            max_batch_memory=batch_size,
            max_batch_samples=batch_size,
            planning_window=256,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=(
                loader.persistent_workers and num_workers > 0
            ),
            collate_fn=collate_fn,
        )
    if loader.costs.max_batch_frames is None:
        raise RuntimeError("enabled dataloader costs require max_batch_frames.")
    return source.dataloader(
        costs=partial(_sample_audio_frame_cost, frame_rate=frame_rate),
        max_batch_memory=loader.costs.max_batch_frames,
        max_batch_samples=batch_size,
        planning_window=loader.costs.planning_window,
        materialize_callable_costs=True,
        distributed_plan_sync="epoch",
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=(
            loader.persistent_workers and num_workers > 0
        ),
        collate_fn=collate_fn,
    )


def _source_dataset(dataset: object) -> MapStyleABC | None:
    return dataset if isinstance(dataset, MapStyleABC) else None


def _reject_enabled_costs(loader: DataLoaderConfig, path: str) -> None:
    if loader.costs.enabled:
        raise ValueError(f"dataloader costs are unsupported for {path}.")


def _sample_audio_frame_cost(row: object, *, frame_rate: float) -> int:
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, (float, int)):
        raise TypeError("codec frame_rate must be numeric for dataloader costs.")
    if not math.isfinite(float(frame_rate)) or frame_rate <= 0:
        raise ValueError("codec frame_rate must be positive for dataloader costs.")
    durations = tuple(_audio_durations(row))
    if not durations:
        raise ValueError("dataloader costs require audio duration metadata.")
    return sum(
        max(1, math.ceil(duration * float(frame_rate)))
        for duration in durations
    )


def _audio_durations(row: object) -> Iterable[float]:
    if isinstance(row, AudioContextCostRow):
        yield from _audio_durations(row.sample)
        yield from _audio_durations(row.audio_context)
        return
    if isinstance(row, Mapping):
        for ref, item in row.items():
            if _is_audio_ref(ref):
                yield _audio_duration(getattr(item, "meta", {}))
        return
    items = getattr(row, "items", None)
    if isinstance(items, tuple):
        for ref, meta in items:
            if _is_audio_ref(ref):
                yield _audio_duration(meta)


def _is_audio_ref(ref: object) -> bool:
    return isinstance(ref, tuple) and len(ref) == 2 and ref[1] is Modality.AUDIO


def _audio_duration(meta: object) -> float:
    if not isinstance(meta, Mapping):
        raise TypeError("audio metadata must be a mapping for dataloader costs.")
    value = meta.get(AudioMeta.DURATION)
    if value is None:
        value = meta.get(AudioMeta.DURATION.value)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("dataloader costs require audio duration metadata.")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("audio duration must be finite and positive.")
    return duration


def _collator(
    shape: DataShape,
    runtime: DatasetRuntime,
    task_weights: Mapping[Task, float],
    *,
    encode_missing_codes: bool = False,
    interleave_audio_frames: int = 25,
    mask_text_ratio: float = 0.5,
    mask_audio_ratio: float = 0.5,
    trace: str | None = None,
    ar_framing: ARFraming = ARFraming.INSTRUCTION,
    tasks: Mapping[Task, TaskConfig] | None = None,
):
    if shape is DataShape.PAIR:
        return Collator(
            runtime,
            task_weights,
            encode_missing_codes=encode_missing_codes,
            interleave_audio_frames=interleave_audio_frames,
            mask_text_ratio=mask_text_ratio,
            mask_audio_ratio=mask_audio_ratio,
            trace=trace,
            ar_framing=ar_framing,
            tasks=tasks,
        )
    if shape is DataShape.SINGLE:
        return SingleCollator(
            runtime,
            task_weights,
            encode_missing_codes=encode_missing_codes,
            interleave_audio_frames=interleave_audio_frames,
            mask_text_ratio=mask_text_ratio,
            mask_audio_ratio=mask_audio_ratio,
            trace=trace,
            ar_framing=ar_framing,
            tasks=tasks,
        )
    raise AssertionError(f"unsupported data shape: {shape}")
