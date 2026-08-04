from __future__ import annotations

import hashlib
import importlib
import json
import traceback
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty
from typing import Protocol, cast

from anydataset.dataset import IndexSelection
from anydataset.provider import CodecProvider
from anydataset.store import ViewMaterializer
from anydataset.store.reader import read_store_manifest
from anydataset.types import (
    AudioView,
    Modality,
    Role,
    Sample,
    TextMeta,
    TextReq,
    TextView,
)
from anytrain.codec import load_frame
from torch.utils.data import Dataset

from .._compat import StrEnum, auto
from .config import AssetMaterializationConfig
from .dataset.speech import DatasetConfig, DatasetName
from .protocol import DatasetRuntime

_ASSET_CONTRACT_VERSION = 1
_RESULT_POLL_SECONDS = 0.1
_RESULT_FINAL_WAIT_SECONDS = 1.0
_MAX_ERROR_MESSAGE_CHARS = 16_384
_MAX_ERROR_TRACEBACK_CHARS = 32_768
_FRAME_CODEC_VIEWS = frozenset(
    {
        AudioView.DAC,
        AudioView.LONGCAT,
        AudioView.STABLE,
        AudioView.UNICODEC,
    }
)


class AssetPhase(StrEnum):
    FALLBACK = auto()
    MATERIALIZING = auto()
    READY = auto()
    FAILED = auto()


class _WorkspaceFilterMissing(FileNotFoundError):
    pass


class _WorkspaceSourceMissing(FileNotFoundError):
    pass


@dataclass(frozen=True)
class AssetRequest:
    """Identity of one filtered codec asset requested by training."""

    dataset: str
    source_root: Path
    output_root: Path
    split: str
    codec: str
    codec_view: AudioView
    filter_policy: str | None
    input_id: str
    provider_id: str
    source_factory: str | None = None

    @property
    def id(self) -> str:
        payload = json.dumps(
            {
                "contract_version": _ASSET_CONTRACT_VERSION,
                "dataset": self.dataset,
                "source_root": str(self.source_root),
                "output_root": str(self.output_root),
                "split": self.split,
                "codec": self.codec,
                "codec_view": self.codec_view.value,
                "filter_policy": self.filter_policy,
                "input_id": self.input_id,
                "provider_id": self.provider_id,
                "source_factory": self.source_factory,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"s2s-asset-{hashlib.sha256(payload).hexdigest()}"

    @property
    def asset_root(self) -> Path:
        """Workspace-shaped root dedicated to this exact logical request."""

        return self.output_root / self.id


class DatasetFactory(Protocol):
    def __call__(self) -> Dataset[Sample]: ...


class SourceFactoryBuilder(Protocol):
    def __call__(self, request: AssetRequest) -> DatasetFactory: ...


class AssetJob(Protocol):
    @property
    def request_id(self) -> str: ...

    @property
    def phase(self) -> AssetPhase: ...

    def start(self, *, owner: bool) -> None: ...

    def finish(self, *, owner: bool) -> None: ...

    def load_ready(self) -> Dataset[Sample]: ...

    def close(self) -> None: ...


class AssetProducer(Protocol):
    def __call__(self) -> None: ...

    def load(self) -> Dataset[Sample]: ...


class _ResultQueue(Protocol):
    def put(self, value: _WorkerResult) -> None: ...

    def get(self, block: bool = True, timeout: float | None = None) -> _WorkerResult: ...

    def close(self) -> None: ...

    def join_thread(self) -> None: ...


@dataclass(frozen=True)
class AssetResolution:
    dataset: Dataset[Sample]
    request_id: str | None = None
    job: AssetJob | None = None


@dataclass(frozen=True)
class WorkspaceWaveformFactory:
    root: Path
    split: str
    filter_policy: str | None

    def __call__(self) -> Dataset[Sample]:
        from zhuyin.datasets.wmt19 import moss_tts

        store = self.root / "base"
        if not store.exists():
            raise _WorkspaceSourceMissing(store)
        read_store_manifest(store)
        try:
            dataset = (
                moss_tts.waveform(
                    root=self.root,
                    split=self.split,
                )
                .filter(self.filter_policy)
                .load()
            )
        except FileNotFoundError as error:
            if _missing_selection(error):
                raise _WorkspaceFilterMissing(self.filter_policy) from error
            raise
        _materialize_length(dataset)
        return cast(Dataset[Sample], cast(object, dataset))


@dataclass(frozen=True)
class FrameCodecProviderFactory:
    codec: str
    output: AudioView

    def __call__(self, device: str) -> CodecProvider:
        return CodecProvider(load_frame(self.codec, device=device), self.output)


@dataclass(frozen=True)
class WorkspaceCodecProducer:
    request: AssetRequest
    source: DatasetFactory
    config: AssetMaterializationConfig

    def __call__(self) -> None:
        output = self.output_dir
        if _load_materialized_dataset(self.request, missing_ok=True) is not None:
            return
        self.request.asset_root.mkdir(parents=True, exist_ok=True)
        try:
            ViewMaterializer(
                output,
                split=self.request.split,
                max_shard_samples=self.config.max_shard_samples,
                batch_size=self.config.batch_size,
                commit_samples=self.config.commit_samples,
                num_workers=0,
                write_workers=self.config.write_workers,
                write_prefetch=self.config.write_prefetch,
                keep_schema=_text_schema(),
                input_id=self.request.input_id,
                provider_id=self.request.provider_id,
            ).write(
                dataset_factory=self.source,
                provider_factory=FrameCodecProviderFactory(
                    self.request.codec,
                    self.request.codec_view,
                ),
                devices=cast(str, self.config.device),
            )
        except Exception:
            # Another run may publish the exact request while this worker waits
            # on ViewMaterializer's output lock.
            if _load_materialized_dataset(self.request, missing_ok=True) is None:
                raise

    @property
    def output_dir(self) -> Path:
        return self.request.asset_root / _store_dir(self.request.codec_view)

    def load(self) -> Dataset[Sample]:
        dataset = _load_materialized_dataset(self.request, missing_ok=False)
        if dataset is None:
            raise AssertionError("materialized codec dataset was not loaded.")
        return dataset


class BackgroundAssetJob:
    """Run one complete materializer on the global owner during epoch zero."""

    def __init__(
        self,
        request: AssetRequest,
        fallback: Dataset[Sample],
        producer: AssetProducer,
    ) -> None:
        self.request = request
        self.fallback = fallback
        self.producer = producer
        self._phase = AssetPhase.FALLBACK
        self._process: BaseProcess | None = None
        self._results: _ResultQueue | None = None
        self._error: Exception | None = None

    @property
    def request_id(self) -> str:
        return self.request.id

    @property
    def phase(self) -> AssetPhase:
        return self._phase

    def start(self, *, owner: bool) -> None:
        if self._phase is not AssetPhase.FALLBACK:
            return
        if not owner:
            self._phase = AssetPhase.MATERIALIZING
            return
        context = get_context("spawn")
        results = cast(_ResultQueue, context.Queue(maxsize=1))
        process = context.Process(
            target=_run_producer,
            args=(self.producer, results),
            name="s2s-asset-materializer",
            daemon=True,
        )
        try:
            process.start()
        except Exception as error:
            results.close()
            results.join_thread()
            self._error = error
            self._phase = AssetPhase.FAILED
            raise
        self._process = process
        self._results = results
        self._phase = AssetPhase.MATERIALIZING

    def finish(self, *, owner: bool) -> None:
        if self._phase is AssetPhase.READY:
            return
        if self._phase is AssetPhase.FAILED:
            if self._error is None:
                raise RuntimeError("asset materialization failed without an exception.")
            raise RuntimeError("asset materialization previously failed.") from self._error
        if not owner:
            return
        process = self._process
        if process is None:
            raise RuntimeError("asset materialization owner did not start its worker.")
        try:
            failure = self._wait_for_worker(process)
            if failure is None:
                self._phase = AssetPhase.READY
                return
            self._error = failure
            self._phase = AssetPhase.FAILED
            raise failure
        finally:
            self._cleanup_worker()

    def _wait_for_worker(self, process: BaseProcess) -> Exception | None:
        results = self._results
        if results is None:
            return RuntimeError("asset materialization worker has no result queue.")
        while True:
            try:
                result = results.get(timeout=_RESULT_POLL_SECONDS)
            except Empty:
                if process.is_alive():
                    continue
                process.join()
                try:
                    result = results.get(timeout=_RESULT_FINAL_WAIT_SECONDS)
                except Empty:
                    return self._missing_worker_result(process)
            break
        process.join()
        if process.exitcode not in (0, None):
            return RuntimeError(
                f"asset materialization worker exited with code {process.exitcode}."
            )
        if result.error_type is None:
            return None
        details = f"{result.error_type}: {result.message}"
        if result.traceback:
            details = f"{details}\n{result.traceback}"
        return RuntimeError(f"asset materialization worker failed: {details}")

    @staticmethod
    def _missing_worker_result(process: BaseProcess) -> Exception:
        if process.exitcode not in (0, None):
            return RuntimeError(
                f"asset materialization worker exited with code {process.exitcode}."
            )
        return RuntimeError("asset materialization worker returned no result.")

    def load_ready(self) -> Dataset[Sample]:
        dataset = self.producer.load()
        self._phase = AssetPhase.READY
        return dataset

    def close(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            self._error = RuntimeError("asset materialization was cancelled during teardown.")
            self._phase = AssetPhase.FAILED
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        process = self._process
        if process is not None and not process.is_alive():
            process.close()
        results = self._results
        if results is not None:
            results.close()
            results.join_thread()
        self._process = None
        self._results = None


@dataclass(frozen=True)
class _WorkerResult:
    error_type: str | None = None
    message: str = ""
    traceback: str = ""


def _run_producer(
    producer: AssetProducer,
    results: _ResultQueue,
) -> None:
    try:
        producer()
    except Exception as error:
        results.put(
            _WorkerResult(
                error_type=type(error).__name__,
                message=_bounded_text(str(error), _MAX_ERROR_MESSAGE_CHARS),
                traceback=_bounded_text(
                    traceback.format_exc(),
                    _MAX_ERROR_TRACEBACK_CHARS,
                ),
            )
        )
    else:
        results.put(_WorkerResult())


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n... <truncated {omitted} characters>"


def resolve_workspace_asset(
    config: DatasetConfig,
    runtime: DatasetRuntime,
    materialization: AssetMaterializationConfig,
) -> AssetResolution:
    """Resolve a prepared codec view or schedule its missing composite asset."""

    if not materialization.enabled:
        raise ValueError("disabled asset materialization must not call its resolver.")
    if config.name is not DatasetName.WMT19_TTS:
        raise ValueError("asset materialization currently supports only wmt19_tts.")
    view = runtime.audio_view
    if materialization.codec_view is not None:
        configured_view = AudioView(materialization.codec_view)
        if configured_view is not view:
            raise ValueError(
                "materialization codec_view must match the runtime audio view: "
                f"{configured_view.value!r} != {view.value!r}."
            )
    if view not in _FRAME_CODEC_VIEWS:
        supported = ", ".join(sorted(entry.value for entry in _FRAME_CODEC_VIEWS))
        raise ValueError(
            "the built-in materializer supports only frame-code codec views "
            f"({supported}); got {view.value!r}. Extend the provider/backend "
            "before materializing another representation."
        )

    from zhuyin.datasets.wmt19 import moss_tts

    source_root = moss_tts.dataset_root(config.root).resolve()
    output_root = Path(cast(str, materialization.output_root)).expanduser().resolve()
    existing = _load_codec_dataset(
        source_root,
        split=config.split,
        view=view,
        filter_policy=config.filter,
        missing_ok=True,
    )
    if existing is not None:
        return AssetResolution(existing)

    workspace_source = WorkspaceWaveformFactory(
        source_root,
        config.split,
        config.filter,
    )
    try:
        fallback = workspace_source()
    except (_WorkspaceFilterMissing, _WorkspaceSourceMissing) as error:
        if materialization.source_factory is None:
            if isinstance(error, _WorkspaceFilterMissing):
                raise FileNotFoundError(
                    f"workspace filter {config.filter!r} is not published for the "
                    "waveform source and no materialization.source_factory was "
                    "configured; register the stream/filter route instead of "
                    "falling back to unfiltered data."
                ) from error
            raise FileNotFoundError(
                f"workspace waveform source is not published at {source_root / 'base'} "
                "and no materialization.source_factory was configured."
            ) from error
        if materialization.input_id is None:
            raise ValueError(
                "materialization.source_factory requires an explicit input_id; "
                "the stream/filter source identity cannot be inferred safely."
            )
        request = _request(
            config,
            runtime,
            materialization,
            source_root=source_root,
            output_root=output_root,
            input_id=materialization.input_id,
            source_factory=materialization.source_factory,
        )
        existing = _load_materialized_dataset(request, missing_ok=True)
        if existing is not None:
            return AssetResolution(existing)
        source = _source_factory(request)
        fallback = source()
    else:
        input_id = materialization.input_id or _workspace_input_id(
            source_root / "base",
            fallback,
            split=config.split,
            filter_policy=config.filter,
        )
        request = _request(
            config,
            runtime,
            materialization,
            source_root=source_root,
            output_root=output_root,
            input_id=input_id,
            source_factory=None,
        )
        source = workspace_source

    _materialize_length(fallback)
    existing = _load_materialized_dataset(request, missing_ok=True)
    if existing is not None:
        return AssetResolution(existing)
    producer = WorkspaceCodecProducer(request, source, materialization)
    job = BackgroundAssetJob(request, fallback, producer)
    return AssetResolution(fallback, request_id=request.id, job=job)


def _request(
    config: DatasetConfig,
    runtime: DatasetRuntime,
    materialization: AssetMaterializationConfig,
    *,
    source_root: Path,
    output_root: Path,
    input_id: str,
    source_factory: str | None,
) -> AssetRequest:
    return AssetRequest(
        dataset=DatasetName.WMT19_TTS.value,
        source_root=source_root,
        output_root=output_root,
        split=config.split,
        codec=runtime.codec_name,
        codec_view=runtime.audio_view,
        filter_policy=config.filter,
        input_id=input_id,
        provider_id=cast(str, materialization.provider_id),
        source_factory=source_factory,
    )


def _source_factory(request: AssetRequest) -> DatasetFactory:
    path = request.source_factory
    if path is None:
        return WorkspaceWaveformFactory(
            request.source_root,
            request.split,
            request.filter_policy,
        )
    builder = _symbol(path)
    factory = builder(request)
    if not callable(factory):
        raise TypeError("materialization source_factory must return a dataset factory.")
    return cast(DatasetFactory, factory)


def _symbol(path: str) -> SourceFactoryBuilder:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("materialization source_factory must use 'module:attribute' syntax.")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"materialization source factory {path!r} must be callable.")
    return cast(SourceFactoryBuilder, value)


def _load_codec_dataset(
    root: Path,
    *,
    split: str,
    view: AudioView,
    filter_policy: str | None,
    missing_ok: bool,
) -> Dataset[Sample] | None:
    from zhuyin.datasets.wmt19 import moss_tts

    store = root / _store_dir(view)
    if not store.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(store)
    read_store_manifest(store)
    try:
        dataset = (
            moss_tts.codec(
                view.value,
                root=root,
                split=split,
            )
            .filter(filter_policy)
            .load()
        )
    except FileNotFoundError as error:
        if missing_ok and _missing_selection(error):
            return None
        raise
    _materialize_length(dataset)
    return cast(Dataset[Sample], cast(object, dataset))


def _missing_selection(error: FileNotFoundError) -> bool:
    message = str(error)
    return message.startswith("WMT19 selection ") and " is not published: " in message


def _load_materialized_dataset(
    request: AssetRequest,
    *,
    missing_ok: bool,
) -> Dataset[Sample] | None:
    output = request.asset_root / _store_dir(request.codec_view)
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        if missing_ok:
            return None
        raise FileNotFoundError(output)
    manifest = read_store_manifest(output)
    expected = {
        "input_id": request.input_id,
        "provider_id": request.provider_id,
    }
    actual = dict(manifest.provenance)
    if actual != expected:
        raise ValueError(
            "materialized workspace asset provenance does not match its request: "
            f"{actual!r} != {expected!r}."
        )
    return _load_codec_dataset(
        request.asset_root,
        split=request.split,
        view=request.codec_view,
        filter_policy=None,
        missing_ok=missing_ok,
    )


def _materialize_length(dataset: object) -> int:
    if not isinstance(dataset, Sized):
        raise TypeError("workspace asset datasets must be finite and expose __len__().")
    return len(dataset)


def _workspace_input_id(
    root: Path,
    dataset: Dataset[Sample],
    *,
    split: str,
    filter_policy: str | None,
) -> str:
    manifest = read_store_manifest(root)
    payload = json.dumps(
        {
            "dataset_id": manifest.dataset_id,
            "sample_count": manifest.sample_count,
            "split": manifest.split,
            "provenance": dict(manifest.provenance),
            "requested_split": split,
            "filter_policy": filter_policy,
            "selection": _selection_id(dataset, filter_policy=filter_policy),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"wmt19-source-{hashlib.sha256(payload).hexdigest()}"


def _selection_id(
    dataset: Dataset[Sample],
    *,
    filter_policy: str | None,
) -> str:
    if filter_policy is None:
        return "all"
    if not isinstance(dataset, IndexSelection):
        raise TypeError(
            "workspace filtered waveform datasets must expose an IndexSelection; "
            "configure materialization.input_id explicitly for another source type."
        )
    digest = hashlib.sha256()
    digest.update(len(dataset.indices).to_bytes(8, "little", signed=False))
    for index in dataset.indices:
        digest.update(index.to_bytes(8, "little", signed=False))
    return f"indexes-sha256-{digest.hexdigest()}"


def _store_dir(view: AudioView) -> str:
    if view is AudioView.STABLE:
        from zhuyin.datasets.wmt19.moss_tts import stable_store_dir

        return stable_store_dir()
    return view.value


def _text_schema() -> Mapping[tuple[Role, Modality], TextReq]:
    requirement = TextReq(
        views=frozenset({TextView.TEXT}),
        meta=frozenset({TextMeta.LANG}),
    )
    return {
        (Role.SOURCE, Modality.TEXT): requirement,
        (Role.TARGET, Modality.TEXT): requirement,
    }


__all__ = [
    "AssetJob",
    "AssetPhase",
    "AssetRequest",
    "AssetResolution",
    "BackgroundAssetJob",
    "resolve_workspace_asset",
]
