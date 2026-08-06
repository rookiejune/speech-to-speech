from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from anydataset.types import AudioView

from ..task import Task
from .dataset.speech import DatasetConfig, DatasetName
from .sample import DataShape


@dataclass
class DataLoaderCostsConfig:
    enabled: bool = False
    max_batch_frames: Optional[int] = None
    planning_window: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("dataloader costs enabled must be a boolean.")
        if self.max_batch_frames is not None:
            if (
                isinstance(self.max_batch_frames, bool)
                or not isinstance(self.max_batch_frames, int)
            ):
                raise TypeError("dataloader costs max_batch_frames must be an integer or None.")
            if self.max_batch_frames <= 0:
                raise ValueError("dataloader costs max_batch_frames must be positive.")
        if isinstance(self.planning_window, bool) or not isinstance(
            self.planning_window,
            int,
        ):
            raise TypeError("dataloader costs planning_window must be an integer.")
        if self.planning_window <= 0:
            raise ValueError("dataloader costs planning_window must be positive.")
        if self.enabled and self.max_batch_frames is None:
            raise ValueError(
                "enabled dataloader costs require max_batch_frames.",
            )


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool = False
    persistent_workers: bool = False
    costs: DataLoaderCostsConfig = field(default_factory=DataLoaderCostsConfig)

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("dataloader batch_size must be an integer.")
        if self.batch_size <= 0:
            raise ValueError("dataloader batch_size must be positive.")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            raise TypeError("dataloader num_workers must be an integer.")
        if self.num_workers < 0:
            raise ValueError("dataloader num_workers must be non-negative.")
        for name in ("pin_memory", "persistent_workers"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"dataloader {name} must be a boolean.")
        if not isinstance(self.costs, DataLoaderCostsConfig):
            raise TypeError("dataloader costs must be a DataLoaderCostsConfig.")


@dataclass
class TaskConfig:
    """Per-task instruction index: ``null`` random, ``int`` fixed."""

    template: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.template is None:
            return
        if isinstance(self.template, bool) or not isinstance(self.template, int):
            raise TypeError("task template must be an integer or null.")
        if self.template < 0:
            raise ValueError("task template must be non-negative.")


@dataclass
class AssetMaterializationConfig:
    """Read-through workspace codec asset materialization options."""

    enabled: bool = False
    codec_view: Optional[str] = None
    output_root: Optional[str] = None
    device: Optional[str] = None
    provider_id: Optional[str] = None
    input_id: Optional[str] = None
    source_factory: Optional[str] = None
    max_shard_samples: int = 100_000
    batch_size: int = 1
    commit_samples: Optional[int] = None
    write_workers: int = 1
    write_prefetch: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("asset materialization enabled must be a boolean.")
        if self.codec_view is not None:
            _non_empty_string("asset materialization codec_view", self.codec_view)
            try:
                AudioView(self.codec_view)
            except ValueError as error:
                raise ValueError(
                    f"unsupported asset materialization codec_view: {self.codec_view!r}."
                ) from error
        for name in ("output_root", "device", "provider_id", "input_id"):
            value = getattr(self, name)
            if value is not None:
                _non_empty_string(f"asset materialization {name}", value)
        if self.source_factory is not None:
            _non_empty_string(
                "asset materialization source_factory",
                self.source_factory,
            )
            module, separator, attribute = self.source_factory.partition(":")
            if not separator or not module or not attribute:
                raise ValueError(
                    "asset materialization source_factory must use "
                    "'module:attribute' syntax."
                )
        for name in ("max_shard_samples", "batch_size"):
            _positive_integer(f"asset materialization {name}", getattr(self, name))
        if self.commit_samples is not None:
            _positive_integer(
                "asset materialization commit_samples",
                self.commit_samples,
            )
        if isinstance(self.write_workers, bool) or not isinstance(
            self.write_workers,
            int,
        ):
            raise TypeError("asset materialization write_workers must be an integer.")
        if self.write_workers < 0:
            raise ValueError(
                "asset materialization write_workers must be non-negative."
            )
        if self.write_prefetch is not None:
            _positive_integer(
                "asset materialization write_prefetch",
                self.write_prefetch,
            )
        if not self.enabled:
            return
        for name in ("output_root", "device", "provider_id"):
            if getattr(self, name) is None:
                raise ValueError(
                    f"enabled asset materialization requires {name}."
                )
        if self.device == "auto":
            raise ValueError(
                "asset materialization device must name one explicit device; "
                "'auto' could start multiple writers."
            )


@dataclass
class StreamingTelemetryConfig:
    """Runtime observability controls for a streaming training loader."""

    enabled: bool = True
    gpu_sample_interval_seconds: float = 1.0
    log_every_n_steps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("streaming telemetry enabled must be a boolean.")
        interval = self.gpu_sample_interval_seconds
        if isinstance(interval, bool) or not isinstance(interval, (float, int)):
            raise TypeError(
                "streaming telemetry gpu_sample_interval_seconds must be numeric."
            )
        if interval < 0:
            raise ValueError(
                "streaming telemetry gpu_sample_interval_seconds must be non-negative."
            )
        self.gpu_sample_interval_seconds = float(interval)
        _positive_integer(
            "streaming telemetry log_every_n_steps",
            self.log_every_n_steps,
        )


@dataclass
class StreamingConfig:
    """Consume immutable synthesis snapshots as one resumable logical epoch."""

    enabled: bool = False
    root: Optional[str] = None
    stream_id: Optional[str] = None
    expected_samples: Optional[int] = None
    poll_seconds: float = 30.0
    status_seconds: float = 60.0
    producer_factory: Optional[str] = None
    producer_options: Dict[str, Any] = field(default_factory=dict)
    telemetry: StreamingTelemetryConfig = field(default_factory=StreamingTelemetryConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("streaming enabled must be a boolean.")
        for name in ("root", "stream_id", "producer_factory"):
            value = getattr(self, name)
            if value is not None:
                _non_empty_string(f"streaming {name}", value)
        if self.expected_samples is not None:
            _positive_integer("streaming expected_samples", self.expected_samples)
        for name in ("poll_seconds", "status_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"streaming {name} must be numeric.")
            if value <= 0:
                raise ValueError(f"streaming {name} must be positive.")
            setattr(self, name, float(value))
        if not isinstance(self.producer_options, dict):
            raise TypeError("streaming producer_options must be a dictionary.")
        if not isinstance(self.telemetry, StreamingTelemetryConfig):
            raise TypeError("streaming telemetry must be a StreamingTelemetryConfig.")
        if self.producer_factory is not None:
            module, separator, attribute = self.producer_factory.partition(":")
            if not separator or not module or not attribute:
                raise ValueError(
                    "streaming producer_factory must use 'module:attribute' syntax."
                )
        elif self.producer_options:
            raise ValueError(
                "streaming producer_options require producer_factory."
            )
        if not self.enabled:
            return
        if self.stream_id is None:
            raise ValueError("enabled streaming requires stream_id.")
        if self.expected_samples is None:
            raise ValueError("enabled streaming requires expected_samples.")


@dataclass
class ToyPerformanceConfig:
    """Bounded real-model performance probe used by an auto-routed toy source."""

    warmup_steps: int = 5
    measure_window_steps: int = 20
    log_every_n_steps: int = 1

    def __post_init__(self) -> None:
        for name in (
            "warmup_steps",
            "measure_window_steps",
            "log_every_n_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"workspace source toy {name} must be an integer.")
        if self.warmup_steps < 0:
            raise ValueError("workspace source toy warmup_steps must be non-negative.")
        if self.measure_window_steps <= 0:
            raise ValueError(
                "workspace source toy measure_window_steps must be positive."
            )
        if self.log_every_n_steps <= 0:
            raise ValueError(
                "workspace source toy log_every_n_steps must be positive."
            )


@dataclass
class WorkspaceSourceConfig:
    """Resolve one workspace source through access, generation, or toy routes."""

    factory: Optional[str] = None
    mode: str = "auto"
    options: Dict[str, Any] = field(default_factory=dict)
    toy: ToyPerformanceConfig = field(default_factory=ToyPerformanceConfig)

    def __post_init__(self) -> None:
        if self.factory is not None:
            _non_empty_string("workspace source factory", self.factory)
            module, separator, attribute = self.factory.partition(":")
            if not separator or not module or not attribute:
                raise ValueError(
                    "workspace source factory must use 'module:attribute' syntax."
                )
        if self.mode not in {"auto", "access", "generate", "toy"}:
            raise ValueError(
                "workspace source mode must be auto, access, generate, or toy."
            )
        if not isinstance(self.options, dict):
            raise TypeError("workspace source options must be a dictionary.")
        if not isinstance(self.toy, ToyPerformanceConfig):
            raise TypeError("workspace source toy must be a ToyPerformanceConfig.")
        if self.factory is None:
            if self.mode != "auto":
                raise ValueError(
                    "workspace source mode requires a configured factory."
                )
            if self.options:
                raise ValueError(
                    "workspace source options require a configured factory."
                )

    @property
    def enabled(self) -> bool:
        return self.factory is not None


@dataclass
class SpeechConfig:
    codec: str
    dataloader: DataLoaderConfig
    shape: DataShape = DataShape.PAIR
    encode_missing_codes: bool = False
    interleave_audio_frames: int = 25
    mask_text_ratio: float = 0.5
    mask_audio_ratio: float = 0.5
    tasks: Optional[Dict[Task, TaskConfig]] = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    materialization: AssetMaterializationConfig = field(
        default_factory=AssetMaterializationConfig
    )
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    source: WorkspaceSourceConfig = field(default_factory=WorkspaceSourceConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.dataloader, DataLoaderConfig):
            raise TypeError("dataloader must be a DataLoaderConfig.")
        if not isinstance(self.shape, DataShape):
            raise TypeError("data shape must be a DataShape.")
        if not isinstance(self.encode_missing_codes, bool):
            raise TypeError("encode_missing_codes must be a boolean.")
        if not isinstance(self.materialization, AssetMaterializationConfig):
            raise TypeError(
                "materialization must be an AssetMaterializationConfig."
            )
        if not isinstance(self.streaming, StreamingConfig):
            raise TypeError("streaming must be a StreamingConfig.")
        if not isinstance(self.source, WorkspaceSourceConfig):
            raise TypeError("source must be a WorkspaceSourceConfig.")
        if self.source.enabled and self.materialization.enabled:
            raise ValueError(
                "workspace source routing and asset materialization are mutually exclusive."
            )
        if self.source.enabled and self.streaming.enabled:
            raise ValueError(
                "workspace source routing and explicit legacy streaming are mutually exclusive."
            )
        if self.materialization.enabled and not self.encode_missing_codes:
            raise ValueError(
                "enabled asset materialization requires encode_missing_codes=true "
                "for the first-epoch waveform fallback."
            )
        if self.streaming.enabled or self.source.enabled:
            if self.dataset.name is not DatasetName.STREAMING_S2ST:
                raise ValueError(
                    "streaming workspace sources require dataset streaming_s2st."
                )
            if self.shape is not DataShape.PAIR:
                raise ValueError("streaming synthesis requires pair-shaped samples.")
            if self.dataset.split_manifest is not None:
                raise ValueError(
                    "streaming synthesis owns the complete 2N membership and does "
                    "not accept split_manifest."
                )
            if self.dataset.speaker is not None:
                raise ValueError("streaming synthesis does not accept dataset speaker.")
            if self.materialization.enabled:
                raise ValueError(
                    "streaming synthesis consumption and asset materialization "
                    "are mutually exclusive."
                )
            if self.encode_missing_codes:
                raise ValueError(
                    "streaming synthesis snapshots must already contain codec views; "
                    "encode_missing_codes must be false."
                )
            if self.dataloader.num_workers != 0:
                raise ValueError(
                    "streaming checkpoint cursors require dataloader num_workers=0."
                )
            if self.dataloader.persistent_workers:
                raise ValueError(
                    "streaming checkpoint cursors require persistent_workers=false."
                )
            if self.dataloader.costs.enabled:
                raise ValueError(
                    "streaming snapshot consumption does not support cost batching."
                )
        if (
            isinstance(self.interleave_audio_frames, bool)
            or not isinstance(self.interleave_audio_frames, int)
            or self.interleave_audio_frames < 1
        ):
            raise ValueError("interleave_audio_frames must be a positive integer.")
        for name in ("mask_text_ratio", "mask_audio_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be a float.")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        self.tasks = None if self.tasks is None else _tasks(self.tasks)
        if (
            self.dataset.name is DatasetName.QWEN_TTS_SPEAKER
            and self.shape is not DataShape.SINGLE
        ):
            raise ValueError("qwen_tts_speaker requires datamodule shape single.")

    def template_index(self, task: Task) -> Optional[int]:
        return task_template_index(self.tasks, task)


def task_template_index(
    tasks: Mapping[Task, TaskConfig] | None,
    task: Task,
) -> Optional[int]:
    if not isinstance(task, Task):
        raise TypeError("task must be a Task.")
    if tasks is None:
        return 0
    if task not in tasks:
        raise KeyError(f"datamodule.tasks missing entry for {task.value}.")
    return tasks[task].template


def _tasks(value: object) -> dict[Task, TaskConfig]:
    if not isinstance(value, Mapping):
        raise TypeError("tasks must be a mapping.")
    normalized: dict[Task, TaskConfig] = {}
    for key, config in value.items():
        task = key if isinstance(key, Task) else _task(key)
        if isinstance(config, TaskConfig):
            resolved = config
        elif isinstance(config, Mapping):
            resolved = TaskConfig(**dict(config))
        else:
            raise TypeError("tasks values must be TaskConfig mappings.")
        normalized[task] = resolved
    return normalized


def _task(value: object) -> Task:
    if isinstance(value, Task):
        return value
    if isinstance(value, str):
        if value in Task.__members__:
            return Task[value]
        return Task(value)
    raise TypeError(f"task key must be a Task or string, got {type(value)}.")


def _non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "AssetMaterializationConfig",
    "DataLoaderConfig",
    "DataLoaderCostsConfig",
    "SpeechConfig",
    "StreamingConfig",
    "StreamingTelemetryConfig",
    "TaskConfig",
    "task_template_index",
]
