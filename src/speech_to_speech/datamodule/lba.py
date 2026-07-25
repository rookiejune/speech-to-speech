from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

from anydataset.types import Sample as RawSample
from anydataset.types import AudioItem, AudioMeta, AudioView, Modality, Role
from lba import LBA
from torch import Tensor

from ..task import Task
from .parser import parse_sample, parse_text_sample
from .protocol import DataRuntime, TextRuntime
from .sample import build_sample, build_text_sample
from .types import AcousticTarget, ModelSample

PlannerMode = Literal["quality", "throughput", "latency"]


@dataclass(frozen=True)
class LBAConfig:
    enabled: bool = False
    max_batch_cost: int = 2048
    token_unit: int = 1
    frame_unit: int = 50
    max_padding_ratio: float = 0.05
    prefetch_batches: int = 4
    planner_mode: str = "quality"
    drop_last_flush: bool = True
    max_sequence_tokens: Optional[int] = None
    max_source_frames: Optional[int] = None
    max_target_frames: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("lba.enabled must be a boolean.")
        for name in ("max_batch_cost", "token_unit", "frame_unit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"lba.{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"lba.{name} must be positive.")
        if isinstance(self.max_padding_ratio, bool) or not isinstance(
            self.max_padding_ratio,
            (int, float),
        ):
            raise TypeError("lba.max_padding_ratio must be a number.")
        if not math.isfinite(self.max_padding_ratio) or not (
            0 <= self.max_padding_ratio <= 1
        ):
            raise ValueError("lba.max_padding_ratio must be between 0 and 1.")
        if isinstance(self.prefetch_batches, bool) or not isinstance(
            self.prefetch_batches,
            int,
        ):
            raise TypeError("lba.prefetch_batches must be an integer.")
        if self.prefetch_batches < 0:
            raise ValueError("lba.prefetch_batches must be non-negative.")
        if self.planner_mode not in {"quality", "throughput", "latency"}:
            raise ValueError(
                "lba.planner_mode must be 'quality', 'throughput', or 'latency'."
            )
        if not isinstance(self.drop_last_flush, bool):
            raise TypeError("lba.drop_last_flush must be a boolean.")
        for name in (
            "max_sequence_tokens",
            "max_source_frames",
            "max_target_frames",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"lba.{name} must be an integer or None.")
            if value <= 0:
                raise ValueError(f"lba.{name} must be positive when set.")


def speech_length(
    sample: RawSample,
    *,
    runtime: DataRuntime,
    tasks: Sequence[Task],
    config: LBAConfig,
) -> int:
    pair = parse_sample(sample, runtime)
    return max(_cost(build_sample(pair, task, runtime), config) for task in tasks)


def metadata_speech_length(
    sample: RawSample,
    *,
    audio_view: AudioView,
    frame_rate: float,
    tasks: Sequence[Task],
    config: LBAConfig,
) -> int:
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)):
        raise TypeError("frame_rate must be a number.")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate must be finite and positive.")
    if config.max_sequence_tokens is not None:
        raise ValueError(
            "metadata LBA length cannot enforce max_sequence_tokens; unset it "
            "or use the parsing-based speech_length path."
        )
    costs: list[int] = []
    for task in tasks:
        source_frames = _task_frames(
            sample,
            task,
            source=True,
            audio_view=audio_view,
            frame_rate=frame_rate,
        )
        target_frames = _task_frames(
            sample,
            task,
            source=False,
            audio_view=audio_view,
            frame_rate=frame_rate,
        )
        _cap(source_frames, config.max_source_frames, name="source frames")
        _cap(target_frames, config.max_target_frames, name="target frames")
        costs.append(math.ceil((source_frames + target_frames) / config.frame_unit))
    if not costs:
        raise ValueError("LBA speech length requires at least one task.")
    return max(costs)


def text_length(
    sample: RawSample,
    *,
    runtime: TextRuntime,
    tasks: Sequence[Task],
    config: LBAConfig,
) -> int:
    pair = parse_text_sample(sample, runtime)
    return max(_cost(build_text_sample(pair, task, runtime), config) for task in tasks)


def _cost(sample: ModelSample, config: LBAConfig) -> int:
    tokens = sample.input_ids.numel()
    source_frames = 0
    target_frames = _frames(sample.acoustic_target)
    _cap(tokens, config.max_sequence_tokens, name="sequence tokens")
    _cap(source_frames, config.max_source_frames, name="source frames")
    _cap(target_frames, config.max_target_frames, name="target frames")
    return math.ceil(tokens / config.token_unit) + math.ceil(
        (source_frames + target_frames) / config.frame_unit
    )


def _frames(value: AcousticTarget | None) -> int:
    if value is None:
        return 0
    return value["codes"].size(0)


def _task_frames(
    sample: RawSample,
    task: Task,
    *,
    source: bool,
    audio_view: AudioView,
    frame_rate: float,
) -> int:
    if source:
        if task.source_modality is not Modality.AUDIO:
            return 0
        role = Role.SOURCE if task.uses_source_role else Role.TARGET
    else:
        if task.target_modality is not Modality.AUDIO:
            return 0
        role = Role.TARGET
    return _audio_frames(sample, role, audio_view=audio_view, frame_rate=frame_rate)


def _audio_frames(
    sample: RawSample,
    role: Role,
    *,
    audio_view: AudioView,
    frame_rate: float,
) -> int:
    item = sample.get((role, Modality.AUDIO))
    if item is None:
        return 0
    if not isinstance(item, AudioItem):
        raise TypeError(f"{role.value} audio sample item must be an AudioItem.")
    value = item.meta.get(AudioMeta.DURATION)
    if value is None:
        return _codec_frames(item, role, audio_view=audio_view)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("AudioMeta.DURATION must be a number of seconds.")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("AudioMeta.DURATION must be finite and non-negative.")
    return round(duration * frame_rate)


def _codec_frames(item: AudioItem, role: Role, *, audio_view: AudioView) -> int:
    codes = item.views.get(audio_view)
    if codes is None:
        raise ValueError(
            f"{role.value} audio sample is missing AudioMeta.DURATION and "
            f"{audio_view.value} codec view."
        )
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError(
            f"{role.value} audio {audio_view.value} codec view must have shape "
            "[frames, codebooks] when AudioMeta.DURATION is absent."
        )
    return codes.size(0)


def _cap(value: int, limit: int | None, *, name: str) -> None:
    if limit is not None and value > limit:
        raise ValueError(
            f"LBA hard cap exceeded: {name}={value} is greater than {limit}."
        )


__all__ = [
    "LBA",
    "LBAConfig",
    "PlannerMode",
    "metadata_speech_length",
    "speech_length",
    "text_length",
]
