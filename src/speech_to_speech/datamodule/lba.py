from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional, Union

from anydataset.types import Sample as RawSample
from lba import LBA

from ..task import Task
from .parser import parse_sample, parse_text_sample
from .protocol import DataRuntime, TextRuntime
from .sample import build_sample, build_text_sample
from .types import AcousticPrompt, AcousticTarget, ModelSample

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
    source_frames = _frames(sample.acoustic_prompt)
    target_frames = _frames(sample.acoustic_target)
    _cap(tokens, config.max_sequence_tokens, name="sequence tokens")
    _cap(source_frames, config.max_source_frames, name="source frames")
    _cap(target_frames, config.max_target_frames, name="target frames")
    return math.ceil(tokens / config.token_unit) + math.ceil(
        (source_frames + target_frames) / config.frame_unit
    )


def _frames(value: Union[AcousticPrompt, AcousticTarget, None]) -> int:
    if value is None:
        return 0
    return value["codes"].size(0)


def _cap(value: int, limit: int | None, *, name: str) -> None:
    if limit is not None and value > limit:
        raise ValueError(
            f"LBA hard cap exceeded: {name}={value} is greater than {limit}."
        )


__all__ = ["LBA", "LBAConfig", "PlannerMode", "speech_length", "text_length"]
