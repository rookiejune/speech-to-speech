"""Data sample and causal-LM batch contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Protocol

from torch import LongTensor, Tensor

IGNORE_INDEX = -100


class Task(StrEnum):
    AUTOREGRESSION = auto()
    TRANSLATION = auto()


class TaskFamily(StrEnum):
    SOURCE_AR = auto()
    TARGET_AR = auto()
    SOURCE_TO_TARGET = auto()
    TARGET_TO_SOURCE = auto()

    @property
    def id(self) -> int:
        match self:
            case TaskFamily.SOURCE_AR:
                return 0
            case TaskFamily.TARGET_AR:
                return 1
            case TaskFamily.SOURCE_TO_TARGET:
                return 2
            case TaskFamily.TARGET_TO_SOURCE:
                return 3


@dataclass(frozen=True)
class SpeechPair:
    source_ids: Tensor
    target_ids: Tensor


@dataclass(frozen=True)
class LongCatSide:
    semantic_ids: Tensor
    acoustic_ids: Tensor


@dataclass(frozen=True)
class LongCatPair:
    source: LongCatSide
    target: LongCatSide


class LongCatBPETokenizer(Protocol):
    def encode_frames(self, frames: Sequence[Sequence[int]]) -> Sequence[int]: ...

    def expand_ids(self, ids: Sequence[int]) -> Sequence[Sequence[int]]: ...


@dataclass(frozen=True)
class LongCatBatchSide:
    semantic_ids: LongTensor
    semantic_mask: Tensor
    acoustic_ids: LongTensor
    acoustic_mask: Tensor


@dataclass(frozen=True)
class AutoregressionExample:
    audio_ids: Tensor
    audio_weights: Tensor | None = None


@dataclass(frozen=True)
class TranslationExample:
    source_ids: Tensor
    target_ids: Tensor
    target_weights: Tensor | None = None


@dataclass
class CausalLMBatch:
    input_ids: LongTensor
    attention_mask: LongTensor
    labels: LongTensor
    logits_to_keep: int | LongTensor
    loss_weights: Tensor | None = None
    source_audio: LongCatBatchSide | None = None
    target_audio: LongCatBatchSide | None = None
    task_family: LongTensor | None = None


@dataclass
class GenerationBatch:
    input_ids: LongTensor
    attention_mask: LongTensor
