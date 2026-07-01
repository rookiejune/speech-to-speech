"""Model-side token, conditioning, and generation contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Protocol

from torch import FloatTensor, LongTensor, Tensor


class SpecialToken(StrEnum):
    # Qwen3
    PAD = "<|endoftext|>"
    BOS = "<|im_start|>"
    EOS = "<|im_end|>"
    USER = "user"
    ASSISTANT = "assistant"
    SEP = "\n"
    BOT = "\x3cthink>"
    EOT = "\x3c/think>"


class AudioBoundary(StrEnum):
    BOA = auto()
    EOA = auto()


@dataclass(frozen=True)
class AcousticCondition:
    hidden_states: FloatTensor
    semantic_ids: LongTensor
    mask: Tensor
    chunk_lengths: tuple[tuple[int, ...], ...] | None = None


class AcousticFeatureGenerator(Protocol):
    def __call__(self, condition: AcousticCondition) -> FloatTensor: ...


class AcousticFeatureExtractor(Protocol):
    def acoustic_codes_to_features(self, acoustic_ids: Tensor) -> FloatTensor: ...


class SemanticBPE(Protocol):
    def expand_ids(self, ids: Sequence[int]) -> Sequence[Sequence[int]]: ...


class WaveformCodec(Protocol):
    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor: ...


class LongCatCodec(WaveformCodec, AcousticFeatureExtractor, Protocol):
    def decode(self, semantic_codes: Tensor, acoustic_codes: Tensor) -> Tensor: ...


@dataclass
class AcousticConditionGeneration:
    hidden_states: FloatTensor
    mask: Tensor
    token_ids: LongTensor | None = None


@dataclass
class SemanticGeneration:
    semantic_ids: LongTensor
    semantic_mask: Tensor
    token_ids: LongTensor


@dataclass
class WaveformGeneration:
    audio: FloatTensor
    audio_mask: Tensor
    semantic_ids: LongTensor
    semantic_mask: Tensor
    acoustic_features: FloatTensor
    condition_hidden_states: FloatTensor
    token_ids: LongTensor


@dataclass
class TeacherForcedWaveformGeneration:
    audio: FloatTensor
    audio_mask: Tensor
    semantic_ids: LongTensor
    semantic_mask: Tensor
    acoustic_features: FloatTensor
    condition_hidden_states: FloatTensor
