from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Protocol

from torch import FloatTensor, LongTensor, Tensor

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


class SpecialToken(StrEnum):
    # Qwen3
    PAD = "<|endoftext|>"
    BOS = "<|im_start|>"
    EOS = "<|im_end|>"
    USER = "user"
    ASSISTANT = "assistant"
    SEP = "\n"
    BOT = "<think>"
    EOT = "</think>"


class AudioBoundary(StrEnum):
    BOA = auto()
    EOA = auto()


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


@dataclass(frozen=True)
class LongCatBatchSide:
    semantic_ids: LongTensor
    semantic_mask: Tensor
    acoustic_ids: LongTensor
    acoustic_mask: Tensor


@dataclass(frozen=True)
class AutoregressionExample:
    audio_ids: Tensor


@dataclass(frozen=True)
class TranslationExample:
    source_ids: Tensor
    target_ids: Tensor


@dataclass(frozen=True)
class BPEArtifactMeta:
    codec_name: str
    vocab_size: int
    min_frequency: int = 0
    max_token_length: int | None = None
    codebook_sizes: tuple[int, ...] = (8192,)
    datasets: tuple[Mapping[str, object], ...] = ()


@dataclass
class CausalLMBatch:
    input_ids: LongTensor
    attention_mask: LongTensor
    labels: LongTensor
    logits_to_keep: int | LongTensor
    source_audio: LongCatBatchSide | None = None
    target_audio: LongCatBatchSide | None = None
    task_family: LongTensor | None = None


@dataclass
class GenerationBatch:
    input_ids: LongTensor
    attention_mask: LongTensor


@dataclass(frozen=True)
class AcousticCondition:
    hidden_states: FloatTensor
    semantic_ids: LongTensor
    mask: Tensor


class AcousticFeatureGenerator(Protocol):
    def __call__(self, condition: AcousticCondition) -> FloatTensor: ...


class SemanticBPE(Protocol):
    def expand_ids(self, ids: Sequence[int]) -> Sequence[Sequence[int]]: ...


class WaveformCodec(Protocol):
    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor: ...


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


# all special tokens:
# ['<|im_end|>',
#  '<|endoftext|>',
#  '<|im_start|>',
#  '<|object_ref_start|>',
#  '<|object_ref_end|>',
#  '<|box_start|>',
#  '<|box_end|>',
#  '<|quad_start|>',
#  '<|quad_end|>',
#  '<|vision_start|>',
#  '<|vision_end|>',
#  '<|vision_pad|>',
#  '<|image_pad|>',
#  '<|video_pad|>']
