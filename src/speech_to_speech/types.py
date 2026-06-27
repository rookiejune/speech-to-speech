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
    max_piece_frames: int
    datasets: tuple[Mapping[str, object], ...] = ()


@dataclass
class CausalLMBatch:
    input_ids: LongTensor
    attention_mask: LongTensor
    labels: LongTensor
    logits_to_keep: int | LongTensor


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
    def expand_ids(self, ids: Sequence[int]) -> Sequence[int]: ...


class WaveformCodec(Protocol):
    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor: ...


@dataclass
class AcousticConditionGeneration:
    hidden_states: FloatTensor
    mask: Tensor
    token_ids: LongTensor | None = None


@dataclass
class WaveformGeneration:
    audio: FloatTensor
    audio_mask: Tensor
    semantic_ids: LongTensor
    semantic_mask: Tensor
    acoustic_features: FloatTensor
    condition_hidden_states: FloatTensor
    token_ids: LongTensor


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
