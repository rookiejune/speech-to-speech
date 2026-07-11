from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import cast

import torch
from anydataset import types
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .._compat import StrEnum, auto
from ..runtime import runtime

ACOUSTIC_PAD_ID = -1


class Task(StrEnum):
    AUDIO_AR = auto()
    ASR = auto()
    S2ST = auto()
    S2TT = auto()
    TEXT_AR = auto()
    T2ST = auto()
    TTS = auto()


class Language(StrEnum):
    ZH = "Chinese"
    EN = "English"

    @classmethod
    def _missing_(cls, value: object) -> Language | None:
        if not isinstance(value, str):
            return None
        normalized = value.lower()
        if normalized in {"zh", "zh-cn", "zh_cn", "chinese"}:
            return cls.ZH
        if normalized in {"en", "en-us", "en_us", "english"}:
            return cls.EN
        return None


@dataclass
class Speech:
    semantic_ids: Tensor
    acoustic_ids: Tensor | None
    text_ids: Tensor
    language: str

    @cached_property
    def bpe_ids(self) -> Tensor:
        return _as_tensor(runtime().audio_tokenizer.encode(self.semantic_ids))

    @cached_property
    def bpe_spans(self) -> Tensor:
        """Number of semantic frames represented by each BPE token."""
        _, counts = runtime().audio_tokenizer.expand_with_counts(self.bpe_ids)
        spans = _as_tensor(counts).to(dtype=torch.long)
        if spans.dim() != 1:
            raise ValueError("audio tokenizer spans must have shape [num_bpe_tokens].")
        if int(spans.sum().item()) != self.semantic_ids.size(0):
            raise ValueError("BPE spans must cover all semantic frames exactly once.")
        return spans

    def __post_init__(self) -> None:
        if self.semantic_ids.dim() != 2:
            raise ValueError("semantic_ids must have shape [frames, codebooks].")
        if self.acoustic_ids is not None:
            if self.acoustic_ids.dim() != 2:
                raise ValueError("acoustic_ids must have shape [frames, codebooks].")
            if self.acoustic_ids.size(0) != self.semantic_ids.size(0):
                raise ValueError(
                    "semantic_ids and acoustic_ids must share the frame axis."
                )


@dataclass
class SpeechPair:
    source: Speech
    target: Speech

    @classmethod
    def from_raw(cls, sample: types.Sample):
        return cls(
            _parse_role(sample, types.Role.SOURCE),
            _parse_role(sample, types.Role.TARGET),
        )


def _parse_audio_item(audio_item: types.AudioItem):
    if runtime().config.audio_view == types.AudioView.LONGCAT:
        sementic_ids = audio_item.views[types.AudioView.LONGCAT]["semantic_codes"]
        acoustic_ids = audio_item.views[types.AudioView.LONGCAT]["acoustic_codes"]
    elif runtime().config.audio_view == types.AudioView.VQ:
        sementic_ids = audio_item.views[types.AudioView.VQ]
        acoustic_ids = None
    else:
        raise NotImplementedError
    return sementic_ids, acoustic_ids


def _parse_role(sample: types.Sample, role: types.Role):
    audio_item = cast(types.AudioItem, sample[(role, types.Modality.AUDIO)])
    semantic_ids, acoustic_ids = _parse_audio_item(audio_item)
    text_item = cast(types.TextItem, sample[(role, types.Modality.TEXT)])
    return Speech(
        semantic_ids=_frame_codes(semantic_ids),
        acoustic_ids=None if acoustic_ids is None else _frame_codes(acoustic_ids),
        text_ids=text_item.views[types.TextView.TEXT],
        language=text_item.meta[types.TextMeta.LANG],
    )


def _frame_codes(codes: Tensor) -> Tensor:
    if codes.dim() == 1:
        return codes.unsqueeze(-1)
    if codes.dim() != 2:
        raise ValueError("audio codes must have shape [frames, codebooks].")
    return codes


def _as_tensor(value: Tensor | list[int]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.long)


@dataclass
class Sample:
    input_ids: Tensor
    labels: Tensor
    acoustic_input_ids: Tensor | None
    acoustic_input_positions: Tensor | None
    acoustic_labels: Tensor | None
    acoustic_label_positions: Tensor | None
    task: Task


@dataclass
class ModelBatch:
    input_ids: Tensor
    labels: Tensor
    acoustic_input_ids: Tensor | None
    acoustic_input_positions: Tensor | None
    acoustic_labels: Tensor | None
    acoustic_label_positions: Tensor | None
    tasks: list[Task]

    @classmethod
    def from_samples(cls, samples: list[Sample]):
        if not samples:
            raise ValueError("ModelBatch requires at least one sample.")
        return cls(
            cast(Tensor, _pad(samples, "input_ids", runtime().pad_token_id)),
            cast(Tensor, _pad(samples, "labels", -100)),
            _pad(samples, "acoustic_input_ids", ACOUSTIC_PAD_ID),
            _pad(samples, "acoustic_input_positions", ACOUSTIC_PAD_ID),
            _pad(samples, "acoustic_labels", ACOUSTIC_PAD_ID),
            _pad(samples, "acoustic_label_positions", ACOUSTIC_PAD_ID),
            [sample.task for sample in samples],
        )

    @cached_property
    def attention_mask(self) -> Tensor:
        return self.input_ids != runtime().pad_token_id

    @cached_property
    def acoustic_input_mask(self) -> Tensor | None:
        if self.acoustic_input_ids is None:
            return None
        if self.acoustic_input_positions is None:
            raise ValueError("acoustic positions are required with acoustic ids.")
        return (self.acoustic_input_ids != ACOUSTIC_PAD_ID).all(dim=-1) & (
            self.acoustic_input_positions >= 0
        )

    @cached_property
    def acoustic_label_mask(self) -> Tensor | None:
        if self.acoustic_labels is None:
            return None
        return (self.acoustic_labels != ACOUSTIC_PAD_ID).all(dim=-1)

    @cached_property
    def acoustic_target_mask(self) -> Tensor | None:
        if self.acoustic_label_positions is None:
            return None
        if self.acoustic_labels is None:
            raise ValueError("acoustic labels are required with target positions.")
        return (self.acoustic_label_positions >= 0) & self.acoustic_label_mask


# Kept as a source-compatible alias for callers that have not migrated yet.
Batch = ModelBatch


def _list_attrs(samples: list[Sample], name: str) -> list[Tensor | None]:
    return [getattr(sample, name) for sample in samples]


def _pad(samples: list[Sample], name: str, padding_value: int):
    values = _list_attrs(samples, name)
    has_none = any(value is None for value in values)
    if has_none:
        if any(value is not None for value in values):
            raise ValueError(f"{name} must be present for every sample or none.")
        return None
    return pad_sequence(
        cast(list[Tensor], values),
        batch_first=True,
        padding_value=padding_value,
    )
