from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import cast

import torch
from anydataset import types
from anydataset.types import Modality
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
    T2TT = auto()
    TTS = auto()

    @property
    def source_modality(self) -> Modality | None:
        if self in {Task.AUDIO_AR, Task.TEXT_AR}:
            return None
        if self in {Task.ASR, Task.S2ST, Task.S2TT}:
            return Modality.AUDIO
        return Modality.TEXT

    @property
    def target_modality(self) -> Modality:
        if self in {Task.ASR, Task.S2TT, Task.TEXT_AR, Task.T2TT}:
            return Modality.TEXT
        return Modality.AUDIO

    @property
    def paired(self) -> bool:
        return self in {Task.S2ST, Task.S2TT, Task.T2ST, Task.T2TT}


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
        tokenizer = runtime().audio_tokenizer
        spans = _as_tensor(
            [len(tokenizer.decode([int(token)])) for token in self.bpe_ids]
        ).to(dtype=torch.long)
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
        codes = audio_item.views[types.AudioView.LONGCAT]
        if not isinstance(codes, Tensor):
            raise TypeError("LongCat view must be a [frame, codebook] Tensor.")
        if codes.dim() != 2 or codes.size(1) < 2:
            raise ValueError(
                "LongCat view must contain semantic and acoustic codebooks."
            )
        semantic_ids = codes[:, :1]
        acoustic_ids = codes[:, 1:]
    elif runtime().config.audio_view == types.AudioView.VQ:
        semantic_ids = audio_item.views[types.AudioView.VQ]
        acoustic_ids = None
    else:
        raise NotImplementedError
    return semantic_ids, acoustic_ids


def _parse_role(sample: types.Sample, role: types.Role):
    audio_item = cast(types.AudioItem, sample[(role, types.Modality.AUDIO)])
    semantic_ids, acoustic_ids = _parse_audio_item(audio_item)
    text_item = cast(types.TextItem, sample[(role, types.Modality.TEXT)])
    text = text_item.views[types.TextView.TEXT]
    return Speech(
        semantic_ids=_frame_codes(semantic_ids),
        acoustic_ids=None if acoustic_ids is None else _frame_codes(acoustic_ids),
        text_ids=_text_ids(text),
        language=text_item.meta[types.TextMeta.LANG],
    )


def _frame_codes(codes: Tensor) -> Tensor:
    if codes.dim() == 1:
        return codes.unsqueeze(-1)
    if codes.dim() != 2:
        raise ValueError("audio codes must have shape [frames, codebooks].")
    return codes


def _text_ids(text: str) -> Tensor:
    ids = _as_tensor(
        runtime().text_tokenizer.encode(text, add_special_tokens=False)
    ).to(dtype=torch.long)
    if ids.dim() != 1:
        raise ValueError("text tokenizer must return a 1D token sequence.")
    return ids


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
        for sample in samples:
            _validate_sample(sample)
        signatures = {
            (sample.task.source_modality, sample.task.target_modality)
            for sample in samples
        }
        if len(signatures) != 1:
            raise ValueError(
                "all samples in a batch must use the same source and target modalities."
            )
        return cls(
            cast(
                Tensor,
                _pad([sample.input_ids for sample in samples], runtime().pad_token_id),
            ),
            cast(Tensor, _pad([sample.labels for sample in samples], -100)),
            _pad(
                [sample.acoustic_input_ids for sample in samples], ACOUSTIC_PAD_ID
            ),
            _pad(
                [sample.acoustic_input_positions for sample in samples],
                ACOUSTIC_PAD_ID,
            ),
            _pad([sample.acoustic_labels for sample in samples], ACOUSTIC_PAD_ID),
            _pad(
                [sample.acoustic_label_positions for sample in samples],
                ACOUSTIC_PAD_ID,
            ),
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


def _pad(values: list[Tensor | None], padding_value: int):
    has_none = any(value is None for value in values)
    if has_none:
        if any(value is not None for value in values):
            raise ValueError("a batch field must be present for every sample or none.")
        return None
    return pad_sequence(
        cast(list[Tensor], values),
        batch_first=True,
        padding_value=padding_value,
    )


def _validate_sample(sample: Sample) -> None:
    if sample.input_ids.dim() != 1 or sample.labels.shape != sample.input_ids.shape:
        raise ValueError("sample input_ids and labels must be aligned 1D tensors.")

    _validate_acoustic_pair(
        sample.input_ids,
        sample.acoustic_input_ids,
        sample.acoustic_input_positions,
        name="acoustic input",
    )
    _validate_acoustic_pair(
        sample.input_ids,
        sample.acoustic_labels,
        sample.acoustic_label_positions,
        name="acoustic target",
    )

    has_target = sample.acoustic_labels is not None
    if sample.task.target_modality is Modality.AUDIO and not has_target:
        raise ValueError("audio-target tasks require acoustic target fields.")
    if sample.task.target_modality is Modality.TEXT and has_target:
        raise ValueError("text-target tasks must not provide acoustic target fields.")
    if sample.acoustic_label_positions is not None:
        labels = sample.labels[sample.acoustic_label_positions]
        if bool(labels.eq(-100).any()):
            raise ValueError("acoustic target positions must point to semantic labels.")


def _validate_acoustic_pair(
    input_ids: Tensor,
    ids: Tensor | None,
    positions: Tensor | None,
    *,
    name: str,
) -> None:
    if (ids is None) != (positions is None):
        raise ValueError(f"{name} ids and positions must be provided together.")
    if ids is None or positions is None:
        return
    if ids.dim() != 2 or positions.dim() != 1:
        raise ValueError(
            f"{name} ids and positions must have shapes [frames, codebooks] and [frames]."
        )
    if ids.size(0) != positions.numel():
        raise ValueError(f"{name} ids and positions must share the frame axis.")
    if bool((positions < 0).any()) or bool((positions >= input_ids.numel()).any()):
        raise ValueError(f"{name} positions must point inside the semantic sequence.")
    if bool(input_ids[positions].eq(runtime().pad_token_id).any()):
        raise ValueError(f"{name} positions must not point to padding tokens.")
