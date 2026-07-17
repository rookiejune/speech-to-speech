from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import cast

from anydataset.types import Modality
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .._compat import StrEnum
from ..task import Task
ACOUSTIC_PAD_ID = -1


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
    semantic_codes: Tensor
    acoustic_codes: Tensor | None
    text_token_ids: Tensor
    audio_token_ids: Tensor
    audio_token_spans: Tensor
    language: Language

    def __post_init__(self) -> None:
        if self.semantic_codes.dim() != 2:
            raise ValueError("semantic_codes must have shape [frames, codebooks].")
        if self.acoustic_codes is not None:
            if self.acoustic_codes.dim() != 2:
                raise ValueError("acoustic_codes must have shape [frames, codebooks].")
            if self.acoustic_codes.size(0) != self.semantic_codes.size(0):
                raise ValueError(
                    "semantic_codes and acoustic_codes must share the frame axis."
                )
        if self.audio_token_ids.dim() != 1 or self.audio_token_spans.shape != (
            self.audio_token_ids.numel(),
        ):
            raise ValueError("audio token ids and spans must be aligned 1D tensors.")
        if int(self.audio_token_spans.sum().item()) != self.semantic_codes.size(0):
            raise ValueError("audio token spans must cover all semantic frames.")


@dataclass
class SpeechPair:
    source: Speech
    target: Speech


@dataclass
class ModelSample:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_prompt_codes: Tensor | None
    acoustic_prompt_positions: Tensor | None
    target_semantic_codes: Tensor | None
    target_acoustic_codes: Tensor | None
    target_audio_token_positions: Tensor | None
    task: Task


@dataclass
class ModelBatch:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_prompt_codes: Tensor | None
    acoustic_prompt_positions: Tensor | None
    target_semantic_codes: Tensor | None
    target_acoustic_codes: Tensor | None
    target_audio_token_positions: Tensor | None
    tasks: list[Task]
    pad_token_id: int

    @classmethod
    def from_samples(
        cls,
        samples: list[ModelSample],
        *,
        pad_token_id: int,
    ) -> ModelBatch:
        if not samples:
            raise ValueError("ModelBatch requires at least one sample.")
        for sample in samples:
            _validate_sample(sample, pad_token_id)
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
                _pad([sample.input_ids for sample in samples], pad_token_id),
            ),
            cast(Tensor, _pad([sample.token_labels for sample in samples], -100)),
            _pad([sample.acoustic_prompt_codes for sample in samples], ACOUSTIC_PAD_ID),
            _pad(
                [sample.acoustic_prompt_positions for sample in samples],
                ACOUSTIC_PAD_ID,
            ),
            _pad([sample.target_semantic_codes for sample in samples], ACOUSTIC_PAD_ID),
            _pad([sample.target_acoustic_codes for sample in samples], ACOUSTIC_PAD_ID),
            _pad(
                [sample.target_audio_token_positions for sample in samples],
                ACOUSTIC_PAD_ID,
            ),
            [sample.task for sample in samples],
            pad_token_id,
        )

    @cached_property
    def attention_mask(self) -> Tensor:
        return self.input_ids != self.pad_token_id

    @cached_property
    def acoustic_prompt_mask(self) -> Tensor | None:
        if self.acoustic_prompt_codes is None:
            return None
        if self.acoustic_prompt_positions is None:
            raise ValueError("acoustic prompt positions are required with codes.")
        return (self.acoustic_prompt_codes != ACOUSTIC_PAD_ID).all(dim=-1) & (
            self.acoustic_prompt_positions >= 0
        )

    @cached_property
    def target_acoustic_mask(self) -> Tensor | None:
        if self.target_audio_token_positions is None:
            return None
        if self.target_acoustic_codes is None:
            raise ValueError("target acoustic codes require token positions.")
        code_mask = (self.target_acoustic_codes != ACOUSTIC_PAD_ID).all(dim=-1)
        return (self.target_audio_token_positions >= 0) & code_mask


def _pad(values: list[Tensor | None], padding_value: int) -> Tensor | None:
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


def _validate_sample(sample: ModelSample, pad_token_id: int) -> None:
    if (
        sample.input_ids.dim() != 1
        or sample.token_labels.shape != sample.input_ids.shape
    ):
        raise ValueError("sample input ids and token labels must be aligned 1D tensors.")

    _validate_acoustic_pair(
        sample.input_ids,
        sample.acoustic_prompt_codes,
        sample.acoustic_prompt_positions,
        name="acoustic prompt",
        pad_token_id=pad_token_id,
    )
    _validate_acoustic_pair(
        sample.input_ids,
        sample.target_acoustic_codes,
        sample.target_audio_token_positions,
        name="acoustic target",
        pad_token_id=pad_token_id,
    )

    has_target = sample.target_acoustic_codes is not None
    if (sample.target_semantic_codes is None) != (sample.target_acoustic_codes is None):
        raise ValueError(
            "target semantic and acoustic codes must be provided together."
        )
    if (
        sample.target_semantic_codes is not None
        and sample.target_acoustic_codes is not None
        and sample.target_semantic_codes.size(0) != sample.target_acoustic_codes.size(0)
    ):
        raise ValueError(
            "target semantic and acoustic codes must share the frame axis."
        )
    if sample.task.target_modality is Modality.TEXT and has_target:
        raise ValueError("text-target tasks must not provide acoustic target fields.")
    if sample.target_audio_token_positions is not None:
        labels = sample.token_labels[sample.target_audio_token_positions]
        if bool(labels.eq(-100).any()):
            raise ValueError("acoustic target positions must point to semantic labels.")


def _validate_acoustic_pair(
    input_ids: Tensor,
    codes: Tensor | None,
    token_positions: Tensor | None,
    *,
    name: str,
    pad_token_id: int,
) -> None:
    if (codes is None) != (token_positions is None):
        raise ValueError(f"{name} codes and token positions must be provided together.")
    if codes is None or token_positions is None:
        return
    if codes.dim() != 2 or token_positions.dim() != 1:
        raise ValueError(
            f"{name} codes and token positions must have shapes "
            "[frames, codebooks] and [frames]."
        )
    if codes.size(0) != token_positions.numel():
        raise ValueError(f"{name} codes and token positions must share the frame axis.")
    if bool((token_positions < 0).any()) or bool(
        (token_positions >= input_ids.numel()).any()
    ):
        raise ValueError(f"{name} positions must point inside the token sequence.")
    if bool(input_ids[token_positions].eq(pad_token_id).any()):
        raise ValueError(f"{name} positions must not point to padding tokens.")
