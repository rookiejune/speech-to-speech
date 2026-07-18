from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, TypedDict

import torch
from anydataset.types import Modality
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .._compat import StrEnum
from .._tensor import is_signed_integer_dtype
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


class AcousticPrompt(TypedDict):
    codes: Tensor
    token_positions: Tensor


class AcousticTarget(TypedDict):
    semantic_codes: Tensor
    codes: Tensor
    token_positions: Tensor


@dataclass
class ModelSample:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_prompt: AcousticPrompt | None
    acoustic_target: AcousticTarget | None
    task: Task


@dataclass
class ModelBatch:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_prompt: AcousticPrompt | None
    acoustic_target: AcousticTarget | None
    tasks: list[Task]
    pad_token_id: int

    def __post_init__(self) -> None:
        if self.input_ids.dim() != 2 or self.token_labels.shape != self.input_ids.shape:
            raise ValueError(
                "batch input ids and token labels must be aligned 2D tensors."
            )
        if not is_signed_integer_dtype(
            self.input_ids.dtype
        ) or not is_signed_integer_dtype(self.token_labels.dtype):
            raise TypeError(
                "batch input ids and token labels must use signed integer dtypes."
            )
        batch_size = self.input_ids.size(0)
        if batch_size < 1:
            raise ValueError("ModelBatch requires at least one row.")
        if len(self.tasks) != batch_size:
            raise ValueError("ModelBatch tasks must provide one Task per row.")
        if any(not isinstance(task, Task) for task in self.tasks):
            raise TypeError("ModelBatch tasks must contain Task values.")
        signatures = {
            (task.source_modality, task.target_modality) for task in self.tasks
        }
        if len(signatures) != 1:
            raise ValueError(
                "all samples in a batch must use the same source and target modalities."
            )

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
        return cls(
            input_ids=_pad([sample.input_ids for sample in samples], pad_token_id),
            token_labels=_pad([sample.token_labels for sample in samples], -100),
            acoustic_prompt=_prompt([sample.acoustic_prompt for sample in samples]),
            acoustic_target=_target([sample.acoustic_target for sample in samples]),
            tasks=[sample.task for sample in samples],
            pad_token_id=pad_token_id,
        )

    @cached_property
    def attention_mask(self) -> Tensor:
        return self.input_ids != self.pad_token_id

    @cached_property
    def acoustic_prompt_mask(self) -> Tensor | None:
        if self.acoustic_prompt is None:
            return None
        return (self.acoustic_prompt["codes"] != ACOUSTIC_PAD_ID).all(dim=-1) & (
            self.acoustic_prompt["token_positions"] >= 0
        )

    @cached_property
    def acoustic_target_mask(self) -> Tensor | None:
        if self.acoustic_target is None:
            return None
        code_mask = (self.acoustic_target["codes"] != ACOUSTIC_PAD_ID).all(dim=-1)
        return (self.acoustic_target["token_positions"] >= 0) & code_mask


def _pad(values: list[Tensor], padding_value: int) -> Tensor:
    return pad_sequence(
        values,
        batch_first=True,
        padding_value=padding_value,
    )


def _prompt(values: list[AcousticPrompt | None]) -> AcousticPrompt | None:
    prompts = _present(values)
    if prompts is None:
        return None
    return AcousticPrompt(
        codes=_pad([value["codes"] for value in prompts], ACOUSTIC_PAD_ID),
        token_positions=_pad(
            [value["token_positions"] for value in prompts], ACOUSTIC_PAD_ID
        ),
    )


def _target(values: list[AcousticTarget | None]) -> AcousticTarget | None:
    targets = _present(values)
    if targets is None:
        return None
    return AcousticTarget(
        semantic_codes=_pad(
            [value["semantic_codes"] for value in targets], ACOUSTIC_PAD_ID
        ),
        codes=_pad([value["codes"] for value in targets], ACOUSTIC_PAD_ID),
        token_positions=_pad(
            [value["token_positions"] for value in targets], ACOUSTIC_PAD_ID
        ),
    )


T = TypeVar("T")


def _present(values: list[T | None]) -> list[T] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("a batch field must be present for every sample or none.")
    return present


def _validate_sample(sample: ModelSample, pad_token_id: int) -> None:
    if (
        sample.input_ids.dim() != 1
        or sample.token_labels.shape != sample.input_ids.shape
    ):
        raise ValueError(
            "sample input ids and token labels must be aligned 1D tensors."
        )

    if sample.acoustic_prompt is not None:
        _validate_acoustic_pair(
            sample.input_ids,
            sample.acoustic_prompt["codes"],
            sample.acoustic_prompt["token_positions"],
            name="acoustic prompt",
            pad_token_id=pad_token_id,
        )

    target = sample.acoustic_target
    if target is not None:
        _validate_acoustic_pair(
            sample.input_ids,
            target["codes"],
            target["token_positions"],
            name="acoustic target",
            pad_token_id=pad_token_id,
        )
        semantic_codes = target["semantic_codes"]
        _validate_codes(semantic_codes, name="target semantic codes")
        if semantic_codes.size(0) != target["codes"].size(0):
            raise ValueError(
                "target semantic and acoustic codes must share the frame axis."
            )
    if sample.task.target_modality is Modality.TEXT and target is not None:
        raise ValueError("text-target tasks must not provide acoustic target fields.")
    if target is not None:
        positions = target["token_positions"].to(
            device=sample.token_labels.device,
            dtype=torch.long,
        )
        labels = sample.token_labels[positions]
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
    _validate_codes(codes, name=f"{name} codes")
    if token_positions.dim() != 1:
        raise ValueError(f"{name} token positions must have shape [frames].")
    if not is_signed_integer_dtype(token_positions.dtype):
        raise TypeError(
            f"{name} token positions must contain integer indices using a signed dtype."
        )
    if codes.size(0) != token_positions.numel():
        raise ValueError(f"{name} codes and token positions must share the frame axis.")
    if bool((token_positions < 0).any()) or bool(
        (token_positions >= input_ids.numel()).any()
    ):
        raise ValueError(f"{name} positions must point inside the token sequence.")
    positions = token_positions.to(device=input_ids.device, dtype=torch.long)
    if bool(input_ids[positions].eq(pad_token_id).any()):
        raise ValueError(f"{name} positions must not point to padding tokens.")


def _validate_codes(codes: Tensor, *, name: str) -> None:
    if codes.dim() != 2:
        raise ValueError(f"{name} must have shape [frames, codebooks].")
    if codes.size(0) == 0 or codes.size(1) == 0:
        raise ValueError(f"{name} must contain at least one frame and codebook.")
    if not is_signed_integer_dtype(codes.dtype):
        raise TypeError(f"{name} must contain integer codec IDs using a signed dtype.")
    if bool((codes < 0).any()):
        raise ValueError(f"{name} must contain non-negative codec IDs.")
