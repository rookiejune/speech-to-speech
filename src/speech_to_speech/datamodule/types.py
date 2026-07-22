from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, TypedDict, Union

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


@dataclass
class Text:
    text_token_ids: Tensor
    language: Language

    def __post_init__(self) -> None:
        if self.text_token_ids.dim() != 1:
            raise ValueError("text_token_ids must have shape [tokens].")
        if not is_signed_integer_dtype(self.text_token_ids.dtype):
            raise TypeError("text_token_ids must use a signed integer dtype.")


@dataclass
class TextPair:
    source: Text
    target: Text


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
        source_modality, target_modality = next(iter(signatures))
        if source_modality is not Modality.AUDIO and self.acoustic_prompt is not None:
            raise ValueError(
                "only audio-source tasks may provide acoustic prompt fields."
            )
        if target_modality is Modality.TEXT and self.acoustic_target is not None:
            raise ValueError(
                "text-target tasks must not provide acoustic target fields."
            )
        _validate_batch_acoustic(
            self.input_ids,
            self.acoustic_prompt,
            name="acoustic prompt",
            minimum_position=0,
        )
        _validate_batch_acoustic(
            self.input_ids,
            self.acoustic_target,
            name="acoustic target",
            minimum_position=1,
        )
        _validate_batch_target_labels(self.token_labels, self.acoustic_target)

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

    def pin_memory(self) -> ModelBatch:
        return ModelBatch(
            input_ids=self.input_ids.pin_memory(),
            token_labels=self.token_labels.pin_memory(),
            acoustic_prompt=_pin_prompt(self.acoustic_prompt),
            acoustic_target=_pin_target(self.acoustic_target),
            tasks=list(self.tasks),
            pad_token_id=self.pad_token_id,
        )


TrainBatch = Union[ModelBatch, tuple[ModelBatch, ...]]


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


def _pin_prompt(value: AcousticPrompt | None) -> AcousticPrompt | None:
    if value is None:
        return None
    return AcousticPrompt(
        codes=value["codes"].pin_memory(),
        token_positions=value["token_positions"].pin_memory(),
    )


def _pin_target(value: AcousticTarget | None) -> AcousticTarget | None:
    if value is None:
        return None
    return AcousticTarget(
        semantic_codes=value["semantic_codes"].pin_memory(),
        codes=value["codes"].pin_memory(),
        token_positions=value["token_positions"].pin_memory(),
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
        if bool((target["token_positions"] < 1).any()):
            raise ValueError(
                "acoustic target positions must be at least 1 so every frame has "
                "a causal predecessor."
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


def _validate_batch_acoustic(
    input_ids: Tensor,
    value: AcousticPrompt | AcousticTarget | None,
    *,
    name: str,
    minimum_position: int,
) -> None:
    if value is None:
        return
    codes = value["codes"]
    positions = value["token_positions"]
    if codes.dim() != 3 or positions.dim() != 2:
        raise ValueError(f"{name} batch fields must have shapes [B, F, Q] and [B, F].")
    if not is_signed_integer_dtype(codes.dtype) or not is_signed_integer_dtype(
        positions.dtype
    ):
        raise TypeError(f"{name} batch fields must use signed integer dtypes.")
    if codes.size(-1) < 1:
        raise ValueError(f"{name} codes must contain at least one codebook.")
    if codes.shape[:2] != positions.shape:
        raise ValueError(f"{name} batch fields must align on batch and frame.")
    if positions.size(0) != input_ids.size(0):
        raise ValueError(f"{name} batch must align with input batch size.")
    if codes.device != input_ids.device or positions.device != input_ids.device:
        raise ValueError(f"{name} batch fields must use the input tensor device.")
    active = positions.ge(0)
    if bool((positions < -1).any()):
        raise ValueError(f"{name} positions may only use -1 as padding.")
    if bool((active & positions.lt(minimum_position)).any()):
        raise ValueError(
            f"{name} positions must be at least {minimum_position} for active frames."
        )
    if bool((active & positions.ge(input_ids.size(1))).any()):
        raise ValueError(f"{name} position exceeds the token sequence length.")
    code_mask = codes.ge(0).all(dim=-1)
    code_padding = codes.eq(ACOUSTIC_PAD_ID).all(dim=-1)
    if not bool((code_mask | code_padding).all()):
        raise ValueError(
            f"{name} codes must be non-negative or use -1 for a whole padded frame."
        )
    if not torch.equal(active, code_mask):
        raise ValueError(
            f"{name} positions and codes must share the same padding mask."
        )
    if positions.size(1) < 1 or not bool(active.any(dim=1).all()):
        raise ValueError(f"each {name} batch row must contain an active frame.")
    semantic = value.get("semantic_codes")
    if semantic is not None:
        if semantic.dim() != 3 or semantic.shape[:2] != positions.shape:
            raise ValueError(
                "acoustic target semantic codes must align on batch and frame."
            )
        if not is_signed_integer_dtype(semantic.dtype):
            raise TypeError("acoustic target semantic codes must use a signed dtype.")
        if semantic.size(-1) < 1:
            raise ValueError("acoustic target semantic codes must contain a codebook.")
        if semantic.device != input_ids.device:
            raise ValueError(
                "acoustic target semantic codes must use the input tensor device."
            )
        semantic_mask = semantic.ge(0).all(dim=-1)
        semantic_padding = semantic.eq(ACOUSTIC_PAD_ID).all(dim=-1)
        if not bool((semantic_mask | semantic_padding).all()) or not torch.equal(
            semantic_mask, active
        ):
            raise ValueError(
                "acoustic target semantic codes must share the frame padding mask."
            )


def _validate_batch_target_labels(
    labels: Tensor,
    target: AcousticTarget | None,
) -> None:
    if target is None:
        return
    positions = target["token_positions"]
    if positions.device != labels.device:
        raise ValueError("acoustic target labels and positions must use one device.")
    active = positions.ge(0)
    rows = torch.arange(labels.size(0), device=positions.device)[:, None]
    selected = labels[rows.expand_as(positions)[active], positions[active]]
    if bool(selected.eq(-100).any()):
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
