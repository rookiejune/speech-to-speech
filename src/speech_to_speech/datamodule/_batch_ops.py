from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .._tensor import is_signed_integer_dtype
from ..task import PredictionModality, Task, uses_source_ctc, uses_target_ctc
from .contract import ACOUSTIC_PAD_ID, CTC_PAD_ID, AcousticTarget, CTCTarget

if TYPE_CHECKING:
    from .batch import ModelSample


@dataclass(frozen=True)
class _PaddedSamples:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_target: AcousticTarget | None
    source_ctc: CTCTarget | None
    target_ctc: CTCTarget | None
    tasks: list[Task]
    predictions: list[PredictionModality]
    audio_seconds: Tensor
    generation_prompt_lengths: Tensor
    audio_input_positions: Tensor | None


@dataclass(frozen=True)
class _BatchGenerationFields:
    audio_seconds: Tensor
    generation_prompt_lengths: Tensor
    audio_input_positions: Tensor | None


@dataclass(frozen=True)
class _BatchUnitCounts:
    tokens: float
    padded_tokens: float
    frames: float | None
    padded_frames: float | None
    audio_seconds: float


def _validate_batch_tensors(
    input_ids: Tensor,
    token_labels: Tensor,
) -> int:
    if input_ids.dim() != 2 or token_labels.shape != input_ids.shape:
        raise ValueError("batch input ids and token labels must be aligned 2D tensors.")
    if not is_signed_integer_dtype(input_ids.dtype) or not is_signed_integer_dtype(
        token_labels.dtype
    ):
        raise TypeError(
            "batch input ids and token labels must use signed integer dtypes."
        )
    batch_size = input_ids.size(0)
    if batch_size < 1:
        raise ValueError("ModelBatch requires at least one row.")
    return batch_size


def _validate_batch_tasks(
    tasks: list[Task],
    predictions: list[PredictionModality],
    batch_size: int,
) -> PredictionModality:
    if len(tasks) != batch_size:
        raise ValueError("ModelBatch tasks must provide one Task per row.")
    if len(predictions) != batch_size:
        raise ValueError("ModelBatch predictions must provide one value per row.")
    if any(not isinstance(task, Task) for task in tasks):
        raise TypeError("ModelBatch tasks must contain Task values.")
    if any(
        not isinstance(prediction, PredictionModality) for prediction in predictions
    ):
        raise TypeError(
            "ModelBatch predictions must contain PredictionModality values."
        )
    for task, prediction in zip(tasks, predictions):
        if prediction not in task.allowed_predictions:
            raise ValueError(
                f"{task.value} does not allow prediction={prediction.value}."
            )
    signatures = {
        (task.source_layout, prediction) for task, prediction in zip(tasks, predictions)
    }
    if len(signatures) != 1:
        raise ValueError(
            "all samples in a batch must use the same execution signature."
        )
    _, prediction = next(iter(signatures))
    return prediction


def _complete_batch_generation_fields(
    input_ids: Tensor,
    token_labels: Tensor,
    *,
    audio_seconds: Tensor | None,
    generation_prompt_lengths: Tensor | None,
    audio_input_positions: Tensor | None,
) -> _BatchGenerationFields:
    batch_size = input_ids.size(0)
    if audio_seconds is None:
        audio_seconds = input_ids.new_zeros(batch_size, dtype=torch.float32)
    _validate_audio_seconds(audio_seconds, batch_size)
    if generation_prompt_lengths is None:
        generation_prompt_lengths = _generation_prompt_lengths(token_labels)
    _validate_generation_prompt_lengths(generation_prompt_lengths, input_ids)
    _validate_batch_audio_input_positions(audio_input_positions, input_ids)
    return _BatchGenerationFields(
        audio_seconds=audio_seconds,
        generation_prompt_lengths=generation_prompt_lengths,
        audio_input_positions=audio_input_positions,
    )


def _checked_batch_generation_fields(
    audio_seconds: Tensor | None,
    generation_prompt_lengths: Tensor | None,
    audio_input_positions: Tensor | None,
) -> _BatchGenerationFields:
    if audio_seconds is None:
        raise RuntimeError("ModelBatch audio_seconds is unavailable after validation.")
    if generation_prompt_lengths is None:
        raise RuntimeError(
            "ModelBatch generation fields are unavailable after validation."
        )
    return _BatchGenerationFields(
        audio_seconds=audio_seconds,
        generation_prompt_lengths=generation_prompt_lengths,
        audio_input_positions=audio_input_positions,
    )


def _padded_samples(samples: list[ModelSample], pad_token_id: int) -> _PaddedSamples:
    if not samples:
        raise ValueError("ModelBatch requires at least one sample.")
    for sample in samples:
        _validate_sample(sample, pad_token_id)
    return _PaddedSamples(
        input_ids=_pad([sample.input_ids for sample in samples], pad_token_id),
        token_labels=_pad([sample.labels.token_labels for sample in samples], -100),
        acoustic_target=_target([sample.labels.acoustic_target for sample in samples]),
        source_ctc=_ctc_target(
            [sample.labels.source_ctc for sample in samples],
            samples,
        ),
        target_ctc=_ctc_target(
            [sample.labels.target_ctc for sample in samples],
            samples,
        ),
        tasks=[sample.request["task"] for sample in samples],
        predictions=[sample.prediction for sample in samples],
        audio_seconds=_audio_seconds(samples),
        generation_prompt_lengths=_sample_prompt_lengths(samples),
        audio_input_positions=_optional_tensor(
            [sample.request["audio_input_positions"] for sample in samples],
            padding_value=-1,
        ),
    )


def _pin_optional(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.pin_memory()


def _to_optional(
    value: Tensor | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> Tensor | None:
    return None if value is None else value.to(device=device, non_blocking=non_blocking)


def _optional_row(value: Tensor | None, index: int) -> Tensor | None:
    return None if value is None else value[index : index + 1]


def _pad(values: list[Tensor], padding_value: int) -> Tensor:
    return pad_sequence(
        values,
        batch_first=True,
        padding_value=padding_value,
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


def _ctc_target(
    values: list[CTCTarget | None],
    samples: list[ModelSample],
) -> CTCTarget | None:
    if not any(value is not None for value in values):
        return None
    positions: list[Tensor] = []
    labels: list[Tensor] = []
    for value, sample in zip(values, samples):
        if value is None:
            positions.append(sample.input_ids.new_empty((0,), dtype=torch.long))
            labels.append(sample.input_ids.new_empty((0,), dtype=torch.long))
        else:
            positions.append(value["token_positions"])
            labels.append(value["text_token_ids"])
    return CTCTarget(
        token_positions=_pad(positions, CTC_PAD_ID),
        text_token_ids=_pad(labels, CTC_PAD_ID),
    )


def _acoustic_target_mask(value: AcousticTarget | None) -> Tensor | None:
    if value is None:
        return None
    code_mask = (value["codes"] != ACOUSTIC_PAD_ID).all(dim=-1)
    return (value["token_positions"] >= 0) & code_mask


def _batch_unit_counts(
    input_ids: Tensor,
    pad_token_id: int,
    acoustic_target: AcousticTarget | None,
    audio_seconds: Tensor,
) -> _BatchUnitCounts:
    token_mask = input_ids != pad_token_id
    frame_mask = _acoustic_target_mask(acoustic_target)
    return _BatchUnitCounts(
        tokens=float(token_mask.sum().item()),
        padded_tokens=float(token_mask.numel()),
        frames=(None if frame_mask is None else float(frame_mask.sum().item())),
        padded_frames=(None if frame_mask is None else float(frame_mask.numel())),
        audio_seconds=float(audio_seconds.sum().item()),
    )


def _optional_tensor(
    values: list[Tensor | None],
    *,
    padding_value: int,
) -> Tensor | None:
    present = _present(values)
    if present is None:
        return None
    return _pad(present, padding_value)


def _pin_target(value: AcousticTarget | None) -> AcousticTarget | None:
    if value is None:
        return None
    return AcousticTarget(
        semantic_codes=value["semantic_codes"].pin_memory(),
        codes=value["codes"].pin_memory(),
        token_positions=value["token_positions"].pin_memory(),
    )


def _pin_ctc_target(value: CTCTarget | None) -> CTCTarget | None:
    if value is None:
        return None
    return CTCTarget(
        token_positions=value["token_positions"].pin_memory(),
        text_token_ids=value["text_token_ids"].pin_memory(),
    )


def _to_target(
    value: AcousticTarget | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> AcousticTarget | None:
    if value is None:
        return None
    return AcousticTarget(
        semantic_codes=value["semantic_codes"].to(
            device=device,
            non_blocking=non_blocking,
        ),
        codes=value["codes"].to(device=device, non_blocking=non_blocking),
        token_positions=value["token_positions"].to(
            device=device,
            non_blocking=non_blocking,
        ),
    )


def _to_ctc_target(
    value: CTCTarget | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> CTCTarget | None:
    if value is None:
        return None
    return CTCTarget(
        token_positions=value["token_positions"].to(
            device=device,
            non_blocking=non_blocking,
        ),
        text_token_ids=value["text_token_ids"].to(
            device=device,
            non_blocking=non_blocking,
        ),
    )


def _target_row(value: AcousticTarget | None, index: int) -> AcousticTarget | None:
    if value is None:
        return None
    return AcousticTarget(
        semantic_codes=value["semantic_codes"][index : index + 1],
        codes=value["codes"][index : index + 1],
        token_positions=value["token_positions"][index : index + 1],
    )


def _ctc_target_row(value: CTCTarget | None, index: int) -> CTCTarget | None:
    if value is None:
        return None
    return CTCTarget(
        token_positions=value["token_positions"][index : index + 1],
        text_token_ids=value["text_token_ids"][index : index + 1],
    )


def _audio_seconds(samples: list[ModelSample]) -> Tensor:
    return (
        samples[0]
        .request["prompt_ids"]
        .new_tensor(
            [sample.labels.audio_seconds for sample in samples],
            dtype=torch.float32,
        )
    )


def _sample_prompt_lengths(samples: list[ModelSample]) -> Tensor:
    return (
        samples[0]
        .request["prompt_ids"]
        .new_tensor(
            [sample.request["prompt_ids"].numel() for sample in samples],
            dtype=torch.long,
        )
    )


def _generation_prompt_lengths(token_labels: Tensor) -> Tensor:
    values = []
    for labels in token_labels:
        positions = labels.ne(-100).nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise ValueError(
                "each token label row must contain at least one target token."
            )
        values.append(int(positions[0].item()))
    return token_labels.new_tensor(values, dtype=torch.long)


def _validate_generation_prompt_lengths(value: Tensor, input_ids: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("ModelBatch generation_prompt_lengths must be a Tensor.")
    if value.shape != (input_ids.size(0),):
        raise ValueError(
            "ModelBatch generation_prompt_lengths must have shape [batch]."
        )
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(
            "ModelBatch generation_prompt_lengths must use a signed integer dtype."
        )
    if bool((value < 1).any()) or bool((value >= input_ids.size(1)).any()):
        raise ValueError(
            "generation prompt lengths must leave a non-empty generated response."
        )


def _validate_batch_audio_input_positions(
    value: Tensor | None,
    input_ids: Tensor,
) -> None:
    if value is None:
        return
    if value.dim() != 2 or value.size(0) != input_ids.size(0):
        raise ValueError(
            "ModelBatch audio_input_positions must have shape [batch, frames]."
        )
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(
            "ModelBatch audio_input_positions must use a signed integer dtype."
        )
    if bool((value < -1).any()) or bool((value >= input_ids.size(1)).any()):
        raise ValueError(
            "ModelBatch audio_input_positions must use -1 padding or valid sequence positions."
        )
    for row in value:
        valid = row[row.ge(0)]
        if valid.numel() != torch.unique(valid).numel():
            raise ValueError(
                "ModelBatch audio_input_positions must not repeat positions."
            )


def _validate_audio_seconds(value: Tensor, batch_size: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("ModelBatch audio_seconds must be a Tensor.")
    if value.shape != (batch_size,):
        raise ValueError("ModelBatch audio_seconds must have shape [batch].")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError("ModelBatch audio_seconds must use a real numeric dtype.")
    if not bool(torch.isfinite(value).all().item()) or bool(value.lt(0).any().item()):
        raise ValueError("ModelBatch audio_seconds must be finite and non-negative.")


T = TypeVar("T")


def _present(values: list[T | None]) -> list[T] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("a batch field must be present for every sample or none.")
    return present


def _validate_sample(sample: ModelSample, pad_token_id: int) -> None:
    prompt_ids = sample.request["prompt_ids"]
    response_ids = sample.labels.response_ids
    input_ids = torch.cat([prompt_ids, response_ids])
    token_labels = sample.labels.token_labels
    if prompt_ids.dim() != 1 or response_ids.dim() != 1:
        raise ValueError("sample prompt_ids and response_ids must be 1D tensors.")
    if prompt_ids.numel() < 1 or response_ids.numel() < 1:
        raise ValueError(
            "generation prompt lengths must leave a non-empty generated response."
        )
    if token_labels.shape != input_ids.shape:
        raise ValueError(
            "sample input ids and token labels must be aligned 1D tensors."
        )
    positions = sample.request["audio_input_positions"]
    if positions is not None:
        if positions.dim() != 1:
            raise ValueError("sample audio_input_positions must have shape [frames].")
        if not is_signed_integer_dtype(positions.dtype):
            raise TypeError(
                "sample audio_input_positions must use a signed integer dtype."
            )
        if bool((positions < 0).any()) or bool((positions >= input_ids.numel()).any()):
            raise ValueError(
                "sample audio_input_positions must be valid sequence positions."
            )
        if positions.numel() != torch.unique(positions).numel():
            raise ValueError("sample audio_input_positions must not repeat positions.")
    target = sample.labels.acoustic_target
    if target is not None:
        _validate_acoustic_pair(
            input_ids,
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
    if not sample.prediction.supervises_audio and target is not None:
        raise ValueError(
            "text-only prediction samples must not provide acoustic target fields."
        )
    if target is not None:
        positions = target["token_positions"].to(
            device=token_labels.device,
            dtype=torch.long,
        )
        labels = token_labels[positions]
        if bool(labels.eq(-100).any()):
            raise ValueError("acoustic target positions must point to semantic labels.")
    _validate_sample_ctc(
        input_ids,
        sample.labels.source_ctc,
        name="source CTC target",
        causal=False,
        allowed=uses_source_ctc(sample.task),
    )
    _validate_sample_ctc(
        input_ids,
        sample.labels.target_ctc,
        name="target CTC target",
        causal=True,
        allowed=uses_target_ctc(sample.task, sample.prediction),
    )


def _validate_batch_acoustic(
    input_ids: Tensor,
    value: AcousticTarget | None,
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


def _validate_sample_ctc(
    input_ids: Tensor,
    value: CTCTarget | None,
    *,
    name: str,
    causal: bool,
    allowed: bool,
) -> None:
    if value is None:
        return
    if not allowed:
        raise ValueError(f"{name} is not allowed by the task transcript visibility route.")
    positions = value["token_positions"]
    labels = value["text_token_ids"]
    if positions.dim() != 1 or labels.dim() != 1:
        raise ValueError(f"{name} fields must be one-dimensional tensors.")
    if positions.numel() == 0 or labels.numel() == 0:
        raise ValueError(f"{name} must contain audio positions and transcript tokens.")
    if not is_signed_integer_dtype(positions.dtype) or not is_signed_integer_dtype(
        labels.dtype
    ):
        raise TypeError(f"{name} fields must use signed integer dtypes.")
    if positions.device != input_ids.device or labels.device != input_ids.device:
        raise ValueError(f"{name} fields must use the input tensor device.")
    minimum = 1 if causal else 0
    if bool((positions < minimum).any()) or bool((positions >= input_ids.numel()).any()):
        relation = "positive" if causal else "non-negative"
        raise ValueError(
            f"{name} positions must be {relation} valid sequence positions."
        )
    if not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError(f"{name} positions must be strictly increasing.")
    if bool((labels < 0).any()):
        raise ValueError(f"{name} transcript ids must be non-negative.")
    _validate_ctc_length(positions.numel(), labels, name=name)


def _validate_batch_ctc(
    input_ids: Tensor,
    value: CTCTarget | None,
    *,
    name: str,
    causal: bool,
    allowed: list[bool],
) -> None:
    if value is None:
        return
    positions = value["token_positions"]
    labels = value["text_token_ids"]
    if positions.dim() != 2 or labels.dim() != 2:
        raise ValueError(f"{name} fields must have shapes [B, A] and [B, U].")
    if positions.size(0) != input_ids.size(0) or labels.size(0) != input_ids.size(0):
        raise ValueError(f"{name} fields must align with the input batch.")
    if not is_signed_integer_dtype(positions.dtype) or not is_signed_integer_dtype(
        labels.dtype
    ):
        raise TypeError(f"{name} fields must use signed integer dtypes.")
    if positions.device != input_ids.device or labels.device != input_ids.device:
        raise ValueError(f"{name} fields must use the input tensor device.")
    position_mask = positions.ne(CTC_PAD_ID)
    label_mask = labels.ne(CTC_PAD_ID)
    if bool((positions < CTC_PAD_ID).any()) or bool((labels < CTC_PAD_ID).any()):
        raise ValueError(f"{name} may only use -1 for padding.")
    if not _right_padded(position_mask) or not _right_padded(label_mask):
        raise ValueError(f"{name} fields must use right padding.")
    active_positions = position_mask.sum(dim=1)
    active_labels = label_mask.sum(dim=1)
    allowed_rows = torch.tensor(allowed, device=positions.device, dtype=torch.bool)
    if allowed_rows.shape != active_positions.shape:
        raise ValueError(f"{name} route flags must align with the input batch.")
    if bool((active_positions.gt(0) & ~allowed_rows).any()):
        raise ValueError(f"{name} is not allowed by the task transcript visibility route.")
    if not torch.equal(active_positions.gt(0), active_labels.gt(0)):
        raise ValueError(
            f"{name} rows must provide both audio positions and transcript tokens."
        )
    minimum = 1 if causal else 0
    active = position_mask & positions.lt(minimum)
    if bool(active.any()):
        relation = "positive" if causal else "non-negative"
        raise ValueError(f"{name} positions must be {relation}.")
    if bool((position_mask & positions.ge(input_ids.size(1))).any()):
        raise ValueError(f"{name} position exceeds the token sequence length.")
    for row, (position_count, label_count) in enumerate(
        zip(active_positions.tolist(), active_labels.tolist())
    ):
        if position_count == 0:
            continue
        row_positions = positions[row, :position_count]
        if not bool((row_positions[1:] > row_positions[:-1]).all()):
            raise ValueError(f"{name} positions must be strictly increasing.")
        row_labels = labels[row, :label_count]
        _validate_ctc_length(position_count, row_labels, name=name)


def _right_padded(mask: Tensor) -> bool:
    if mask.size(1) < 2:
        return True
    return not bool((~mask[:, :-1] & mask[:, 1:]).any())


def _validate_ctc_length(input_length: int, labels: Tensor, *, name: str) -> None:
    repeats = int(labels[1:].eq(labels[:-1]).sum().item())
    minimum = labels.numel() + repeats
    if input_length < minimum:
        raise ValueError(
            f"{name} requires at least {minimum} audio positions for "
            f"{labels.numel()} transcript tokens, got {input_length}."
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
