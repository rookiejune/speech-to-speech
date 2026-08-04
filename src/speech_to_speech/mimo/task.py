"""Kimi-style aligned task composition for the dual-stream data contract.

The regular datamodule builds a serialized :class:`ModelBatch`.  This module
keeps the MIMO task grammar separate and turns prepared segment records into a
single :class:`MimoSample`.  Token ids are local to their respective text and
audio heads; no global id-space offset is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .._compat import StrEnum, auto
from .._tensor import is_signed_integer_dtype
from .contract import MIMO_IGNORE_INDEX, MimoSample


class MimoTask(StrEnum):
    TEXT_ONLY = auto()
    AUDIO_ONLY = auto()
    AUDIO_TO_TEXT = auto()
    TEXT_TO_AUDIO = auto()
    AUDIO_TO_NEXT_SEMANTIC = auto()
    AUDIO_TO_NEXT_TEXT = auto()
    AUDIO_TO_NEXT_SEMANTIC_AND_TEXT = auto()


KIMI_PRETRAIN_TASK_WEIGHTS: dict[MimoTask, float] = {
    MimoTask.AUDIO_ONLY: 1.0,
    MimoTask.TEXT_ONLY: 7.0,
    MimoTask.AUDIO_TO_TEXT: 1.0,
    MimoTask.TEXT_TO_AUDIO: 1.0,
    MimoTask.AUDIO_TO_NEXT_SEMANTIC: 1.0,
    MimoTask.AUDIO_TO_NEXT_TEXT: 1.0,
    MimoTask.AUDIO_TO_NEXT_SEMANTIC_AND_TEXT: 2.0,
}


@dataclass(frozen=True)
class MimoSpecialTokens:
    """Structural ids used by the aligned task composer."""

    text_bos: int
    text_eos: int
    text_blank: int
    audio_bos: int
    audio_eos: int
    audio_blank: int
    audio_delay_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "text_bos",
            "text_eos",
            "text_blank",
            "audio_bos",
            "audio_eos",
            "audio_blank",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            isinstance(self.audio_delay_tokens, bool)
            or not isinstance(self.audio_delay_tokens, int)
            or self.audio_delay_tokens < 0
        ):
            raise ValueError("audio_delay_tokens must be a non-negative integer.")


@dataclass(frozen=True)
class MimoSegment:
    """One prepared recording segment on the local text/audio vocabularies."""

    text_input_ids: Tensor
    audio_input_ids: Tensor
    audio_features: Tensor | None = None
    recording_id: str | None = None
    segment_index: int | None = None

    def __post_init__(self) -> None:
        _token_vector(self.text_input_ids, "text_input_ids")
        _token_vector(self.audio_input_ids, "audio_input_ids")
        if self.text_input_ids.device != self.audio_input_ids.device:
            raise ValueError("MimoSegment token streams must share a device.")
        if self.audio_features is not None:
            if self.audio_features.dim() != 2:
                raise ValueError("audio_features must have shape [audio_tokens, D].")
            if self.audio_features.shape[0] != self.audio_input_ids.numel():
                raise ValueError("audio_features must align with audio_input_ids.")
            if self.audio_features.size(1) < 1 or not self.audio_features.is_floating_point():
                raise ValueError("audio_features must be floating-point with D > 0.")
            if self.audio_features.device != self.text_input_ids.device:
                raise ValueError("audio_features must share the token device.")
        if self.recording_id is not None and not self.recording_id:
            raise ValueError("recording_id must not be empty.")
        if self.segment_index is not None:
            if isinstance(self.segment_index, bool) or not isinstance(self.segment_index, int):
                raise TypeError("segment_index must be an integer or None.")
            if self.segment_index < 0:
                raise ValueError("segment_index must be non-negative.")


def build_mimo_sample(
    task: MimoTask,
    segments: Iterable[MimoSegment],
    special: MimoSpecialTokens,
    *,
    ignore_index: int = MIMO_IGNORE_INDEX,
) -> MimoSample:
    """Compose one of Kimi's seven causal dual-stream pretraining tasks.

    For contextual tasks all segments except the last are observed audio
    context and the last segment is the supervised target.  Source spans are
    never supervised, and continuous features are enabled only on observed
    audio tokens.  The returned masks use the unshifted label convention;
    :class:`MimoObjective` performs the one-token causal shift.
    """

    if not isinstance(task, MimoTask):
        raise TypeError("task must be a MimoTask.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer.")
    values = tuple(segments)
    if not values:
        raise ValueError("MIMO task composition requires at least one segment.")
    if any(not isinstance(value, MimoSegment) for value in values):
        raise TypeError("segments must contain MimoSegment values.")
    contextual = task in {
        MimoTask.AUDIO_TO_NEXT_SEMANTIC,
        MimoTask.AUDIO_TO_NEXT_TEXT,
        MimoTask.AUDIO_TO_NEXT_SEMANTIC_AND_TEXT,
    }
    if contextual and len(values) != 2:
        raise ValueError(f"{task.value} requires exactly two ordered segments.")
    _validate_order(values, require_adjacent=contextual)

    if task in {MimoTask.TEXT_ONLY, MimoTask.AUDIO_ONLY}:
        blocks = (
            _target_text(values[-1], special, ignore_index)
            if task is MimoTask.TEXT_ONLY
            else _target_audio(values[-1], special, ignore_index)
        ,)
    elif task is MimoTask.AUDIO_TO_TEXT:
        source, target = _source_target(values, task)
        blocks = (_source_audio(source, special), _target_text(target, special, ignore_index))
    elif task is MimoTask.TEXT_TO_AUDIO:
        source, target = _source_target(values, task)
        blocks = (_source_text(source, special), _target_audio(target, special, ignore_index))
    elif contextual:
        context = tuple(_source_audio(value, special) for value in values[:-1])
        target = values[-1]
        if task is MimoTask.AUDIO_TO_NEXT_SEMANTIC:
            target_block = _target_audio(target, special, ignore_index)
        elif task is MimoTask.AUDIO_TO_NEXT_TEXT:
            target_block = _target_text(target, special, ignore_index)
        else:
            target_block = _target_parallel(target, special, ignore_index)
        blocks = context + (target_block,)
    else:
        raise AssertionError(f"unsupported MIMO task: {task}")

    recording_id = values[0].recording_id
    return _pack_blocks(blocks, task, ignore_index, recording_id=recording_id)


@dataclass(frozen=True)
class _Block:
    text: Tensor
    audio: Tensor
    text_labels: Tensor
    audio_labels: Tensor
    text_mask: Tensor
    audio_mask: Tensor
    features: Tensor | None
    feature_mask: Tensor | None


def _source_audio(segment: MimoSegment, special: MimoSpecialTokens) -> _Block:
    audio = _with_boundaries(segment.audio_input_ids, special.audio_bos, special.audio_eos)
    text = audio.new_full((audio.numel(),), special.text_blank)
    features, feature_mask = _audio_features(segment, audio.numel())
    return _Block(
        text=text,
        audio=audio,
        text_labels=text.new_full(text.shape, MIMO_IGNORE_INDEX),
        audio_labels=audio.new_full(audio.shape, MIMO_IGNORE_INDEX),
        text_mask=torch.zeros_like(text, dtype=torch.bool),
        audio_mask=torch.zeros_like(audio, dtype=torch.bool),
        features=features,
        feature_mask=feature_mask,
    )


def _source_text(segment: MimoSegment, special: MimoSpecialTokens) -> _Block:
    text = _with_boundaries(segment.text_input_ids, special.text_bos, special.text_eos)
    audio = text.new_full((text.numel(),), special.audio_blank)
    return _Block(
        text=text,
        audio=audio,
        text_labels=text.new_full(text.shape, MIMO_IGNORE_INDEX),
        audio_labels=audio.new_full(audio.shape, MIMO_IGNORE_INDEX),
        text_mask=torch.zeros_like(text, dtype=torch.bool),
        audio_mask=torch.zeros_like(audio, dtype=torch.bool),
        features=None,
        feature_mask=None,
    )


def _target_text(segment: MimoSegment, special: MimoSpecialTokens, ignore_index: int) -> _Block:
    text = _with_boundaries(segment.text_input_ids, special.text_bos, special.text_eos)
    audio = text.new_full((text.numel(),), special.audio_blank)
    labels = text.new_full(text.shape, ignore_index)
    labels[1:] = text[1:]
    return _Block(
        text=text,
        audio=audio,
        text_labels=labels,
        audio_labels=audio.new_full(audio.shape, ignore_index),
        text_mask=torch.arange(text.numel(), device=text.device).ge(1),
        audio_mask=torch.zeros_like(audio, dtype=torch.bool),
        features=None,
        feature_mask=None,
    )


def _target_audio(segment: MimoSegment, special: MimoSpecialTokens, ignore_index: int) -> _Block:
    delay = special.audio_delay_tokens
    payload = _with_boundaries(segment.audio_input_ids, special.audio_bos, special.audio_eos)
    audio = payload.new_full((delay + payload.numel(),), special.audio_blank)
    audio[delay:] = payload
    text = audio.new_full(audio.shape, special.text_blank)
    labels = audio.new_full(audio.shape, ignore_index)
    labels[delay + 1 :] = audio[delay + 1 :]
    return _Block(
        text=text,
        audio=audio,
        text_labels=text.new_full(text.shape, ignore_index),
        audio_labels=labels,
        text_mask=torch.zeros_like(text, dtype=torch.bool),
        audio_mask=torch.arange(audio.numel(), device=audio.device).ge(delay + 1),
        # Target audio must not expose continuous features from the answer.
        features=None,
        feature_mask=None,
    )


def _target_parallel(segment: MimoSegment, special: MimoSpecialTokens, ignore_index: int) -> _Block:
    text = _with_boundaries(segment.text_input_ids, special.text_bos, special.text_eos)
    payload = _with_boundaries(segment.audio_input_ids, special.audio_bos, special.audio_eos)
    delay = special.audio_delay_tokens
    length = max(text.numel(), delay + payload.numel())
    text_ids = text.new_full((length,), special.text_blank)
    audio_ids = payload.new_full((length,), special.audio_blank)
    text_ids[: text.numel()] = text
    audio_ids[delay : delay + payload.numel()] = payload
    text_labels = text_ids.new_full(text_ids.shape, ignore_index)
    audio_labels = audio_ids.new_full(audio_ids.shape, ignore_index)
    text_labels[1 : text.numel()] = text[1:]
    audio_labels[delay + 1 : delay + payload.numel()] = payload[1:]
    return _Block(
        text=text_ids,
        audio=audio_ids,
        text_labels=text_labels,
        audio_labels=audio_labels,
        text_mask=torch.arange(length, device=text.device).lt(text.numel()) & torch.arange(length, device=text.device).ge(1),
        audio_mask=torch.arange(length, device=audio_ids.device).ge(delay + 1)
        & torch.arange(length, device=audio_ids.device).lt(delay + payload.numel()),
        # Continuous features are injected only for observed source audio.
        features=None,
        feature_mask=None,
    )


def _pack_blocks(
    blocks: tuple[_Block, ...],
    task: MimoTask,
    ignore_index: int,
    *,
    recording_id: str | None,
) -> MimoSample:
    feature_widths = {
        block.features.size(-1)
        for block in blocks
        if block.features is not None
    }
    if len(feature_widths) > 1:
        raise ValueError("all MIMO source audio features must share a width.")
    feature_width = next(iter(feature_widths), None)
    if feature_width is None:
        features = None
        feature_mask = None
    else:
        rows: list[Tensor] = []
        masks: list[Tensor] = []
        present_dtype = next(
            block.features.dtype for block in blocks if block.features is not None
        )
        for block in blocks:
            if block.features is None:
                rows.append(
                    torch.zeros(
                        (block.text.numel(), feature_width),
                        dtype=present_dtype,
                        device=block.text.device,
                    )
                )
                masks.append(torch.zeros(block.text.shape, dtype=torch.bool, device=block.text.device))
            else:
                rows.append(block.features)
                masks.append(block.feature_mask if block.feature_mask is not None else torch.zeros(block.text.shape, dtype=torch.bool, device=block.text.device))
        features = torch.cat(rows)
        feature_mask = torch.cat(masks)
    first = blocks[0]
    text_labels = torch.cat([block.text_labels for block in blocks])
    audio_labels = torch.cat([block.audio_labels for block in blocks])
    length = text_labels.numel()
    attention_mask = torch.ones(
        (length,),
        dtype=torch.bool,
        device=first.text.device,
    )
    text_labels = text_labels.masked_fill(text_labels.eq(MIMO_IGNORE_INDEX), ignore_index)
    audio_labels = audio_labels.masked_fill(audio_labels.eq(MIMO_IGNORE_INDEX), ignore_index)
    return MimoSample(
        text_input_ids=torch.cat([block.text for block in blocks]),
        audio_input_ids=torch.cat([block.audio for block in blocks]),
        text_labels=text_labels,
        audio_labels=audio_labels,
        text_loss_mask=torch.cat([block.text_mask for block in blocks]),
        audio_loss_mask=torch.cat([block.audio_mask for block in blocks]),
        attention_mask=attention_mask,
        audio_features=features,
        audio_feature_mask=feature_mask,
        task_id=task.value,
        recording_id=recording_id,
        ignore_index=ignore_index,
    )


def _audio_features(
    segment: MimoSegment,
    length: int,
    *,
    offset: int = 0,
) -> tuple[Tensor | None, Tensor | None]:
    if segment.audio_features is None:
        return None, None
    features = torch.zeros(
        (length, segment.audio_features.size(-1)),
        dtype=segment.audio_features.dtype,
        device=segment.audio_features.device,
    )
    mask = torch.zeros((length,), dtype=torch.bool, device=features.device)
    start = offset + 1
    end = start + segment.audio_features.size(0)
    if end > length:
        raise ValueError("audio feature rows do not fit the target audio span.")
    features[start:end] = segment.audio_features
    mask[start:end] = True
    return features, mask


def _with_boundaries(values: Tensor, bos: int, eos: int) -> Tensor:
    return torch.cat((values.new_tensor([bos]), values, values.new_tensor([eos])))


def _source_target(values: tuple[MimoSegment, ...], task: MimoTask) -> tuple[MimoSegment, MimoSegment]:
    resolved = list(values)
    if not resolved:
        raise ValueError(f"{task.value} requires at least one segment.")
    if len(resolved) > 2:
        raise ValueError(f"{task.value} accepts at most two segments.")
    return (resolved[0], resolved[-1])


def _validate_order(
    values: tuple[MimoSegment, ...],
    *,
    require_adjacent: bool = False,
) -> None:
    ids = [value.recording_id for value in values]
    if require_adjacent and any(value is None for value in ids):
        raise ValueError(
            "contextual MIMO tasks require recording_id for every segment."
        )
    present_ids = [value for value in ids if value is not None]
    if present_ids and len(set(present_ids)) != 1:
        raise ValueError("all segments in one MIMO sample must share recording_id.")
    indexes = [value.segment_index for value in values]
    if require_adjacent and any(value is None for value in indexes):
        raise ValueError(
            "contextual MIMO tasks require segment_index for every segment."
        )
    present_indexes = [value for value in indexes if value is not None]
    if present_indexes != sorted(present_indexes) or len(present_indexes) != len(
        set(present_indexes)
    ):
        raise ValueError("MIMO segment_index values must be ordered and unique.")
    if require_adjacent and any(
        right != left + 1
        for left, right in zip(present_indexes, present_indexes[1:])
    ):
        raise ValueError(
            "contextual MIMO tasks require consecutive segment_index values."
        )


def _token_vector(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.dim() != 1:
        raise ValueError(f"{name} must have shape [tokens].")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must use a signed integer dtype.")


__all__ = [
    "KIMI_PRETRAIN_TASK_WEIGHTS",
    "MimoSegment",
    "MimoSpecialTokens",
    "MimoTask",
    "build_mimo_sample",
]
