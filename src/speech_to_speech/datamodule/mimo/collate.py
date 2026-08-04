"""Collation for aligned dual-stream MIMO samples."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ...mimo import MIMO_IGNORE_INDEX, MimoBatch, MimoSample


def collate_mimo(
    samples: Sequence[MimoSample],
    *,
    text_pad_token_id: int,
    audio_pad_token_id: int,
    ignore_index: int = MIMO_IGNORE_INDEX,
) -> MimoBatch:
    """Right-pad aligned MIMO samples into a batch."""

    values = list(samples)
    if not values:
        raise ValueError("collate_mimo requires at least one sample.")
    if any(not isinstance(sample, MimoSample) for sample in values):
        raise TypeError("collate_mimo samples must be MimoSample values.")
    if any(sample.ignore_index != ignore_index for sample in values):
        raise ValueError("all MimoSample ignore_index values must match collate_mimo.")
    if len({sample.text_input_ids.device for sample in values}) != 1:
        raise ValueError("all MimoSample values must share a device.")
    for name, tensors in (
        ("text_input_ids", [sample.text_input_ids for sample in values]),
        ("audio_input_ids", [sample.audio_input_ids for sample in values]),
        ("text_labels", [sample.text_labels for sample in values]),
        ("audio_labels", [sample.audio_labels for sample in values]),
    ):
        if len({value.dtype for value in tensors}) != 1:
            raise TypeError(f"all MimoSample {name} tensors must share a dtype.")

    text_input_ids = pad_sequence(
        [sample.text_input_ids for sample in values],
        batch_first=True,
        padding_value=text_pad_token_id,
    )
    audio_input_ids = pad_sequence(
        [sample.audio_input_ids for sample in values],
        batch_first=True,
        padding_value=audio_pad_token_id,
    )
    text_labels = pad_sequence(
        [sample.text_labels for sample in values],
        batch_first=True,
        padding_value=ignore_index,
    )
    audio_labels = pad_sequence(
        [sample.audio_labels for sample in values],
        batch_first=True,
        padding_value=ignore_index,
    )
    attention_mask = pad_sequence(
        [sample.effective_attention_mask for sample in values],
        batch_first=True,
        padding_value=False,
    )
    text_loss_mask = pad_sequence(
        [sample.effective_text_loss_mask for sample in values],
        batch_first=True,
        padding_value=False,
    )
    audio_loss_mask = pad_sequence(
        [sample.effective_audio_loss_mask for sample in values],
        batch_first=True,
        padding_value=False,
    )
    audio_features, audio_feature_mask = _features(values)
    return MimoBatch(
        text_input_ids=text_input_ids,
        audio_input_ids=audio_input_ids,
        text_labels=text_labels,
        audio_labels=audio_labels,
        text_pad_token_id=text_pad_token_id,
        audio_pad_token_id=audio_pad_token_id,
        attention_mask=attention_mask,
        text_loss_mask=text_loss_mask,
        audio_loss_mask=audio_loss_mask,
        audio_features=audio_features,
        audio_feature_mask=audio_feature_mask,
        task_ids=tuple(sample.task_id for sample in values),
        recording_ids=tuple(sample.recording_id for sample in values),
        ignore_index=ignore_index,
    )


def _features(samples: list[MimoSample]) -> tuple[Tensor | None, Tensor | None]:
    values = [sample.audio_features for sample in samples]
    if not any(value is not None for value in values):
        return None, None
    present = [value for value in values if value is not None]
    if not present:
        raise AssertionError("feature presence check was inconsistent.")
    feature_width = present[0].size(1)
    if any(value.size(1) != feature_width for value in present):
        raise ValueError("all audio_features must share the feature dimension.")
    if any(value.dtype != present[0].dtype for value in present):
        raise TypeError("all audio_features must share a dtype.")
    if any(value.device != present[0].device for value in present):
        raise ValueError("all audio_features must share a device.")

    feature_rows: list[Tensor] = []
    mask_rows: list[Tensor] = []
    for sample, value in zip(samples, values):
        if value is None:
            feature_rows.append(
                torch.zeros(
                    (sample.text_input_ids.numel(), feature_width),
                    dtype=present[0].dtype,
                    device=present[0].device,
                )
            )
            mask_rows.append(
                torch.zeros(
                    sample.text_input_ids.shape,
                    dtype=torch.bool,
                    device=sample.text_input_ids.device,
                )
            )
            continue
        if sample.audio_feature_mask is None:
            raise AssertionError("validated samples must carry feature masks.")
        feature_rows.append(value)
        mask_rows.append(sample.audio_feature_mask)
    return (
        pad_sequence(feature_rows, batch_first=True, padding_value=0.0),
        pad_sequence(mask_rows, batch_first=True, padding_value=False),
    )


__all__ = ["collate_mimo"]
