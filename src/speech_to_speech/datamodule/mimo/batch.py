"""Contracts for aligned dual-stream (MIMO) autoregressive batches.

The existing :class:`ModelBatch` represents one serialized token stream.  Kimi
style training keeps text and semantic-audio tokens aligned at every time
step, so it has a deliberately separate batch contract in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..._tensor import is_signed_integer_dtype


MIMO_IGNORE_INDEX = -100


@dataclass(frozen=True)
class MimoSample:
    """One unpadded aligned text/audio training example.

    All four token vectors describe the same time axis.  ``audio_feature_mask``
    is intentionally explicit whenever continuous audio features are supplied:
    target/generated audio positions must be masked out by the caller.
    """

    text_input_ids: Tensor
    audio_input_ids: Tensor
    text_labels: Tensor
    audio_labels: Tensor
    attention_mask: Tensor | None = None
    text_loss_mask: Tensor | None = None
    audio_loss_mask: Tensor | None = None
    audio_features: Tensor | None = None
    audio_feature_mask: Tensor | None = None
    task_id: str = "mimo"
    recording_id: str | None = None
    ignore_index: int = MIMO_IGNORE_INDEX

    def __post_init__(self) -> None:
        token_values = {
            "text_input_ids": self.text_input_ids,
            "audio_input_ids": self.audio_input_ids,
            "text_labels": self.text_labels,
            "audio_labels": self.audio_labels,
        }
        for name, value in token_values.items():
            _validate_token_vector(value, name=name)
        length = self.text_input_ids.numel()
        if length == 0:
            raise ValueError("MimoSample token streams must not be empty.")
        if any(value.numel() != length for value in token_values.values()):
            raise ValueError("MimoSample token streams and labels must share length.")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("ignore_index must be an integer.")

        if self.attention_mask is not None:
            _validate_mask(self.attention_mask, (length,), name="attention_mask")
            if not bool(self.attention_mask.any()):
                raise ValueError("MimoSample attention_mask must select a position.")
        if self.text_loss_mask is not None:
            _validate_mask(self.text_loss_mask, (length,), name="text_loss_mask")
            _validate_label_mask(
                self.text_labels,
                self.text_loss_mask,
                "text",
                ignore_index=self.ignore_index,
            )
        if self.audio_loss_mask is not None:
            _validate_mask(self.audio_loss_mask, (length,), name="audio_loss_mask")
            _validate_label_mask(
                self.audio_labels,
                self.audio_loss_mask,
                "audio",
                ignore_index=self.ignore_index,
            )

        if self.audio_features is None:
            if self.audio_feature_mask is not None:
                raise ValueError("audio_feature_mask requires audio_features to be provided.")
        else:
            audio_features = self.audio_features
            if audio_features.dim() != 2:
                raise ValueError("audio_features must have shape [T, D].")
            if audio_features.size(0) != length or audio_features.size(1) < 1:
                raise ValueError("audio_features must align with tokens and have D > 0.")
            if not audio_features.is_floating_point():
                raise TypeError("audio_features must use a floating-point dtype.")
            if self.audio_feature_mask is None:
                raise ValueError("audio_feature_mask is required when audio_features are provided.")
            _validate_mask(
                self.audio_feature_mask,
                (length,),
                name="audio_feature_mask",
            )
            if bool((self.audio_feature_mask & self.effective_audio_loss_mask).any()):
                raise ValueError(
                    "audio_feature_mask cannot select supervised audio target positions."
                )

        _validate_string(self.task_id, name="task_id", allow_none=False)
        _validate_string(self.recording_id, name="recording_id", allow_none=True)
        _validate_same_device(tuple(value for value in self._tensors() if value is not None))

    def _tensors(self) -> tuple[Tensor | None, ...]:
        return (
            self.text_input_ids,
            self.audio_input_ids,
            self.text_labels,
            self.audio_labels,
            self.attention_mask,
            self.text_loss_mask,
            self.audio_loss_mask,
            self.audio_features,
            self.audio_feature_mask,
        )

    @property
    def effective_attention_mask(self) -> Tensor:
        if self.attention_mask is None:
            return torch.ones(
                self.text_input_ids.shape,
                dtype=torch.bool,
                device=self.text_input_ids.device,
            )
        return self.attention_mask

    @property
    def effective_text_loss_mask(self) -> Tensor:
        if self.text_loss_mask is None:
            return self.text_labels.ne(self.ignore_index)
        return self.text_loss_mask

    @property
    def effective_audio_loss_mask(self) -> Tensor:
        if self.audio_loss_mask is None:
            return self.audio_labels.ne(self.ignore_index)
        return self.audio_loss_mask


@dataclass
class MimoBatch:
    """A padded batch of aligned text/audio token streams.

    ``text_loss_mask`` and ``audio_loss_mask`` are independent.  This permits
    text-only/audio-only examples and Kimi contextual tasks in one data path.
    The masks are normalized during construction and always become boolean
    tensors with shape ``[B, T]``.
    """

    text_input_ids: Tensor
    audio_input_ids: Tensor
    text_labels: Tensor
    audio_labels: Tensor
    text_pad_token_id: int = 0
    audio_pad_token_id: int = 0
    attention_mask: Tensor | None = None
    text_loss_mask: Tensor | None = None
    audio_loss_mask: Tensor | None = None
    audio_features: Tensor | None = None
    audio_feature_mask: Tensor | None = None
    task_ids: tuple[str, ...] | None = None
    recording_ids: tuple[str | None, ...] | None = None
    ignore_index: int = MIMO_IGNORE_INDEX

    def __post_init__(self) -> None:
        token_values = {
            "text_input_ids": self.text_input_ids,
            "audio_input_ids": self.audio_input_ids,
            "text_labels": self.text_labels,
            "audio_labels": self.audio_labels,
        }
        for name, value in token_values.items():
            _validate_token_matrix(value, name=name)
        if self.text_input_ids.shape != self.audio_input_ids.shape:
            raise ValueError("text_input_ids and audio_input_ids must be aligned.")
        if self.text_labels.shape != self.text_input_ids.shape:
            raise ValueError("text_labels must align with text_input_ids.")
        if self.audio_labels.shape != self.text_input_ids.shape:
            raise ValueError("audio_labels must align with audio_input_ids.")
        batch_size, sequence_length = self.text_input_ids.shape
        if batch_size == 0 or sequence_length == 0:
            raise ValueError("MimoBatch must contain at least one token row.")

        _validate_id(self.text_pad_token_id, name="text_pad_token_id")
        _validate_id(self.audio_pad_token_id, name="audio_pad_token_id")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("ignore_index must be an integer.")

        if self.attention_mask is None:
            self.attention_mask = self.text_input_ids.ne(
                self.text_pad_token_id
            ) | self.audio_input_ids.ne(self.audio_pad_token_id)
        _validate_mask(
            self.attention_mask,
            (batch_size, sequence_length),
            name="attention_mask",
        )
        if not bool(self.attention_mask.any(dim=1).all()):
            raise ValueError("each MimoBatch row must contain an attended position.")

        if self.text_loss_mask is None:
            self.text_loss_mask = self.text_labels.ne(self.ignore_index)
        if self.audio_loss_mask is None:
            self.audio_loss_mask = self.audio_labels.ne(self.ignore_index)
        _validate_mask(
            self.text_loss_mask,
            (batch_size, sequence_length),
            name="text_loss_mask",
        )
        _validate_mask(
            self.audio_loss_mask,
            (batch_size, sequence_length),
            name="audio_loss_mask",
        )
        _validate_label_mask(
            self.text_labels,
            self.text_loss_mask,
            "text",
            ignore_index=self.ignore_index,
        )
        _validate_label_mask(
            self.audio_labels,
            self.audio_loss_mask,
            "audio",
            ignore_index=self.ignore_index,
        )
        for name, mask in (
            ("text_loss_mask", self.text_loss_mask),
            ("audio_loss_mask", self.audio_loss_mask),
        ):
            if bool((mask & ~self.attention_mask).any()):
                raise ValueError(f"{name} must be false on padded positions.")
        causal_target_mask = (
            (self.text_loss_mask[:, 1:] | self.audio_loss_mask[:, 1:])
            & self.attention_mask[:, 1:]
            & self.attention_mask[:, :-1]
        )
        if not bool(causal_target_mask.any(dim=1).all()):
            raise ValueError("each MimoBatch row must contain a supervised target.")

        if self.audio_features is None:
            if self.audio_feature_mask is not None:
                raise ValueError("audio_feature_mask requires audio_features to be provided.")
        else:
            if self.audio_features.dim() != 3:
                raise ValueError("audio_features must have shape [B, T, D].")
            if self.audio_features.shape[:2] != (batch_size, sequence_length):
                raise ValueError("audio_features must align with token streams.")
            if self.audio_features.size(-1) < 1:
                raise ValueError("audio_features must have a non-empty feature dimension.")
            if not self.audio_features.is_floating_point():
                raise TypeError("audio_features must use a floating-point dtype.")
            if self.audio_feature_mask is None:
                self.audio_feature_mask = torch.zeros(
                    (batch_size, sequence_length),
                    dtype=torch.bool,
                    device=self.audio_features.device,
                )
            _validate_mask(
                self.audio_feature_mask,
                (batch_size, sequence_length),
                name="audio_feature_mask",
            )
            if bool((self.audio_feature_mask & ~self.attention_mask).any()):
                raise ValueError("audio_feature_mask must be false on padded positions.")
            if self.audio_loss_mask is None:
                raise RuntimeError("MimoBatch audio_loss_mask was not normalized.")
            if bool((self.audio_feature_mask & self.audio_loss_mask).any()):
                raise ValueError(
                    "audio_feature_mask cannot select supervised audio target positions."
                )

        _validate_metadata(
            self.task_ids,
            batch_size,
            name="task_ids",
            allow_none=True,
            allow_none_values=False,
        )
        _validate_metadata(
            self.recording_ids,
            batch_size,
            name="recording_ids",
            allow_none=True,
            allow_none_values=True,
        )
        _validate_same_device(tuple(value for value in self._tensors() if value is not None))

    def _tensors(self) -> tuple[Tensor | None, ...]:
        return (
            self.text_input_ids,
            self.audio_input_ids,
            self.text_labels,
            self.audio_labels,
            self.attention_mask,
            self.text_loss_mask,
            self.audio_loss_mask,
            self.audio_features,
            self.audio_feature_mask,
        )

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[MimoSample],
        *,
        text_pad_token_id: int,
        audio_pad_token_id: int,
        ignore_index: int = MIMO_IGNORE_INDEX,
    ) -> MimoBatch:
        return collate_mimo(
            samples,
            text_pad_token_id=text_pad_token_id,
            audio_pad_token_id=audio_pad_token_id,
            ignore_index=ignore_index,
        )

    @property
    def batch_size(self) -> int:
        return self.text_input_ids.size(0)

    @property
    def sequence_length(self) -> int:
        return self.text_input_ids.size(1)

    @property
    def text_target_mask(self) -> Tensor:
        """Mask aligned with causal predictions (the first position is dropped)."""
        attention_mask = self.attention_mask
        text_loss_mask = self.text_loss_mask
        if attention_mask is None or text_loss_mask is None:
            raise RuntimeError("MimoBatch masks were not normalized.")
        return text_loss_mask[:, 1:] & attention_mask[:, :-1]

    @property
    def audio_target_mask(self) -> Tensor:
        attention_mask = self.attention_mask
        audio_loss_mask = self.audio_loss_mask
        if attention_mask is None or audio_loss_mask is None:
            raise RuntimeError("MimoBatch masks were not normalized.")
        return audio_loss_mask[:, 1:] & attention_mask[:, :-1]

    @property
    def supervised_token_counts(self) -> tuple[Tensor, Tensor]:
        return self.text_target_mask.sum(dim=1), self.audio_target_mask.sum(dim=1)

    @property
    def supervised_token_count(self) -> int:
        """Total text plus audio targets on the shifted causal axis."""
        text, audio = self.supervised_token_counts
        return int(text.sum().item() + audio.sum().item())

    def row(self, index: int) -> MimoBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("MimoBatch row index must be an integer.")
        if index < 0 or index >= self.batch_size:
            raise IndexError(f"MimoBatch row index is out of range: {index}.")
        attention_mask = self.attention_mask
        text_loss_mask = self.text_loss_mask
        audio_loss_mask = self.audio_loss_mask
        if attention_mask is None or text_loss_mask is None or audio_loss_mask is None:
            raise RuntimeError("MimoBatch masks were not normalized.")
        task_ids = None if self.task_ids is None else (self.task_ids[index],)
        recording_ids = None if self.recording_ids is None else (self.recording_ids[index],)
        return MimoBatch(
            text_input_ids=self.text_input_ids[index : index + 1],
            audio_input_ids=self.audio_input_ids[index : index + 1],
            text_labels=self.text_labels[index : index + 1],
            audio_labels=self.audio_labels[index : index + 1],
            text_pad_token_id=self.text_pad_token_id,
            audio_pad_token_id=self.audio_pad_token_id,
            attention_mask=attention_mask[index : index + 1],
            text_loss_mask=text_loss_mask[index : index + 1],
            audio_loss_mask=audio_loss_mask[index : index + 1],
            audio_features=(
                None if self.audio_features is None else self.audio_features[index : index + 1]
            ),
            audio_feature_mask=(
                None
                if self.audio_feature_mask is None
                else self.audio_feature_mask[index : index + 1]
            ),
            task_ids=task_ids,
            recording_ids=recording_ids,
            ignore_index=self.ignore_index,
        )

    def pin_memory(self) -> MimoBatch:
        return self._transfer(pin=True)

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> MimoBatch:
        return self._transfer(device=device, non_blocking=non_blocking)

    def _transfer(
        self,
        *,
        device: torch.device | None = None,
        non_blocking: bool = False,
        pin: bool = False,
    ) -> MimoBatch:
        def move(value: Tensor) -> Tensor:
            if pin:
                return value.pin_memory()
            return value.to(device=device, non_blocking=non_blocking)

        def move_optional(value: Tensor | None) -> Tensor | None:
            return None if value is None else move(value)

        attention_mask = self.attention_mask
        text_loss_mask = self.text_loss_mask
        audio_loss_mask = self.audio_loss_mask
        if (
            attention_mask is None
            or text_loss_mask is None
            or audio_loss_mask is None
        ):
            raise RuntimeError("MimoBatch masks were not normalized.")
        result = MimoBatch.__new__(MimoBatch)
        result.text_input_ids = move(self.text_input_ids)
        result.audio_input_ids = move(self.audio_input_ids)
        result.text_labels = move(self.text_labels)
        result.audio_labels = move(self.audio_labels)
        result.text_pad_token_id = self.text_pad_token_id
        result.audio_pad_token_id = self.audio_pad_token_id
        result.attention_mask = move(attention_mask)
        result.text_loss_mask = move(text_loss_mask)
        result.audio_loss_mask = move(audio_loss_mask)
        result.audio_features = move_optional(self.audio_features)
        result.audio_feature_mask = move_optional(self.audio_feature_mask)
        result.task_ids = self.task_ids
        result.recording_ids = self.recording_ids
        result.ignore_index = self.ignore_index
        return result


def collate_mimo(
    samples: Sequence[MimoSample],
    *,
    text_pad_token_id: int,
    audio_pad_token_id: int,
    ignore_index: int = MIMO_IGNORE_INDEX,
) -> MimoBatch:
    """Right-pad aligned MIMO samples into a :class:`MimoBatch`.

    Feature tensors are optional per sample, but when present every sample must
    use the same feature width.  Missing rows receive zero features and a false
    feature mask, making accidental target-feature leakage explicit.
    """

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

    feature_values = [sample.audio_features for sample in values]
    audio_features: Tensor | None = None
    audio_feature_mask: Tensor | None = None
    if any(value is not None for value in feature_values):
        present = [value for value in feature_values if value is not None]
        if not present:
            raise AssertionError("feature presence check was inconsistent.")
        feature_width = present[0].size(1)
        if any(value.size(1) != feature_width for value in present):
            raise ValueError("all audio_features must share the feature dimension.")
        if any(value.dtype != present[0].dtype for value in present):
            raise TypeError("all audio_features must share a dtype.")
        if any(value.device != present[0].device for value in present):
            raise ValueError("all audio_features must share a device.")
        padded_features: list[Tensor] = []
        padded_masks: list[Tensor] = []
        for sample, feature in zip(values, feature_values):
            if feature is None:
                padded_features.append(
                    torch.zeros(
                        (sample.text_input_ids.numel(), feature_width),
                        dtype=present[0].dtype,
                        device=present[0].device,
                    )
                )
                padded_masks.append(
                    torch.zeros(
                        sample.text_input_ids.shape,
                        dtype=torch.bool,
                        device=sample.text_input_ids.device,
                    )
                )
            else:
                if sample.audio_feature_mask is None:
                    raise AssertionError("validated samples must carry feature masks.")
                padded_features.append(feature)
                padded_masks.append(sample.audio_feature_mask)
        audio_features = pad_sequence(
            padded_features,
            batch_first=True,
            padding_value=0.0,
        )
        audio_feature_mask = pad_sequence(
            padded_masks,
            batch_first=True,
            padding_value=False,
        )

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


def _validate_token_vector(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if value.dim() != 1:
        raise ValueError(f"{name} must have shape [T].")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must use a signed integer dtype.")


def _validate_token_matrix(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if value.dim() != 2:
        raise ValueError(f"{name} must have shape [B, T].")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must use a signed integer dtype.")


def _validate_mask(value: Tensor, shape: tuple[int, ...], *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use boolean dtype.")


def _validate_label_mask(
    labels: Tensor,
    mask: Tensor,
    modality: str,
    *,
    ignore_index: int,
) -> None:
    if bool((mask & labels.eq(ignore_index)).any()):
        raise ValueError(f"{modality}_loss_mask cannot select ignore-index labels.")


def _validate_id(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _validate_string(value: str | None, *, name: str, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")


def _validate_metadata(
    values: tuple[str, ...] | tuple[str | None, ...] | None,
    batch_size: int,
    *,
    name: str,
    allow_none: bool,
    allow_none_values: bool,
) -> None:
    if values is None and allow_none:
        return
    if values is None or len(values) != batch_size:
        raise ValueError(f"{name} must contain one value per batch row.")
    for value in values:
        if value is None and allow_none_values:
            continue
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} values must be non-empty strings.")


def _validate_same_device(values: tuple[Tensor, ...]) -> None:
    devices = {value.device for value in values}
    if len(devices) > 1:
        raise ValueError("all MIMO batch tensors must share a device.")


__all__ = [
    "MIMO_IGNORE_INDEX",
    "MimoBatch",
    "MimoSample",
    "collate_mimo",
]
