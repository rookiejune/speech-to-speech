"""Model-independent contracts for aligned dual-stream backbones."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch
from torch import Tensor
from transformers.cache_utils import Cache

from ..backbone.contract import (
    BackboneOutput,
    BackboneReadout,
)


@dataclass(frozen=True)
class DualStreamHiddenStates:
    """Aligned hidden states produced by a MIMO encoder.

    ``text`` and ``audio`` may use different final hidden widths when a model
    has branch-specific projections, but their batch and sequence dimensions
    must remain aligned.  ``shared`` optionally retains the common transformer
    representation for auxiliary objectives.
    """

    text: Tensor
    audio: Tensor
    shared: Tensor | None = None

    def __post_init__(self) -> None:
        _validate_hidden(self.text, name="text hidden states")
        _validate_hidden(self.audio, name="audio hidden states")
        if self.text.shape[:2] != self.audio.shape[:2]:
            raise ValueError("dual hidden states must align on [B, T].")
        if self.shared is not None:
            _validate_hidden(self.shared, name="shared hidden states")
            if self.shared.shape[:2] != self.text.shape[:2]:
                raise ValueError("shared hidden states must align on [B, T].")
        devices = {
            value.device for value in (self.text, self.audio, self.shared) if value is not None
        }
        if len(devices) > 1:
            raise ValueError("dual hidden states must share a device.")

    @property
    def batch_size(self) -> int:
        return self.text.size(0)

    @property
    def sequence_length(self) -> int:
        return self.text.size(1)

    def as_tuple(self) -> tuple[Tensor, Tensor]:
        return self.text, self.audio


@dataclass(frozen=True)
class DualStreamLogits:
    """Pair of local-vocabulary logits for a dual hidden state."""

    text: Tensor
    audio: Tensor

    def __post_init__(self) -> None:
        _validate_logits(self.text, name="text logits")
        _validate_logits(self.audio, name="audio logits")
        if self.text.shape[:2] != self.audio.shape[:2]:
            raise ValueError("dual logits must align on [B, T].")
        if self.text.device != self.audio.device:
            raise ValueError("dual logits must share a device.")

    def as_tuple(self) -> tuple[Tensor, Tensor]:
        return self.text, self.audio


@dataclass(frozen=True)
class DualStreamOutput:
    """Dual hidden states plus the raw backbone output/cache."""

    hidden_states: DualStreamHiddenStates
    output: BackboneOutput

    @property
    def text(self) -> Tensor:
        return self.hidden_states.text

    @property
    def audio(self) -> Tensor:
        return self.hidden_states.audio

    @property
    def shared(self) -> Tensor | None:
        return self.hidden_states.shared

    @property
    def past_key_values(self) -> Cache | None:
        return self.output.past_key_values


class DualStreamReadout(Protocol):
    """Read two local output heads from one aligned hidden-state object."""

    def __call__(
        self,
        hidden_states: DualStreamHiddenStates,
        *,
        attention_mask: Tensor | None = None,
    ) -> DualStreamLogits: ...


class DualStreamEncoder(Protocol):
    """Minimal protocol consumed by a MIMO training model."""

    def encode_dual(
        self,
        *,
        text_inputs_embeds: Tensor,
        audio_inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: Tensor | None = None,
        cache_position: Tensor | None = None,
        audio_features: Tensor | None = None,
        audio_feature_mask: Tensor | None = None,
        feature_projection: Callable[[Tensor], Tensor] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> DualStreamOutput: ...


class MimoBackbone(DualStreamEncoder, Protocol):
    """Alias with the terminology used by training configuration."""


@dataclass(frozen=True)
class DualStreamBodyAdapter:
    """Adapt a callable HF-style body to :class:`DualStreamEncoder`.

    The body is called exactly once with the sum of text/audio embeddings.
    Branch selection is configured through two existing ``BackboneReadout``
    values, e.g. ``last_hidden_state[0]`` and ``last_hidden_state[1]`` for a
    body that returns a tuple of branch states.
    """

    body: Callable[..., object]
    text_readout: BackboneReadout = field(default_factory=BackboneReadout)
    audio_readout: BackboneReadout = field(default_factory=BackboneReadout)
    supports_cache_position: bool = True

    def __post_init__(self) -> None:
        if not callable(self.body):
            raise TypeError("dual stream body must be callable.")
        if not isinstance(self.text_readout, BackboneReadout):
            raise TypeError("text_readout must be a BackboneReadout.")
        if not isinstance(self.audio_readout, BackboneReadout):
            raise TypeError("audio_readout must be a BackboneReadout.")
        if not isinstance(self.supports_cache_position, bool):
            raise TypeError("supports_cache_position must be a bool.")

    def encode_dual(
        self,
        *,
        text_inputs_embeds: Tensor,
        audio_inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: Tensor | None = None,
        cache_position: Tensor | None = None,
        audio_features: Tensor | None = None,
        audio_feature_mask: Tensor | None = None,
        feature_projection: Callable[[Tensor], Tensor] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> DualStreamOutput:
        fused = fuse_dual_embeddings(
            text_inputs_embeds,
            audio_inputs_embeds,
            audio_features=audio_features,
            audio_feature_mask=audio_feature_mask,
            feature_projection=feature_projection,
        )
        kwargs: dict[str, object] = {
            "inputs_embeds": fused,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden_states
            or self.text_readout.requires_hidden_states
            or self.audio_readout.requires_hidden_states,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
        }
        if self.supports_cache_position:
            kwargs["cache_position"] = cache_position
        if extra:
            kwargs.update(extra)
        raw_output = self.body(**kwargs)
        text = self.text_readout.select(cast(BackboneOutput, raw_output))
        audio = self.audio_readout.select(cast(BackboneOutput, raw_output))
        hidden = DualStreamHiddenStates(text=text, audio=audio)
        return DualStreamOutput(hidden_states=hidden, output=cast(BackboneOutput, raw_output))


def fuse_dual_embeddings(
    text_inputs_embeds: Tensor,
    audio_inputs_embeds: Tensor,
    *,
    audio_features: Tensor | None = None,
    audio_feature_mask: Tensor | None = None,
    feature_projection: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Sum aligned token embeddings and optionally masked continuous features."""

    _validate_embedding(text_inputs_embeds, name="text_inputs_embeds")
    _validate_embedding(audio_inputs_embeds, name="audio_inputs_embeds")
    if text_inputs_embeds.shape != audio_inputs_embeds.shape:
        raise ValueError("text and audio embeddings must share shape [B, T, H].")
    if text_inputs_embeds.device != audio_inputs_embeds.device:
        raise ValueError("text and audio embeddings must share a device.")
    fused = text_inputs_embeds + audio_inputs_embeds
    if audio_features is None:
        if audio_feature_mask is not None:
            raise ValueError("audio_feature_mask requires audio_features.")
        return fused
    if audio_features.dim() != 3:
        raise ValueError("audio_features must have shape [B, T, D].")
    if audio_features.shape[:2] != text_inputs_embeds.shape[:2]:
        raise ValueError("audio_features must align with [B, T].")
    if not audio_features.is_floating_point():
        raise TypeError("audio_features must use a floating-point dtype.")
    if audio_feature_mask is None:
        raise ValueError("audio_feature_mask is required for continuous features.")
    if audio_feature_mask.shape != text_inputs_embeds.shape[:2]:
        raise ValueError("audio_feature_mask must align with [B, T].")
    if audio_feature_mask.dtype != torch.bool:
        raise TypeError("audio_feature_mask must use boolean dtype.")
    if audio_feature_mask.device != text_inputs_embeds.device:
        raise ValueError("audio_feature_mask must share the embeddings device.")
    if audio_features.device != text_inputs_embeds.device:
        raise ValueError("audio_features must share the embeddings device.")
    if feature_projection is None:
        if audio_features.size(-1) != fused.size(-1):
            raise ValueError(
                "feature_projection is required when audio feature width differs "
                "from embedding width."
            )
        projected = audio_features
    else:
        if not callable(feature_projection):
            raise TypeError("feature_projection must be callable.")
        projected = feature_projection(audio_features)
        if not isinstance(projected, Tensor):
            raise TypeError("feature_projection must return a tensor.")
        if projected.shape != fused.shape:
            raise ValueError("projected audio features must have shape [B, T, H].")
    if projected.device != fused.device:
        raise ValueError("projected audio features must share the embeddings device.")
    projected = projected.to(dtype=fused.dtype)
    return fused + projected.masked_fill(~audio_feature_mask.unsqueeze(-1), 0)


def shared_dual_hidden_states(hidden_states: Tensor) -> DualStreamHiddenStates:
    """Represent a conventional shared body output as both MIMO branches."""

    _validate_hidden(hidden_states, name="hidden_states")
    return DualStreamHiddenStates(
        text=hidden_states,
        audio=hidden_states,
        shared=hidden_states,
    )


def _validate_embedding(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.dim() != 3:
        raise ValueError(f"{name} must have shape [B, T, H].")
    if value.size(-1) < 1:
        raise ValueError(f"{name} must have a non-empty hidden width.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")


def _validate_hidden(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.dim() != 3:
        raise ValueError(f"{name} must have shape [B, T, H].")
    if value.size(-1) < 1:
        raise ValueError(f"{name} must have a non-empty hidden width.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")


def _validate_logits(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.dim() != 3:
        raise ValueError(f"{name} must have shape [B, T, V].")
    if value.size(-1) < 1:
        raise ValueError(f"{name} must have a non-empty vocabulary.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")


__all__ = [
    "DualStreamBodyAdapter",
    "DualStreamEncoder",
    "DualStreamHiddenStates",
    "DualStreamLogits",
    "DualStreamOutput",
    "DualStreamReadout",
    "MimoBackbone",
    "fuse_dual_embeddings",
    "shared_dual_hidden_states",
]
