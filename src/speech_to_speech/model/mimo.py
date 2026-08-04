"""Trainable model wrapper for aligned text/audio autoregression."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.cache_utils import Cache

from ..datamodule.mimo import MimoBatch
from ..generation.mimo import MimoGenerationStep
from ..runtime.backbone.mimo import (
    DualStreamBodyAdapter,
    DualStreamHiddenStates,
    DualStreamLogits,
    DualStreamOutput,
)
from ..runtime.types import BackboneReadout


@dataclass(frozen=True)
class MimoModelConfig:
    """Numerical options that must be persisted with a MIMO checkpoint."""

    audio_feature_scale: float = math.sqrt(2.0)
    supports_cache_position: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.audio_feature_scale, bool)
            or not isinstance(self.audio_feature_scale, (int, float))
            or not math.isfinite(float(self.audio_feature_scale))
            or self.audio_feature_scale <= 0
        ):
            raise ValueError("audio_feature_scale must be finite and positive.")
        if not isinstance(self.supports_cache_position, bool):
            raise TypeError("supports_cache_position must be a boolean.")


class TiedEmbeddingHead(nn.Module):
    """Tie logits to a prefix of an embedding without a second owner.

    Runtime backbones often expose one global input table while MIMO heads use
    local text/audio vocabularies.  A weak reference keeps the table registered
    only at its canonical embedding path (usually ``backbone.embed_tokens``).
    """

    def __init__(self, embedding: nn.Embedding, vocab_size: int) -> None:
        super().__init__()
        if not isinstance(embedding, nn.Embedding):
            raise TypeError("TiedEmbeddingHead embedding must be nn.Embedding.")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("TiedEmbeddingHead vocab_size must be positive.")
        if vocab_size > embedding.num_embeddings:
            raise ValueError("TiedEmbeddingHead vocab_size exceeds the embedding table.")
        self._embedding_ref = weakref.ref(embedding)
        self.vocab_size = vocab_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        embedding = self._embedding_ref()
        if embedding is None:
            raise RuntimeError("the tied embedding no longer exists.")
        if hidden_states.size(-1) != embedding.embedding_dim:
            raise ValueError("hidden states do not match the tied embedding width.")
        return F.linear(
            hidden_states.to(dtype=embedding.weight.dtype),
            embedding.weight[: self.vocab_size],
        )


class MimoModel(nn.Module):
    """Own a Kimi-style body, two input embeddings, and two output routes.

    The body receives ``text_embedding + audio_embedding + audio_features``
    once.  ``text_readout`` and ``audio_readout`` select branch outputs from
    that same call.  When an output head is omitted, its embedding weight is
    tied directly to the corresponding local-vocabulary logits.
    """

    def __init__(
        self,
        body: nn.Module,
        *,
        text_embedding: nn.Embedding,
        audio_embedding: nn.Embedding,
        text_readout: BackboneReadout,
        audio_readout: BackboneReadout,
        text_head: nn.Module | None = None,
        audio_head: nn.Module | None = None,
        audio_feature_projection: nn.Module | None = None,
        config: MimoModelConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(body, nn.Module):
            raise TypeError("MIMO body must be a torch.nn.Module.")
        for name, value in (
            ("text_embedding", text_embedding),
            ("audio_embedding", audio_embedding),
        ):
            if not isinstance(value, nn.Embedding):
                raise TypeError(f"{name} must be a torch.nn.Embedding.")
        if text_embedding.embedding_dim != audio_embedding.embedding_dim:
            raise ValueError("text and audio embeddings must share their hidden width.")
        if not isinstance(text_readout, BackboneReadout) or not isinstance(
            audio_readout, BackboneReadout
        ):
            raise TypeError("MIMO branch readouts must be BackboneReadout values.")
        for name, value in (
            ("text_head", text_head),
            ("audio_head", audio_head),
            ("audio_feature_projection", audio_feature_projection),
        ):
            if value is not None and not isinstance(value, nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module or None.")

        if config is not None and not isinstance(config, MimoModelConfig):
            raise TypeError("config must be a MimoModelConfig or None.")
        self.config = MimoModelConfig() if config is None else config
        self.body = body
        self.text_embedding = text_embedding
        self.audio_embedding = audio_embedding
        self.text_head = text_head
        self.audio_head = audio_head
        self.audio_feature_projection = audio_feature_projection
        self.text_readout = text_readout
        self.audio_readout = audio_readout
        self._encoder = DualStreamBodyAdapter(
            self.body,
            text_readout=text_readout,
            audio_readout=audio_readout,
            supports_cache_position=self.config.supports_cache_position,
        )

    def dual_hidden_states(self, batch: MimoBatch) -> DualStreamHiddenStates:
        if not isinstance(batch, MimoBatch):
            raise TypeError("MimoModel.dual_hidden_states expects a MimoBatch.")
        attention_mask = batch.attention_mask
        if attention_mask is None:
            raise RuntimeError("MimoBatch masks were not normalized.")
        output = self._encode(
            batch.text_input_ids,
            batch.audio_input_ids,
            attention_mask=attention_mask,
            audio_features=batch.audio_features,
            audio_feature_mask=batch.audio_feature_mask,
            use_cache=False,
        )
        return output.hidden_states

    def dual_logits(
        self,
        hidden_states: DualStreamHiddenStates,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(hidden_states, DualStreamHiddenStates):
            raise TypeError("dual_logits expects DualStreamHiddenStates.")
        return (
            self.text_logits(hidden_states.text),
            self.audio_logits(hidden_states.audio),
        )

    def text_logits(self, hidden_states: Tensor) -> Tensor:
        if self.text_head is not None:
            logits = self.text_head(hidden_states)
            return _logits(logits, hidden_states, "text")
        if hidden_states.size(-1) != self.text_embedding.embedding_dim:
            raise ValueError("tied text head requires the text embedding hidden width.")
        return F.linear(
            hidden_states.to(dtype=self.text_embedding.weight.dtype),
            self.text_embedding.weight,
        )

    def audio_logits(self, hidden_states: Tensor) -> Tensor:
        if self.audio_head is not None:
            logits = self.audio_head(hidden_states)
            return _logits(logits, hidden_states, "audio")
        if hidden_states.size(-1) != self.audio_embedding.embedding_dim:
            raise ValueError("tied audio head requires the audio embedding hidden width.")
        return F.linear(
            hidden_states.to(dtype=self.audio_embedding.weight.dtype),
            self.audio_embedding.weight,
        )

    def forward(self, batch: MimoBatch) -> DualStreamLogits:
        hidden = self.dual_hidden_states(batch)
        text, audio = self.dual_logits(hidden)
        return DualStreamLogits(text=text, audio=audio)

    def mimo_generation_step(
        self,
        text_input_ids: Tensor,
        audio_input_ids: Tensor,
        *,
        attention_mask: Tensor,
        past_key_values: object | None,
        use_cache: bool,
        audio_features: Tensor | None = None,
        audio_feature_mask: Tensor | None = None,
    ) -> MimoGenerationStep:
        output = self._encode(
            text_input_ids,
            audio_input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_mask=audio_feature_mask,
            past_key_values=cast(Cache | None, past_key_values),
            use_cache=use_cache,
        )
        text, audio = self.dual_logits(output.hidden_states)
        return MimoGenerationStep(
            text_logits=text,
            audio_logits=audio,
            past_key_values=output.past_key_values,
        )

    def _encode(
        self,
        text_input_ids: Tensor,
        audio_input_ids: Tensor,
        *,
        attention_mask: Tensor,
        audio_features: Tensor | None = None,
        audio_feature_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool,
    ) -> DualStreamOutput:
        _input_ids(text_input_ids, self.text_embedding, "text_input_ids")
        _input_ids(audio_input_ids, self.audio_embedding, "audio_input_ids")
        if text_input_ids.shape != audio_input_ids.shape:
            raise ValueError("MIMO text/audio input ids must be aligned.")
        if attention_mask.shape != text_input_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError("MIMO attention_mask must be boolean and align with inputs.")
        if attention_mask.device != text_input_ids.device:
            raise ValueError("MIMO attention_mask must share the input device.")
        text = self.text_embedding(text_input_ids)
        audio = self.audio_embedding(audio_input_ids)
        return self._encoder.encode_dual(
            text_inputs_embeds=text,
            audio_inputs_embeds=audio,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            audio_features=audio_features,
            audio_feature_mask=audio_feature_mask,
            feature_projection=(
                None if audio_features is None else self._project_audio_features
            ),
        )

    def _project_audio_features(self, features: Tensor) -> Tensor:
        projected = (
            features
            if self.audio_feature_projection is None
            else self.audio_feature_projection(features)
        )
        if not isinstance(projected, Tensor):
            raise TypeError("audio_feature_projection must return a Tensor.")
        return projected * float(self.config.audio_feature_scale)


def _input_ids(value: Tensor, embedding: nn.Embedding, name: str) -> None:
    if value.dim() != 2 or value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError(f"{name} must be a signed integer tensor [B, T].")
    if bool((value < 0).any()) or bool((value >= embedding.num_embeddings).any()):
        raise ValueError(f"{name} contains an id outside its local vocabulary.")


def _logits(value: object, hidden: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name}_head must return a Tensor.")
    if value.dim() != 3 or value.shape[:2] != hidden.shape[:2] or value.size(-1) < 1:
        raise ValueError(f"{name}_head must return logits with shape [B, T, V].")
    return value


__all__ = ["MimoModel", "MimoModelConfig", "TiedEmbeddingHead"]
