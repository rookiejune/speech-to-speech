from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..runtime import runtime
from .adapter import create_adapter
from .embedding import create_embedding
from .embedding.audio import merge_by_positions


@dataclass
class Config:
    audio_embed_adapter: str | None = "linear"
    audio_output_adapter: str | None = "linear"
    acoustic_adapter: str | None = "linear"
    acoustic_codebooks: int | None = None


class SemanticModel(nn.Module):
    """Shared loading, semantic modeling, and acoustic-prompt logic."""

    def __init__(self, config: Config | None = None, runtime_snapshot=None) -> None:
        super().__init__()

        self.config = config or Config()
        self.runtime = runtime() if runtime_snapshot is None else runtime_snapshot
        self.layout = self.runtime.layout
        self.embed_tokens = create_embedding(
            self.config.audio_embed_adapter, self.runtime
        )
        self.backbone = self.runtime.backbone
        hidden_size = self.backbone.config.hidden_size
        backbone_weight = self.backbone.embed_tokens.weight
        self.acoustic_adapter = create_adapter(
            self.config.acoustic_adapter,
            self.runtime.codec.acoustic_feature_dim,
            hidden_size,
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)
        self.acoustic_gate = nn.Parameter(
            torch.zeros(
                hidden_size, device=backbone_weight.device, dtype=backbone_weight.dtype
            )
        )
        audio_weight = self.embed_tokens.embeddings["audio"].weight
        self._audio_output_adapter = create_adapter(
            self.config.audio_output_adapter,
            hidden_size,
            audio_weight.size(1),
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def text_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return logits in the local text vocabulary."""
        return F.linear(hidden_state, self.backbone.embed_tokens.weight)

    def audio_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return logits in the local audio-tokenizer vocabulary."""
        audio_embedding = self.embed_tokens.embeddings["audio"]
        projected = self._audio_output_adapter(hidden_state)
        return F.linear(projected, audio_embedding.weight)

    def _logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        logits = hidden_state.new_full(
            (*hidden_state.shape[:-1], self.runtime.layout.vocab_size),
            float("-inf"),
        )
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        logits[..., text_start:text_end] = self.text_logits(hidden_state)
        logits[..., audio_start:audio_end] = self.audio_logits(hidden_state)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        acoustic_input_ids: torch.Tensor | None = None,
        acoustic_input_positions: torch.Tensor | None = None,
        acoustic_input_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        inputs_embeds = self.embed_tokens(input_ids)
        if acoustic_input_ids is not None:
            if acoustic_input_positions is None:
                raise ValueError(
                    "acoustic_input_positions is required with acoustic_input_ids."
                )
            acoustic = self._acoustic_prompt_embedding(
                input_ids,
                acoustic_input_ids,
                acoustic_input_positions,
                acoustic_input_mask,
            )
            inputs_embeds = inputs_embeds + acoustic

        backbone_output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = backbone_output.hidden_states[-1]
        logits = self._logits(hidden_states)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=backbone_output.past_key_values,
            hidden_states=backbone_output.hidden_states
            if output_hidden_states
            else None,
            attentions=getattr(backbone_output, "attentions", None),
        )

    def generate_semantic(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_input_ids: torch.Tensor | None = None,
        acoustic_input_positions: torch.Tensor | None = None,
        acoustic_input_mask: torch.Tensor | None = None,
        stop_token_id: int | None = None,
        token_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("invalid generation parameters")
        generated = prompt_ids
        for step in range(max_new_tokens):
            output = self(
                generated,
                acoustic_input_ids=acoustic_input_ids if step == 0 else None,
                acoustic_input_positions=acoustic_input_positions
                if step == 0
                else None,
                acoustic_input_mask=acoustic_input_mask if step == 0 else None,
            )
            logits = output.logits[:, -1] / temperature
            if token_range is not None:
                start, end = token_range
                if not 0 <= start < end <= logits.size(-1):
                    raise ValueError("token_range must be a valid logits interval.")
                allowed = logits.new_full(logits.shape, float("-inf"))
                allowed[..., start:end] = logits[..., start:end]
                logits = allowed
            if top_p < 1.0:
                logits = _top_p_filter(logits, top_p)
            next_ids = (
                torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
            )
            generated = torch.cat((generated, next_ids), dim=-1)
            if stop_token_id is not None and bool(next_ids.eq(stop_token_id).all()):
                break
        return generated

    def target_frame_condition(
        self,
        hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3 or target_positions.dim() != 2:
            raise ValueError(
                "hidden states and target positions must be [B, S, H] and [B, F]."
            )
        if hidden_states.size(0) != target_positions.size(0):
            raise ValueError("hidden states and target positions must align on batch.")
        mask = target_positions.ge(0)
        safe_positions = target_positions.clamp_min(0)
        if bool((safe_positions >= hidden_states.size(1)).any()):
            raise ValueError("target hidden position exceeds the sequence length.")
        condition = hidden_states.gather(
            1,
            safe_positions[..., None].expand(-1, -1, hidden_states.size(-1)),
        )
        return condition.masked_fill(~mask[..., None], 0)

    def target_frame_label_condition(
        self,
        labels: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Embed teacher-forced semantic labels at target acoustic frames."""
        if labels.dim() != 2 or target_positions.dim() != 2:
            raise ValueError("labels and target positions must be [B, S] and [B, F].")
        if labels.size(0) != target_positions.size(0):
            raise ValueError("labels and target positions must align on batch.")

        label_positions = target_positions + 1
        valid = label_positions.ge(0) & label_positions.lt(labels.size(1))
        safe_positions = label_positions.clamp(0, labels.size(1) - 1)
        safe_labels = labels.gather(1, safe_positions)
        valid = valid & safe_labels.ne(-100)
        safe_labels = safe_labels.masked_fill(~valid, 0)
        condition = self.embed_tokens(safe_labels)
        return condition.masked_fill(~valid[..., None], 0)

    def _acoustic_prompt_embedding(
        self,
        input_ids: torch.Tensor,
        acoustic_input_ids: torch.Tensor,
        positions: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if acoustic_input_ids.dim() != 3 or positions.dim() != 2:
            raise ValueError("acoustic inputs must have shapes [B, F, M] and [B, F].")
        if acoustic_input_ids.shape[:2] != positions.shape:
            raise ValueError(
                "acoustic ids and positions must align on batch and frame dimensions."
            )
        if frame_mask is None:
            frame_mask = (acoustic_input_ids != -1).all(dim=-1)
        if frame_mask.shape != positions.shape:
            raise ValueError("acoustic frame mask must align with acoustic positions.")
        safe_ids = acoustic_input_ids.masked_fill(acoustic_input_ids == -1, 0)
        frame_features = self.runtime.codec.acoustic_codes_to_features(safe_ids)
        frame_features = merge_by_positions(
            frame_features,
            positions,
            input_ids.size(1),
            frame_mask,
        )
        frame_features = self.acoustic_adapter(frame_features)
        return frame_features * self.acoustic_gate.to(dtype=frame_features.dtype)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative - sorted_logits.softmax(dim=-1) >= top_p
    remove[..., 0] = False
    filtered = logits.new_full(logits.shape, float("-inf"))
    filtered.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits.masked_fill(remove, float("-inf")),
    )
    return filtered
