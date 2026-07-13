from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..runtime import Runtime, runtime
from ._sampling import top_p_filter
from .adapter import AdapterType, create_adapter
from .embedding import create_semantic_audio_modules
from .embedding.audio import merge_by_positions
from .protocol import ModelRuntime


@dataclass
class Config:
    semantic_audio_adapter: AdapterType | None = AdapterType.LINEAR
    semantic_audio_output_adapter: AdapterType | None = AdapterType.LINEAR
    acoustic_prompt_adapter: AdapterType | None = AdapterType.LINEAR
    acoustic_decoder_dim: int | None = None
    acoustic_decoder_layers: int = 8
    acoustic_decoder_heads: int = 8
    acoustic_decoder_ffn_ratio: int = 4
    acoustic_repa_dim: int = 768
    acoustic_repa_layer: int | None = None


class SemanticModel(nn.Module):
    """Shared loading, semantic modeling, and acoustic-prompt logic."""

    def __init__(
        self,
        config: Config | None = None,
        runtime_snapshot: Runtime | ModelRuntime | None = None,
    ) -> None:
        super().__init__()

        self.config = config or Config()
        snapshot = runtime() if runtime_snapshot is None else runtime_snapshot
        self.runtime = cast(
            ModelRuntime,
            cast(object, snapshot),
        )
        self.layout = self.runtime.layout
        (
            self.semantic_audio_embedding,
            self.semantic_audio_adapter,
        ) = create_semantic_audio_modules(
            self.config.semantic_audio_adapter, self.runtime
        )
        self.backbone = self.runtime.backbone
        hidden_size = self.backbone.config.hidden_size
        input_embedding = self.backbone.get_input_embeddings()
        output_embedding = self.backbone.get_output_embeddings()
        _, text_end = self.layout.blocks["text"]
        if input_embedding.weight.size(0) < text_end:
            raise ValueError(
                "backbone input embedding does not cover the text layout vocabulary."
            )
        if output_embedding.weight.size(0) < text_end:
            raise ValueError(
                "backbone output embedding does not cover the text layout vocabulary."
            )
        backbone_weight = input_embedding.weight
        self.acoustic_prompt_adapter = (
            create_adapter(
                self.config.acoustic_prompt_adapter,
                self.runtime.codec.acoustic_feature_dim,
                hidden_size,
            )
            if self.runtime.codec.acoustic_codebook_sizes
            else nn.Identity()
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)
        self.acoustic_prompt_gate = nn.Parameter(
            torch.zeros(
                hidden_size, device=backbone_weight.device, dtype=backbone_weight.dtype
            )
        )
        semantic_audio_weight = self.semantic_audio_embedding.weight
        self._semantic_audio_output_adapter = create_adapter(
            self.config.semantic_audio_output_adapter,
            hidden_size,
            semantic_audio_weight.size(1),
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def text_logits(
        self,
        hidden_state: torch.Tensor,
        local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits in the local text vocabulary."""
        _, text_end = self.layout.blocks["text"]
        output = self.backbone.get_output_embeddings()
        weight = output.weight[:text_end]
        output_bias = getattr(output, "bias", None)
        bias = None if output_bias is None else output_bias[:text_end]
        if local_ids is not None:
            weight = weight.index_select(0, local_ids)
            bias = None if bias is None else bias.index_select(0, local_ids)
        return F.linear(hidden_state, weight, bias)

    def semantic_audio_logits(
        self,
        hidden_state: torch.Tensor,
        local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits in the local audio-tokenizer vocabulary."""
        projected = self._semantic_audio_output_adapter(hidden_state)
        weight = self.semantic_audio_embedding.weight
        if local_ids is not None:
            weight = weight.index_select(0, local_ids)
        return F.linear(projected, weight)

    def _logits(
        self,
        hidden_state: torch.Tensor,
        token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids is not None:
            return self._selected_logits(hidden_state, token_ids)
        logits = hidden_state.new_full(
            (*hidden_state.shape[:-1], self.runtime.layout.vocab_size),
            float("-inf"),
        )
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        logits[..., text_start:text_end] = self.text_logits(hidden_state)
        logits[..., audio_start:audio_end] = self.semantic_audio_logits(hidden_state)
        return logits

    def _selected_logits(
        self,
        hidden_state: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = hidden_state.new_empty(*hidden_state.shape[:-1], token_ids.numel())
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        text_mask = token_ids.ge(text_start) & token_ids.lt(text_end)
        audio_mask = token_ids.ge(audio_start) & token_ids.lt(audio_end)
        if not bool((text_mask | audio_mask).all()):
            raise ValueError("selected token ids contain an invalid vocabulary id.")
        if bool(text_mask.any()):
            text_ids = token_ids[text_mask] - text_start
            logits[..., text_mask] = self.text_logits(hidden_state, text_ids)
        if bool(audio_mask.any()):
            audio_ids = token_ids[audio_mask] - audio_start
            logits[..., audio_mask] = self.semantic_audio_logits(
                hidden_state, audio_ids
            )
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
        _generation_token_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        inputs_embeds = self._input_embedding(input_ids)
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

        backbone_output = self.backbone.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            **kwargs,
        )
        hidden_states = backbone_output.last_hidden_state
        logit_hidden_states = (
            hidden_states if _generation_token_ids is None else hidden_states[:, -1:]
        )
        logits = self._logits(
            logit_hidden_states,
            _generation_token_ids,
        )
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,  # pyright: ignore[reportArgumentType]
            past_key_values=backbone_output.past_key_values,
            hidden_states=(hidden_states,)  # pyright: ignore[reportArgumentType]
            if output_hidden_states
            else None,
            attentions=backbone_output.attentions,  # pyright: ignore[reportArgumentType]
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
        prompt_attention_mask: torch.Tensor | None = None,
        stop_token_id: int | None = None,
        allowed_token_ids: Sequence[int] | torch.Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> torch.Tensor:
        generated, _, _ = self._generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            acoustic_input_ids=acoustic_input_ids,
            acoustic_input_positions=acoustic_input_positions,
            acoustic_input_mask=acoustic_input_mask,
            prompt_attention_mask=prompt_attention_mask,
            stop_token_id=stop_token_id,
            allowed_token_ids=allowed_token_ids,
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=False,
        )
        return generated

    def _generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        acoustic_input_ids: torch.Tensor | None,
        acoustic_input_positions: torch.Tensor | None,
        acoustic_input_mask: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        stop_token_id: int | None,
        allowed_token_ids: Sequence[int] | torch.Tensor | None,
        do_sample: bool,
        use_cache: bool,
        collect_audio_condition: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if max_new_tokens < 0 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("invalid generation parameters")
        if prompt_ids.dim() != 2 or prompt_ids.size(0) < 1:
            raise ValueError("generation requires at least one prompt row.")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        if prompt_attention_mask.shape != prompt_ids.shape:
            raise ValueError("prompt attention mask must align with prompt ids.")
        if not bool(prompt_attention_mask.any(dim=1).all()):
            raise ValueError("each generation prompt must contain at least one token.")

        generation_token_ids = None
        if allowed_token_ids is not None:
            generation_token_ids = torch.as_tensor(
                allowed_token_ids,
                device=prompt_ids.device,
                dtype=torch.long,
            )
            if generation_token_ids.dim() != 1 or generation_token_ids.numel() == 0:
                raise ValueError("allowed_token_ids must be a non-empty 1D sequence.")
            if generation_token_ids.unique().numel() != generation_token_ids.numel():
                raise ValueError("allowed_token_ids must not contain duplicates.")
            text_start, text_end = self.runtime.layout.blocks["text"]
            audio_start, audio_end = self.runtime.layout.blocks["audio"]
            text_mask = generation_token_ids.ge(text_start) & generation_token_ids.lt(
                text_end
            )
            audio_mask = generation_token_ids.ge(audio_start) & generation_token_ids.lt(
                audio_end
            )
            if not bool((text_mask | audio_mask).all()):
                raise ValueError("allowed_token_ids contains an invalid vocabulary id.")

        generated = prompt_ids
        input_ids = prompt_ids
        attention_mask = prompt_attention_mask
        past_key_values = None
        conditions: list[list[torch.Tensor]] = [[] for _ in range(prompt_ids.size(0))]
        spans: list[list[int]] = [[] for _ in range(prompt_ids.size(0))]
        finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=prompt_ids.device)
        for _ in range(max_new_tokens):
            inject_acoustic = past_key_values is None
            output = self(
                input_ids,
                attention_mask=attention_mask,
                acoustic_input_ids=acoustic_input_ids if inject_acoustic else None,
                acoustic_input_positions=acoustic_input_positions
                if inject_acoustic
                else None,
                acoustic_input_mask=acoustic_input_mask if inject_acoustic else None,
                output_hidden_states=collect_audio_condition,
                _generation_token_ids=generation_token_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = output.logits[:, -1] / temperature
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)
            next_indices = (
                torch.distributions.Categorical(logits=logits).sample()
                if do_sample
                else logits.argmax(dim=-1)
            )
            next_ids = (
                next_indices
                if generation_token_ids is None
                else generation_token_ids.index_select(0, next_indices)
            )
            next_ids = next_ids.unsqueeze(-1)

            if collect_audio_condition:
                if output.hidden_states is None:
                    raise RuntimeError("model did not return generation hidden states.")
                for row, token_id in enumerate(next_ids[:, 0].tolist()):
                    if finished[row] or not self.runtime.is_codec_audio_id(token_id):
                        continue
                    local_id = token_id - self.runtime.codec_audio_range[0]
                    span = int(self.runtime.audio_tokenizer.frame_spans([local_id])[0])
                    if span <= 0:
                        raise ValueError(
                            "generated audio token decoded to no semantic frames."
                        )
                    conditions[row].append(
                        output.hidden_states[-1][row, -1].expand(span, -1)
                    )
                    spans[row].append(span)

            generated = torch.cat((generated, next_ids), dim=-1)
            if stop_token_id is not None:
                finished |= next_ids[:, 0].eq(stop_token_id)
                if bool(finished.all()):
                    break
            if use_cache:
                past_key_values = output.past_key_values
                if past_key_values is None:
                    raise RuntimeError("backbone did not return a generation cache.")
                input_ids = next_ids
            else:
                input_ids = generated
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_ids, dtype=torch.bool)), dim=1
            )

        if not any(conditions):
            return generated, None, None
        if any(not row for row in conditions):
            raise ValueError("an audio generation row produced no codec-decodable tokens.")
        condition = pad_sequence(
            [torch.cat(row, dim=0) for row in conditions], batch_first=True
        )
        frame_spans = pad_sequence(
            [generated.new_tensor(row) for row in spans], batch_first=True
        )
        return generated, condition, frame_spans

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
        if bool((mask & target_positions.lt(1)).any()):
            raise ValueError("target token positions must have a causal predictor.")
        safe_positions = (target_positions - 1).clamp_min(0)
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

        valid = target_positions.ge(0) & target_positions.lt(labels.size(1))
        safe_positions = target_positions.clamp(0, labels.size(1) - 1)
        safe_labels = labels.gather(1, safe_positions)
        valid = valid & safe_labels.ne(-100)
        safe_labels = safe_labels.masked_fill(~valid, 0)
        condition = self._input_embedding(safe_labels)
        return condition.masked_fill(~valid[..., None], 0)

    def _input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        text_start, text_end = self.layout.blocks["text"]
        audio_start, audio_end = self.layout.blocks["audio"]
        text_mask = input_ids.ge(text_start) & input_ids.lt(text_end)
        audio_mask = input_ids.ge(audio_start) & input_ids.lt(audio_end)
        if not bool((text_mask | audio_mask).all()):
            raise ValueError("semantic ids contain an id outside the runtime layout.")
        output = self.backbone.get_input_embeddings()(
            input_ids.clamp(text_start, text_end - 1) - text_start
        )
        if bool(audio_mask.any()):
            audio_ids = input_ids[audio_mask] - audio_start
            output[audio_mask] = self.semantic_audio_adapter(
                self.semantic_audio_embedding(audio_ids)
            )
        return output

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
        frame_features = self._acoustic_features(safe_ids)
        frame_features, token_mask = merge_by_positions(
            frame_features,
            positions,
            input_ids.size(1),
            frame_mask,
        )
        frame_features = self.acoustic_prompt_adapter(frame_features)
        frame_features = frame_features.masked_fill(~token_mask[..., None], 0)
        return frame_features * self.acoustic_prompt_gate.to(
            dtype=frame_features.dtype
        )

    def _acoustic_features(self, codes: torch.Tensor) -> torch.Tensor:
        features = self.runtime.codec.acoustic_codes_to_features(codes)
        weight = self.backbone.get_input_embeddings().weight
        return features.to(device=weight.device, dtype=weight.dtype)
