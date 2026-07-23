from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional

import torch
from anydataset.types import Modality
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from ._generation import generate_sequence
from ._head import VocabularyHeadMixin
from .adapter import AdapterType, create_adapter
from .embedding import create_semantic_audio_modules
from .embedding.audio import merge_by_positions
from .protocol import TokenModelRuntime
from .toy import ToyConfig, create_toy_backbone
from ..runtime.types import BackboneOutput


@dataclass
class Config:
    semantic_audio_adapter: Optional[AdapterType] = AdapterType.LINEAR
    semantic_audio_output_adapter: Optional[AdapterType] = AdapterType.LINEAR
    acoustic_prompt_adapter: Optional[AdapterType] = AdapterType.LINEAR
    toy: Optional[ToyConfig] = None


class TokenModel(VocabularyHeadMixin, nn.Module):
    """Shared token modeling and acoustic-prompt logic."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        runtime: TokenModelRuntime,
    ) -> None:
        super().__init__()

        self.config = config or Config()
        self.runtime = runtime
        self.layout = self.runtime.layout
        text_start, text_end = self.layout.blocks["text"]
        self.backbone = (
            self.runtime.backbone
            if self.config.toy is None
            else create_toy_backbone(self.config.toy, text_end - text_start)
        )
        (
            self.semantic_audio_embedding,
            self.semantic_audio_adapter,
        ) = create_semantic_audio_modules(
            self.config.semantic_audio_adapter,
            self.runtime,
            self.backbone,
        )
        hidden_size = self.backbone.config.hidden_size
        input_embedding = self.backbone.get_input_embeddings()
        output_embedding = self.backbone.get_output_embeddings()
        text_vocab_size = text_end - text_start
        if input_embedding.weight.size(0) < text_vocab_size:
            raise ValueError(
                "backbone input embedding does not cover the text layout vocabulary."
            )
        if output_embedding.weight.size(0) < text_vocab_size:
            raise ValueError(
                "backbone output embedding does not cover the text layout vocabulary."
            )
        backbone_weight = input_embedding.weight
        self.audio_token_frame_spans = nn.Buffer(
            _frame_span_lookup(self.runtime).to(device=backbone_weight.device),
            persistent=False,
        )
        self.acoustic_prompt_adapter = (
            create_adapter(
                self.config.acoustic_prompt_adapter,
                self.runtime.codec.acoustic_feature_dim,
                hidden_size,
            )
            if self.runtime.codec.acoustic_codebook_sizes
            else nn.Identity()
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)
        acoustic_prompt_gate = torch.zeros(
            hidden_size, device=backbone_weight.device, dtype=backbone_weight.dtype
        )
        self.acoustic_prompt_gate = (
            nn.Parameter(acoustic_prompt_gate)
            if self.runtime.codec.acoustic_codebook_sizes
            else nn.Buffer(acoustic_prompt_gate, persistent=False)
        )
        semantic_audio_weight = self.semantic_audio_embedding.weight
        self.semantic_audio_output_adapter = create_adapter(
            self.config.semantic_audio_output_adapter,
            hidden_size,
            semantic_audio_weight.size(1),
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        acoustic_prompt_codes: torch.Tensor | None = None,
        acoustic_prompt_positions: torch.Tensor | None = None,
        acoustic_prompt_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        backbone_output = self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            acoustic_prompt_codes=acoustic_prompt_codes,
            acoustic_prompt_positions=acoustic_prompt_positions,
            acoustic_prompt_mask=acoustic_prompt_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        hidden_states = backbone_output.last_hidden_state
        logits = self.token_logits(hidden_states)
        return self._output(
            backbone_output, hidden_states, logits, output_hidden_states
        )

    def generation_step(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        acoustic_prompt_codes: torch.Tensor | None,
        acoustic_prompt_positions: torch.Tensor | None,
        acoustic_prompt_mask: torch.Tensor | None,
        output_hidden_states: bool,
        token_ids: torch.Tensor | None,
        modality: Modality | None,
        past_key_values: Cache | None,
        use_cache: bool,
    ) -> CausalLMOutputWithPast:
        """Run one autoregressive step with an explicit output-head selection."""
        if token_ids is not None and modality is not None:
            raise ValueError(
                "generation token ids and modality cannot both be provided."
            )
        backbone_output = self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            acoustic_prompt_codes=acoustic_prompt_codes,
            acoustic_prompt_positions=acoustic_prompt_positions,
            acoustic_prompt_mask=acoustic_prompt_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = backbone_output.last_hidden_state
        last_hidden_state = hidden_states[:, -1:]
        if modality is not None:
            logits = self.modality_logits(last_hidden_state, modality)
        elif token_ids is not None:
            logits = self.selected_logits(last_hidden_state, token_ids)
        else:
            logits = self.token_logits(last_hidden_state)
        return self._output(
            backbone_output, hidden_states, logits, output_hidden_states
        )

    @staticmethod
    def _output(
        backbone_output: BackboneOutput,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        output_hidden_states: bool,
    ) -> CausalLMOutputWithPast:
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,  # pyright: ignore[reportArgumentType]
            past_key_values=backbone_output.past_key_values,
            hidden_states=(hidden_states,)  # pyright: ignore[reportArgumentType]
            if output_hidden_states
            else None,
            attentions=backbone_output.attentions,  # pyright: ignore[reportArgumentType]
        )

    def token_hidden_states(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        acoustic_prompt_codes: torch.Tensor | None = None,
        acoustic_prompt_positions: torch.Tensor | None = None,
        acoustic_prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one training batch without constructing vocabulary logits."""
        return self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            acoustic_prompt_codes=acoustic_prompt_codes,
            acoustic_prompt_positions=acoustic_prompt_positions,
            acoustic_prompt_mask=acoustic_prompt_mask,
            use_cache=False,
        ).last_hidden_state

    def _backbone_output(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        acoustic_prompt_codes: torch.Tensor | None,
        acoustic_prompt_positions: torch.Tensor | None,
        acoustic_prompt_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> BackboneOutput:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        inputs_embeds = self._input_embedding(input_ids)
        if acoustic_prompt_codes is not None:
            if acoustic_prompt_positions is None:
                raise ValueError(
                    "acoustic_prompt_positions is required with acoustic_prompt_codes."
                )
            acoustic = self._acoustic_prompt_embedding(
                input_ids,
                acoustic_prompt_codes,
                acoustic_prompt_positions,
                acoustic_prompt_mask,
            )
            inputs_embeds = inputs_embeds + acoustic

        return self.backbone.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
        )

    def generate_tokens(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_prompt_codes: torch.Tensor | None = None,
        acoustic_prompt_positions: torch.Tensor | None = None,
        acoustic_prompt_mask: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        allowed_token_ids: Sequence[int] | torch.Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> torch.Tensor:
        generated, _, _ = generate_sequence(
            self,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            acoustic_prompt_codes=acoustic_prompt_codes,
            acoustic_prompt_positions=acoustic_prompt_positions,
            acoustic_prompt_mask=acoustic_prompt_mask,
            prompt_attention_mask=prompt_attention_mask,
            stop_token_id=stop_token_id,
            generation_modality=generation_modality,
            allowed_token_ids=allowed_token_ids,
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=False,
        )
        return generated

    def generate_audio_condition(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_prompt_codes: torch.Tensor | None = None,
        acoustic_prompt_positions: torch.Tensor | None = None,
        acoustic_prompt_mask: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate audio tokens and their frame-aligned acoustic condition."""
        generated, condition, frame_spans = generate_sequence(
            self,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            acoustic_prompt_codes=acoustic_prompt_codes,
            acoustic_prompt_positions=acoustic_prompt_positions,
            acoustic_prompt_mask=acoustic_prompt_mask,
            prompt_attention_mask=prompt_attention_mask,
            stop_token_id=self.runtime.eoa_token_id,
            generation_modality=Modality.AUDIO,
            allowed_token_ids=None,
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=True,
        )
        if condition is None or frame_spans is None:
            raise ValueError(
                "token generation produced no codec-decodable audio tokens."
            )
        frame_counts = frame_spans.sum(dim=1)
        frame_mask = (
            torch.arange(condition.size(1), device=condition.device)[None]
            < frame_counts[:, None]
        )
        return generated, condition, frame_mask

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
        safe_positions = (target_positions - 1).clamp_min(0)
        condition = hidden_states.gather(
            1,
            safe_positions[..., None].expand(-1, -1, hidden_states.size(-1)),
        )
        return condition.masked_fill(~mask[..., None], 0)

    def target_frame_label_condition(
        self,
        token_labels: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Embed teacher-forced token labels at target acoustic frames."""
        if token_labels.dim() != 2 or target_positions.dim() != 2:
            raise ValueError(
                "token labels and target positions must be [B, S] and [B, F]."
            )
        if token_labels.size(0) != target_positions.size(0):
            raise ValueError("token labels and target positions must align on batch.")

        valid = target_positions.ge(0) & target_positions.lt(token_labels.size(1))
        safe_positions = target_positions.clamp(0, token_labels.size(1) - 1)
        safe_labels = token_labels.gather(1, safe_positions)
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
            raise ValueError(
                "input token ids contain an id outside the runtime layout."
            )
        output = self.backbone.get_input_embeddings()(
            input_ids.clamp(text_start, text_end - 1) - text_start
        )
        audio_token_ids = input_ids[audio_mask] - audio_start
        output[audio_mask] = self.semantic_audio_adapter(
            self.semantic_audio_embedding(audio_token_ids)
        )
        return output

    def _acoustic_prompt_embedding(
        self,
        input_ids: torch.Tensor,
        acoustic_prompt_codes: torch.Tensor,
        token_positions: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if acoustic_prompt_codes.dim() != 3 or token_positions.dim() != 2:
            raise ValueError("acoustic inputs must have shapes [B, F, M] and [B, F].")
        if acoustic_prompt_codes.shape[:2] != token_positions.shape:
            raise ValueError(
                "acoustic codes and token positions must align on batch and frame."
            )
        if frame_mask is None:
            frame_mask = (acoustic_prompt_codes != -1).all(dim=-1)
        if frame_mask.shape != token_positions.shape:
            raise ValueError("acoustic frame mask must align with token positions.")
        safe_codes = acoustic_prompt_codes.masked_fill(acoustic_prompt_codes == -1, 0)
        frame_features = self.acoustic_code_features(safe_codes)
        frame_features, token_mask = merge_by_positions(
            frame_features,
            token_positions,
            input_ids.size(1),
            frame_mask,
        )
        frame_features = self.acoustic_prompt_adapter(frame_features)
        frame_features = frame_features.masked_fill(~token_mask[..., None], 0)
        return frame_features * self.acoustic_prompt_gate.to(dtype=frame_features.dtype)

    def acoustic_code_features(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert codec-local acoustic codes to model-aligned features."""
        features = self.runtime.codec.acoustic_codes_to_features(codes)
        weight = self.backbone.get_input_embeddings().weight
        return features.to(device=weight.device, dtype=weight.dtype)


def _frame_span_lookup(runtime: TokenModelRuntime) -> torch.Tensor:
    spans = torch.as_tensor(
        runtime.audio_tokenizer.frame_spans(range(runtime.audio_tokenizer.vocab_size)),
        dtype=torch.long,
    )
    if spans.shape != (runtime.audio_tokenizer.vocab_size,):
        raise ValueError("audio token frame spans must cover the tokenizer vocabulary.")
    if bool((spans < 0).any()) or not bool((spans > 0).any()):
        raise ValueError("audio token frame spans must be non-negative and non-empty.")
    return spans
