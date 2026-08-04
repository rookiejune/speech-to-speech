from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch
from anydataset.types import Modality
from peft import LoraConfig
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .._tensor import is_signed_integer_dtype
from ..task import PredictionModality
from ..runtime.backbone import BackboneOutputView
from . import _assembly
from .contract import ModelCheckpointContract, build_model_contract
from .generation import (
    GenerationEngine,
    GenerationOptions,
    GenerationOutput,
    GenerationRequest,
    GenerationStepResult,
    TokenKind,
)
from .adapter import AdapterType
from ._helper import register
from .audio_input import AudioInputAdapterConfig, AudioInputTower
from .audio_output import AudioOutputAdapterConfig
from .ctc import CTCConfig, CTCDecoderRoutes, CTCRoute, ObjectiveHiddenOutput
from .embedding.fsq import FsqEmbeddingConfig, FsqNeighbors
from ..runtime.protocol import TokenModelRuntime
from .token import TokenInterface
from .toy import ToyConfig


@dataclass
class Config:
    semantic_audio_adapter: Optional[AdapterType] = AdapterType.LINEAR
    audio_output_adapter: AudioOutputAdapterConfig = field(
        default_factory=AudioOutputAdapterConfig
    )
    audio_input_adapter: AudioInputAdapterConfig = field(
        default_factory=AudioInputAdapterConfig
    )
    fsq_embedding: FsqEmbeddingConfig = field(default_factory=FsqEmbeddingConfig)
    ctc: CTCConfig = field(default_factory=CTCConfig)
    toy: Optional[ToyConfig] = None
    lora: Optional[LoraConfig] = None


class Model(nn.Module):
    """Text and semantic-audio model; Flow/RVQ compositions subclass this entry."""

    audio_token_frame_spans: torch.Tensor
    source_audio_encoder: AudioInputTower | None
    tokens: TokenInterface

    @property
    def lora_config(self) -> Optional[LoraConfig]:
        return self.config.lora

    @property
    def checkpoint_contract(self) -> ModelCheckpointContract:
        return build_model_contract(
            self,
            self._acoustic_checkpoint_components(),
        )

    def _acoustic_checkpoint_components(self) -> Mapping[str, object]:
        return {"type": "none"}

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
        text_vocab_size = text_end - text_start
        self.backbone = _assembly.backbone(self.runtime, self.config, text_vocab_size)
        text_embedding = _assembly.text_embedding(self.backbone, text_vocab_size)
        text_embedding.requires_grad_(False)
        self._encoder = _assembly.backbone_adapter(
            self.runtime,
            self.backbone,
            prefer_runtime=self.config.toy is None,
        )
        hidden_size = _assembly.backbone_hidden_size(
            self.runtime,
            self.backbone,
            prefer_runtime=self.config.toy is None,
        )
        self.tokens = _assembly.tokens(
            self.runtime,
            self.config,
            text_embedding,
            hidden_size,
        )
        backbone_weight = text_embedding.weight
        register(
            self,
            "audio_token_frame_spans",
            _assembly.frame_span_lookup(self.runtime).to(device=backbone_weight.device),
            persistent=False,
        )
        semantic_audio_dim = self.tokens.semantic_audio_embedding.embedding_dim
        self.source_audio_encoder = _assembly.audio_input_adapter(
            self.config,
            semantic_audio_dim,
            hidden_size,
            device=backbone_weight.device,
        )
        self.ctc_decoders = CTCDecoderRoutes(
            self.config.ctc,
            hidden_size,
        ).to(device=backbone_weight.device, dtype=torch.float32)
        if _assembly.runtime_gradient_checkpointing(self.runtime):
            _assembly.enable_interface_gradient_checkpointing(
                self.tokens,
                self.source_audio_encoder,
                self.ctc_decoders,
            )

    @property
    def text_embedding(self) -> nn.Embedding:
        text_start, text_end = self.layout.blocks[Modality.TEXT.value]
        return _assembly.text_embedding(self.backbone, text_end - text_start)

    def select_audio_head_cache(
        self,
        past_key_values: object | None,
        indices: torch.Tensor,
    ) -> object | None:
        return self.tokens.select_audio_head_cache(past_key_values, indices)

    def text_logits(
        self,
        hidden_state: torch.Tensor,
        local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.tokens.text_logits(self.text_embedding, hidden_state, local_ids)

    def semantic_audio_logits(
        self,
        hidden_state: torch.Tensor,
        local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.tokens.semantic_audio_logits(hidden_state, local_ids)

    def audio_neighbor_targets(self, local_ids: torch.Tensor) -> FsqNeighbors | None:
        return self.tokens.audio_neighbor_targets(local_ids)

    def project_audio_hidden(
        self,
        hidden_state: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        selection_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, object | None]:
        return self.tokens.project_audio_hidden(
            hidden_state,
            attention_mask=attention_mask,
            selection_mask=selection_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def token_logits(
        self,
        hidden_state: torch.Tensor,
        modality: Modality | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        audio_hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.tokens.token_logits(
            self.text_embedding,
            hidden_state,
            modality,
            attention_mask=attention_mask,
            audio_hidden_state=audio_hidden_state,
        )

    def modality_logits(
        self,
        hidden_state: torch.Tensor,
        modality: Modality,
        *,
        attention_mask: torch.Tensor | None = None,
        audio_hidden_state: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, object | None]:
        blocked_token_ids = (
            (self.runtime.pad_token_id, self.runtime.bos_token_id)
            if modality is Modality.TEXT
            else (self.runtime.boa_token_id, self.runtime.mask_token_id)
        )
        return self.tokens.modality_logits(
            self.text_embedding,
            hidden_state,
            modality,
            blocked_token_ids=blocked_token_ids,
            attention_mask=attention_mask,
            audio_hidden_state=audio_hidden_state,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def selected_logits(
        self,
        hidden_state: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        token_kind: TokenKind | None = None,
        attention_mask: torch.Tensor | None = None,
        audio_hidden_state: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        validate: bool = True,
    ) -> tuple[torch.Tensor, object | None]:
        return self.tokens.selected_logits(
            self.text_embedding,
            hidden_state,
            token_ids,
            token_kind=token_kind,
            attention_mask=attention_mask,
            audio_hidden_state=audio_hidden_state,
            past_key_values=past_key_values,
            use_cache=use_cache,
            validate=validate,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        backbone_output = self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            audio_input_positions=audio_input_positions,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        hidden_states = backbone_output.last_hidden_state
        logits = self.token_logits(hidden_states, attention_mask=attention_mask)
        return self._output(
            backbone_output, hidden_states, logits, output_hidden_states
        )

    def generation_step(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        token_ids: torch.Tensor | None,
        token_kind: TokenKind | None = None,
        modality: Modality | None,
        past_key_values: Cache | None,
        use_cache: bool,
        audio_input_positions: torch.Tensor | None = None,
        audio_head_past: object | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> GenerationStepResult:
        """Run one autoregressive step with an explicit output-head selection."""
        if token_ids is not None and modality is not None:
            raise ValueError(
                "generation token ids and modality cannot both be provided."
            )
        if token_kind is not None and token_ids is None:
            raise ValueError("generation token kind requires explicit token ids.")
        readout_modality = _generation_readout_modality(
            modality,
            token_kind,
            has_modality_readouts=self._encoder.has_modality_readouts,
        )
        backbone_output = self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            audio_input_positions=audio_input_positions,
            past_key_values=past_key_values,
            use_cache=use_cache,
            input_modalities=input_modalities,
            validate_input=validate_input,
            validate_audio_input_positions=validate_audio_input_positions,
            modality=readout_modality,
        )
        hidden_states = backbone_output.last_hidden_state
        uses_audio_head = (
            modality is not Modality.TEXT
            and token_kind != Modality.TEXT.value
        )
        head_hidden_states = (
            hidden_states
            if uses_audio_head and self._audio_head_uses_sequence_context()
            else hidden_states[:, -1:]
        )
        head_attention_mask = attention_mask[:, -head_hidden_states.size(1) :]
        audio_past = None
        if modality is not None:
            logits, audio_past = self.modality_logits(
                head_hidden_states,
                modality,
                attention_mask=head_attention_mask,
                past_key_values=audio_head_past,
                use_cache=use_cache,
            )
        elif token_ids is not None:
            logits, audio_past = self.selected_logits(
                head_hidden_states,
                token_ids,
                token_kind=token_kind,
                attention_mask=head_attention_mask,
                past_key_values=audio_head_past,
                use_cache=use_cache,
                validate=validate_input,
            )
        else:
            adapted, audio_past = self.project_audio_hidden(
                head_hidden_states,
                attention_mask=head_attention_mask,
                past_key_values=audio_head_past,
                use_cache=use_cache,
            )
            logits = self.token_logits(
                head_hidden_states,
                audio_hidden_state=adapted,
            )
        return GenerationStepResult(
            logits=logits[:, -1:],
            past_key_values=backbone_output.past_key_values,
            audio_head_past=audio_past,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )

    def _audio_head_uses_sequence_context(self) -> bool:
        return not self.tokens.audio_head.is_pointwise

    @staticmethod
    def _output(
        backbone_output: BackboneOutputView,
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
        audio_input_positions: torch.Tensor | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        prediction: PredictionModality | None = None,
    ) -> torch.Tensor:
        """Encode one training batch without constructing vocabulary logits."""
        modality = _prediction_readout_modality(
            prediction,
            has_modality_readouts=self._encoder.has_modality_readouts,
        )
        return self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            audio_input_positions=audio_input_positions,
            input_modalities=input_modalities,
            validate_input=validate_input,
            validate_audio_input_positions=validate_audio_input_positions,
            use_cache=False,
            modality=modality,
        ).last_hidden_state

    def objective_hidden_output(
        self,
        input_ids: torch.Tensor,
        *,
        ctc_routes: frozenset[CTCRoute],
        attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        prediction: PredictionModality | None = None,
    ) -> ObjectiveHiddenOutput:
        """Encode token and route-specific CTC readouts in one backbone call."""
        if not isinstance(ctc_routes, frozenset) or any(
            not isinstance(route, CTCRoute) for route in ctc_routes
        ):
            raise TypeError("ctc_routes must be a frozenset of CTCRoute values.")
        modality = _prediction_readout_modality(
            prediction,
            has_modality_readouts=self._encoder.has_modality_readouts,
        )
        output = self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            audio_input_positions=audio_input_positions,
            input_modalities=input_modalities,
            validate_input=validate_input,
            validate_audio_input_positions=validate_audio_input_positions,
            output_hidden_states=self.ctc_decoders.requires_hidden_states(ctc_routes),
            use_cache=False,
            modality=modality,
        )
        source = (
            self.ctc_decoders.hidden_states(output, CTCRoute.SOURCE)
            if CTCRoute.SOURCE in ctc_routes
            else None
        )
        target = (
            self.ctc_decoders.hidden_states(output, CTCRoute.TARGET)
            if CTCRoute.TARGET in ctc_routes
            else None
        )
        return ObjectiveHiddenOutput(
            token=output.last_hidden_state,
            source_ctc=source,
            target_ctc=target,
        )

    def ctc_logits(
        self,
        route: CTCRoute,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one CTC route, then apply the frozen tied text readout."""
        decoded, pooled_mask = self.ctc_decoders(route, hidden_states, mask)
        return self.text_logits(decoded), pooled_mask

    def training_input_hints(
        self,
        input_ids: torch.Tensor,
        audio_input_positions: torch.Tensor | None,
    ) -> tuple[frozenset[Modality], bool] | None:
        """Validate CPU batch routing once before it is asynchronously transferred."""
        if input_ids.device.type != "cpu":
            return None
        modalities = self.tokens.selected_modalities(input_ids)
        if audio_input_positions is None:
            return modalities, True
        _validate_audio_input_positions(
            input_ids,
            audio_input_positions,
            self.runtime.codec_audio_range,
        )
        return modalities, True

    def _backbone_output(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        audio_input_positions: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        modality: Modality | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutputView:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        inputs_embeds = self._input_embedding(
            input_ids,
            audio_input_positions,
            input_modalities=input_modalities,
            validate_input=validate_input,
            validate_audio_input_positions=validate_audio_input_positions,
        )

        return self._encoder.encode(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            modality=modality,
        )

    def generate_tokens(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        allowed_token_ids: Sequence[int] | torch.Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> torch.Tensor:
        output = GenerationEngine(self).generate(
            GenerationRequest(
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                audio_input_positions=audio_input_positions,
                stop_token_id=stop_token_id,
                generation_modality=generation_modality,
                allowed_token_ids=allowed_token_ids,
            ),
            GenerationOptions(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
            ),
        )
        return output.sequences

    def generate_tokens_with_logprobs(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        allowed_token_ids: Sequence[int] | torch.Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> GenerationOutput:
        return GenerationEngine(self).generate(
            GenerationRequest(
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                audio_input_positions=audio_input_positions,
                stop_token_id=stop_token_id,
                generation_modality=generation_modality,
                allowed_token_ids=allowed_token_ids,
            ),
            GenerationOptions(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
                collect_logprobs=True,
            ),
        )

    def generate_audio_condition(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate audio tokens and their frame-aligned acoustic condition."""
        output = GenerationEngine(self).generate(
            GenerationRequest(
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                audio_input_positions=audio_input_positions,
                stop_token_id=self.runtime.eoa_token_id,
                generation_modality=Modality.AUDIO,
            ),
            GenerationOptions(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
                collect_audio_condition=True,
                min_new_tokens=1,
            ),
        )
        generated = output.sequences
        condition = output.audio_condition
        frame_spans = output.frame_spans
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
        text_start, _ = self.layout.blocks[Modality.TEXT.value]
        safe_labels = safe_labels.masked_fill(~valid, text_start)
        condition = self._input_embedding(safe_labels)
        return condition.masked_fill(~valid[..., None], 0)

    def _input_embedding(
        self,
        input_ids: torch.Tensor,
        audio_input_positions: torch.Tensor | None = None,
        *,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> torch.Tensor:
        override_mask = None
        if self.source_audio_encoder is not None and audio_input_positions is not None:
            if validate_audio_input_positions:
                _validate_audio_input_positions(
                    input_ids,
                    audio_input_positions,
                    self.runtime.codec_audio_range,
                )
            override_mask = _audio_override_mask(input_ids, audio_input_positions)
        output = self.tokens.embed(
            input_ids,
            self.text_embedding,
            input_modalities=input_modalities,
            validate=validate_input,
            audio_override_mask=override_mask,
        )
        if self.source_audio_encoder is not None and audio_input_positions is not None:
            output = self._overlay_audio_input(
                output,
                input_ids,
                audio_input_positions,
            )
        return output

    def _overlay_audio_input(
        self,
        output: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return output

        valid = positions.ge(0)
        safe_positions = positions.clamp(0, input_ids.size(1) - 1)
        selected_ids = input_ids.gather(1, safe_positions)
        audio_start, _ = self.layout.blocks["audio"]
        local_ids = (selected_ids - audio_start).clamp(
            0,
            self.tokens.semantic_audio_embedding.num_embeddings - 1,
        )
        features = self.tokens.audio_rows(local_ids)
        adapter = self.source_audio_encoder
        if adapter is None:
            raise RuntimeError("audio input adapter is unavailable.")
        projected = adapter(features, mask=valid)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)[:, None]
        rows = rows.expand_as(safe_positions)
        output[rows[valid], safe_positions[valid]] = projected[valid].to(
            dtype=output.dtype
        )
        return output


def _audio_override_mask(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    override = torch.zeros_like(input_ids, dtype=torch.bool)
    if positions.numel() == 0:
        return override
    valid = positions.ge(0)
    safe_positions = positions.clamp(0, input_ids.size(1) - 1)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)[:, None]
    rows = rows.expand_as(safe_positions)
    override[rows[valid], safe_positions[valid]] = True
    return override


def _validate_audio_input_positions(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    codec_audio_range: tuple[int, int],
) -> None:
    if positions.dim() != 2 or positions.size(0) != input_ids.size(0):
        raise ValueError("audio_input_positions must have shape [batch, frames].")
    if not is_signed_integer_dtype(positions.dtype):
        raise TypeError("audio_input_positions must use a signed integer dtype.")
    if positions.device != input_ids.device:
        raise ValueError("audio_input_positions must be on the input device.")
    if positions.numel() == 0:
        return
    if bool((positions < -1).any()) or bool((positions >= input_ids.size(1)).any()):
        raise ValueError(
            "audio_input_positions must use -1 padding or valid sequence positions."
        )
    ordered = positions.sort(dim=1).values
    repeated = ordered[:, 1:].eq(ordered[:, :-1]) & ordered[:, 1:].ge(0)
    if bool(repeated.any()):
        raise ValueError(
            "audio_input_positions must not repeat valid positions within a row."
        )
    valid = positions.ge(0)
    safe_positions = positions.clamp_min(0)
    selected_ids = input_ids.gather(1, safe_positions)
    codec_start, codec_end = codec_audio_range
    if bool((valid & (selected_ids < codec_start)).any()) or bool(
        (valid & (selected_ids >= codec_end)).any()
    ):
        raise ValueError(
            "audio_input_positions must point to visible codec audio payload tokens."
        )


def _prediction_readout_modality(
    prediction: PredictionModality | None,
    *,
    has_modality_readouts: bool,
) -> Modality | None:
    if prediction is None:
        return None
    if not isinstance(prediction, PredictionModality):
        raise TypeError("prediction must be a PredictionModality or None.")
    modalities = prediction.supervised_modalities()
    if len(modalities) == 1:
        return next(iter(modalities))
    if has_modality_readouts:
        raise ValueError(
            "mixed prediction modalities require a shared backbone readout; "
            "modality-specific readouts only support homogeneous text or audio batches."
        )
    return None


def _generation_readout_modality(
    modality: Modality | None,
    token_kind: str | None,
    *,
    has_modality_readouts: bool,
) -> Modality | None:
    if modality is not None:
        return modality
    if token_kind == Modality.TEXT.value:
        return Modality.TEXT
    if token_kind == Modality.AUDIO.value:
        return Modality.AUDIO
    if token_kind == "mixed" and has_modality_readouts:
        raise ValueError(
            "mixed generation tokens require a shared backbone readout; "
            "modality-specific readouts only support homogeneous text or audio generation."
        )
    return None
