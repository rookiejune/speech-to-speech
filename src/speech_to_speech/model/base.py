from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Embedding, Layout
from peft import LoraConfig, inject_adapter_in_model
from peft.utils.other import cast_mixed_precision_params
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from .._tensor import is_signed_integer_dtype
from ._generation import (
    GenerationOutput,
    GenerationStepResult,
    generate_sequence,
    generate_sequence_full,
)
from ..runtime.backbone import BackboneBodyAdapter, BackboneEncoder, BackboneOutputView
from ._checkpointing import enable_gradient_checkpointing
from ._head import VocabularyHeadMixin
from ._helper import (
    AdapterType,
    CastOutput,
    EmbeddingView,
    create_adapter,
    register,
)
from .audio_input import (
    AudioInputAdapterConfig,
    AudioInputAdapterType,
    AudioInputTower,
    create_audio_input_adapter,
)
from .audio_output import (
    AudioOutputAdapter,
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
    create_audio_output_adapter,
)
from .embedding.audio import (
    create_semantic_audio_embedding,
    require_semantic_audio_embedding,
)
from .protocol import TokenModelRuntime
from .toy import ToyConfig, create_toy_backbone
from ..prediction import PredictionModality
from ..runtime.types import Backbone, BackboneOutput, BackboneReadout


@dataclass
class Config:
    semantic_audio_adapter: Optional[AdapterType] = AdapterType.LINEAR
    audio_output_adapter: AudioOutputAdapterConfig = field(
        default_factory=AudioOutputAdapterConfig
    )
    audio_input_adapter: AudioInputAdapterConfig = field(
        default_factory=AudioInputAdapterConfig
    )
    toy: Optional[ToyConfig] = None
    lora: Optional[LoraConfig] = None


class Model(VocabularyHeadMixin, nn.Module):
    """Text and semantic-audio model; Flow/RVQ compositions subclass this entry."""

    audio_token_frame_spans: torch.Tensor
    audio_input_adapter: AudioInputTower | None
    token_embedding: Embedding
    audio_output_adapter: AudioOutputAdapter

    @property
    def lora_config(self) -> Optional[LoraConfig]:
        return self.config.lora

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
        self.backbone = _backbone(self.runtime, self.config, text_vocab_size)
        text_embedding = _text_embedding(self.backbone, text_vocab_size)
        self._backbone_body = _backbone_adapter(
            self.runtime,
            self.backbone,
            prefer_runtime=self.config.toy is None,
        )
        hidden_size = _backbone_hidden_size(
            self.runtime,
            self.backbone,
            prefer_runtime=self.config.toy is None,
        )
        self.token_embedding = _token_embedding(
            self.runtime,
            self.config,
            text_embedding,
            hidden_size,
        )
        backbone_weight = text_embedding.weight
        register(
            self,
            "audio_token_frame_spans",
            _frame_span_lookup(self.runtime).to(device=backbone_weight.device),
            persistent=False,
        )
        semantic_audio_weight = require_semantic_audio_embedding(
            self.token_embedding.embeddings["audio"],
            "semantic audio embedding",
        ).weight
        self.audio_input_adapter = _audio_input_adapter(
            self.config,
            semantic_audio_weight.size(1),
            hidden_size,
            device=backbone_weight.device,
        )
        self.audio_output_adapter = _audio_output_adapter(
            self.config.audio_output_adapter,
            semantic_audio_weight.size(1),
            hidden_size,
            device=backbone_weight.device,
        )
        if _runtime_gradient_checkpointing(self.runtime):
            _enable_external_gradient_checkpointing(self)

    def audio_output_adapter_batch_select(
        self,
        past_key_values: object | None,
        indices: torch.Tensor,
    ) -> object | None:
        return self.audio_output_adapter.batch_select_past(past_key_values, indices)

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
        token_kind: str | None = None,
        modality: Modality | None,
        past_key_values: Cache | None,
        use_cache: bool,
        audio_input_positions: torch.Tensor | None = None,
        audio_output_past: object | None = None,
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
            has_modality_readouts=self._backbone_body.has_modality_readouts,
        )
        backbone_output = self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            audio_input_positions=audio_input_positions,
            past_key_values=past_key_values,
            use_cache=use_cache,
            modality=readout_modality,
        )
        hidden_states = backbone_output.last_hidden_state
        last_hidden_state = hidden_states[:, -1:]
        step_mask = attention_mask[:, -1:] if attention_mask is not None else None
        audio_past = None
        if modality is not None:
            logits, audio_past = self.modality_logits(
                last_hidden_state,
                modality,
                attention_mask=step_mask,
                past_key_values=audio_output_past,
                use_cache=use_cache,
            )
        elif token_ids is not None:
            logits, audio_past = self.selected_logits(
                last_hidden_state,
                token_ids,
                token_kind=token_kind,
                attention_mask=step_mask,
                past_key_values=audio_output_past,
                use_cache=use_cache,
            )
        else:
            adapted, audio_past = self.project_audio_hidden(
                last_hidden_state,
                attention_mask=step_mask,
                past_key_values=audio_output_past,
                use_cache=use_cache,
            )
            logits = self.token_logits(
                last_hidden_state,
                audio_hidden_state=adapted,
            )
        return GenerationStepResult(
            logits=logits,
            past_key_values=backbone_output.past_key_values,
            audio_output_past=audio_past,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )

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
        embedding_blocks: frozenset[str] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        prediction: PredictionModality | None = None,
    ) -> torch.Tensor:
        """Encode one training batch without constructing vocabulary logits."""
        modality = _prediction_readout_modality(
            prediction,
            has_modality_readouts=self._backbone_body.has_modality_readouts,
        )
        return self._backbone_output(
            input_ids,
            attention_mask=attention_mask,
            audio_input_positions=audio_input_positions,
            embedding_blocks=embedding_blocks,
            validate_input=validate_input,
            validate_audio_input_positions=validate_audio_input_positions,
            use_cache=False,
            modality=modality,
        ).last_hidden_state

    def training_input_hints(
        self,
        input_ids: torch.Tensor,
        audio_input_positions: torch.Tensor | None,
    ) -> tuple[frozenset[str], bool] | None:
        """Validate CPU batch routing once before it is asynchronously transferred."""
        if input_ids.device.type != "cpu":
            return None
        blocks = self.token_embedding.selected_blocks(input_ids)
        if audio_input_positions is None:
            return blocks, True
        _validate_audio_input_positions(
            input_ids,
            audio_input_positions,
            self.runtime.codec_audio_range,
        )
        return blocks, True

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
        embedding_blocks: frozenset[str] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        modality: Modality | None = None,
    ) -> BackboneOutputView:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        inputs_embeds = self._input_embedding(
            input_ids,
            audio_input_positions,
            embedding_blocks=embedding_blocks,
            validate_input=validate_input,
            validate_audio_input_positions=validate_audio_input_positions,
        )

        return self._backbone_body.encode(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
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
        generated, _, _ = generate_sequence(
            self,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            audio_input_positions=audio_input_positions,
            stop_token_id=stop_token_id,
            generation_modality=generation_modality,
            allowed_token_ids=allowed_token_ids,
            token_kind=_allowed_token_kind(allowed_token_ids, self.layout),
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=False,
        )
        return generated

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
        return generate_sequence_full(
            self,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            audio_input_positions=audio_input_positions,
            stop_token_id=stop_token_id,
            generation_modality=generation_modality,
            allowed_token_ids=allowed_token_ids,
            token_kind=_allowed_token_kind(allowed_token_ids, self.layout),
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=False,
            collect_logprobs=True,
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
        generated, condition, frame_spans = generate_sequence(
            self,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            audio_input_positions=audio_input_positions,
            stop_token_id=self.runtime.eoa_token_id,
            generation_modality=Modality.AUDIO,
            allowed_token_ids=None,
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=True,
            min_new_tokens=1,
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

    def _input_embedding(
        self,
        input_ids: torch.Tensor,
        audio_input_positions: torch.Tensor | None = None,
        *,
        embedding_blocks: frozenset[str] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> torch.Tensor:
        output = self.token_embedding(
            input_ids,
            selected_blocks=embedding_blocks,
            validate=validate_input,
        )
        if self.audio_input_adapter is not None and audio_input_positions is not None:
            output = self._overlay_audio_input(
                output,
                input_ids,
                audio_input_positions,
                validate=validate_audio_input_positions,
            )
        return output

    def _overlay_audio_input(
        self,
        output: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        if validate:
            _validate_audio_input_positions(
                input_ids,
                positions,
                self.runtime.codec_audio_range,
            )
        if positions.numel() == 0:
            return output

        valid = positions.ge(0)
        safe_positions = positions.clamp(0, input_ids.size(1) - 1)
        selected_ids = input_ids.gather(1, safe_positions)
        audio_start, _ = self.layout.blocks["audio"]
        audio_embedding = require_semantic_audio_embedding(
            self.token_embedding.embeddings["audio"],
            "semantic audio embedding",
        )
        local_ids = (selected_ids - audio_start).clamp(
            0,
            audio_embedding.num_embeddings - 1,
        )
        features = audio_embedding(local_ids)
        adapter = self.audio_input_adapter
        if adapter is None:
            raise RuntimeError("audio input adapter is unavailable.")
        projected = adapter(features, mask=valid)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)[:, None]
        rows = rows.expand_as(safe_positions)
        output[rows[valid], safe_positions[valid]] = projected[valid].to(
            dtype=output.dtype
        )
        return output


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
    valid = positions.ge(0)
    safe_positions = positions.clamp(0, input_ids.size(1) - 1)
    selected_ids = input_ids.gather(1, safe_positions)
    codec_start, codec_end = codec_audio_range
    if bool((valid & (selected_ids < codec_start)).any()) or bool(
        (valid & (selected_ids >= codec_end)).any()
    ):
        raise ValueError(
            "audio_input_positions must point to visible codec audio payload tokens."
        )


def _backbone(
    runtime: TokenModelRuntime,
    config: Config,
    text_vocab_size: int,
) -> Backbone:
    backbone = (
        runtime.backbone
        if config.toy is None
        else create_toy_backbone(config.toy, text_vocab_size)
    )
    module = cast(nn.Module, cast(object, backbone))
    if config.lora is not None:
        adapted = inject_adapter_in_model(
            config.lora,
            module,
            adapter_name="speech",
        )
        if adapted is not module:
            raise RuntimeError("PEFT adapter injection must preserve the backbone object.")
        reference = next(module.parameters(), None)
        if reference is not None and reference.dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            cast_mixed_precision_params(module, reference.dtype)
    return cast(Backbone, cast(object, module))


def _text_embedding(backbone: Backbone, text_vocab_size: int) -> nn.Embedding:
    text_source = backbone.get_input_embeddings()
    if text_source.weight.size(0) < text_vocab_size:
        raise ValueError(
            "backbone input embedding does not cover the text layout vocabulary."
        )
    if text_source.num_embeddings != text_vocab_size:
        # Layout text block may be a prefix of a larger backbone table.
        text_embedding = nn.Embedding.from_pretrained(
            text_source.weight[:text_vocab_size].detach().clone(),
            freeze=False,
        )
    else:
        text_embedding = text_source
    # Backbone keeps a non-Module view; idspace owns the real embedding once.
    _install_text_embedding_view(backbone, EmbeddingView(text_embedding))
    return text_embedding


def _token_embedding(
    runtime: TokenModelRuntime,
    config: Config,
    text_embedding: nn.Embedding,
    hidden_size: int,
) -> Embedding:
    audio_embedding = create_semantic_audio_embedding(
        runtime,
        reference=text_embedding.weight,
        embedding_dim=hidden_size,
    ).to(device=text_embedding.weight.device, dtype=torch.float32)
    audio_weight = cast(torch.Tensor, audio_embedding.weight)
    audio_feature_dim = int(audio_weight.shape[-1])
    audio_adapter = CastOutput(
        create_adapter(
            _aligned_audio_adapter(
                config.semantic_audio_adapter,
                audio_feature_dim,
                hidden_size,
            ),
            audio_feature_dim,
            hidden_size,
        ).to(device=text_embedding.weight.device, dtype=torch.float32),
        dtype=text_embedding.weight.dtype,
    )
    return Embedding(
        runtime.layout,
        text=text_embedding,
        audio=audio_embedding,  # pyright: ignore[reportArgumentType]
        adapters={"audio": audio_adapter},
    )


def _audio_input_adapter(
    config: Config,
    semantic_audio_dim: int,
    hidden_size: int,
    *,
    device: torch.device,
) -> AudioInputTower | None:
    if config.audio_input_adapter.type is AudioInputAdapterType.NONE:
        return None
    return create_audio_input_adapter(
        config.audio_input_adapter,
        semantic_audio_dim,
        hidden_size,
    ).to(device=device)


def _audio_output_adapter(
    config: AudioOutputAdapterConfig,
    semantic_audio_dim: int,
    hidden_size: int,
    *,
    device: torch.device,
) -> AudioOutputAdapter:
    audio_output_adapter = _aligned_audio_output_adapter(
        config,
        hidden_size,
        semantic_audio_dim,
    )
    audio_output_dim = (
        hidden_size
        if audio_output_adapter.type is AudioOutputAdapterType.NONE
        else semantic_audio_dim
    )
    return create_audio_output_adapter(
        audio_output_adapter,
        hidden_size,
        audio_output_dim,
    ).to(
        device=device,
        dtype=torch.float32,
    )


def _install_text_embedding_view(backbone: Backbone, view: EmbeddingView) -> None:
    """Point backbone text lookup at a non-owning view of the idspace table.

    HuggingFace keeps ``embed_tokens`` in ``_modules``, so assigning a plain view
    through ``set_input_embeddings`` fails. Drop the Module registration first,
    then store the view as a normal attribute.
    """
    current = backbone.get_input_embeddings()
    if _replace_registered_embedding(cast(nn.Module, backbone), current, view, seen=set()):
        return
    if getattr(backbone, "input_embeddings", None) is current:
        backbone.input_embeddings = view  # type: ignore[attr-defined]
        if cast(object, backbone.get_input_embeddings()) is view:
            return
        raise RuntimeError(
            "backbone input_embeddings replacement did not update "
            "get_input_embeddings()."
        )
    raise RuntimeError(
        "backbone must expose a replaceable input embedding attribute so the "
        "shared text table is referenced without dual Module ownership."
    )


def _backbone_adapter(
    runtime: object,
    backbone: Backbone,
    *,
    prefer_runtime: bool,
) -> BackboneEncoder:
    adapter = getattr(runtime, "backbone_adapter", None) if prefer_runtime else None
    if adapter is not None:
        return cast(BackboneEncoder, adapter)
    body = getattr(backbone, "base_model", backbone)
    if not callable(body):
        raise TypeError("backbone fallback body must be callable.")
    return cast(
        BackboneEncoder,
        BackboneBodyAdapter(
            cast(Callable[..., BackboneOutput], body),
            readout=BackboneReadout(
                cast(str, getattr(runtime, "backbone_readout", "last_hidden_state"))
            ),
            supports_cache_position=bool(
                getattr(runtime, "backbone_supports_cache_position", True)
            ),
        ),
    )


def _backbone_hidden_size(
    runtime: object,
    backbone: Backbone,
    *,
    prefer_runtime: bool,
) -> int:
    adapter = getattr(runtime, "backbone_adapter", None) if prefer_runtime else None
    value = getattr(adapter, "hidden_size", None) if adapter is not None else None
    if isinstance(value, bool):
        raise TypeError("backbone hidden size must be an integer.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("backbone hidden size must be positive.")
        return value
    config_value = getattr(backbone.config, "hidden_size", None)
    if isinstance(config_value, bool) or not isinstance(config_value, int):
        raise AttributeError("backbone config does not expose hidden_size.")
    if config_value <= 0:
        raise ValueError("backbone hidden size must be positive.")
    return config_value


def _replace_registered_embedding(
    parent: nn.Module,
    current: nn.Module,
    view: EmbeddingView,
    *,
    seen: set[int],
) -> bool:
    parent_id = id(parent)
    if parent_id in seen:
        return False
    seen.add(parent_id)
    for name, child in list(parent._modules.items()):
        if child is None:
            continue
        if child is current:
            parent._modules.pop(name)
            setattr(parent, name, view)
            return True
        if _replace_registered_embedding(child, current, view, seen=seen):
            return True
    return False


def _runtime_gradient_checkpointing(runtime: object) -> bool:
    config = getattr(runtime, "config", None)
    return bool(getattr(config, "gradient_checkpointing", False))


def _enable_external_gradient_checkpointing(model: Model) -> None:
    count = enable_gradient_checkpointing(model.token_embedding)
    if model.audio_input_adapter is not None:
        count += enable_gradient_checkpointing(model.audio_input_adapter)
    if model.audio_output_adapter.config.type is not AudioOutputAdapterType.NONE:
        count += enable_gradient_checkpointing(model.audio_output_adapter)
    if count == 0:
        raise RuntimeError(
            "runtime.gradient_checkpointing requested custom adapter checkpointing, "
            "but no checkpointable adapter layers were found."
        )


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


def _aligned_audio_adapter(
    configured: AdapterType | None,
    in_features: int,
    out_features: int,
) -> AdapterType | None:
    """Use identity when dims already match and config asks for a plain linear map."""
    if in_features == out_features and configured is AdapterType.LINEAR:
        return None
    return configured


def _aligned_audio_output_adapter(
    configured: AudioOutputAdapterConfig,
    in_features: int,
    out_features: int,
) -> AudioOutputAdapterConfig:
    """Drop the default linear projector when audio space already matches hidden."""
    if in_features == out_features and configured.type is AudioOutputAdapterType.LINEAR:
        return AudioOutputAdapterConfig(
            type=AudioOutputAdapterType.NONE,
            layers=configured.layers,
            heads=configured.heads,
            ffn_ratio=configured.ffn_ratio,
            dropout=configured.dropout,
        )
    return configured


def _allowed_token_kind(
    token_ids: Sequence[int] | torch.Tensor | None,
    layout: Layout,
) -> str | None:
    """Classify CPU constrained ids once for the per-token output-head route."""
    if token_ids is None:
        return None
    if isinstance(token_ids, torch.Tensor) and token_ids.device.type != "cpu":
        return None
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if ids.dim() != 1 or ids.numel() == 0:
        return None
    text_start, text_end = layout.blocks["text"]
    audio_start, audio_end = layout.blocks["audio"]
    text = ids.ge(text_start) & ids.lt(text_end)
    audio = ids.ge(audio_start) & ids.lt(audio_end)
    if not bool((text | audio).all()):
        return None
    if bool(text.all()):
        return "text"
    if bool(audio.all()):
        return "audio"
    return "mixed"


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
