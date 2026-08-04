from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from peft import inject_adapter_in_model
from peft.utils.other import cast_mixed_precision_params
from torch import nn

from ..runtime.backbone import BackboneBodyAdapter, BackboneEncoder
from ..runtime.types import Backbone, BackboneOutput, BackboneReadout
from ._checkpointing import enable_gradient_checkpointing
from ._helper import AdapterType, CastOutput, create_adapter
from .audio_input import (
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
from .embedding.audio import create_semantic_audio_embedding
from .protocol import TokenModelRuntime
from .token import TokenInterface
from .toy import create_toy_backbone

if TYPE_CHECKING:
    from .base import Config


def backbone(
    runtime: TokenModelRuntime,
    config: Config,
    text_vocab_size: int,
) -> Backbone:
    value = (
        runtime.backbone
        if config.toy is None
        else create_toy_backbone(config.toy, text_vocab_size)
    )
    module = cast(nn.Module, cast(object, value))
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


def text_embedding(backbone: Backbone, text_vocab_size: int) -> nn.Embedding:
    source = backbone.get_input_embeddings()
    if not isinstance(source, nn.Embedding):
        raise TypeError("backbone input embedding must be a torch.nn.Embedding.")
    if source.weight.size(0) < text_vocab_size:
        raise ValueError(
            "backbone input embedding does not cover the text layout vocabulary."
        )
    return source


def tokens(
    runtime: TokenModelRuntime,
    config: Config,
    text_embedding: nn.Embedding,
    hidden_size: int,
) -> TokenInterface:
    audio_embedding = create_semantic_audio_embedding(
        runtime,
        reference=text_embedding.weight,
        embedding_dim=hidden_size,
        fsq=config.fsq_embedding,
    ).to(device=text_embedding.weight.device, dtype=torch.float32)
    audio_feature_dim = audio_embedding.embedding_dim
    audio_projection = CastOutput(
        create_adapter(
            aligned_audio_adapter(
                config.semantic_audio_adapter,
                audio_feature_dim,
                hidden_size,
            ),
            audio_feature_dim,
            hidden_size,
        ).to(device=text_embedding.weight.device, dtype=torch.float32),
        reference=text_embedding.weight,
    )
    audio_head = audio_output_adapter(
        config.audio_output_adapter,
        audio_feature_dim,
        hidden_size,
        device=text_embedding.weight.device,
    )
    return TokenInterface(
        runtime.layout,
        audio_embedding=audio_embedding,
        audio_projection=audio_projection,
        audio_head=audio_head,
    )


def audio_input_adapter(
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


def audio_output_adapter(
    config: AudioOutputAdapterConfig,
    semantic_audio_dim: int,
    hidden_size: int,
    *,
    device: torch.device,
) -> AudioOutputAdapter:
    options = aligned_audio_output_adapter(
        config,
        hidden_size,
        semantic_audio_dim,
    )
    output_dim = (
        hidden_size
        if options.type is AudioOutputAdapterType.NONE
        else semantic_audio_dim
    )
    return create_audio_output_adapter(
        options,
        hidden_size,
        output_dim,
    ).to(
        device=device,
        dtype=torch.float32,
    )


def backbone_adapter(
    runtime: object,
    backbone: Backbone,
    *,
    prefer_runtime: bool,
) -> BackboneEncoder:
    adapter = getattr(runtime, "backbone_adapter", None) if prefer_runtime else None
    if adapter is not None:
        if isinstance(adapter, nn.Module):
            raise TypeError(
                "runtime backbone adapter must be a non-Module service; "
                "the model backbone is the only registered owner."
            )
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


def backbone_hidden_size(
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


def runtime_gradient_checkpointing(runtime: object) -> bool:
    value = getattr(runtime, "gradient_checkpointing", False)
    if not isinstance(value, bool):
        raise TypeError("runtime gradient_checkpointing must be a bool.")
    return value


def enable_interface_gradient_checkpointing(
    tokens: TokenInterface,
    source_audio_encoder: AudioInputTower | None,
) -> None:
    count = enable_gradient_checkpointing(tokens)
    if source_audio_encoder is not None:
        count += enable_gradient_checkpointing(source_audio_encoder)
    if count == 0:
        raise RuntimeError(
            "runtime.gradient_checkpointing requested custom adapter checkpointing, "
            "but no checkpointable adapter layers were found."
        )


def frame_span_lookup(runtime: TokenModelRuntime) -> torch.Tensor:
    spans = torch.as_tensor(
        runtime.audio_tokenizer.frame_spans(range(runtime.audio_tokenizer.vocab_size)),
        dtype=torch.long,
    )
    if spans.shape != (runtime.audio_tokenizer.vocab_size,):
        raise ValueError("audio token frame spans must cover the tokenizer vocabulary.")
    if bool((spans < 0).any()) or not bool((spans > 0).any()):
        raise ValueError("audio token frame spans must be non-negative and non-empty.")
    return spans


def aligned_audio_adapter(
    configured: AdapterType | None,
    in_features: int,
    out_features: int,
) -> AdapterType | None:
    """Use identity when dimensions match and config asks for a linear map."""
    if in_features == out_features and configured is AdapterType.LINEAR:
        return None
    return configured


def aligned_audio_output_adapter(
    configured: AudioOutputAdapterConfig,
    in_features: int,
    out_features: int,
) -> AudioOutputAdapterConfig:
    """Drop the default linear projector when audio space matches hidden space."""
    if in_features == out_features and configured.type is AudioOutputAdapterType.LINEAR:
        return AudioOutputAdapterConfig(
            type=AudioOutputAdapterType.NONE,
            layers=configured.layers,
            heads=configured.heads,
            ffn_ratio=configured.ffn_ratio,
            dropout=configured.dropout,
        )
    return configured
