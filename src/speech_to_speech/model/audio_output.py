from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
from torch import Tensor, nn

from .._compat import StrEnum, auto
from ._checkpointing import GradientCheckpointingLayer
from ._helper import (
    MLPAdapter,
    safe_transformer_mask,
    tower_fields,
    valid_mask,
    validate_tower_fields,
)


class AudioOutputAdapterType(StrEnum):
    """Legacy audio pre-head adapter family.

    ``none`` / ``linear`` / ``mlp`` are pointwise special cases with no sequence
    mixing. ``transformer`` is a causal self-attention stack with its own KV cache.
    The default tied-head path uses ``none`` and computes logits against the
    effective audio input embedding table instead of adding an output-side tower.
    """

    NONE = auto()
    LINEAR = auto()
    MLP = auto()
    TRANSFORMER = auto()


@dataclass(frozen=True)
class AudioOutputAdapterConfig:
    """Configuration for the legacy semantic-audio pre-head adapter."""

    type: AudioOutputAdapterType = AudioOutputAdapterType.NONE
    layers: int = 2
    heads: int = 8
    ffn_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.type, AudioOutputAdapterType):
            raise TypeError("audio output adapter type must be AudioOutputAdapterType.")
        validate_tower_fields(
            "audio output adapter",
            layers=self.layers,
            heads=self.heads,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
        )


def audio_output_options(
    config: Optional[Union[AudioOutputAdapterConfig, Mapping[str, object]]],
) -> AudioOutputAdapterConfig:
    """Normalize a config object or plain mapping at the model boundary."""
    if config is None:
        return AudioOutputAdapterConfig()
    if isinstance(config, AudioOutputAdapterConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("audio output adapter config must be a config or mapping.")
    adapter_type = config.get("type", AudioOutputAdapterType.NONE)
    return AudioOutputAdapterConfig(
        type=AudioOutputAdapterType(cast(str, adapter_type)),
        **tower_fields(config),
    )


class AudioOutputAdapter(GradientCheckpointingLayer):
    """Optional legacy projection before the semantic-audio tied head.

    Pointwise variants ignore cache arguments. The transformer variant is causal
    and returns an independent past for autoregressive generation.
    """

    def __init__(
        self,
        config: AudioOutputAdapterConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("audio output adapter in_features must be positive.")
        if out_features <= 0:
            raise ValueError("audio output adapter out_features must be positive.")

        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.input_projection: nn.Module = nn.Identity()
        self.layers: nn.ModuleList | None = None
        if config.type is AudioOutputAdapterType.NONE:
            if in_features != out_features:
                raise ValueError(
                    "identity audio output adapter requires matching feature dimensions."
                )
            self.adapter: nn.Module = nn.Identity()
        elif config.type is AudioOutputAdapterType.LINEAR:
            self.adapter = nn.Linear(in_features, out_features)
        elif config.type is AudioOutputAdapterType.MLP:
            self.adapter = MLPAdapter(in_features, out_features)
        elif config.type is AudioOutputAdapterType.TRANSFORMER:
            if out_features % config.heads != 0:
                raise ValueError(
                    "audio output transformer out_features must be divisible by heads."
                )
            intermediate = max(1, int(round(config.ffn_ratio * out_features)))
            self.input_projection = nn.Linear(in_features, out_features)
            self.layers = nn.ModuleList(
                [
                    _CausalTransformerLayer(
                        out_features,
                        heads=config.heads,
                        intermediate=intermediate,
                        dropout=config.dropout,
                    )
                    for _ in range(config.layers)
                ]
            )
            self.adapter = nn.Identity()
        else:
            raise AssertionError(f"unsupported audio output adapter type: {config.type}")
        self.to(dtype=torch.float32)

    @property
    def is_pointwise(self) -> bool:
        return self.config.type is not AudioOutputAdapterType.TRANSFORMER

    def forward(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        if hidden_state.dim() < 2:
            raise ValueError("audio output hidden state must have at least two dimensions.")
        if hidden_state.size(-1) != self.in_features:
            raise ValueError(
                "audio output hidden dimension does not match the adapter config."
            )
        values = hidden_state.to(dtype=torch.float32)
        if self.is_pointwise:
            if past_key_values is not None:
                raise ValueError("pointwise audio output adapter does not use cache.")
            return self.adapter(values), None

        if values.dim() != 3:
            raise ValueError(
                "causal audio output transformer requires shape [batch, sequence, hidden]."
            )
        return self._forward_transformer(
            values,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def batch_select_past(
        self,
        past_key_values: object | None,
        indices: Tensor,
    ) -> object | None:
        if past_key_values is None:
            return None
        if self.is_pointwise:
            raise ValueError("pointwise audio output adapter does not use cache.")
        layers = cast(Sequence[tuple[Tensor, Tensor]], past_key_values)
        return tuple(
            (key.index_select(0, indices), value.index_select(0, indices))
            for key, value in layers
        )

    def _forward_transformer(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None,
        past_key_values: object | None,
        use_cache: bool,
    ) -> tuple[Tensor, object | None]:
        assert self.layers is not None
        values = self.input_projection(hidden_state)
        valid = valid_mask(values, attention_mask, name="audio output")
        past_layers = cast(
            Optional[Sequence[tuple[Tensor, Tensor]]],
            past_key_values,
        )
        if past_layers is not None and len(past_layers) != len(self.layers):
            raise ValueError("audio output cache depth does not match adapter layers.")
        next_past: list[tuple[Tensor, Tensor]] = []
        for index, layer in enumerate(self.layers):
            layer_past = None if past_layers is None else past_layers[index]
            values, layer_cache = layer(
                values,
                attention_mask=valid,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                if layer_cache is None:
                    raise RuntimeError("causal audio output layer did not return cache.")
                next_past.append(layer_cache)
        values = values.masked_fill(~valid[..., None], 0)
        return values, tuple(next_past) if use_cache else None


class _CausalTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        intermediate: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, hidden_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None,
        use_cache: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        residual = hidden_state
        query = self.norm1(hidden_state)
        if past_key_value is None:
            key = query
            value = query
            key_padding_mask = ~safe_transformer_mask(attention_mask)
            attn_mask = torch.ones(
                query.size(1),
                query.size(1),
                device=query.device,
                dtype=torch.bool,
            ).triu(1)
        else:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, query), dim=1)
            value = torch.cat((past_value, query), dim=1)
            past_len = past_key.size(1)
            full_mask = torch.cat(
                (
                    attention_mask.new_ones(attention_mask.size(0), past_len),
                    attention_mask,
                ),
                dim=1,
            )
            key_padding_mask = ~safe_transformer_mask(full_mask)
            attn_mask = None
        attended, _ = self.attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=attn_mask is not None,
        )
        hidden_state = residual + self.dropout(attended)
        hidden_state = hidden_state + self.ffn(self.norm2(hidden_state))
        cache = (key, value) if use_cache else None
        return hidden_state, cache


def create_audio_output_adapter(
    config: Union[
        AudioOutputAdapterConfig,
        AudioOutputAdapterType,
        Mapping[str, object],
        str,
    ],
    in_features: int,
    out_features: int,
) -> AudioOutputAdapter:
    """Create a causal-family semantic-audio output adapter with FP32 parameters."""
    if isinstance(config, AudioOutputAdapterConfig):
        options = config
    elif isinstance(config, (AudioOutputAdapterType, str)):
        options = AudioOutputAdapterConfig(type=AudioOutputAdapterType(config))
    else:
        options = audio_output_options(config)
    return AudioOutputAdapter(options, in_features, out_features)


__all__ = [
    "AudioOutputAdapter",
    "AudioOutputAdapterConfig",
    "AudioOutputAdapterType",
    "audio_output_options",
    "create_audio_output_adapter",
]
