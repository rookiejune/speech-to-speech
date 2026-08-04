from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
from torch import Tensor, nn

from .._compat import StrEnum, auto, register
from .adapter import MLPAdapter
from .tower import (
    GradientCheckpointingLayer,
    safe_transformer_mask,
    tower_fields,
    valid_mask,
    validate_tower_fields,
)


class AudioInputAdapterType(StrEnum):
    """Architecture used by the source-audio input tower."""

    NONE = auto()
    MLP = auto()
    TRANSFORMER = auto()


@dataclass(frozen=True)
class AudioInputAdapterConfig:
    """Configuration for a same-length source-audio input adapter.

    ``mask`` passed to :class:`AudioInputTower` uses ``True`` for active
    frames. ``causal`` controls whether the transformer variant can use future
    codec frames; it is ignored by pointwise adapters.
    """

    type: AudioInputAdapterType = AudioInputAdapterType.MLP
    layers: int = 2
    heads: int = 8
    ffn_ratio: float = 4.0
    dropout: float = 0.0
    causal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.type, AudioInputAdapterType):
            raise TypeError("audio input adapter type must be AudioInputAdapterType.")
        if not isinstance(self.causal, bool):
            raise TypeError("audio input adapter causal must be a bool.")
        validate_tower_fields(
            "audio input adapter",
            layers=self.layers,
            heads=self.heads,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
        )


def audio_input_options(
    config: Optional[Union[AudioInputAdapterConfig, Mapping[str, object]]],
) -> AudioInputAdapterConfig:
    """Normalize a config object or a plain mapping at the module boundary."""
    if config is None:
        return AudioInputAdapterConfig()
    if isinstance(config, AudioInputAdapterConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("audio input adapter config must be a config or mapping.")

    adapter_type = config.get("type", AudioInputAdapterType.MLP)
    return AudioInputAdapterConfig(
        type=AudioInputAdapterType(cast(str, adapter_type)),
        causal=cast(bool, config.get("causal", False)),
        **tower_fields(config),
    )


class AudioInputTower(GradientCheckpointingLayer):
    """Encode source-audio features without changing their frame length.

    The tower accepts ``[B, F, D]`` features and returns ``[B, F, H]``. A
    valid-frame mask can be supplied as ``[B, F]``; inactive frames are never
    used as transformer keys and are always zero in the returned tensor.
    Parameters are initialized in FP32 so this module can sit at the boundary
    of a lower-precision language-model backbone. Explicit model-wide dtype
    conversions are honored by casting inputs to the tower's current dtype.
    """

    def __init__(
        self,
        config: AudioInputAdapterConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("audio input adapter in_features must be positive.")
        if out_features <= 0:
            raise ValueError("audio input adapter out_features must be positive.")
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self._dtype_reference: Tensor
        register(
            self,
            "_dtype_reference",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.input_projection: nn.Module = nn.Identity()
        self.adapter: nn.Module

        if config.type is AudioInputAdapterType.MLP:
            self.adapter = MLPAdapter(in_features, out_features)
        elif config.type is AudioInputAdapterType.TRANSFORMER:
            if out_features % config.heads != 0:
                raise ValueError(
                    "audio input transformer out_features must be divisible by heads."
                )
            intermediate = max(1, int(round(config.ffn_ratio * out_features)))
            self.input_projection = nn.Linear(in_features, out_features)
            layer = nn.TransformerEncoderLayer(
                d_model=out_features,
                nhead=config.heads,
                dim_feedforward=intermediate,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.adapter = nn.TransformerEncoder(layer, num_layers=config.layers)
        elif config.type is AudioInputAdapterType.NONE:
            raise ValueError("audio input adapter type=none does not build a tower.")
        else:
            raise AssertionError(f"unsupported audio input adapter type: {config.type}")

        self.to(dtype=torch.float32)

    def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        if features.dim() != 3:
            raise ValueError(
                "audio input features must have shape [batch, frames, dim]."
            )
        if features.size(-1) != self.in_features:
            raise ValueError(
                "audio input feature dimension does not match the adapter config."
            )
        if features.size(1) <= 0:
            raise ValueError("audio input must contain at least one frame.")

        valid = valid_mask(features, mask, name="audio input")
        dtype = self._dtype_reference.dtype
        if self.config.type is AudioInputAdapterType.TRANSFORMER:
            values = features.to(dtype=dtype)
            values = values.masked_fill(~valid[..., None], 0)
            values = self.input_projection(values)
            key_padding_mask = ~safe_transformer_mask(valid)
            causal_mask = (
                torch.ones(
                    values.size(1),
                    values.size(1),
                    device=values.device,
                    dtype=torch.bool,
                ).triu(1)
                if self.config.causal
                else None
            )
            values = self.adapter(
                values,
                mask=causal_mask,
                src_key_padding_mask=key_padding_mask,
                is_causal=self.config.causal,
            )
            return values.masked_fill(~valid[..., None], 0)

        selected = self.adapter(features[valid].to(dtype=dtype))
        values = selected.new_zeros((*features.shape[:2], self.out_features))
        values[valid] = selected
        return values


def create_audio_input_adapter(
    config: Union[
        AudioInputAdapterConfig,
        AudioInputAdapterType,
        Mapping[str, object],
        str,
    ],
    in_features: int,
    out_features: int,
) -> AudioInputTower:
    """Create a source-audio input tower with FP32 parameters."""
    if isinstance(config, AudioInputAdapterConfig):
        options = config
    elif isinstance(config, (AudioInputAdapterType, str)):
        options = AudioInputAdapterConfig(type=AudioInputAdapterType(config))
    else:
        options = audio_input_options(config)
    return AudioInputTower(options, in_features, out_features)


__all__ = [
    "AudioInputAdapterConfig",
    "AudioInputAdapterType",
    "AudioInputTower",
    "audio_input_options",
    "create_audio_input_adapter",
]
