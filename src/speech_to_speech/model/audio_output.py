from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
from torch import Tensor, nn

from .._compat import StrEnum, auto
from .adapter import MLPAdapter


class AudioOutputAdapterType(StrEnum):
    """Architecture used by the semantic-audio output adapter."""

    NONE = auto()
    LINEAR = auto()
    MLP = auto()


@dataclass(frozen=True)
class AudioOutputAdapterConfig:
    """Configuration for a pointwise semantic-audio output adapter.

    The adapter preserves the last hidden dimension contract, so it works for
    both teacher forcing and the one-token autoregressive generation step.
    """

    type: AudioOutputAdapterType = AudioOutputAdapterType.LINEAR

    def __post_init__(self) -> None:
        if not isinstance(self.type, AudioOutputAdapterType):
            raise TypeError("audio output adapter type must be AudioOutputAdapterType.")


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
    adapter_type = config.get("type", AudioOutputAdapterType.LINEAR)
    return AudioOutputAdapterConfig(
        type=AudioOutputAdapterType(cast(str, adapter_type)),
    )


class AudioOutputAdapter(nn.Module):
    """Project backbone states into the semantic-audio feature space.

    This adapter is deliberately pointwise. The generation head receives one
    hidden state at a time after the KV-cache step, so a sequence-mixing
    transformer here would require a separate causal cache contract.
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
        else:
            raise AssertionError(f"unsupported audio output adapter type: {config.type}")
        self.to(dtype=torch.float32)

    def forward(self, hidden_state: Tensor) -> Tensor:
        if hidden_state.dim() < 2:
            raise ValueError("audio output hidden state must have at least two dimensions.")
        if hidden_state.size(-1) != self.in_features:
            raise ValueError(
                "audio output hidden dimension does not match the adapter config."
            )
        return self.adapter(hidden_state.to(dtype=torch.float32))


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
    """Create an explicit pointwise semantic-audio output adapter."""
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
