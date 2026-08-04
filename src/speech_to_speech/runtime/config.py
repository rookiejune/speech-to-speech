from __future__ import annotations

import warnings
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Union

from anydataset.types import AudioView, Modality
import torch

from .._compat import StrEnum, auto
from .backbone import BackboneInitialization, BackboneType, validate_backbone_readout


_FLOW_METHODS = frozenset(
    {
        "adaptive_heun",
        "bosh3",
        "dopri5",
        "dopri8",
        "euler",
        "explicit_adams",
        "fehlberg2",
        "fixed_adams",
        "heun2",
        "heun3",
        "implicit_adams",
        "midpoint",
        "rk4",
        "scipy_solver",
    }
)

_GENERATOR_ARTIFACT_FIELD = "acoustic_generator_artifact"
_LEGACY_GENERATOR_ARTIFACT_FIELD = "semantic_codec_artifact"


def migrate_config_fields(fields: MutableMapping[str, Any]) -> None:
    """Migrate the legacy artifact field at the external config boundary."""
    if _LEGACY_GENERATOR_ARTIFACT_FIELD not in fields:
        return
    legacy = fields.pop(_LEGACY_GENERATOR_ARTIFACT_FIELD)
    current = fields.get(_GENERATOR_ARTIFACT_FIELD)
    if current is not None and legacy is not None and current != legacy:
        raise ValueError(
            "runtime config contains conflicting acoustic_generator_artifact and "
            "legacy semantic_codec_artifact values."
        )
    if current is None:
        fields[_GENERATOR_ARTIFACT_FIELD] = legacy
    warnings.warn(
        "runtime.semantic_codec_artifact is deprecated; use "
        "runtime.acoustic_generator_artifact.",
        FutureWarning,
        stacklevel=2,
    )


class AudioSequenceLayout(StrEnum):
    FLATTENED = auto()
    SEMANTIC = auto()


@dataclass(frozen=True)
class Config:
    codec: str = "longcat"
    backbone_type: BackboneType = BackboneType.HF_CAUSAL_LM
    backbone: str = "Qwen/Qwen3-0.6B"
    backbone_initialization: BackboneInitialization = BackboneInitialization.PRETRAINED
    backbone_trust_remote_code: bool = False
    backbone_chat_template: Optional[str] = None
    backbone_readout: str = "last_hidden_state"
    backbone_readouts: dict[str, str] = field(default_factory=dict)
    backbone_supports_cache_position: bool = True
    backbone_module: str = ""
    backbone_body: str = "base_model"
    audio_tokenizer: Optional[Union[str, Path]] = None
    acoustic_generator_artifact: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    gradient_checkpointing: bool = False
    flow_method: str = "midpoint"
    flow_nfe: int = 20
    flow_num_steps: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.backbone_type, BackboneType):
            raise TypeError("backbone_type must be a BackboneType.")
        if not isinstance(self.backbone_initialization, BackboneInitialization):
            raise TypeError("backbone_initialization must be a BackboneInitialization.")
        if not isinstance(self.backbone_trust_remote_code, bool):
            raise TypeError("backbone_trust_remote_code must be a bool.")
        _validate_optional_nonempty_string(
            self.backbone_chat_template,
            "backbone_chat_template",
        )
        validate_backbone_readout(self.backbone_readout)
        _validate_backbone_readouts(self.backbone_readouts)
        if not isinstance(self.backbone_supports_cache_position, bool):
            raise TypeError("backbone_supports_cache_position must be a bool.")
        _validate_path(self.backbone_module, "backbone_module")
        _validate_path(self.backbone_body, "backbone_body", allow_empty=False)
        if self.acoustic_generator_artifact is not None:
            if not self.acoustic_generator_artifact:
                raise ValueError("acoustic_generator_artifact must not be empty.")
            if self.audio_view is AudioView.BICODEC:
                raise ValueError(
                    "BiCodec cannot use runtime.acoustic_generator_artifact; "
                    "its global units belong to the token sequence."
                )
            if self.audio_view is not AudioView.LONGCAT:
                raise ValueError(
                    "acoustic generator artifacts currently require LongCat."
                )
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a bool.")
        if self.flow_method not in _FLOW_METHODS:
            raise ValueError(f"unsupported flow method: {self.flow_method}")
        if self.flow_nfe <= 0:
            raise ValueError(f"flow_nfe must be positive, got {self.flow_nfe}.")
        if self.flow_num_steps < 2:
            raise ValueError(
                "flow_num_steps must be at least 2, "
                f"got {self.flow_num_steps}."
            )

    @property
    def audio_view(self) -> AudioView:
        if self.codec == "stable_codec":
            return AudioView.STABLE
        try:
            return AudioView(self.codec)
        except ValueError as error:
            raise ValueError(f"unsupported codec: {self.codec}") from error


def config_for_local_rank(config: Config) -> Config:
    """Bind an unindexed CUDA device to the current distributed local rank."""
    device = None if config.device is None else torch.device(config.device)
    if device is not None and device.type == "cuda" and device.index is None:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return replace(config, device=None if device is None else str(device))

def validate_sequence_layout_config(config: Config, layout: AudioSequenceLayout) -> None:
    if not isinstance(layout, AudioSequenceLayout):
        raise TypeError("audio_sequence_layout must be an AudioSequenceLayout.")
    if config.audio_view is AudioView.BICODEC:
        if layout is not AudioSequenceLayout.FLATTENED:
            raise ValueError(
                "BiCodec uses one self-describing structured sequence layout; "
                "set audio_sequence_layout=flattened."
            )
        if config.acoustic_generator_artifact is not None:
            raise ValueError(
                "BiCodec global tokens are generated or reused in the token sequence "
                "and cannot use runtime.acoustic_generator_artifact."
            )
        return
    if layout is AudioSequenceLayout.FLATTENED:
        if config.audio_tokenizer is not None:
            raise ValueError(
                "audio_sequence_layout=flattened cannot use a BPE audio tokenizer "
                "for frame-code codecs."
            )
        if config.acoustic_generator_artifact is not None:
            raise ValueError(
                "runtime.acoustic_generator_artifact requires audio_sequence_layout=semantic."
            )




def _validate_path(value: object, name: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        if allow_empty:
            return
        raise ValueError(f"{name} must not be empty.")
    if value.startswith(".") or value.endswith(".") or ".." in value:
        raise ValueError(f"{name} must be a dotted attribute path.")
    for part in value.split("."):
        if not part.isidentifier():
            raise ValueError(f"{name} must contain identifier path components.")


def _validate_backbone_readouts(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("backbone_readouts must be a mapping.")
    for modality, readout in value.items():
        if not isinstance(modality, str):
            raise TypeError("backbone_readouts keys must be strings.")
        if modality not in {Modality.TEXT.value, Modality.AUDIO.value}:
            raise ValueError("backbone_readouts keys must be 'text' or 'audio'.")
        validate_backbone_readout(readout)


def _validate_optional_nonempty_string(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")

__all__ = [
    "AudioSequenceLayout",
    "Config",
    "config_for_local_rank",
    "migrate_config_fields",
    "validate_sequence_layout_config",
]
