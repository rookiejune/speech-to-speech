from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

from ..._compat import StrEnum, auto
from ..types import validate_backbone_readout


class BackboneInitialization(StrEnum):
    PRETRAINED = auto()
    RANDOM = auto()


class BackboneType(StrEnum):
    HF_CAUSAL_LM = auto()
    KIMI_AUDIO = auto()
    QWEN2_5_OMNI_TEXT = auto()


@dataclass(frozen=True)
class AdapterConfig:
    type: BackboneType = BackboneType.HF_CAUSAL_LM
    path: str = "Qwen/Qwen3-0.6B"
    initialization: BackboneInitialization = BackboneInitialization.PRETRAINED
    trust_remote_code: bool = False
    chat_template: Optional[str] = None
    readout: str = "last_hidden_state"
    readouts: Mapping[str, str] = field(default_factory=dict)
    supports_cache_position: bool = True
    module: str = ""
    body: str = "base_model"
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.type, BackboneType):
            raise TypeError("backbone adapter type must be a BackboneType.")
        if not isinstance(self.path, str):
            raise TypeError("backbone adapter path must be a string.")
        if not self.path:
            raise ValueError("backbone adapter path must not be empty.")
        if not isinstance(self.initialization, BackboneInitialization):
            raise TypeError(
                "backbone adapter initialization must be a BackboneInitialization."
            )
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("backbone adapter trust_remote_code must be a bool.")
        _optional_nonempty_string(
            self.chat_template,
            "backbone adapter chat_template",
        )
        validate_backbone_readout(self.readout)
        _readouts(self.readouts, "backbone adapter readouts")
        if not isinstance(self.supports_cache_position, bool):
            raise TypeError("backbone adapter supports_cache_position must be a bool.")
        _path(self.module, "backbone adapter module")
        _path(self.body, "backbone adapter body", allow_empty=False)
        _optional_string(self.device, "backbone adapter device")
        _optional_string(self.dtype, "backbone adapter dtype")
        _optional_string(
            self.attn_implementation,
            "backbone adapter attention implementation",
        )
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("backbone adapter gradient_checkpointing must be a bool.")


def _optional_string(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")


def _optional_nonempty_string(value: object, name: str) -> None:
    _optional_string(value, name)
    if value == "":
        raise ValueError(f"{name} must not be empty.")


def _readouts(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for modality, readout in value.items():
        if not isinstance(modality, str):
            raise TypeError(f"{name} keys must be strings.")
        if modality not in {"text", "audio"}:
            raise ValueError(f"{name} keys must be 'text' or 'audio'.")
        validate_backbone_readout(readout)


def _path(value: object, name: str, *, allow_empty: bool = True) -> None:
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


__all__ = [
    "AdapterConfig",
    "BackboneInitialization",
    "BackboneType",
]
