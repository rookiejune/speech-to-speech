from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, TypeVar, TypedDict, cast

import torch
from torch import nn


ModuleT = TypeVar("ModuleT", bound=nn.Module)
_ADAPTER_NAME = "speech"
_BACKEND = "huggingface-peft"
_BIAS = "none"
_GRAMMAR = "peft-lora-v1"


class CheckpointPayload(TypedDict):
    grammar: str
    backend: str
    adapter_name: str
    bias: str
    enabled: bool
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    use_rslora: bool


@dataclass
class LoraConfig:
    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_rslora: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("LoRA enabled must be a boolean.")
        _positive_int(self.rank, "LoRA rank")
        _positive_int(self.alpha, "LoRA alpha")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (float, int))
        ):
            raise TypeError("LoRA dropout must be a number.")
        if not math.isfinite(float(self.dropout)) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be finite and in [0, 1).")
        if not self.target_modules:
            raise ValueError("LoRA target_modules must not be empty.")
        if any(not isinstance(name, str) or not name for name in self.target_modules):
            raise TypeError("LoRA target_modules must contain non-empty strings.")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target_modules must not contain duplicates.")
        if not isinstance(self.use_rslora, bool):
            raise TypeError("LoRA use_rslora must be a boolean.")


class LoraModel(Protocol):
    @property
    def lora_config(self) -> LoraConfig: ...


def inject(backbone: ModuleT, config: LoraConfig) -> ModuleT:
    if not config.enabled:
        return backbone
    peft = _peft()
    peft_config = peft.LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=float(config.dropout),
        target_modules=list(config.target_modules),
        bias=_BIAS,
        use_rslora=config.use_rslora,
    )
    adapted = peft.inject_adapter_in_model(
        peft_config,
        backbone,
        adapter_name=_ADAPTER_NAME,
    )
    if adapted is not backbone:
        raise RuntimeError("PEFT adapter injection must preserve the backbone object.")
    reference = next(backbone.parameters(), None)
    if reference is not None and reference.dtype in {torch.float16, torch.bfloat16}:
        peft.cast_mixed_precision_params(backbone, reference.dtype)
    return backbone


def checkpoint_payload(config: LoraConfig) -> CheckpointPayload:
    return {
        "grammar": _GRAMMAR,
        "backend": _BACKEND,
        "adapter_name": _ADAPTER_NAME,
        "bias": _BIAS,
        "enabled": config.enabled,
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": float(config.dropout),
        "target_modules": sorted(config.target_modules),
        "use_rslora": config.use_rslora,
    }


def _peft() -> Any:
    try:
        module = cast(Any, import_module("peft"))
    except ModuleNotFoundError as error:
        if error.name not in {"peft", "accelerate"}:
            raise
        raise RuntimeError(
            "model.lora requires Hugging Face PEFT; install peft and accelerate."
        ) from error
    required = (
        "LoraConfig",
        "inject_adapter_in_model",
        "cast_mixed_precision_params",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            "model.lora requires a compatible Hugging Face PEFT release; "
            f"missing APIs: {', '.join(missing)}."
        )
    return module


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "CheckpointPayload",
    "LoraConfig",
    "LoraModel",
    "checkpoint_payload",
    "inject",
]
