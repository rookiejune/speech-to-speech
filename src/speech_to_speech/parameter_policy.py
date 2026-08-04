from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from torch import nn

from ._compat import StrEnum, auto
from .model.ctc import CTCConfig, CTCRoute


class ParameterGroup(StrEnum):
    BACKBONE = auto()
    BACKBONE_ADAPTER = auto()
    SEMANTIC_AUDIO_EMBEDDING = auto()
    SEMANTIC_AUDIO_ADAPTER = auto()
    AUDIO_INPUT_ADAPTER = auto()
    AUDIO_OUTPUT = auto()
    ALIGNMENT_DECODER = auto()
    ACOUSTIC_DECODER = auto()


class ParameterPolicyName(StrEnum):
    FULL = auto()
    LORA = auto()
    SPEECH_INTERFACE = auto()
    SEMANTIC_ONLY = auto()
    ACOUSTIC_ONLY = auto()
    SPEECH_INTERFACE_TOP_THIRD = auto()


class ParameterPolicyModel(Protocol):
    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterable[tuple[str, nn.Parameter]]: ...


@runtime_checkable
class _InputEmbeddingOwner(Protocol):
    def get_input_embeddings(self) -> nn.Module: ...


@dataclass(frozen=True)
class ParameterPolicySpec:
    name: ParameterPolicyName
    trainable_groups: frozenset[ParameterGroup]
    frozen_groups: frozenset[ParameterGroup]
    backbone_top_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, ParameterPolicyName):
            raise TypeError("parameter policy name must be a ParameterPolicyName.")
        if not self.trainable_groups:
            raise ValueError("parameter policy must train at least one group.")
        groups = self.trainable_groups | self.frozen_groups
        if groups != frozenset(ParameterGroup):
            raise ValueError(
                "parameter policy trainable and frozen groups must cover every "
                "parameter group."
            )
        if self.trainable_groups & self.frozen_groups:
            raise ValueError(
                "parameter policy trainable and frozen groups must be disjoint."
            )
        if any(not isinstance(group, ParameterGroup) for group in groups):
            raise TypeError("parameter policy groups must be ParameterGroup values.")
        if self.backbone_top_fraction is not None:
            value = self.backbone_top_fraction
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("backbone_top_fraction must be in [0, 1].")


@dataclass
class ParameterPolicyConfig:
    name: ParameterPolicyName = ParameterPolicyName.FULL
    trainable_groups: list[ParameterGroup] = field(
        default_factory=lambda: list(ParameterGroup)
    )
    frozen_groups: list[ParameterGroup] = field(default_factory=list)
    backbone_top_fraction: Optional[float] = 1.0

    def __post_init__(self) -> None:
        self.spec()

    def spec(self) -> ParameterPolicySpec:
        spec = ParameterPolicySpec(
            self.name,
            frozenset(self.trainable_groups),
            frozenset(self.frozen_groups),
            backbone_top_fraction=self.backbone_top_fraction,
        )
        preset = PARAMETER_POLICY_SPECS.get(self.name)
        if preset is not None and spec != preset:
            raise ValueError(
                "parameter_policy.name must match its trainable_groups, "
                "frozen_groups, and backbone_top_fraction preset."
            )
        return spec


SPEECH_INTERFACE_GROUPS = frozenset(
    {
        ParameterGroup.SEMANTIC_AUDIO_EMBEDDING,
        ParameterGroup.SEMANTIC_AUDIO_ADAPTER,
        ParameterGroup.AUDIO_INPUT_ADAPTER,
        ParameterGroup.AUDIO_OUTPUT,
        ParameterGroup.ALIGNMENT_DECODER,
        ParameterGroup.ACOUSTIC_DECODER,
    }
)

SEMANTIC_GROUPS = frozenset(
    {
        ParameterGroup.SEMANTIC_AUDIO_EMBEDDING,
        ParameterGroup.SEMANTIC_AUDIO_ADAPTER,
        ParameterGroup.AUDIO_INPUT_ADAPTER,
        ParameterGroup.AUDIO_OUTPUT,
        ParameterGroup.ALIGNMENT_DECODER,
    }
)

ACOUSTIC_GROUPS = frozenset(
    {
        ParameterGroup.ACOUSTIC_DECODER,
    }
)

PARAMETER_POLICY_SPECS: Mapping[ParameterPolicyName, ParameterPolicySpec] = {
    ParameterPolicyName.FULL: ParameterPolicySpec(
        ParameterPolicyName.FULL,
        frozenset(ParameterGroup),
        frozenset(),
        backbone_top_fraction=1.0,
    ),
    ParameterPolicyName.LORA: ParameterPolicySpec(
        ParameterPolicyName.LORA,
        SPEECH_INTERFACE_GROUPS | {ParameterGroup.BACKBONE_ADAPTER},
        frozenset(ParameterGroup)
        - SPEECH_INTERFACE_GROUPS
        - {ParameterGroup.BACKBONE_ADAPTER},
        backbone_top_fraction=0.0,
    ),
    ParameterPolicyName.SPEECH_INTERFACE: ParameterPolicySpec(
        ParameterPolicyName.SPEECH_INTERFACE,
        SPEECH_INTERFACE_GROUPS,
        frozenset(ParameterGroup) - SPEECH_INTERFACE_GROUPS,
        backbone_top_fraction=0.0,
    ),
    ParameterPolicyName.SEMANTIC_ONLY: ParameterPolicySpec(
        ParameterPolicyName.SEMANTIC_ONLY,
        SEMANTIC_GROUPS,
        frozenset(ParameterGroup) - SEMANTIC_GROUPS,
        backbone_top_fraction=0.0,
    ),
    ParameterPolicyName.ACOUSTIC_ONLY: ParameterPolicySpec(
        ParameterPolicyName.ACOUSTIC_ONLY,
        ACOUSTIC_GROUPS,
        frozenset(ParameterGroup) - ACOUSTIC_GROUPS,
        backbone_top_fraction=0.0,
    ),
    ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD: ParameterPolicySpec(
        ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD,
        SPEECH_INTERFACE_GROUPS
        | {ParameterGroup.BACKBONE, ParameterGroup.BACKBONE_ADAPTER},
        frozenset(),
        backbone_top_fraction=1.0 / 3.0,
    ),
}

def default_parameter_policy_config(
    name: ParameterPolicyName,
) -> ParameterPolicyConfig:
    spec = PARAMETER_POLICY_SPECS[name]
    return ParameterPolicyConfig(
        name=spec.name,
        trainable_groups=list(spec.trainable_groups),
        frozen_groups=list(spec.frozen_groups),
        backbone_top_fraction=spec.backbone_top_fraction,
    )


# Backbone bodies are registered directly under ``model.backbone``. Kimi's
# MIMO branch follows the same layer-index contract as the main branch.
_LAYER_PATTERN = re.compile(r"^backbone\.(?:layers|mimo_layers)\.(\d+)\.")
_FINAL_NORM_PATTERN = re.compile(r"^backbone\.norm\.")


@dataclass(frozen=True)
class ParameterPolicyTrainability:
    spec: ParameterPolicySpec

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ParameterPolicySpec):
            raise TypeError("parameter policy trainability requires a spec.")

    def __call__(
        self,
        module: nn.Module,
        name: str,
        parameter: nn.Parameter,
    ) -> bool:
        group = _policy_group(name, parameter, self.spec)
        if _structurally_frozen(name, module, parameter):
            return False
        trainable = (
            _peft_trainable(name, parameter, self.spec)
            if self.spec.name is ParameterPolicyName.LORA
            and name.startswith("backbone.")
            else group in self.spec.trainable_groups
        )
        if group is ParameterGroup.BACKBONE and trainable:
            return _backbone_trainable(
                name,
                module,
                self.spec.backbone_top_fraction,
            )
        return trainable


def _policy_group(
    name: str,
    parameter: nn.Parameter,
    spec: ParameterPolicySpec,
) -> ParameterGroup:
    if _peft_trainable(name, parameter, spec):
        return ParameterGroup.BACKBONE_ADAPTER
    return parameter_group(name)


def _peft_trainable(
    name: str,
    parameter: nn.Parameter,
    spec: ParameterPolicySpec,
) -> bool:
    return (
        spec.name is ParameterPolicyName.LORA
        and name.startswith("backbone.")
        and parameter.requires_grad
    )


def parameter_group(name: str) -> ParameterGroup:
    legacy = next(
        (
            prefix
            for prefix in (
                "token_embedding.",
                "audio_input_adapter.",
                "audio_output_adapter.",
            )
            if name.startswith(prefix)
        ),
        None,
    )
    if legacy is not None:
        raise ValueError(
            f"parameter {name!r} uses legacy model ownership prefix {legacy!r}."
        )
    if name.startswith("backbone."):
        if ".lora_A." in name or ".lora_B." in name:
            return ParameterGroup.BACKBONE_ADAPTER
        return ParameterGroup.BACKBONE
    if name.startswith("tokens.audio_embedding."):
        return ParameterGroup.SEMANTIC_AUDIO_EMBEDDING
    if name.startswith("tokens.audio_projection."):
        return ParameterGroup.SEMANTIC_AUDIO_ADAPTER
    if name.startswith("source_audio_encoder."):
        return ParameterGroup.AUDIO_INPUT_ADAPTER
    if name.startswith("tokens.audio_head."):
        return ParameterGroup.AUDIO_OUTPUT
    if name.startswith(("ctc_decoders.source.", "ctc_decoders.target.")):
        return ParameterGroup.ALIGNMENT_DECODER
    if (
        name.startswith("acoustic_condition.")
        or name.startswith("acoustic_decoder.")
        or name.startswith("acoustic_flow.")
    ):
        return ParameterGroup.ACOUSTIC_DECODER
    raise ValueError(f"parameter {name!r} does not belong to a parameter group.")


def _backbone_trainable(
    name: str, model: ParameterPolicyModel, top_fraction: float | None
) -> bool:
    if top_fraction is None or top_fraction >= 1:
        return True
    if top_fraction <= 0:
        return False
    match = _LAYER_PATTERN.match(name)
    if match is None:
        return _is_final_norm(name)
    layer = int(match.group(1))
    layers = _num_layers(model, layer + 1)
    trainable_layers = max(1, math.ceil(layers * top_fraction))
    return layer >= layers - trainable_layers


def _is_final_norm(name: str) -> bool:
    return _FINAL_NORM_PATTERN.match(name) is not None


def _num_layers(model: ParameterPolicyModel, minimum: int) -> int:
    backbone = getattr(model, "backbone", None)
    config = None if backbone is None else getattr(backbone, "config", None)
    value = None if config is None else getattr(config, "num_hidden_layers", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return minimum
    return value


def _structurally_frozen(
    name: str,
    model: ParameterPolicyModel,
    parameter: nn.Parameter,
) -> bool:
    if _is_text_embedding(model, parameter):
        return True
    if _inactive_ctc_decoder(name, model):
        return True
    if name.startswith("acoustic_decoder.decoder.embed_tokens."):
        return True
    decoder = getattr(model, "acoustic_decoder", None)
    last = _last_module_index(getattr(decoder, "codebook_embeddings", None))
    if last is not None and name.startswith(
        f"acoustic_decoder.codebook_embeddings.{last}."
    ):
        return True
    if last is not None and name.startswith(
        f"acoustic_decoder.embedding_projections.{last}."
    ):
        return True
    return False


def _inactive_ctc_decoder(name: str, model: ParameterPolicyModel) -> bool:
    route: CTCRoute | None = None
    if name.startswith("ctc_decoders.source."):
        route = CTCRoute.SOURCE
    elif name.startswith("ctc_decoders.target."):
        route = CTCRoute.TARGET
    if route is None:
        return False
    model_config = getattr(model, "config", None)
    config = getattr(model_config, "ctc", None)
    return isinstance(config, CTCConfig) and not config.route(route).enabled


def _is_text_embedding(
    model: ParameterPolicyModel,
    parameter: nn.Parameter,
) -> bool:
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, _InputEmbeddingOwner):
        return False
    embedding = backbone.get_input_embeddings()
    return isinstance(embedding, nn.Embedding) and parameter is embedding.weight


def _last_module_index(value: object) -> int | None:
    if not isinstance(value, nn.ModuleList):
        return None
    if len(value) == 0:
        return None
    return len(value) - 1


__all__ = [
    "ACOUSTIC_GROUPS",
    "PARAMETER_POLICY_SPECS",
    "SEMANTIC_GROUPS",
    "SPEECH_INTERFACE_GROUPS",
    "ParameterGroup",
    "ParameterPolicyConfig",
    "ParameterPolicyName",
    "ParameterPolicySpec",
    "ParameterPolicyTrainability",
    "default_parameter_policy_config",
    "parameter_group",
]
