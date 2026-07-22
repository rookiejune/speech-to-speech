from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Optional, Protocol

from torch import nn

from ._compat import StrEnum, auto


class ParameterGroup(StrEnum):
    BACKBONE = auto()
    SEMANTIC_AUDIO_EMBEDDING = auto()
    SEMANTIC_AUDIO_ADAPTER = auto()
    ACOUSTIC_PROMPT = auto()
    SEMANTIC_AUDIO_OUTPUT = auto()
    ACOUSTIC_DECODER = auto()


class StageName(StrEnum):
    STAGE_0 = auto()
    STAGE_1 = auto()
    STAGE_2 = auto()
    STAGE_3 = auto()
    STAGE_4 = auto()


class StagedModel(Protocol):
    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterable[tuple[str, nn.Parameter]]: ...


@dataclass(frozen=True)
class StageSpec:
    name: StageName
    trainable_groups: frozenset[ParameterGroup]
    frozen_groups: frozenset[ParameterGroup]
    backbone_top_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, StageName):
            raise TypeError("stage name must be a StageName.")
        if not self.trainable_groups:
            raise ValueError("stage must train at least one parameter group.")
        groups = self.trainable_groups | self.frozen_groups
        if groups != frozenset(ParameterGroup):
            raise ValueError(
                "stage trainable and frozen groups must cover every parameter group."
            )
        if self.trainable_groups & self.frozen_groups:
            raise ValueError("stage trainable and frozen groups must be disjoint.")
        if any(not isinstance(group, ParameterGroup) for group in groups):
            raise TypeError("stage parameter groups must be ParameterGroup values.")
        if self.backbone_top_fraction is not None:
            value = self.backbone_top_fraction
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("backbone_top_fraction must be in [0, 1].")


@dataclass
class StageLoaderConfig:
    weight: float
    task_weights: dict[str, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (float, int))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("stage loader weight must be finite and positive.")
        _validate_weights(self.task_weights, name="stage loader task weights")


@dataclass
class StageConfig:
    name: StageName = StageName.STAGE_0
    trainable_groups: list[ParameterGroup] = field(
        default_factory=lambda: list(ParameterGroup)
    )
    frozen_groups: list[ParameterGroup] = field(default_factory=list)
    backbone_top_fraction: Optional[float] = 1.0
    loaders: dict[str, StageLoaderConfig] = field(default_factory=dict)
    batches_per_step: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.batches_per_step, bool)
            or not isinstance(self.batches_per_step, int)
        ):
            raise TypeError("stage batches_per_step must be an integer.")
        if self.batches_per_step < 1:
            raise ValueError("stage batches_per_step must be positive.")
        if not isinstance(self.loaders, Mapping):
            raise TypeError("stage loaders must be a mapping.")
        if self.loaders:
            _validate_weights(self.loader_weights(), name="stage loader weights")
            for name in self.loaders:
                if not name:
                    raise ValueError("stage loader names must not be empty.")
        self.spec()

    def spec(self) -> StageSpec:
        return StageSpec(
            self.name,
            frozenset(self.trainable_groups),
            frozenset(self.frozen_groups),
            backbone_top_fraction=self.backbone_top_fraction,
        )

    def loader_weights(self) -> dict[str, float]:
        return {name: loader.weight for name, loader in self.loaders.items()}

    def task_weights_by_loader(self) -> dict[str, dict[str, float]]:
        return {
            name: dict(loader.task_weights)
            for name, loader in self.loaders.items()
        }


SPEECH_INTERFACE_GROUPS = frozenset(
    {
        ParameterGroup.SEMANTIC_AUDIO_EMBEDDING,
        ParameterGroup.SEMANTIC_AUDIO_ADAPTER,
        ParameterGroup.ACOUSTIC_PROMPT,
        ParameterGroup.SEMANTIC_AUDIO_OUTPUT,
        ParameterGroup.ACOUSTIC_DECODER,
    }
)

STAGE_SPECS: Mapping[StageName, StageSpec] = {
    StageName.STAGE_0: StageSpec(
        StageName.STAGE_0,
        frozenset(ParameterGroup),
        frozenset(),
        backbone_top_fraction=1.0,
    ),
    StageName.STAGE_1: StageSpec(
        StageName.STAGE_1,
        SPEECH_INTERFACE_GROUPS,
        frozenset(ParameterGroup) - SPEECH_INTERFACE_GROUPS,
        backbone_top_fraction=0.0,
    ),
    StageName.STAGE_2: StageSpec(
        StageName.STAGE_2,
        SPEECH_INTERFACE_GROUPS,
        frozenset(ParameterGroup) - SPEECH_INTERFACE_GROUPS,
        backbone_top_fraction=0.0,
    ),
    StageName.STAGE_3: StageSpec(
        StageName.STAGE_3,
        SPEECH_INTERFACE_GROUPS | {ParameterGroup.BACKBONE},
        frozenset(),
        backbone_top_fraction=1.0 / 3.0,
    ),
    StageName.STAGE_4: StageSpec(
        StageName.STAGE_4,
        SPEECH_INTERFACE_GROUPS | {ParameterGroup.BACKBONE},
        frozenset(),
        backbone_top_fraction=1.0,
    ),
}


def default_stage_config(name: StageName) -> StageConfig:
    spec = STAGE_SPECS[name]
    return StageConfig(
        name=spec.name,
        trainable_groups=list(spec.trainable_groups),
        frozen_groups=list(spec.frozen_groups),
        backbone_top_fraction=spec.backbone_top_fraction,
    )

_LAYER_PATTERN = re.compile(r"^backbone\.model\.layers\.(\d+)\.")


def apply_stage(model: StagedModel, spec: StageSpec) -> dict[ParameterGroup, int]:
    counts = {group: 0 for group in ParameterGroup}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        counts[group] += parameter.numel()
        if _structurally_frozen(name, model):
            parameter.requires_grad_(False)
            continue
        trainable = group in spec.trainable_groups
        if group is ParameterGroup.BACKBONE and trainable:
            trainable = _backbone_trainable(name, model, spec.backbone_top_fraction)
        parameter.requires_grad_(trainable)
    return counts


def parameter_group(name: str) -> ParameterGroup:
    if name.startswith("backbone."):
        return ParameterGroup.BACKBONE
    if name.startswith("semantic_audio_embedding."):
        return ParameterGroup.SEMANTIC_AUDIO_EMBEDDING
    if name.startswith("semantic_audio_adapter."):
        return ParameterGroup.SEMANTIC_AUDIO_ADAPTER
    if name.startswith("acoustic_prompt_adapter.") or name == "acoustic_prompt_gate":
        return ParameterGroup.ACOUSTIC_PROMPT
    if name.startswith("semantic_audio_output_adapter."):
        return ParameterGroup.SEMANTIC_AUDIO_OUTPUT
    if name.startswith("acoustic_decoder.") or name.startswith("acoustic_flow."):
        return ParameterGroup.ACOUSTIC_DECODER
    raise ValueError(f"parameter {name!r} does not belong to a stage group.")


def _backbone_trainable(
    name: str, model: StagedModel, top_fraction: float | None
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
    return name.startswith("backbone.model.norm.") or name.startswith(
        "backbone.model.final_layernorm."
    )


def _num_layers(model: StagedModel, minimum: int) -> int:
    backbone = getattr(model, "backbone", None)
    config = None if backbone is None else getattr(backbone, "config", None)
    value = None if config is None else getattr(config, "num_hidden_layers", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return minimum
    return value


def _structurally_frozen(name: str, model: StagedModel) -> bool:
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


def _last_module_index(value: object) -> int | None:
    if not isinstance(value, nn.ModuleList):
        return None
    if len(value) == 0:
        return None
    return len(value) - 1


def _validate_weights(weights: Mapping[str, float], *, name: str) -> None:
    if not weights:
        raise ValueError(f"{name} must contain at least one item.")
    if any(not key for key in weights):
        raise ValueError(f"{name} names must not be empty.")
    values = list(weights.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must be finite and positive.")
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{name} must have a finite positive total.")


__all__ = [
    "ParameterGroup",
    "SPEECH_INTERFACE_GROUPS",
    "STAGE_SPECS",
    "StageConfig",
    "StageLoaderConfig",
    "StageName",
    "StageSpec",
    "apply_stage",
    "default_stage_config",
    "parameter_group",
]
