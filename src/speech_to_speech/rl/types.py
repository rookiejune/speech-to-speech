from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from semantic_acoustic_generator.rl import GeneratorCandidate
from torch import Tensor

from ..datamodule.batch import ModelBatch
from ..generation.result import AudioOutput
from ..task import Request


@dataclass
class PreferenceBatch:
    chosen: ModelBatch
    rejected: ModelBatch
    ref_chosen_logps: Tensor | None = None
    ref_rejected_logps: Tensor | None = None

    def __post_init__(self) -> None:
        batch_size = self.chosen.input_ids.size(0)
        if self.rejected.input_ids.size(0) != batch_size:
            raise ValueError("chosen and rejected batches must have the same batch size.")
        if self.chosen.input_ids.shape != self.chosen.token_labels.shape:
            raise ValueError("chosen input ids and token labels must align.")
        if self.rejected.input_ids.shape != self.rejected.token_labels.shape:
            raise ValueError("rejected input ids and token labels must align.")
        if self.chosen.input_ids.shape != self.rejected.input_ids.shape:
            raise ValueError("chosen and rejected batches must have aligned shapes.")
        if (self.ref_chosen_logps is None) != (self.ref_rejected_logps is None):
            raise ValueError("reference chosen and rejected logps must be provided together.")
        for name, logps in (
            ("ref_chosen_logps", self.ref_chosen_logps),
            ("ref_rejected_logps", self.ref_rejected_logps),
        ):
            if logps is not None and logps.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [batch].")


@dataclass
class GRPOBatch:
    sequences: ModelBatch
    old_token_logps: Tensor
    rewards: Tensor
    ref_token_logps: Tensor | None = None
    group_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.old_token_logps.dim() != 3:
            raise ValueError("old token logps must have shape [batch, group, tokens].")
        batch_size, group_size, token_count = self.old_token_logps.shape
        if self.sequences.input_ids.size(0) != batch_size * group_size:
            raise ValueError("sequence rows must equal batch times group.")
        if self.sequences.input_ids.size(1) - 1 != token_count:
            raise ValueError("old token logps must align with next-token targets.")
        if self.rewards.shape != (batch_size, group_size):
            raise ValueError("rewards must have shape [batch, group].")
        if (
            self.ref_token_logps is not None
            and self.ref_token_logps.shape != self.old_token_logps.shape
        ):
            raise ValueError("reference token logps must align with old token logps.")
        if self.group_mask is not None and self.group_mask.shape != self.rewards.shape:
            raise ValueError("group mask must have shape [batch, group].")


@dataclass(frozen=True)
class S2SGenerationResult:
    sample_id: int
    group_id: int
    candidate_id: int
    request: Request
    response_ids: Tensor
    audio: AudioOutput | None = None
    sac_candidate: GeneratorCandidate | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_negative(self.sample_id, name="sample_id")
        _non_negative(self.group_id, name="group_id")
        _non_negative(self.candidate_id, name="candidate_id")
        if self.response_ids.dim() != 1:
            raise ValueError("response_ids must have shape [tokens].")
        if self.response_ids.dtype.is_floating_point or self.response_ids.dtype == torch.bool:
            raise TypeError("response_ids must use an integer dtype.")


@dataclass(frozen=True)
class S2SRollout:
    results: tuple[S2SGenerationResult, ...]
    batch_size: int
    group_size: int

    def __post_init__(self) -> None:
        _positive(self.batch_size, name="batch_size")
        _positive(self.group_size, name="group_size")
        if len(self.results) != self.batch_size * self.group_size:
            raise ValueError("result count must equal batch_size * group_size.")
        for expected, result in enumerate(self.results):
            sample_id = expected // self.group_size
            candidate_id = expected % self.group_size
            if result.sample_id != sample_id or result.group_id != sample_id:
                raise ValueError("result sample_id/group_id must follow rollout order.")
            if result.candidate_id != candidate_id:
                raise ValueError("candidate_id must follow rollout order within each group.")


@dataclass(frozen=True)
class S2SRewardBatch:
    rewards: Tensor
    group_mask: Tensor | None = None
    components: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rewards.dim() != 2:
            raise ValueError("rewards must have shape [batch, group].")
        if self.rewards.dtype == torch.bool or self.rewards.is_complex():
            raise TypeError("rewards must be real-valued.")
        if self.group_mask is not None:
            if self.group_mask.shape != self.rewards.shape:
                raise ValueError("group_mask must align with rewards.")
            if self.group_mask.dtype != torch.bool:
                raise TypeError("group_mask must be boolean.")
        for name, value in self.components.items():
            if value.shape != self.rewards.shape:
                raise ValueError(f"reward component {name!r} must align with rewards.")


def _positive(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _non_negative(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "GRPOBatch",
    "PreferenceBatch",
    "S2SGenerationResult",
    "S2SRewardBatch",
    "S2SRollout",
]
