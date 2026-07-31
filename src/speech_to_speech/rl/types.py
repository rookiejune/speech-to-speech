from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from semantic_acoustic_codec.rl import SACCandidate
from torch import Tensor

from speech_to_speech.generation.types import AudioOutput, Request


@dataclass(frozen=True)
class S2SGenerationResult:
    sample_id: int
    group_id: int
    candidate_id: int
    request: Request
    response_ids: Tensor
    audio: AudioOutput | None = None
    sac_candidate: SACCandidate | None = None
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


__all__ = ["S2SGenerationResult", "S2SRewardBatch", "S2SRollout"]
