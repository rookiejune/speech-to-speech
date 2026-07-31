from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .types import ModelBatch


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
        if self.ref_token_logps is not None and self.ref_token_logps.shape != self.old_token_logps.shape:
            raise ValueError("reference token logps must align with old token logps.")
        if self.group_mask is not None and self.group_mask.shape != self.rewards.shape:
            raise ValueError("group mask must have shape [batch, group].")
