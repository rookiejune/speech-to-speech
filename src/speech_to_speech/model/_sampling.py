from __future__ import annotations

import torch
from torch import Tensor


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = sorted_logits.softmax(dim=-1)
    remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
    remove[..., 0] = False
    filtered = logits.new_full(logits.shape, float("-inf"))
    filtered.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits.masked_fill(remove, float("-inf")),
    )
    return filtered
