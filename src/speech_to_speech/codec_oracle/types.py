from __future__ import annotations

from enum import auto

import torch
from torch import Tensor

from .._compat import StrEnum


class Objective(StrEnum):
    FLOW = auto()
    RVQ = auto()


class Initialization(StrEnum):
    CODEC = auto()
    RANDOM = auto()

    def weight(self, codebook: Tensor, *, seed: int) -> Tensor:
        if codebook.dim() != 2 or not torch.is_floating_point(codebook):
            raise ValueError(
                "codec codebook must have shape [vocab, dim] and floating dtype."
            )
        if self is Initialization.CODEC:
            return codebook.clone()
        return matched_random_weight(codebook, seed=seed)


def matched_random_weight(
    reference: Tensor,
    *,
    seed: int,
    rows: int | None = None,
) -> Tensor:
    shape = reference.shape if rows is None else (rows, reference.size(-1))
    output = reference.new_empty(shape)
    generator = torch.Generator(device=output.device).manual_seed(seed)
    return output.normal_(
        mean=float(reference.mean()),
        std=float(reference.std(correction=0)),
        generator=generator,
    )


__all__ = ["Initialization", "Objective", "matched_random_weight"]
