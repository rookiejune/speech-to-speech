from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor
from typing_extensions import NotRequired


@dataclass(frozen=True)
class LossItem:
    loss: Tensor
    details: dict[str, Tensor] | None

    def mask_by(self, mask: Tensor):
        loss = self.loss[mask].mean()

        details = self.details
        if details is not None:
            details = {key: value[mask].mean() for key, value in details.items()}
        return LossItem(loss, details)


class Outputs(TypedDict):
    loss: Tensor
    semantic: NotRequired[LossItem]
    flow_matching: NotRequired[LossItem]
    repa: NotRequired[LossItem]
    causal_lm: NotRequired[LossItem]
