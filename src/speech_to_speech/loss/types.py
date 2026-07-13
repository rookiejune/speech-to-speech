from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor
from typing_extensions import NotRequired


@dataclass(frozen=True)
class LossItem:
    loss: Tensor
    details: dict[str, Tensor] | None

    def mean(self, mask: Tensor) -> LossItem:
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


def loss_items(outputs: Outputs) -> Iterator[tuple[str, LossItem]]:
    for name, item in (
        ("semantic", outputs.get("semantic")),
        ("flow_matching", outputs.get("flow_matching")),
        ("repa", outputs.get("repa")),
        ("causal_lm", outputs.get("causal_lm")),
    ):
        if item is not None:
            yield name, item
