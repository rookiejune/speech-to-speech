from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypedDict

import torch
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

    def weighted_mean(self, weight: Tensor) -> Tensor:
        if weight.shape != self.loss.shape:
            raise ValueError("loss weights must align with loss rows.")
        total = weight.sum()
        if bool(total.le(0)):
            raise ValueError("loss weights must contain a positive total.")
        return (self.loss * weight).sum() / total


class Outputs(TypedDict):
    loss: Tensor
    token: NotRequired[LossItem]
    flow_matching: NotRequired[LossItem]
    repa: NotRequired[LossItem]
    rvq: NotRequired[LossItem]
    loss_weights: NotRequired[dict[str, float]]


_UNITS = {
    "token": "tokens",
    "flow_matching": "frames",
    "repa": "frames",
    "rvq": "frames",
}


def combine_outputs(outputs: Sequence[Outputs]) -> Outputs:
    if not outputs:
        raise ValueError("cannot combine an empty output sequence.")
    result: Outputs = {"loss": outputs[0]["loss"].new_zeros(())}
    for name, unit in _UNITS.items():
        items = [output[name] for output in outputs if name in output]
        if not items:
            continue
        item = _cat(items)
        result[name] = item
        result["loss"] = result["loss"] + _loss_weight(outputs, name) * _mean(
            item,
            unit,
        )
    return result


def loss_items(outputs: Outputs) -> Iterator[tuple[str, LossItem]]:
    for name, item in (
        ("token", outputs.get("token")),
        ("flow_matching", outputs.get("flow_matching")),
        ("repa", outputs.get("repa")),
        ("rvq", outputs.get("rvq")),
    ):
        if item is not None:
            yield name, item


def _cat(items: Iterable[LossItem]) -> LossItem:
    values = list(items)
    if not values:
        raise ValueError("cannot concatenate an empty loss item sequence.")
    details = _cat_details([item.details for item in values])
    return LossItem(
        torch.cat([item.loss.reshape(-1) for item in values]),
        details,
    )


def _cat_details(values: list[dict[str, Tensor] | None]) -> dict[str, Tensor] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("loss item details must be present for every shard or none.")
    keys = set(present[0])
    if any(set(value) != keys for value in present):
        raise ValueError("loss item details must use consistent keys.")
    return {
        key: torch.cat([value[key].reshape(-1) for value in present])
        for key in sorted(keys)
    }


def _mean(item: LossItem, unit: str) -> Tensor:
    details = item.details
    if details is None or unit not in details:
        return item.loss.mean()
    return item.weighted_mean(details[unit].to(dtype=item.loss.dtype))


def _loss_weight(outputs: Sequence[Outputs], name: str) -> float:
    weights = [
        output.get("loss_weights", {}).get(name, 1.0)
        for output in outputs
        if name in output
    ]
    first = weights[0]
    if any(weight != first for weight in weights):
        raise ValueError(f"{name} loss weight changed within a joint batch.")
    return first
