from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypedDict, cast

from anytrain.loss import LossItem, combine_loss_outputs, iter_loss_items
from torch import Tensor
from typing_extensions import NotRequired


class Outputs(TypedDict):
    loss: Tensor
    dpo: NotRequired[LossItem]
    grpo: NotRequired[LossItem]
    token: NotRequired[LossItem]
    ctc: NotRequired[LossItem]
    flow_matching: NotRequired[LossItem]
    repa: NotRequired[LossItem]
    rvq: NotRequired[LossItem]
    mimo: NotRequired[LossItem]
    loss_weights: NotRequired[dict[str, float]]


_UNITS = {
    "dpo": "preferences",
    "grpo": "preferences",
    "token": "tokens",
    "ctc": "sequences",
    "flow_matching": "frames",
    "repa": "frames",
    "rvq": "frames",
    "mimo": "tokens",
}
_OBJECTIVES = tuple(_UNITS)


def combine_outputs(
    outputs: Sequence[Outputs],
    *,
    total_loss: Tensor | None = None,
) -> Outputs:
    generic_outputs = cast(Sequence[dict[str, Any]], outputs)
    return cast(
        Outputs,
        combine_loss_outputs(
            generic_outputs,
            _UNITS,
            validate_item_weights=False,
            total_loss=total_loss,
        ),
    )


def loss_items(outputs: Mapping[str, Any]) -> Iterator[tuple[str, LossItem]]:
    yield from iter_loss_items(outputs, _OBJECTIVES)


def loss_unit(name: str) -> str:
    try:
        return _UNITS[name]
    except KeyError as error:
        raise ValueError(f"unsupported loss objective: {name}") from error
