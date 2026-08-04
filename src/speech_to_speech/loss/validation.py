from __future__ import annotations

from anytrain.evaluator.weighted import Metric
from torch import Tensor

from .types import LossItem, Outputs, loss_items, loss_unit

_OBJECTIVE_NAME = {
    "token": "token/loss",
    "ctc": "alignment/ctc/loss",
    "flow_matching": "acoustic/flow_matching/loss",
    "repa": "acoustic/repa/loss",
    "rvq": "acoustic/rvq/loss",
}


def validation_metrics(outputs: Outputs) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {}
    for objective, item in loss_items(outputs):
        name = _OBJECTIVE_NAME[objective]
        metrics[name] = _metric(item, item.loss, loss_unit(objective))
        if objective == "rvq":
            metrics.update(_rvq_metrics(item))
    return metrics


def _rvq_metrics(item: LossItem) -> dict[str, Metric]:
    details = item.details
    if details is None:
        raise TypeError("RVQ validation requires loss details.")
    metrics = {}
    for key, values in sorted(details.items()):
        name = _rvq_name(key)
        if name is not None:
            metrics[name] = _metric(item, values, "frames")
    return metrics


def _rvq_name(key: str) -> str | None:
    parts = key.split("_")
    if len(parts) == 2 and parts[0] == "codebook" and parts[1].isdigit():
        return f"acoustic/rvq/{key}"
    if (
        len(parts) == 3
        and parts[0] == "codebook"
        and parts[1].isdigit()
        and parts[2] == "top1"
    ):
        return f"acoustic/rvq/{key}"
    return None


def _metric(item: LossItem, values: Tensor, unit: str) -> Metric:
    details = item.details
    if details is None or unit not in details:
        raise TypeError(f"validation loss item requires {unit!r} details.")
    return Metric(values, details[unit])
