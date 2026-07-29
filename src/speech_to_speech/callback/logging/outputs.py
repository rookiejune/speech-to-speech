from __future__ import annotations

from typing import Any, cast

from anytrain.lightning import LossItemLoggerCallback

from ...datamodule.types import ModelBatch
from ...loss.types import loss_items


class OutputsLogger(LossItemLoggerCallback):
    def __init__(self, template: str = "{objective}_{key}/{task}") -> None:
        super().__init__(
            _tasks,
            template=template,
            label_name="task",
            loss_items_fn=loss_items,
        )


def _tasks(objective: str, batch: Any) -> list[object]:
    return _batch_tasks(cast(ModelBatch, batch), objective)


def _batch_tasks(batch: ModelBatch, objective: str) -> list[object]:
    if objective == "token":
        return list(batch.tasks)
    if objective in {"flow_matching", "repa", "rvq"}:
        if batch.acoustic_target is not None:
            return list(batch.tasks)
        return []
    raise ValueError(f"unsupported loss objective: {objective}")
