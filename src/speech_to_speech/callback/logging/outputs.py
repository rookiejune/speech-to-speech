from __future__ import annotations

from typing import Any, cast

from anytrain.lightning import LossItemLoggerCallback

from ...datamodule.types import ModelBatch, TrainBatch
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
    train_batch = cast(TrainBatch, batch)
    return _batch_or_tuple_tasks(train_batch, objective)


def _batch_or_tuple_tasks(batch: TrainBatch, objective: str) -> list[object]:
    if not isinstance(batch, tuple):
        return _batch_tasks(batch, objective)
    tasks = []
    for item in batch:
        tasks.extend(_batch_tasks(item, objective))
    return tasks


def _batch_tasks(batch: ModelBatch, objective: str) -> list[object]:
    if objective == "token":
        return list(batch.tasks)
    if objective in {"flow_matching", "repa", "rvq"}:
        if batch.acoustic_target is not None:
            return list(batch.tasks)
        return []
    raise ValueError(f"unsupported loss objective: {objective}")
