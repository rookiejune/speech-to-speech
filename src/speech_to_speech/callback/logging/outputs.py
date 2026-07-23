from __future__ import annotations

from typing import Any, Mapping, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...datamodule.types import ModelBatch, TrainBatch
from ...loss.types import Outputs, loss_items


class OutputsLogger(Callback):
    def __init__(self, template: str = "{objective}_{key}/{task}") -> None:
        super().__init__()

        self.template = template

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # 不同rank的task列表可能不一样，不做同步
        if not isinstance(outputs, Mapping):
            raise TypeError("OutputsLogger requires mapping training outputs.")
        typed_outputs = cast(Outputs, outputs)
        device = typed_outputs["loss"].device
        train_batch = cast(TrainBatch, batch)
        for objective, loss_item in loss_items(typed_outputs):
            tasks = _tasks(train_batch, objective)
            if loss_item.loss.numel() != len(tasks):
                raise ValueError(
                    f"{objective} loss rows must align with logged task rows."
                )

            for task in dict.fromkeys(tasks):
                mask = torch.tensor(
                    [value == task for value in tasks],
                    device=device,
                    dtype=torch.bool,
                )
                mean_item = loss_item.mean(mask)

                pl_module.log(
                    self.template.format(
                        objective=objective,
                        key="loss",
                        task=task,
                    ),
                    mean_item.loss,
                )

                if mean_item.details is not None:
                    for key, scalar in mean_item.details.items():
                        pl_module.log(
                            self.template.format(
                                objective=objective,
                                key=key,
                                task=task,
                            ),
                            scalar,
                        )


def _tasks(batch: TrainBatch, objective: str) -> list[object]:
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
