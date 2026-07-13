from __future__ import annotations

from typing import Any, Mapping, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...datamodule.types import ModelBatch
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
        batch = cast(ModelBatch, batch)

        for task in dict.fromkeys(batch.tasks):
            mask = torch.tensor(
                [value == task for value in batch.tasks],
                device=device,
                dtype=torch.bool,
            )

            for objective, loss_item in loss_items(typed_outputs):
                loss_item = loss_item.mean(mask)

                pl_module.log(
                    self.template.format(
                        objective=objective,
                        key="loss",
                        task=task,
                    ),
                    loss_item.loss,
                )

                if loss_item.details is not None:
                    for key, scalar in loss_item.details.items():
                        pl_module.log(
                            self.template.format(
                                objective=objective,
                                key=key,
                                task=task,
                            ),
                            scalar,
                        )
