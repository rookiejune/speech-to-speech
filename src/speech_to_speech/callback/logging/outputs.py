from typing import Any, Mapping, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...datamodule.types import ModelBatch
from ...loss.types import LossItem, Outputs


class OutputsLogger(Callback):
    def __init__(self, tag="{stage}_{key}/{task}") -> None:
        super().__init__()

        self.tag = tag

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # 不同rank的task列表可能不一样，不做同步
        outputs = dict(cast(Outputs, outputs))
        loss = cast(Tensor, outputs["loss"])
        loss_items = cast(
            dict[str, LossItem],
            {key: value for key, value in outputs.items() if key != "loss"},
        )
        pl_module.log("loss", loss, prog_bar=True, on_step=True)

        batch = cast(ModelBatch, batch)

        task_set = set(batch.tasks)

        for task in task_set:
            mask = torch.tensor(
                [value == task for value in batch.tasks],
                device=loss.device,
            ).bool()

            for stage, loss_item in loss_items.items():
                loss_item = loss_item.mask_by(mask)

                pl_module.log(
                    self.tag.format(stage=stage, key="loss", task=task),
                    loss_item.loss,
                )

                if loss_item.details is not None:
                    for key, scalar in loss_item.details.items():
                        pl_module.log(
                            self.tag.format(stage=stage, key=key, task=task),
                            scalar,
                        )
