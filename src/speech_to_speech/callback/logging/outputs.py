from __future__ import annotations

import math
from typing import Any, cast

import torch
from anytrain.lightning import LossItemLoggerCallback
from anytrain.loss import LossItem
from lightning import LightningModule, Trainer
from torch import Tensor

from ...datamodule.types import ModelBatch
from ...loss.types import loss_items

_OBJECTIVE_PREFIX = {
    "token": "token",
    "rvq": "acoustic/rvq",
    "flow_matching": "acoustic/flow_matching",
    "repa": "acoustic/repa",
}
_COUNT_KEYS = frozenset({"tokens", "text_tokens", "audio_tokens", "frames"})


class OutputsLogger(LossItemLoggerCallback):
    """Log per-task loss means and cumulative supervised token/frame counts."""

    def __init__(self) -> None:
        super().__init__(
            _tasks,
            template="{objective}/{key}/{task}",
            label_name="task",
            loss_items_fn=loss_items,
        )
        self._counts: dict[str, float] = {}

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch_idx
        if not isinstance(outputs, dict):
            raise TypeError("OutputsLogger requires mapping training outputs.")

        for objective, item in self._loss_items(outputs):
            labels = list(self.labels_fn(objective, batch))
            if item.loss.numel() != len(labels):
                raise ValueError(
                    f"{objective} loss rows must align with logged label rows."
                )
            self._log_item(trainer, pl_module, objective, item, labels)

    def _log_item(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        objective: str,
        item: LossItem,
        labels: list[object],
    ) -> None:
        for label in dict.fromkeys(labels):
            mask = torch.tensor(
                [value == label for value in labels],
                device=item.loss.device,
                dtype=torch.bool,
            )
            pl_module.log(
                self._tag(objective=objective, key="loss", label=label),
                item.loss[mask].mean(),
                on_step=True,
                on_epoch=False,
            )
            if item.details is None:
                continue
            for key, values in item.details.items():
                tag = self._tag(objective=objective, key=key, label=label)
                if key in _COUNT_KEYS:
                    step = _reduce_sum(trainer, pl_module, values[mask].sum())
                    total = self._counts.get(tag, 0.0) + float(step.detach().cpu())
                    self._counts[tag] = total
                    pl_module.log(
                        tag,
                        total,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=False,
                    )
                    continue
                pl_module.log(
                    tag,
                    values[mask].mean(),
                    on_step=True,
                    on_epoch=False,
                )

    def _tag(self, *, objective: str, key: str, label: object) -> str:
        try:
            prefix = _OBJECTIVE_PREFIX[objective]
        except KeyError as error:
            raise ValueError(f"unsupported loss objective: {objective}") from error
        return f"{prefix}/{key}/{label}"

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"counts": dict(self._counts)}

    def load_state_dict(self, state_dict: dict[str, dict[str, float]]) -> None:
        counts = state_dict.get("counts", {})
        if not isinstance(counts, dict):
            raise TypeError("OutputsLogger state counts must be a mapping.")
        resolved: dict[str, float] = {}
        for key, value in counts.items():
            if not isinstance(key, str):
                raise TypeError("OutputsLogger count keys must be strings.")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(
                    f"OutputsLogger count {key!r} must be finite and non-negative."
                )
            resolved[key] = number
        self._counts = resolved


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


def _reduce_sum(
    trainer: Trainer,
    pl_module: LightningModule,
    value: Tensor,
) -> Tensor:
    del pl_module
    world_size = int(getattr(trainer, "world_size", 1))
    if world_size <= 1:
        return value
    strategy = getattr(trainer, "strategy", None)
    reduce = getattr(strategy, "reduce", None)
    if not callable(reduce):
        raise RuntimeError("distributed token counts require trainer.strategy.reduce.")
    return cast(Tensor, reduce(value.detach(), reduce_op="sum"))
