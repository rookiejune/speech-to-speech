from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from anytrain.lightning import LossItemLoggerCallback
from anytrain.loss import LossItem
from lightning import pytorch as pl
from torch import Tensor

from ...datamodule.types import (
    FusedBatch,
    LoaderBatch,
    ModelBatch,
    RawSpeechBatch,
    TrainInput,
)
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
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
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
            self._log_output_item(trainer, pl_module, objective, item, labels)

    def _log_output_item(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        objective: str,
        item: LossItem,
        labels: list[object],
    ) -> None:
        for label in _distributed_labels(trainer, labels):
            mask = torch.tensor(
                [value == label for value in labels],
                device=item.loss.device,
                dtype=torch.bool,
            )
            loss, count = _distributed_mean(trainer, pl_module, item.loss, mask)
            if _empty(count):
                continue
            pl_module.log(
                self._tag(objective=objective, key="loss", label=label),
                loss,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
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
                mean, count = _distributed_mean(trainer, pl_module, values, mask)
                if _empty(count):
                    continue
                pl_module.log(
                    tag,
                    mean,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )

    def _tag(self, *, objective: str, key: str, label: object) -> str:
        try:
            prefix = _OBJECTIVE_PREFIX[objective]
        except KeyError as error:
            raise ValueError(f"unsupported loss objective: {objective}") from error
        return f"{prefix}/{key}/{label}"

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"counts": dict(self._counts)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        counts = state_dict.get("counts", {})
        if not isinstance(counts, Mapping):
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
    if isinstance(batch, LoaderBatch):
        return _tasks(objective, batch.batch)
    if isinstance(batch, FusedBatch):
        labels: list[object] = []
        for child in batch.batches:
            labels.extend(_batch_tasks(child, objective))
        return labels
    return _batch_tasks(cast(TrainInput, batch), objective)


def _batch_tasks(batch: TrainInput, objective: str) -> list[object]:
    if objective == "token":
        return list(batch.tasks)
    if objective in {"flow_matching", "repa", "rvq"}:
        if _has_acoustic_target(batch):
            return list(batch.tasks)
        return []
    raise ValueError(f"unsupported loss objective: {objective}")


def _has_acoustic_target(batch: TrainInput) -> bool:
    if isinstance(batch, ModelBatch):
        return batch.acoustic_target is not None
    if isinstance(batch, RawSpeechBatch):
        return any(sample.prediction.supervises_audio for sample in batch.samples)
    raise TypeError("training batch must be ModelBatch or RawSpeechBatch.")


def _reduce_sum(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
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


def _distributed_labels(trainer: pl.Trainer, labels: list[object]) -> list[object]:
    del trainer
    local = _ordered(labels)
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return local
    gathered: list[list[object] | None] = [
        None for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(gathered, local)
    ordered: list[object] = []
    for rank_labels in gathered:
        if rank_labels is None:
            continue
        for label in rank_labels:
            if not any(existing == label for existing in ordered):
                ordered.append(label)
    return ordered


def _distributed_mean(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    values: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    dtype = values.dtype if values.is_floating_point() else torch.float32
    local_sum = values[mask].sum().to(dtype=dtype)
    local_count = mask.sum().to(device=values.device, dtype=dtype)
    global_sum = _reduce_sum(trainer, pl_module, local_sum)
    global_count = _reduce_sum(trainer, pl_module, local_count)
    return global_sum / global_count.clamp_min(1), global_count


def _empty(count: Tensor) -> bool:
    return bool((count <= 0).detach().cpu())


def _ordered(labels: list[object]) -> list[object]:
    ordered: list[object] = []
    for label in labels:
        if not any(existing == label for existing in ordered):
            ordered.append(label)
    return ordered
