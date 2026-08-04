from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
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
from ...task_spec import uses_source_ctc, uses_target_ctc

_OBJECTIVE_PREFIX = {
    "token": "token",
    "ctc": "alignment/ctc",
    "rvq": "acoustic/rvq",
    "flow_matching": "acoustic/flow_matching",
    "repa": "acoustic/repa",
    "mimo": "mimo",
}
_COUNT_KEYS = frozenset(
    {
        "tokens",
        "text_tokens",
        "audio_tokens",
        "frames",
        "sequences",
        "source_tokens",
        "target_tokens",
        "source_steps",
        "target_steps",
    }
)


@dataclass
class _PendingStat:
    total: Tensor
    weight: Tensor | None


class OutputsLogger(LossItemLoggerCallback):
    """Log cadence-window task means and cumulative supervised unit counts."""

    def __init__(self) -> None:
        super().__init__(
            _tasks,
            template="{objective}/{key}/{task}",
            label_name="task",
            loss_items_fn=loss_items,
        )
        self._counts: dict[str, float] = {}
        self._pending: dict[str, _PendingStat] = {}
        self._last_logged_step: int | None = None

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
            self._accumulate_output_item(objective, item, labels)
        if self._should_log(trainer):
            self._flush(trainer, pl_module)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._flush(trainer, pl_module)

    def _accumulate_output_item(
        self,
        objective: str,
        item: LossItem,
        labels: list[object],
    ) -> None:
        details = item.details
        for label in _ordered(labels):
            task_mask = torch.tensor(
                [value == label for value in labels],
                device=item.loss.device,
                dtype=torch.bool,
            )
            loss_mask = _observation_mask(
                objective,
                "loss",
                details,
                task_mask,
            )
            if bool(loss_mask.any()):
                self._accumulate(
                    self._tag(objective=objective, key="loss", label=label),
                    item.loss,
                    loss_mask,
                    count=False,
                )
            if details is None:
                continue
            for key, values in details.items():
                if values.shape != item.loss.shape:
                    raise ValueError(
                        f"{objective} detail {key!r} rows must align with loss rows."
                    )
                mask = _observation_mask(
                    objective,
                    key,
                    details,
                    task_mask,
                )
                if not bool(mask.any()):
                    continue
                self._accumulate(
                    self._tag(objective=objective, key=key, label=label),
                    values,
                    mask,
                    count=key in _COUNT_KEYS,
                )

    def _accumulate(
        self,
        tag: str,
        values: Tensor,
        mask: Tensor,
        *,
        count: bool,
    ) -> None:
        total = values.detach()[mask].sum(dtype=torch.float32)
        weight = None if count else mask.sum().to(dtype=torch.float32)
        pending = self._pending.get(tag)
        if pending is None:
            self._pending[tag] = _PendingStat(total, weight)
            return
        if (pending.weight is None) != count:
            raise RuntimeError(
                f"metric {tag!r} changed count semantics within a window."
            )
        if pending.total.device != total.device:
            pending.total = pending.total.to(device=total.device)
            if pending.weight is not None:
                pending.weight = pending.weight.to(device=total.device)
        pending.total = pending.total + total
        if weight is not None:
            if pending.weight is None:
                raise RuntimeError(f"metric {tag!r} lost its mean weight.")
            pending.weight = pending.weight + weight

    def _should_log(self, trainer: pl.Trainer) -> bool:
        step = getattr(trainer, "global_step", None)
        if step is None:
            return True
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("trainer.global_step must be an integer.")
        every = getattr(trainer, "log_every_n_steps", 1)
        if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
            raise ValueError("trainer.log_every_n_steps must be a positive integer.")
        if step <= 0 or step % every != 0 or self._last_logged_step == step:
            return False
        self._last_logged_step = step
        return True

    def _flush(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        schema = _distributed_schema(self._pending)
        if not schema:
            return
        device = _pending_device(self._pending, pl_module)
        payload = torch.zeros(
            (len(schema), 2),
            dtype=torch.float32,
            device=device,
        )
        for index, (tag, is_count) in enumerate(schema):
            pending = self._pending.get(tag)
            if pending is None:
                continue
            payload[index, 0] = pending.total.to(device=device)
            if not is_count:
                if pending.weight is None:
                    raise RuntimeError(f"metric {tag!r} is missing its mean weight.")
                payload[index, 1] = pending.weight.to(device=device)
        reduced = _reduce_sum(trainer, payload)
        values = reduced.detach().cpu().tolist()
        self._pending.clear()
        for (tag, is_count), (total, weight) in zip(schema, values):
            if is_count:
                cumulative = self._counts.get(tag, 0.0) + float(total)
                self._counts[tag] = cumulative
                value = cumulative
            else:
                if weight <= 0:
                    continue
                value = float(total) / float(weight)
            pl_module.log(
                tag,
                value,
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

    def state_dict(self) -> dict[str, object]:
        return {
            "counts": dict(self._counts),
            "last_logged_step": self._last_logged_step,
            "pending": {
                tag: {
                    "total": stat.total.detach().cpu(),
                    "weight": (
                        None if stat.weight is None else stat.weight.detach().cpu()
                    ),
                }
                for tag, stat in self._pending.items()
            },
        }

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
        last_logged_step = state_dict.get("last_logged_step")
        if last_logged_step is not None and (
            isinstance(last_logged_step, bool)
            or not isinstance(last_logged_step, int)
            or last_logged_step < 0
        ):
            raise ValueError(
                "OutputsLogger last_logged_step must be non-negative or None."
            )
        self._last_logged_step = last_logged_step
        pending = state_dict.get("pending", {})
        if not isinstance(pending, Mapping):
            raise TypeError("OutputsLogger pending stats must be a mapping.")
        restored: dict[str, _PendingStat] = {}
        for tag, raw in pending.items():
            if not isinstance(tag, str) or not isinstance(raw, Mapping):
                raise TypeError("OutputsLogger pending entries must be named mappings.")
            total = raw.get("total")
            weight = raw.get("weight")
            if not isinstance(total, Tensor) or total.numel() != 1:
                raise TypeError("OutputsLogger pending totals must be scalar tensors.")
            if weight is not None and (
                not isinstance(weight, Tensor) or weight.numel() != 1
            ):
                raise TypeError(
                    "OutputsLogger pending weights must be scalar tensors or None."
                )
            restored[tag] = _PendingStat(
                total.detach(), None if weight is None else weight.detach()
            )
        self._pending = restored


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
    if objective == "ctc":
        return list(batch.tasks) if _has_ctc_target(batch) else []
    if objective in {"flow_matching", "repa", "rvq"}:
        if _has_acoustic_target(batch):
            return list(batch.tasks)
        return []
    if objective == "mimo":
        # MIMO batches carry their task labels directly and deliberately do
        # not implement the single-stream TrainInput protocol.
        task_ids = getattr(batch, "task_ids", None)
        if task_ids is None:
            raise TypeError("MIMO batches must expose task_ids for logging.")
        return list(task_ids)
    raise ValueError(f"unsupported loss objective: {objective}")


def _has_acoustic_target(batch: TrainInput) -> bool:
    if isinstance(batch, ModelBatch):
        return batch.acoustic_target is not None
    if isinstance(batch, RawSpeechBatch):
        return any(sample.prediction.supervises_audio for sample in batch.samples)
    raise TypeError("training batch must be ModelBatch or RawSpeechBatch.")


def _has_ctc_target(batch: TrainInput) -> bool:
    if isinstance(batch, ModelBatch):
        return batch.source_ctc is not None or batch.target_ctc is not None
    if isinstance(batch, RawSpeechBatch):
        return any(
            uses_source_ctc(sample.task)
            or uses_target_ctc(sample.task, sample.prediction)
            for sample in batch.samples
        )
    raise TypeError("training batch must be ModelBatch or RawSpeechBatch.")


def _observation_mask(
    objective: str,
    key: str,
    details: Mapping[str, Tensor] | None,
    task_mask: Tensor,
) -> Tensor:
    if objective != "ctc":
        return task_mask
    if details is None:
        raise TypeError("CTC logging requires loss details.")
    if key.startswith("source_"):
        unit = "source_tokens"
    elif key.startswith("target_"):
        unit = "target_tokens"
    else:
        unit = "sequences"
    counts = details.get(unit)
    if counts is None or counts.shape != task_mask.shape:
        raise ValueError(f"CTC logging requires row-aligned {unit!r} details.")
    return task_mask & counts.gt(0)


def _reduce_sum(trainer: pl.Trainer, value: Tensor) -> Tensor:
    world_size = int(getattr(trainer, "world_size", 1))
    if world_size <= 1:
        return value
    strategy = getattr(trainer, "strategy", None)
    reduce = getattr(strategy, "reduce", None)
    if not callable(reduce):
        raise RuntimeError("distributed token counts require trainer.strategy.reduce.")
    return cast(Tensor, reduce(value.detach(), reduce_op="sum"))


def _distributed_schema(
    pending: Mapping[str, _PendingStat],
) -> list[tuple[str, bool]]:
    local = [(tag, stat.weight is None) for tag, stat in pending.items()]
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local
    gathered: list[list[tuple[str, bool]] | None] = [
        None for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(gathered, local)
    kinds: dict[str, bool] = {}
    for rank_schema in gathered:
        if rank_schema is None:
            continue
        for tag, is_count in rank_schema:
            existing = kinds.get(tag)
            if existing is not None and existing != is_count:
                raise RuntimeError(
                    f"metric {tag!r} has inconsistent count semantics across ranks."
                )
            kinds[tag] = is_count
    return list(kinds.items())


def _pending_device(
    pending: Mapping[str, _PendingStat],
    pl_module: pl.LightningModule,
) -> torch.device:
    first = next(iter(pending.values()), None)
    if first is not None:
        return first.total.device
    try:
        return pl_module.device
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "distributed output logging requires a module device on ranks without metrics."
        ) from error


def _ordered(labels: list[object]) -> list[object]:
    ordered: list[object] = []
    for label in labels:
        if not any(existing == label for existing in ordered):
            ordered.append(label)
    return ordered
