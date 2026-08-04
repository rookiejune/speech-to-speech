"""Standalone Lightning wrapper for aligned dual-stream pretraining."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar, TypedDict, Union, runtime_checkable

import torch
from anytrain.lightning.schedule import ScheduleRuntime
from anytrain.loss import LossItem
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from ..mimo import MimoBatch
from ..loss.mimo import MimoObjective
from .optim import Config as OptimConfig
from ..runtime.backbone.mimo import (
    DualStreamHiddenStates,
    DualStreamLogits,
    DualStreamOutput,
)


DualHidden = Union[DualStreamHiddenStates, DualStreamOutput]


@runtime_checkable
class MimoModel(Protocol):
    """Model surface required by :class:`MimoModule`."""

    def dual_hidden_states(self, batch: MimoBatch) -> DualHidden: ...

    def dual_logits(
        self,
        hidden_states: DualHidden,
    ) -> DualStreamLogits | tuple[Tensor, Tensor]: ...


class MimoStepOutput(TypedDict):
    loss: Tensor
    mimo: LossItem


ModelT = TypeVar("ModelT", bound=nn.Module)


class MimoModule(LightningModule, Generic[ModelT]):
    """Train a dual-stream model without coupling it to the single-stream model.

    The wrapped model must be an ``nn.Module`` so Lightning owns its parameters,
    and it must satisfy :class:`MimoModel` for the objective forward.  The
    objective performs route-specific token normalization; this wrapper reduces
    it to the scalar loss consumed by automatic optimization.
    """

    def __init__(
        self,
        *,
        model: ModelT,
        objective: MimoObjective | None = None,
        optim: OptimConfig | None = None,
        schedule_runtime: ScheduleRuntime | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("MimoModule model must be a torch.nn.Module.")
        if not isinstance(model, MimoModel):
            raise TypeError("MimoModule model must expose dual_hidden_states and dual_logits.")
        if objective is not None and not isinstance(objective, MimoObjective):
            raise TypeError("MimoModule objective must be a MimoObjective or None.")
        if optim is not None and not isinstance(optim, OptimConfig):
            raise TypeError("MimoModule optim must be an OptimConfig or None.")

        self.model = model
        self.objective = MimoObjective() if objective is None else objective
        self.optim = OptimConfig() if optim is None else optim
        self.schedule_runtime = schedule_runtime

    def training_step(
        self,
        batch: MimoBatch,
        batch_idx: int = 0,
    ) -> MimoStepOutput:
        del batch_idx
        output = self._step(batch)
        self.log(
            "loss",
            output["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            batch_size=batch.batch_size,
        )
        self._log_routes(
            output["mimo"],
            prefix="train",
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        return output

    def validation_step(
        self,
        batch: MimoBatch,
        batch_idx: int = 0,
    ) -> MimoStepOutput:
        del batch_idx
        output = self._step(batch)
        self.log(
            "val/loss",
            output["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch.batch_size,
        )
        self._log_routes(
            output["mimo"],
            prefix="val",
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        return output

    def on_train_epoch_start(self) -> None:
        """Advance deterministic task datasets with the Lightning epoch.

        ``MimoTaskDataset`` deliberately keeps its sampling state outside the
        worker RNG.  Lightning may receive either a datamodule or a direct
        dataloader, so discover the dataset from the trainer rather than
        relying on one particular entry path.
        """

        loader = getattr(self.trainer, "train_dataloader", None)
        _set_dataset_epoch(loader, self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        loader = getattr(self.trainer, "val_dataloaders", None)
        _set_dataset_epoch(loader, self.current_epoch)

    def _step(self, batch: MimoBatch) -> MimoStepOutput:
        if not isinstance(batch, MimoBatch):
            raise TypeError("MimoModule requires MimoBatch inputs.")
        # MimoBatch validates masks on CPU before transfer; trusted loss avoids
        # repeating device-to-host predicates in every training step.
        item = self.objective.from_batch(batch, self.model, validate=False)
        return {"loss": self.objective.mean(item, distributed=True), "mimo": item}

    def _log_routes(
        self,
        item: LossItem,
        *,
        prefix: str,
        on_step: bool,
        on_epoch: bool,
        batch_size: int,
    ) -> None:
        details = item.details
        if details is None:
            raise RuntimeError("MimoObjective did not return route details.")
        route_means = self.objective.route_means(item, distributed=True)
        for route, loss in zip(("text", "audio"), route_means):
            self.log(
                f"{prefix}/{route}_loss",
                loss,
                on_step=on_step,
                on_epoch=on_epoch,
                # route_means() has already reduced its denominator globally.
                sync_dist=False,
                batch_size=batch_size,
            )
            self.log(
                f"{prefix}/{route}_tokens",
                details[f"{route}_tokens"].sum(),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                reduce_fx="sum",
                batch_size=batch_size,
            )

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, MimoBatch):
            return batch.to(device, non_blocking=True)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self.model,
            preset="pretrain",
            optimizer=self.optim.name,
            lr=self.optim.learning_rate,
            weight_decay=self.optim.weight_decay,
        )
        if self.schedule_runtime is None:
            return optimizer
        return self.schedule_runtime.configure_optimizers(optimizer)


MimoPretrainingModule = MimoModule


def _set_dataset_epoch(value: object, epoch: int) -> None:
    """Set an epoch on a loader/dataset, including Lightning wrappers."""

    if value is None:
        return
    if isinstance(value, Mapping):
        for loader in value.values():
            _set_dataset_epoch(loader, epoch)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for loader in value:
            _set_dataset_epoch(loader, epoch)
        return
    setter = getattr(value, "set_epoch", None)
    if callable(setter):
        setter(epoch)
        return
    dataset = getattr(value, "dataset", None)
    if dataset is not None and dataset is not value:
        _set_dataset_epoch(dataset, epoch)
    loaders = getattr(value, "loaders", None)
    if loaders is not None and loaders is not value:
        if isinstance(loaders, dict):
            for loader in loaders.values():
                _set_dataset_epoch(loader, epoch)
        else:
            _set_dataset_epoch(loaders, epoch)


__all__ = [
    "DualHidden",
    "MimoModel",
    "MimoModule",
    "MimoPretrainingModule",
    "MimoStepOutput",
]
