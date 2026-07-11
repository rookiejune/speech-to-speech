from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...loss.types import LossItem


class _FlowRuntime(Protocol):
    time_sampler: Any


class _FlowModel(Protocol):
    runtime: Any


class FlowMatchingLogger(Callback):
    """Log flow-matching sampler configuration and the sampled training times."""

    def __init__(self, every_n_steps: int = 100) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        self.every_n_steps = every_n_steps

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        flow_runtime = cast(
            _FlowRuntime, cast(_FlowModel, pl_module.model).runtime.flow_matching
        )
        sampler = flow_runtime.time_sampler
        config = vars(sampler)
        values = [f"sampler={type(sampler).__name__}"]
        values.extend(
            f"{name}={config[name]}"
            for name in ("mean", "std", "t_min", "t_max")
            if name in config
        )
        if hasattr(logger.experiment, "add_text"):
            logger.experiment.add_text("flow/config", "\n".join(values), 0)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        if trainer.global_step % self.every_n_steps != 0:
            return
        if not isinstance(outputs, Mapping):
            return
        flow = outputs.get("flow_matching")
        if not isinstance(flow, LossItem) or flow.details is None:
            return

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment
        t = flow.details.get("t")
        if isinstance(t, Tensor) and hasattr(experiment, "add_histogram"):
            experiment.add_histogram(
                "flow/time",
                t.detach().float().cpu(),
                trainer.global_step,
            )
