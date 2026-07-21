from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from anytrain.stats import TimeBucketedMean, time_bucketed_mean
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from .._lightning import histogram_experiment, scalar_experiment, text_experiment
from ...loss.types import LossItem


class _FlowRuntime(Protocol):
    @property
    def time_sampler(self) -> object: ...


class FlowMatchingLogger(Callback):
    """Log flow-matching sampler configuration and the sampled training times."""

    def __init__(
        self,
        runtime: _FlowRuntime,
        every_n_steps: int = 100,
        time_bucket_count: int = 10,
    ) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        if time_bucket_count < 1:
            raise ValueError("time_bucket_count must be positive")
        self.runtime = runtime
        self.every_n_steps = every_n_steps
        self.time_bucket_count = time_bucket_count

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        experiment = text_experiment(trainer)
        if experiment is None:
            return

        sampler = self.runtime.time_sampler
        config = vars(sampler)
        values = [f"sampler={type(sampler).__name__}"]
        values.extend(
            f"{name}={config[name]}"
            for name in ("mean", "std", "t_min", "t_max")
            if name in config
        )
        experiment.add_text("flow/config", "\n".join(values), 0)

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

        t = flow.details.get("t")
        if not isinstance(t, Tensor):
            return

        experiment = histogram_experiment(trainer)
        if experiment is not None:
            experiment.add_histogram(
                "flow/time",
                t.detach().float().cpu(),
                trainer.global_step,
            )
        self._bucket_loss(trainer, flow.loss, t)

    def _bucket_loss(self, trainer: Trainer, loss: Tensor, t: Tensor) -> None:
        experiment = scalar_experiment(trainer)
        if experiment is None:
            return

        bucketed = time_bucketed_mean(
            loss.detach(),
            t.detach(),
            bucket_count=self.time_bucket_count,
            t_min=_sampler_bound(self.runtime.time_sampler, "t_min", 0.0),
            t_max=_sampler_bound(self.runtime.time_sampler, "t_max", 1.0),
        )
        bucketed = _sync_bucketed(trainer, bucketed)
        if not _is_global_zero(trainer):
            return

        mean = bucketed.mean.detach().cpu()
        count = bucketed.count.detach().cpu()
        edges = bucketed.edges.detach().cpu()
        for index in range(self.time_bucket_count):
            if count[index] <= 0:
                continue
            experiment.add_scalar(
                _bucket_tag(edges[index], edges[index + 1]),
                float(mean[index]),
                trainer.global_step,
            )


def _sampler_bound(sampler: object, name: str, default: float) -> float:
    return float(getattr(sampler, name, default))


def _sync_bucketed(trainer: Trainer, bucketed: TimeBucketedMean) -> TimeBucketedMean:
    strategy = getattr(trainer, "strategy", None)
    reduce = getattr(strategy, "reduce", None)
    if not callable(reduce):
        return bucketed
    return TimeBucketedMean(
        edges=bucketed.edges,
        total=reduce(bucketed.total, reduce_op="sum"),
        count=reduce(bucketed.count, reduce_op="sum"),
    )


def _is_global_zero(trainer: Trainer) -> bool:
    return bool(getattr(trainer, "is_global_zero", True))


def _bucket_tag(lower: Tensor, upper: Tensor) -> str:
    return f"flow/loss_t/{float(lower):.2f}_{float(upper):.2f}"
