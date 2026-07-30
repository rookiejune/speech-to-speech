from __future__ import annotations

from typing import Protocol

from anytrain.lightning import LossTimeBucketLoggerCallback, experiment
from lightning import LightningModule, Trainer

from ..interval import TrainInterval


class _FlowRuntime(Protocol):
    @property
    def time_sampler(self) -> object: ...


class FlowMatchingLogger(LossTimeBucketLoggerCallback):
    """Log flow-matching sampler configuration and sampled training times."""

    def __init__(
        self,
        runtime: _FlowRuntime,
        every_n_steps: int = 100,
        time_bucket_count: int = 10,
    ) -> None:
        self.runtime = runtime
        self.interval = TrainInterval(every_n_steps=every_n_steps)
        super().__init__(
            item_name="flow_matching",
            detail_key="t",
            histogram_tag="acoustic/flow_matching/time",
            scalar_template=(
                "acoustic/flow_matching/loss_t/{lower:.2f}_{upper:.2f}"
            ),
            every_n_steps=every_n_steps,
            bucket_count=time_bucket_count,
            t_min=_sampler_bound(runtime.time_sampler, "t_min", 0.0),
            t_max=_sampler_bound(runtime.time_sampler, "t_max", 1.0),
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        writer = experiment.text(trainer)
        if writer is None:
            return

        sampler = self.runtime.time_sampler
        config = vars(sampler)
        values = [f"sampler={type(sampler).__name__}"]
        values.extend(
            f"{name}={config[name]}"
            for name in ("mean", "std", "t_min", "t_max")
            if name in config
        )
        writer.add_text("acoustic/flow_matching/config", "\n".join(values), 0)

    def should_log(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
    ) -> bool:
        del pl_module, batch
        return self.interval.should_run(int(trainer.global_step))

    def state_dict(self) -> dict[str, dict[str, int | None]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, int | None]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))


def _sampler_bound(sampler: object, name: str, default: float) -> float:
    return float(getattr(sampler, name, default))
