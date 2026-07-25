from __future__ import annotations

from typing import Protocol

from anytrain.lightning import LossTimeBucketLoggerCallback
from lightning import LightningModule, Trainer

from .._lightning import text_experiment
from ..interval import TrainInterval


class _FlowRuntime(Protocol):
    @property
    def time_sampler(self) -> object: ...


class FlowMatchingLogger(LossTimeBucketLoggerCallback):
    """Log flow-matching sampler configuration and sampled training times."""

    def __init__(
        self,
        runtime: _FlowRuntime,
        every_n_steps: int | None = 100,
        every_audio_seconds: float | None = None,
        time_bucket_count: int = 10,
    ) -> None:
        self.runtime = runtime
        self.interval = TrainInterval(
            every_n_steps=every_n_steps,
            every_audio_seconds=every_audio_seconds,
        )
        super().__init__(
            item_name="flow_matching",
            detail_key="t",
            histogram_tag="flow/time",
            scalar_template="flow/loss_t/{lower:.2f}_{upper:.2f}",
            every_n_steps=every_n_steps,
            bucket_count=time_bucket_count,
            t_min=_sampler_bound(runtime.time_sampler, "t_min", 0.0),
            t_max=_sampler_bound(runtime.time_sampler, "t_max", 1.0),
        )

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

    def should_log(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
    ) -> bool:
        return self.interval.should_run(trainer, pl_module, batch)

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, float]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))


def _sampler_bound(sampler: object, name: str, default: float) -> float:
    return float(getattr(sampler, name, default))
