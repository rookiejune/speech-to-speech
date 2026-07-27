from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from anytrain.lightning import experiment
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ..interval import TrainInterval
from ...generation import TextProbe, TextProbeResult


class _Module(Protocol):
    def evaluate_text(
        self,
        probes: Mapping[str, TextProbe],
        *,
        max_new_tokens: int,
    ) -> dict[str, TextProbeResult]: ...


class TextRetentionLogger(Callback):
    """Log deterministic text generation and reference NLL during training."""

    def __init__(
        self,
        probes: Mapping[str, TextProbe],
        *,
        every_n_steps: int | None = 1_000,
        every_audio_seconds: float | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__()
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        self.probes = dict(probes)
        self.every_n_steps = every_n_steps
        self.interval = TrainInterval(
            every_n_steps=every_n_steps,
            every_audio_seconds=every_audio_seconds,
        )
        self.max_new_tokens = max_new_tokens
        self._baseline_nll: dict[str, float] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        module = cast(_Module, cast(object, pl_module))
        self._log(trainer, module, baseline=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        if not self.interval.should_run(trainer, pl_module, batch):
            return
        module = cast(_Module, cast(object, pl_module))
        self._log(trainer, module, baseline=False)

    def state_dict(self) -> dict[str, dict[str, float] | dict[str, float]]:
        return {
            "interval": self.interval.state_dict(),
            "baseline_nll": dict(self._baseline_nll),
        }

    def load_state_dict(
        self,
        state_dict: dict[str, dict[str, float] | dict[str, float]],
    ) -> None:
        self.interval.load_state_dict(
            cast(dict[str, float], state_dict.get("interval", {}))
        )
        self._baseline_nll = dict(
            cast(dict[str, float], state_dict.get("baseline_nll", {}))
        )

    def _log(self, trainer: Trainer, module: _Module, *, baseline: bool) -> None:
        if not trainer.is_global_zero:
            return
        scalar_writer = experiment.scalar(trainer)
        text_writer = experiment.text(trainer)
        if scalar_writer is None and text_writer is None:
            return

        results = module.evaluate_text(
            self.probes,
            max_new_tokens=self.max_new_tokens,
        )
        if baseline:
            self._baseline_nll = {
                name: result["nll"] for name, result in results.items()
            }

        for name, probe in self.probes.items():
            result = results[name]
            nll = result["nll"]
            if scalar_writer is not None:
                scalar_writer.add_scalar(
                    f"text_retention/{name}/nll", nll, trainer.global_step
                )
                baseline_nll = self._baseline_nll.get(name)
                if baseline_nll is not None:
                    scalar_writer.add_scalar(
                        f"text_retention/{name}/nll_delta",
                        nll - baseline_nll,
                        trainer.global_step,
                    )
            if text_writer is not None:
                text_writer.add_text(
                    f"text_retention/{name}/generation",
                    _text(probe, result),
                    trainer.global_step,
                )


def _text(probe: TextProbe, result: TextProbeResult) -> str:
    return "\n\n".join(
        (
            f"Instruction: {probe['instruction']}",
            f"Reference: {probe['reference']}",
            f"Generated: {result['generated']}",
        )
    )


__all__ = ["TextProbe", "TextRetentionLogger"]
