from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ...pl_module.text import TextProbe, TextProbeResult


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
        every_n_steps: int = 1_000,
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        self.probes = dict(probes)
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self._baseline_nll: dict[str, float] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log(trainer, cast(_Module, pl_module), baseline=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if trainer.global_step % self.every_n_steps != 0:
            return
        self._log(trainer, cast(_Module, pl_module), baseline=False)

    def _log(self, trainer: Trainer, module: _Module, *, baseline: bool) -> None:
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        results = module.evaluate_text(
            self.probes,
            max_new_tokens=self.max_new_tokens,
        )
        if baseline:
            self._baseline_nll = {
                name: result["nll"] for name, result in results.items()
            }
        if not trainer.is_global_zero:
            return

        experiment = logger.experiment
        for name, probe in self.probes.items():
            result = results[name]
            nll = result["nll"]
            if hasattr(experiment, "add_scalar"):
                experiment.add_scalar(
                    f"text_retention/{name}/nll", nll, trainer.global_step
                )
                baseline_nll = self._baseline_nll.get(name)
                if baseline_nll is not None:
                    experiment.add_scalar(
                        f"text_retention/{name}/nll_delta",
                        nll - baseline_nll,
                        trainer.global_step,
                    )
            if hasattr(experiment, "add_text"):
                experiment.add_text(
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
