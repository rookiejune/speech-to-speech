from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...loss.types import LossItem
from .trace import event, stage


class _FlowOracle(Protocol):
    device: torch.device

    def sample(self, semantic_codes: Tensor, *, seed: int) -> Tensor: ...

    def target(
        self,
        acoustic_codes: Tensor,
        *,
        normalize: bool,
        log_first: bool = True,
    ) -> Tensor: ...


class _TokenOracle(Protocol):
    device: torch.device

    def predict(self, codes: Tensor) -> Tensor: ...


class CodecOracleLogger(Callback):
    """Log fixed-sample metrics and waveforms for codec oracle experiments."""

    def __init__(
        self,
        *,
        objective: str,
        codec: Any,
        codes: Tensor,
        output_dir: Path,
        sample_rate: int,
        seed: int,
        sample_every_n_steps: int,
        histogram_every_n_steps: int,
        save_audio: bool,
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.objective = objective
        self.codec = codec
        self.codes = codes
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.seed = seed
        self.sample_every_n_steps = sample_every_n_steps
        self.histogram_every_n_steps = histogram_every_n_steps
        self.save_audio = save_audio
        self.metadata = dict(metadata)
        self.losses: list[float] = []
        self.samples: list[dict[str, float | int]] = []

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        event("trainer.fit", "started", objective=self.objective)
        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_text"):
            experiment.add_text(
                "oracle/config",
                json.dumps(self.metadata, indent=2, sort_keys=True),
                0,
            )
        with stage("callback.oracle_decode", objective=self.objective):
            waveform = self._oracle_waveform(pl_module)
        self._audio(trainer, "oracle/reconstruction", waveform, 0)
        if self.save_audio:
            self._save(waveform, "reconstruction.wav")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        if isinstance(loss, Tensor):
            self.losses.append(float(loss.detach().float()))
        if isinstance(outputs, Mapping):
            item = outputs.get("flow_matching")
            if (
                isinstance(item, LossItem)
                and item.details is not None
                and self.histogram_every_n_steps > 0
                and trainer.global_step % self.histogram_every_n_steps == 0
            ):
                self._histogram(trainer, item)
        if (
            trainer.is_global_zero
            and self.sample_every_n_steps > 0
            and trainer.global_step % self.sample_every_n_steps == 0
        ):
            self._sample(trainer, pl_module)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del trainer, optimizer
        gradients = [
            parameter.grad.detach().norm(2)
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            pl_module.log(
                "train/grad_norm",
                torch.stack(gradients).norm(2),
                on_step=True,
                sync_dist=True,
            )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if not self.samples or self.samples[-1]["step"] != trainer.global_step:
            self._sample(trainer, pl_module)
        report = {
            **self.metadata,
            "steps": len(self.losses),
            "loss": _loss_report(self.losses),
            "samples": self.samples,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        event("metrics.write", "done", path=str(self.output_dir / "metrics.json"))

    def _sample(self, trainer: Trainer, module: LightningModule) -> None:
        with stage(
            "callback.sample",
            objective=self.objective,
            step=trainer.global_step,
        ):
            if self.objective == "flow":
                value, waveform = self._flow_sample(cast(_FlowOracle, module))
                tag = "oracle/sample_feature_mse"
            else:
                value, waveform = self._token_sample(cast(_TokenOracle, module))
                tag = "oracle/token_accuracy"
        self.samples.append({"step": trainer.global_step, "value": value})
        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_scalar"):
            experiment.add_scalar(tag, value, trainer.global_step)
        self._audio(trainer, "oracle/sample", waveform, trainer.global_step)
        if self.save_audio:
            self._save(waveform, f"sample-step-{trainer.global_step:06d}.wav")

    def _flow_sample(self, module: _FlowOracle) -> tuple[float, Tensor]:
        codes = self.codes.unsqueeze(0).to(module.device)
        semantic_codes = codes[..., 0]
        acoustic_codes = codes[..., 1:]
        sampled = module.sample(semantic_codes, seed=self.seed)
        target = module.target(acoustic_codes, normalize=False, log_first=False)
        value = float(F.mse_loss(sampled.float(), target.float()))
        with stage("callback.waveform_decode", objective=self.objective):
            waveform = self.codec.decode_features(
                semantic_codes.unsqueeze(-1),
                sampled,
            )
        return value, waveform

    def _token_sample(self, module: _TokenOracle) -> tuple[float, Tensor]:
        codes = self.codes[:, 0].unsqueeze(0).to(module.device)
        predicted = module.predict(codes)
        value = float(predicted.eq(codes).float().mean())
        with stage("callback.waveform_decode", objective=self.objective):
            waveform = self.codec.decode(predicted.unsqueeze(-1))
        return value, waveform

    def _oracle_waveform(self, module: LightningModule) -> Tensor:
        codes = self.codes.unsqueeze(0).to(module.device)
        if self.objective == "flow":
            semantic_codes = codes[..., :1]
            acoustic_codes = codes[..., 1:]
            with stage("callback.dequantize", objective=self.objective):
                features = cast(_FlowOracle, module).target(
                    acoustic_codes,
                    normalize=False,
                    log_first=False,
                )
            with stage("callback.waveform_decode", objective=self.objective):
                return self.codec.decode_features(semantic_codes, features)
        with stage("callback.waveform_decode", objective=self.objective):
            return self.codec.decode(codes)

    def _audio(
        self,
        trainer: Trainer,
        tag: str,
        waveform: Tensor,
        step: int,
    ) -> None:
        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_audio"):
            return
        experiment.add_audio(
            tag,
            waveform.detach().float().cpu()[0],
            step,
            sample_rate=self.sample_rate,
        )

    def _histogram(self, trainer: Trainer, item: LossItem) -> None:
        experiment = getattr(trainer.logger, "experiment", None)
        values = item.details.get("t") if item.details is not None else None
        if (
            experiment is not None
            and hasattr(experiment, "add_histogram")
            and isinstance(values, Tensor)
        ):
            experiment.add_histogram(
                "flow/time", values.detach().float().cpu(), trainer.global_step
            )

    def _save(self, waveform: Tensor, filename: str) -> None:
        import torchaudio

        directory = self.output_dir / "audio"
        directory.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            directory / filename,
            waveform.detach().float().cpu()[0],
            self.sample_rate,
        )


def _loss_report(
    values: Sequence[float],
    window: int = 20,
) -> dict[str, float | int]:
    if not values:
        return {"steps": 0}
    size = min(window, len(values))
    first = sum(values[:size]) / size
    last = sum(values[-size:]) / size
    return {
        "steps": len(values),
        "window": size,
        "first": values[0],
        "last": values[-1],
        "first_mean": first,
        "last_mean": last,
        "last_to_first": last / first,
    }


__all__ = ["CodecOracleLogger"]
