from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torch.utils.data import DistributedSampler

from ..callback._lightning import (
    attached_datamodule,
    audio_experiment,
    histogram_experiment,
    scalar_experiment,
    text_experiment,
)
from ..loss.types import LossItem
from ..reporting import window_summary
from .trace import event, timed
from .types import Objective


class _FlowOracle(Protocol):
    device: torch.device

    def sample(self, semantic_codes: Tensor, *, seed: int) -> Tensor: ...

    def features(self, acoustic_codes: Tensor) -> Tensor: ...


class _TokenOracle(Protocol):
    device: torch.device

    def teacher_forced_ids(self, codes: Tensor) -> Tensor: ...


class Logger(Callback):
    """Log fixed-sample metrics and waveforms for codec oracle experiments."""

    def __init__(
        self,
        *,
        objective: Objective,
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
        experiment = text_experiment(trainer)
        if experiment is not None:
            experiment.add_text(
                "oracle/config",
                json.dumps(self.metadata, indent=2, sort_keys=True),
                0,
            )
        with timed("callback.oracle_decode", objective=self.objective):
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

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if not self.samples or self.samples[-1]["step"] != trainer.global_step:
            self._sample(trainer, pl_module)
        report = {
            **self.metadata,
            "steps": len(self.losses),
            "loss": window_summary(self.losses),
            "samples": self.samples,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        event("metrics.write", "done", path=str(self.output_dir / "metrics.json"))

    def _sample(self, trainer: Trainer, module: LightningModule) -> None:
        with timed(
            "callback.sample",
            objective=self.objective,
            step=trainer.global_step,
        ):
            if self.objective is Objective.FLOW:
                flow_oracle = cast(_FlowOracle, cast(object, module))
                value, waveform = self._flow_sample(flow_oracle)
                tag = "oracle/sample_feature_mse"
                audio_tag = "oracle/sample"
                filename = f"sample-step-{trainer.global_step:06d}.wav"
            elif self.objective is Objective.TOKEN:
                token_oracle = cast(_TokenOracle, cast(object, module))
                value, waveform = self._token_probe(token_oracle)
                tag = "oracle/teacher_forced_accuracy"
                audio_tag = "oracle/teacher_forced"
                filename = f"teacher-forced-step-{trainer.global_step:06d}.wav"
            else:
                raise AssertionError(f"unsupported objective: {self.objective}")
        self.samples.append({"step": trainer.global_step, "value": value})
        experiment = scalar_experiment(trainer)
        if experiment is not None:
            experiment.add_scalar(tag, value, trainer.global_step)
        self._audio(trainer, audio_tag, waveform, trainer.global_step)
        if self.save_audio:
            self._save(waveform, filename)

    def _flow_sample(self, module: _FlowOracle) -> tuple[float, Tensor]:
        codes = self.codes.unsqueeze(0).to(module.device)
        semantic_codes = codes[..., 0]
        acoustic_codes = codes[..., 1:]
        sampled = module.sample(semantic_codes, seed=self.seed)
        target = module.features(acoustic_codes)
        value = float(F.mse_loss(sampled.float(), target.float()))
        with timed("callback.waveform_decode", objective=self.objective):
            waveform = self.codec.decode_features(
                semantic_codes.unsqueeze(-1),
                sampled,
            )
        return value, waveform

    def _token_probe(self, module: _TokenOracle) -> tuple[float, Tensor]:
        codes = self.codes[:, 0].unsqueeze(0).to(module.device)
        predicted = module.teacher_forced_ids(codes)
        value = float(predicted.eq(codes).float().mean())
        with timed("callback.waveform_decode", objective=self.objective):
            waveform = self.codec.decode(predicted.unsqueeze(-1))
        return value, waveform

    def _oracle_waveform(self, module: LightningModule) -> Tensor:
        codes = self.codes.unsqueeze(0).to(module.device)
        if self.objective is Objective.FLOW:
            semantic_codes = codes[..., :1]
            acoustic_codes = codes[..., 1:]
            with timed("callback.dequantize", objective=self.objective):
                flow_oracle = cast(_FlowOracle, cast(object, module))
                features = flow_oracle.features(acoustic_codes)
            with timed("callback.waveform_decode", objective=self.objective):
                return self.codec.decode_features(semantic_codes, features)
        if self.objective is Objective.TOKEN:
            with timed("callback.waveform_decode", objective=self.objective):
                return self.codec.decode(codes)
        raise AssertionError(f"unsupported objective: {self.objective}")

    def _audio(
        self,
        trainer: Trainer,
        tag: str,
        waveform: Tensor,
        step: int,
    ) -> None:
        experiment = audio_experiment(trainer)
        if experiment is None:
            return
        experiment.add_audio(
            tag,
            waveform.detach().float().cpu()[0],
            step,
            sample_rate=self.sample_rate,
        )

    def _histogram(self, trainer: Trainer, item: LossItem) -> None:
        experiment = histogram_experiment(trainer)
        values = item.details.get("t") if item.details is not None else None
        if experiment is not None and isinstance(values, Tensor):
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


class WorldSizeContract(Callback):
    def __init__(self, expected: int) -> None:
        super().__init__()
        self.expected = expected

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.world_size != self.expected:
            raise RuntimeError(
                f"expected DDP world size {self.expected}, got {trainer.world_size}."
            )
        if trainer.is_global_zero:
            event(
                "distributed.contract",
                "ready",
                strategy=type(trainer.strategy).__name__,
                world_size=trainer.world_size,
            )


class SamplerEpochSetter(Callback):
    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        sampler = getattr(attached_datamodule(trainer), "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(trainer.current_epoch)


__all__ = ["Logger", "SamplerEpochSetter", "WorldSizeContract"]
