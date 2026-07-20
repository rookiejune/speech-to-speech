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

from ..callback import WorldSizeContract as BaseWorldSizeContract
from ..callback._lightning import (
    audio_experiment,
    histogram_experiment,
    scalar_experiment,
    text_experiment,
)
from ..loss.types import LossItem
from ..reporting import window_summary
from .trace import event, timed
from .types import Objective


class _AcousticFlowScreening(Protocol):
    device: torch.device

    def sample(self, semantic_codes: Tensor, *, seed: int) -> Tensor: ...

    def features(self, acoustic_codes: Tensor) -> Tensor: ...


class _AcousticRVQScreening(Protocol):
    device: torch.device

    def sample(self, semantic_codes: Tensor, *, seed: int) -> Tensor: ...

    def features(self, acoustic_codes: Tensor) -> Tensor: ...


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
            reduced = trainer.strategy.reduce(
                loss.detach().float(),
                reduce_op="mean",
            )
            if trainer.is_global_zero:
                self.losses.append(float(reduced))
        if trainer.is_global_zero and isinstance(outputs, Mapping):
            item = outputs.get(
                "flow_matching" if self.objective is Objective.FLOW else "rvq"
            )
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
                flow_module = cast(_AcousticFlowScreening, cast(object, module))
                feature_mse, waveform = self._flow_sample(flow_module)
                primary = feature_mse
                metrics = {"feature_mse": feature_mse}
            elif self.objective is Objective.RVQ:
                rvq_module = cast(_AcousticRVQScreening, cast(object, module))
                metrics, waveform = self._rvq_sample(rvq_module)
                primary = metrics["code_accuracy"]
            else:
                raise AssertionError(f"unsupported objective: {self.objective}")
        self.samples.append(
            {"step": trainer.global_step, "value": primary, **metrics}
        )
        experiment = scalar_experiment(trainer)
        if experiment is not None:
            for name, value in metrics.items():
                experiment.add_scalar(
                    f"oracle/sample_{name}",
                    value,
                    trainer.global_step,
                )
        self._audio(trainer, "oracle/sample", waveform, trainer.global_step)
        if self.save_audio:
            self._save(waveform, f"sample-step-{trainer.global_step:06d}.wav")

    def _flow_sample(self, module: _AcousticFlowScreening) -> tuple[float, Tensor]:
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

    def _rvq_sample(
        self,
        module: _AcousticRVQScreening,
    ) -> tuple[dict[str, float], Tensor]:
        codes = self.codes.unsqueeze(0).to(module.device)
        semantic_codes = codes[..., 0]
        target_codes = codes[..., 1:]
        sampled_codes = module.sample(semantic_codes, seed=self.seed)
        matches = sampled_codes.eq(target_codes)
        codebook_accuracy = matches.float().mean(dim=(0, 1))
        target_features = module.features(target_codes)
        sampled_features = module.features(sampled_codes)
        metrics = {
            "code_accuracy": float(matches.float().mean()),
            "feature_mse": float(
                F.mse_loss(sampled_features.float(), target_features.float())
            ),
            **{
                f"codebook_{index}_accuracy": float(value)
                for index, value in enumerate(codebook_accuracy)
            },
        }
        with timed("callback.waveform_decode", objective=self.objective):
            waveform = self.codec.decode_features(
                semantic_codes.unsqueeze(-1),
                sampled_features,
            )
        return metrics, waveform

    def _oracle_waveform(self, module: LightningModule) -> Tensor:
        codes = self.codes.unsqueeze(0).to(module.device)
        if self.objective is Objective.FLOW:
            semantic_codes = codes[..., :1]
            acoustic_codes = codes[..., 1:]
            with timed("callback.dequantize", objective=self.objective):
                flow_module = cast(_AcousticFlowScreening, cast(object, module))
                features = flow_module.features(acoustic_codes)
            with timed("callback.waveform_decode", objective=self.objective):
                return self.codec.decode_features(semantic_codes, features)
        if self.objective is Objective.RVQ:
            semantic_codes = codes[..., :1]
            acoustic_codes = codes[..., 1:]
            with timed("callback.dequantize", objective=self.objective):
                rvq_module = cast(_AcousticRVQScreening, cast(object, module))
                features = rvq_module.features(acoustic_codes)
            with timed("callback.waveform_decode", objective=self.objective):
                return self.codec.decode_features(semantic_codes, features)
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
        if experiment is None or item.details is None:
            return
        if self.objective is Objective.FLOW:
            values = item.details.get("t")
            if isinstance(values, Tensor):
                experiment.add_histogram(
                    "flow/time", values.detach().float().cpu(), trainer.global_step
                )
            return
        if self.objective is Objective.RVQ:
            for name, values in item.details.items():
                experiment.add_histogram(
                    f"rvq/{name}_loss",
                    values.detach().float().cpu(),
                    trainer.global_step,
                )
            return
        raise AssertionError(f"unsupported objective: {self.objective}")

    def _save(self, waveform: Tensor, filename: str) -> None:
        import torchaudio

        directory = self.output_dir / "audio"
        directory.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            directory / filename,
            waveform.detach().float().cpu()[0],
            self.sample_rate,
        )


class WorldSizeContract(BaseWorldSizeContract):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        if trainer.is_global_zero:
            event(
                "distributed.contract",
                "ready",
                strategy=type(trainer.strategy).__name__,
                world_size=trainer.world_size,
            )


__all__ = ["Logger", "WorldSizeContract"]
