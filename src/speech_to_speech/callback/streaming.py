from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from anydataset import types
from anytrain.lightning import experiment
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ..datamodule.streaming import PublishedSample, StreamingTelemetry
from ..task import Task
from .interval import TrainInterval
from .gpu import GpuTelemetrySampler


class _StreamingDataModule(Protocol):
    runtime: object

    @property
    def streaming_enabled(self) -> bool: ...

    @property
    def streaming_synthesis_enabled(self) -> bool: ...

    def start_streaming_synthesis(self, *, owner: bool) -> None: ...

    def check_streaming_synthesis(self, *, owner: bool) -> None: ...

    def close_streaming_synthesis(self, *, owner: bool) -> None: ...

    def set_streaming_global_step(self, step: int) -> None: ...

    def acknowledge_streaming_batch(self, global_step: int) -> None: ...

    def set_streaming_stop_requested(self, requested: Callable[[], bool]) -> None: ...

    def streaming_telemetry(
        self,
        *,
        loader_name: str | None = None,
    ) -> StreamingTelemetry | None: ...

    def published_streaming_samples(
        self,
        indices: Sequence[int],
        *,
        loader_name: str,
    ) -> list[PublishedSample]: ...


class _StreamingSynthesisService:
    def __init__(self, datamodule: _StreamingDataModule) -> None:
        self.datamodule = datamodule

    def start(self, *, owner: bool) -> None:
        self.datamodule.start_streaming_synthesis(owner=owner)

    def check(self, *, owner: bool) -> None:
        self.datamodule.check_streaming_synthesis(owner=owner)

    def close(self, *, owner: bool) -> None:
        self.datamodule.close_streaming_synthesis(owner=owner)


def streaming_synthesis_service(
    trainer: Trainer,
) -> _StreamingSynthesisService | None:
    """Expose the stream producer through anytrain's managed service API."""

    datamodule = _datamodule(trainer)
    enabled = getattr(
        datamodule,
        "streaming_synthesis_enabled",
        datamodule.streaming_enabled,
    )
    if not enabled:
        return None
    return _StreamingSynthesisService(datamodule)


class StreamingSynthesis(Callback):
    """Connect trainer stop state and optimizer boundaries to the stream cursor."""

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        datamodule = _datamodule(trainer)
        if not datamodule.streaming_enabled:
            return
        datamodule.set_streaming_stop_requested(
            lambda: bool(trainer.received_sigterm)
        )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        datamodule = _datamodule(trainer)
        if datamodule.streaming_enabled:
            datamodule.set_streaming_global_step(int(trainer.global_step))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        datamodule = _datamodule(trainer)
        if datamodule.streaming_enabled:
            datamodule.acknowledge_streaming_batch(int(trainer.global_step))


class StreamingTelemetryCallback(Callback):
    """Log streaming wait/load/step timing and best-effort GPU utilization."""

    def __init__(
        self,
        *,
        loader_name: str | None = None,
        gpu_sample_interval_seconds: float = 1.0,
        log_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        if loader_name is not None and (not isinstance(loader_name, str) or not loader_name):
            raise ValueError("streaming telemetry loader_name must be non-empty when set.")
        if isinstance(gpu_sample_interval_seconds, bool) or not isinstance(
            gpu_sample_interval_seconds,
            (int, float),
        ):
            raise TypeError("streaming telemetry GPU interval must be numeric.")
        if gpu_sample_interval_seconds < 0:
            raise ValueError("streaming telemetry GPU interval must be non-negative.")
        if type(log_every_n_steps) is not int or log_every_n_steps < 1:
            raise ValueError("streaming telemetry log_every_n_steps must be positive.")
        self.loader_name = loader_name
        self.gpu_sample_interval_seconds = float(gpu_sample_interval_seconds)
        self.log_every_n_steps = log_every_n_steps
        self._sampler: GpuTelemetrySampler | None = None
        self._step_started_at: float | None = None
        self._summary_path: Path | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        datamodule = _datamodule(trainer)
        if not datamodule.streaming_enabled or not trainer.is_global_zero:
            return
        log_dir = _trainer_log_dir(trainer)
        self._summary_path = log_dir / "streaming_gpu_summary.json"
        self._sampler = GpuTelemetrySampler(
            log_dir / "streaming_gpu.csv",
            interval_seconds=self.gpu_sample_interval_seconds,
        )
        self._sampler.start()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        self._step_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        started = self._step_started_at
        self._step_started_at = None
        if started is None or trainer.global_step % self.log_every_n_steps != 0:
            return
        datamodule = _datamodule(trainer)
        telemetry = datamodule.streaming_telemetry(loader_name=self.loader_name)
        if telemetry is None:
            return
        step_seconds = time.perf_counter() - started
        if step_seconds < 0:
            raise RuntimeError("streaming telemetry step timer moved backwards.")
        denominator = telemetry.batch_fetch_seconds + step_seconds
        metrics = {
            "streaming/batch_fetch_seconds": telemetry.batch_fetch_seconds,
            "streaming/batch_wait_seconds": telemetry.batch_wait_seconds,
            "streaming/batch_load_seconds": telemetry.batch_load_seconds,
            "streaming/step_seconds": step_seconds,
            "streaming/wait_seconds_total": telemetry.total_wait_seconds,
            "streaming/fetch_seconds_total": telemetry.total_fetch_seconds,
            "streaming/load_seconds_total": telemetry.total_load_seconds,
            "streaming/wait_events_total": float(telemetry.wait_events),
            "streaming/poll_count_total": float(telemetry.poll_count),
            "streaming/read_position": float(telemetry.read_position),
            "streaming/committed_position": float(telemetry.committed_position),
            "streaming/committed_batches": float(telemetry.committed_batches),
            "streaming/published_samples": float(telemetry.published_samples),
            "streaming/expected_samples": float(telemetry.expected_samples),
            "streaming/wait_ratio": (
                telemetry.batch_wait_seconds / denominator if denominator > 0 else 0.0
            ),
        }
        metrics = _maximum_metrics(trainer, pl_module, metrics)
        if not trainer.is_global_zero:
            return
        sampler = self._sampler
        if sampler is not None:
            latest = sampler.latest()
            gpu_tags = {
                "utilization_gpu_percent": "gpu_utilization_percent",
                "utilization_memory_percent": "gpu_memory_utilization_percent",
                "memory_used_mb": "gpu_memory_used_mb",
                "memory_total_mb": "gpu_memory_total_mb",
                "power_draw_w": "gpu_power_draw_w",
                "power_limit_w": "gpu_power_limit_w",
            }
            metrics.update(
                {
                    f"streaming/{gpu_tags[name]}": value
                    for name, value in latest.items()
                    if name in gpu_tags
                }
            )
        _log_scalars(trainer, metrics)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._stop_sampler(trainer)

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module, exception
        self._stop_sampler(trainer)

    def _stop_sampler(self, trainer: Trainer) -> None:
        del trainer
        sampler = self._sampler
        if sampler is None:
            return
        sampler.stop()
        summary_path = self._summary_path
        if summary_path is not None:
            summary_path.write_text(
                json.dumps(sampler.summary(), ensure_ascii=True, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        self._sampler = None


class SynthesisSampleLogger(Callback):
    """Log generated artifacts alongside their dataset translation references."""

    def __init__(
        self,
        indices: Sequence[int],
        every_n_steps: int,
        *,
        loader_name: str,
    ) -> None:
        super().__init__()
        if not indices:
            raise ValueError("synthesis sample indices must not be empty.")
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError(
                "synthesis sample indices must be non-negative integers."
            )
        if len(set(indices)) != len(indices):
            raise ValueError("synthesis sample indices must be unique.")
        if type(every_n_steps) is not int or every_n_steps < 1:
            raise ValueError("synthesis sample every_n_steps must be positive.")
        if not isinstance(loader_name, str) or not loader_name:
            raise ValueError("synthesis sample loader_name must be non-empty.")
        self.indices = tuple(indices)
        self.loader_name = loader_name
        self.interval = TrainInterval(every_n_steps=every_n_steps)
        self._logged: set[int] = set()

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            loader_name=self.loader_name,
            indices=self.indices,
            every_n_steps=self.interval.every_n_steps,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        if not trainer.is_global_zero:
            return
        if not self.interval.should_run(int(trainer.global_step)):
            return
        self._log_pending(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            self._log_pending(trainer)

    def _log_pending(self, trainer: Trainer) -> None:
        pending = [index for index in self.indices if index not in self._logged]
        if not pending:
            return
        datamodule = _datamodule(trainer)
        samples = datamodule.published_streaming_samples(
            pending,
            loader_name=self.loader_name,
        )
        if not samples:
            return
        audio_writer = experiment.audio(trainer)
        text_writer = experiment.text(trainer)
        if audio_writer is None and text_writer is None:
            return
        for published in samples:
            self._log_sample(
                datamodule,
                published,
                audio_writer=audio_writer,
                text_writer=text_writer,
                step=int(trainer.global_step),
            )
            self._logged.add(published.index)

    def _log_sample(
        self,
        datamodule: _StreamingDataModule,
        published: PublishedSample,
        *,
        audio_writer: Any | None,
        text_writer: Any | None,
        step: int,
    ) -> None:
        tag = f"synthesis/{published.index}"
        source_text = _text(published.sample, types.Role.SOURCE)
        model_translation = _text(published.sample, types.Role.TARGET)
        dataset_translation = published.reference_translation
        if text_writer is not None:
            text_writer.add_text(f"{tag}/source_text", source_text, step)
            text_writer.add_text(
                f"{tag}/model_translation",
                model_translation,
                step,
            )
            text_writer.add_text(
                f"{tag}/dataset_translation",
                dataset_translation,
                step,
            )
            text_writer.add_text(
                f"{tag}/translation_comparison",
                _translation_comparison(
                    source_text,
                    model_translation,
                    dataset_translation,
                ),
                step,
            )
            text_writer.add_text(
                f"{tag}/metadata",
                json.dumps(
                    {
                        "dataset_index": published.index,
                        "snapshot_id": published.snapshot_id,
                        "source_text": source_text,
                        "model_translation": model_translation,
                        "dataset_translation": dataset_translation,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                step,
            )
        if audio_writer is None:
            return
        from .logging.sample_report import sample_audio

        logging_datamodule = cast(Any, datamodule)
        source, source_rate = sample_audio(
            logging_datamodule,
            published.sample,
            Task.S2ST,
            source=True,
        )
        target, target_rate = sample_audio(
            logging_datamodule,
            published.sample,
            Task.S2ST,
            source=False,
        )
        audio_writer.add_audio(
            f"{tag}/source_audio",
            source,
            step,
            sample_rate=source_rate,
        )
        audio_writer.add_audio(
            f"{tag}/target_audio",
            target,
            step,
            sample_rate=target_rate,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "interval": self.interval.state_dict(),
            "logged": sorted(self._logged),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        interval = state_dict.get("interval", {})
        if not isinstance(interval, Mapping):
            raise TypeError("synthesis sample interval state must be a mapping.")
        self.interval.load_state_dict(interval)
        logged = state_dict.get("logged", [])
        if not isinstance(logged, list) or any(
            type(index) is not int or index < 0 for index in logged
        ):
            raise TypeError(
                "synthesis sample logged state must be non-negative integers."
            )
        self._logged = set(cast(list[int], logged))


def _trainer_log_dir(trainer: Trainer) -> Path:
    logger = getattr(trainer, "logger", None)
    log_dir = getattr(logger, "log_dir", None)
    if log_dir is None:
        log_dir = getattr(trainer, "default_root_dir", None)
    if log_dir is None:
        raise RuntimeError("streaming telemetry requires a trainer log directory.")
    path = Path(str(log_dir)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maximum_metrics(
    trainer: Trainer,
    pl_module: LightningModule,
    metrics: Mapping[str, float],
) -> dict[str, float]:
    world_size = int(getattr(trainer, "world_size", 1))
    if world_size <= 1:
        return dict(metrics)
    device = getattr(pl_module, "device", torch.device("cpu"))
    values = torch.tensor(
        list(metrics.values()),
        dtype=torch.float64,
        device=device,
    )
    reduce = getattr(trainer.strategy, "reduce", None)
    if not callable(reduce):
        raise RuntimeError("distributed streaming telemetry requires strategy.reduce().")
    reduced = reduce(values, reduce_op="max")
    if not isinstance(reduced, torch.Tensor) or reduced.shape != values.shape:
        raise TypeError("distributed streaming telemetry reduction returned an invalid tensor.")
    return dict(zip(metrics, (float(value) for value in reduced.detach().cpu().tolist())))


def _log_scalars(trainer: Trainer, metrics: Mapping[str, float]) -> None:
    writer = experiment.scalar(trainer)
    step = int(trainer.global_step)
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(name, value, step)
        return
    logger = getattr(trainer, "logger", None)
    log_metrics = getattr(logger, "log_metrics", None)
    if callable(log_metrics):
        log_metrics(dict(metrics), step=step)


def _datamodule(trainer: Trainer) -> _StreamingDataModule:
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        raise RuntimeError("streaming callback requires Trainer.fit(..., datamodule=...).")
    return cast(_StreamingDataModule, cast(object, datamodule))


def _text(sample: types.Sample, role: types.Role) -> str:
    reference = (role, types.Modality.TEXT)
    try:
        item = sample[reference]
    except KeyError as error:
        raise KeyError(f"synthesis sample is missing {role.value} text.") from error
    if not isinstance(item, types.TextItem):
        raise TypeError(f"synthesis sample {role.value} text must be a TextItem.")
    value = item.views.get(types.TextView.TEXT)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"synthesis sample {role.value} TextView.TEXT must be non-empty."
        )
    return value


def _translation_comparison(
    source_text: str,
    model_translation: str,
    dataset_translation: str,
) -> str:
    rows = (
        ("source", source_text),
        ("model translation", model_translation),
        ("dataset translation", dataset_translation),
    )
    return "\n".join(
        (
            "| artifact | text |",
            "| --- | --- |",
            *(f"| {name} | {_markdown_cell(value)} |" for name, value in rows),
        )
    )


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


__all__ = [
    "StreamingSynthesis",
    "StreamingTelemetryCallback",
    "SynthesisSampleLogger",
    "streaming_synthesis_service",
]
