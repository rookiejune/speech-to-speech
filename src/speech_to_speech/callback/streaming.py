from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any, Protocol, cast

from anydataset import types
from anytrain.lightning import experiment
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ..datamodule.streaming import PublishedSample
from ..task import Task
from .interval import TrainInterval


class _StreamingDataModule(Protocol):
    runtime: object

    @property
    def streaming_enabled(self) -> bool: ...

    def start_streaming_synthesis(self, *, owner: bool) -> None: ...

    def check_streaming_synthesis(self, *, owner: bool) -> None: ...

    def close_streaming_synthesis(self, *, owner: bool) -> None: ...

    def set_streaming_global_step(self, step: int) -> None: ...

    def acknowledge_streaming_batch(self, global_step: int) -> None: ...

    def published_streaming_samples(
        self,
        indices: Sequence[int],
        *,
        loader_name: str,
    ) -> list[PublishedSample]: ...


class StreamingSynthesis(Callback):
    """Resume the producer and commit the training cursor at optimizer boundaries."""

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        datamodule = _datamodule(trainer)
        if not datamodule.streaming_enabled:
            return
        self._owner_call(trainer, "start", datamodule.start_streaming_synthesis)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        datamodule = _datamodule(trainer)
        if datamodule.streaming_enabled:
            datamodule.set_streaming_global_step(int(trainer.global_step))

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        datamodule = _datamodule(trainer)
        if datamodule.streaming_enabled:
            self._owner_call(trainer, "check", datamodule.check_streaming_synthesis)

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

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module, exception
        self._close(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._close(trainer)

    def _close(self, trainer: Trainer) -> None:
        datamodule = _datamodule(trainer)
        if not datamodule.streaming_enabled:
            return
        with suppress(Exception):
            datamodule.close_streaming_synthesis(owner=bool(trainer.is_global_zero))

    def _owner_call(
        self,
        trainer: Trainer,
        operation: str,
        method: Any,
    ) -> None:
        error: Exception | None = None
        if trainer.is_global_zero:
            try:
                method(owner=True)
            except Exception as caught:
                error = caught
        payload = (
            None
            if error is None
            else (type(error).__name__, str(error))
        )
        received = trainer.strategy.broadcast(payload, src=0)
        if received is None:
            return
        if (
            not isinstance(received, tuple)
            or len(received) != 2
            or any(not isinstance(value, str) for value in received)
        ):
            raise TypeError("streaming synthesis error broadcast was invalid.")
        error_type, message = cast(tuple[str, str], received)
        failure = RuntimeError(
            f"streaming synthesis {operation} failed on the global owner: "
            f"{error_type}: {message}"
        )
        if error is not None:
            raise failure from error
        raise failure


class SynthesisSampleLogger(Callback):
    """Log persisted teacher artifacts without running the backbone again."""

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
        target_text = _text(published.sample, types.Role.TARGET)
        if text_writer is not None:
            text_writer.add_text(f"{tag}/source_text", source_text, step)
            text_writer.add_text(f"{tag}/target_text", target_text, step)
            text_writer.add_text(
                f"{tag}/metadata",
                json.dumps(
                    {
                        "dataset_index": published.index,
                        "snapshot_id": published.snapshot_id,
                        "source_text": source_text,
                        "target_text": target_text,
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


__all__ = ["StreamingSynthesis", "SynthesisSampleLogger"]
