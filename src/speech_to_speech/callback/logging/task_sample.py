from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from anydataset import types
from anytrain.lightning import experiment
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ...datamodule.batch import ModelBatch
from ...datamodule.diagnostic import SampleSplit
from ...generation import decode_response_text
from ...generation.service import requests_from_batch
from ...generation.contract import Result
from ...task import PredictionModality, Request, Task
from .._oom import batch_report, generation_report, report_oom
from ..interval import TrainInterval
from .sample_report import (
    DataModule,
    GenerationKwargs,
    Module,
    audio_metrics,
    build_result_metadata,
    log_source_audio,
    log_target_reference_audio,
    metadata_json,
    prediction_modality,
    reference_text,
    sample_log_record,
    text_metrics,
)


def _attached_datamodule(trainer: Trainer) -> object:
    value = getattr(trainer, "datamodule", None)
    if value is None:
        raise RuntimeError("callback requires Trainer.fit(..., datamodule=...).")
    return value


class TaskSampleLogger(Callback):
    def __init__(
        self,
        indices: Sequence[int],
        every_n_steps: int | None,
        *,
        loader_name: str,
        task: Task,
        split: SampleSplit = SampleSplit.TRAIN,
        seed: int = 0,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if not indices:
            raise ValueError("indices must contain at least one sample index.")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("indices must contain integer sample indices.")
        if not isinstance(split, SampleSplit):
            raise TypeError("task sample split must be a SampleSplit.")
        if not loader_name:
            raise ValueError("task sample loader_name must not be empty.")
        if not isinstance(task, Task):
            raise TypeError("task sample task must be a Task.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("task sample seed must be an integer.")
        if seed < 0:
            raise ValueError("task sample seed must be non-negative.")
        if every_n_steps is None:
            raise ValueError("task sample every_n_steps must be set.")
        self.indices = list(indices)
        self.loader_name = loader_name
        self.split = split
        self.task = task
        self.seed = seed
        self.every_n_steps = every_n_steps
        self.interval = TrainInterval(every_n_steps=every_n_steps)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.samples: list[types.Sample] = []

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            loader_name=self.loader_name,
            split=self.split.value,
            task=self.task.value,
            seed=self.seed,
            indices=tuple(self.indices),
            every_n_steps=self.interval.every_n_steps,
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        datamodule = cast(DataModule, _attached_datamodule(trainer))
        self.samples = datamodule.diagnostic_samples(
            self.indices,
            split=self.split,
            loader_name=self.loader_name,
        )

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        del batch, batch_idx
        if not self._should_log(trainer):
            return
        audio_writer, scalar_writer, text_writer = self._writers(trainer)
        if audio_writer is None and scalar_writer is None and text_writer is None:
            return
        module = cast(Module, cast(object, pl_module))
        datamodule = cast(DataModule, _attached_datamodule(trainer))
        sample_batch = self._materialize_samples(trainer, pl_module, module, datamodule)
        requests = requests_from_batch(sample_batch)
        generation = self._generation_kwargs()
        generation_metadata = {**generation, "seed": self.seed}
        results = self._generate_samples(
            trainer,
            pl_module,
            module,
            datamodule,
            sample_batch,
            requests,
            generation,
            generation_metadata,
            text_writer,
        )
        if len(results) != len(requests):
            error = RuntimeError("task sample generation returned the wrong row count.")
            self._log_failed_rows(
                text_writer,
                datamodule,
                sample_batch,
                requests,
                generation_metadata,
                error,
                step=trainer.global_step,
            )
            raise error
        for row, (dataset_index, sample, request, result) in enumerate(
            zip(self.indices, self.samples, requests, results)
        ):
            log_result_row(
                RowLogContext(
                    audio_writer=audio_writer,
                    scalar_writer=scalar_writer,
                    text_writer=text_writer,
                    datamodule=datamodule,
                    module=module,
                    row_batch=sample_batch.row(row),
                    dataset_index=dataset_index,
                    sample=sample,
                    request=request,
                    result=result,
                    generation_metadata=generation_metadata,
                    tag=self._tag(dataset_index),
                    step=trainer.global_step,
                ),
                max_new_tokens=self.max_new_tokens,
            )

    def _should_log(self, trainer: Trainer) -> bool:
        return self.interval.should_run(int(trainer.global_step)) and trainer.is_global_zero

    def _writers(self, trainer: Trainer) -> tuple[Any | None, Any | None, Any | None]:
        return experiment.audio(trainer), experiment.scalar(trainer), experiment.text(trainer)

    def _materialize_samples(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        module: Module,
        datamodule: DataModule,
    ) -> ModelBatch:
        collator = datamodule.diagnostic_collator(
            self.task,
            split=self.split,
            loader_name=self.loader_name,
        )
        diagnostic_batch = collator(self.samples)
        try:
            materialized = module.materialize_batch(diagnostic_batch)
        except torch.OutOfMemoryError as error:
            report_oom(
                trainer,
                pl_module,
                error,
                phase="task_sample_materialize",
                inputs=batch_report(diagnostic_batch),
            )
            raise
        if not isinstance(materialized, ModelBatch):
            raise TypeError("task sample logging requires one materialized ModelBatch.")
        return materialized

    def _generate_samples(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        module: Module,
        datamodule: DataModule,
        sample_batch: ModelBatch,
        requests: Sequence[Request],
        generation: GenerationKwargs,
        generation_metadata: Mapping[str, Any],
        text_writer: Any | None,
    ) -> list[Result]:
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.set_rng_state(torch.Generator().manual_seed(self.seed).get_state())
                if cuda_devices:
                    torch.cuda.manual_seed(self.seed)
                return module.generate(requests, **generation)
        except Exception as error:
            if report_oom(
                trainer,
                pl_module,
                error,
                phase="task_sample_generation",
                inputs=generation_report(
                    requests,
                    max_new_tokens=generation["max_new_tokens"],
                    do_sample=generation["do_sample"],
                    use_cache=generation["use_cache"],
                ),
            ):
                raise
            self._log_failed_rows(
                text_writer,
                datamodule,
                sample_batch,
                requests,
                generation_metadata,
                error,
                step=trainer.global_step,
            )
            raise

    def _log_failed_rows(
        self,
        text_writer: Any | None,
        datamodule: DataModule,
        sample_batch: ModelBatch,
        requests: Sequence[Request],
        generation_metadata: Mapping[str, Any],
        error: Exception,
        *,
        step: int,
    ) -> None:
        if text_writer is None:
            return
        failure = {"type": type(error).__name__, "message": str(error)}
        for row, (dataset_index, sample, request) in enumerate(
            zip(self.indices, self.samples, requests)
        ):
            text_writer.add_text(
                f"{self._tag(dataset_index)}/metadata",
                metadata_json(
                    sample_log_record(
                        datamodule,
                        sample_batch.row(row),
                        dataset_index,
                        sample,
                        request,
                        status="failed",
                        generation_settings=generation_metadata,
                        error=failure,
                    )
                ),
                step,
            )

    def _tag(self, dataset_index: int) -> str:
        return f"sample/{self.task.value}/{dataset_index}"

    def _generation_kwargs(self) -> GenerationKwargs:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }

    def state_dict(self) -> dict[str, dict[str, int | None]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, int | None]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))


@dataclass(frozen=True)
class RowLogContext:
    audio_writer: Any | None
    scalar_writer: Any | None
    text_writer: Any | None
    datamodule: DataModule
    module: Module
    row_batch: ModelBatch
    dataset_index: int
    sample: types.Sample
    request: Request
    result: Result
    generation_metadata: Mapping[str, Any]
    tag: str
    step: int


def log_result_row(
    context: RowLogContext,
    *,
    max_new_tokens: int,
) -> None:
    metrics: dict[str, float] = {}
    prediction = prediction_modality(context.request)
    decode_error = context.result.get("decode_error")
    status = "partial" if decode_error is not None else "ok"
    result_metadata = build_result_metadata(
        context.result,
        max_new_tokens=max_new_tokens,
        prediction=prediction,
        runtime=context.datamodule.runtime,
    )
    metrics.update(
        {
            "generation/response_tokens": float(result_metadata["response_tokens"]),
            "generation/reached_max_new_tokens": float(
                result_metadata["reached_max_new_tokens"]
            ),
        }
    )
    if context.audio_writer is not None:
        log_source_audio(
            context.audio_writer,
            context.datamodule,
            context.sample,
            context.request["task"],
            context.tag,
            context.step,
        )
    generated_text = log_generation_payload(
        context,
        result_metadata,
        prediction,
        metrics,
    )
    write_row_outputs(
        context,
        result_metadata,
        metrics,
        status=status,
        generated_text=generated_text,
    )


def log_generation_payload(
    context: RowLogContext,
    result_metadata: Mapping[str, Any],
    prediction: PredictionModality,
    metrics: dict[str, float],
) -> str | None:
    audio = context.result["audio"]
    decode_error = context.result.get("decode_error")
    generated_text = decode_response_text(
        context.datamodule.runtime,
        context.result["response_ids"],
        prediction=prediction,
    )
    if generated_text is not None:
        target_text = reference_text(
            context.sample,
            context.request["task"],
            prediction,
        )
        if target_text is not None:
            metrics.update(text_metrics(target_text, generated_text))
    if audio is None:
        return log_text_or_partial_audio(
            context,
            result_metadata,
            prediction,
            metrics,
            decode_error,
            generated_text,
        )
    metrics["generation/stopped_without_eoa"] = float(
        result_metadata["stopped_without_eoa"]
    )
    metrics["generation/audio_available"] = 1.0
    metrics["generation/audio_decode_failed"] = 0.0
    target_audio = log_target_reference_audio(
        context.audio_writer,
        context.datamodule,
        context.module,
        context.row_batch,
        context.sample,
        context.request["task"],
        context.tag,
        context.step,
    )
    metrics.update(
        audio_metrics(
            audio["waveform"],
            audio["sample_rate"],
            target_duration=(
                None
                if target_audio is None
                else target_audio[0].size(-1) / target_audio[1]
            ),
        )
    )
    return generated_text


def log_text_or_partial_audio(
    context: RowLogContext,
    result_metadata: Mapping[str, Any],
    prediction: PredictionModality,
    metrics: dict[str, float],
    decode_error: Mapping[str, str] | None,
    generated_text: str | None,
) -> str | None:
    if prediction.supervises_audio:
        metrics["generation/stopped_without_eoa"] = float(
            result_metadata["stopped_without_eoa"]
        )
        metrics["generation/audio_available"] = 0.0
        if decode_error is not None:
            metrics["generation/audio_decode_failed"] = 1.0
        if context.audio_writer is not None:
            log_target_reference_audio(
                context.audio_writer,
                context.datamodule,
                context.module,
                context.row_batch,
                context.sample,
                context.request["task"],
                context.tag,
                context.step,
            )
        return generated_text
    if not prediction.supervises_text:
        return None
    metrics["generation/stopped_without_eos"] = float(
        result_metadata["stopped_without_eos"]
    )
    return generated_text


def write_row_outputs(
    context: RowLogContext,
    result_metadata: Mapping[str, Any],
    metrics: Mapping[str, float],
    *,
    status: str,
    generated_text: str | None,
) -> None:
    if context.scalar_writer is not None:
        for name, value in metrics.items():
            context.scalar_writer.add_scalar(
                f"{context.tag}/{name}",
                value,
                context.step,
            )
    if context.text_writer is not None:
        context.text_writer.add_text(
            f"{context.tag}/metadata",
            metadata_json(
                sample_log_record(
                    context.datamodule,
                    context.row_batch,
                    context.dataset_index,
                    context.sample,
                    context.request,
                    status=status,
                    generation_settings=context.generation_metadata,
                    result_metadata=result_metadata,
                    response_ids=context.result["response_ids"],
                    generated_text=generated_text,
                    metrics=metrics,
                )
            ),
            context.step,
        )
    audio = context.result["audio"]
    if audio is not None and context.audio_writer is not None:
        context.audio_writer.add_audio(
            f"{context.tag}/generated",
            audio["waveform"].detach().cpu(),
            context.step,
            sample_rate=audio["sample_rate"],
        )
    if context.text_writer is not None:
        write_text_outputs(
            context,
            generated_text,
            include_ids=audio is None,
        )


def write_text_outputs(
    context: RowLogContext,
    generated_text: str | None,
    *,
    include_ids: bool,
) -> None:
    if context.text_writer is None:
        return
    prediction = prediction_modality(context.request)
    if prediction.supervises_text:
        target_text = reference_text(
            context.sample,
            context.request["task"],
            prediction,
        )
        if target_text is not None:
            context.text_writer.add_text(
                f"{context.tag}/target",
                target_text,
                context.step,
            )
    if include_ids:
        context.text_writer.add_text(
            f"{context.tag}/generated_ids",
            " ".join(str(value) for value in context.result["response_ids"].tolist()),
            context.step,
        )
    if generated_text is not None:
        context.text_writer.add_text(
            f"{context.tag}/generated",
            generated_text,
            context.step,
        )


__all__ = ["RowLogContext", "TaskSampleLogger", "log_result_row"]
