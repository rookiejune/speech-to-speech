from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypedDict, cast

import torch
from anydataset import types
from anytrain.codec import SemanticAcousticCodes
from anytrain.module.idspace import Layout
from anytrain.lightning import experiment
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...generation import Request, Result
from ...generation.batch import requests_from_batch
from ...generation.decode import decode_reference_codes
from ...generation.eval.acoustic import reference_audio
from ...prediction import PredictionModality
from ...runtime.audio_tokenizer import BiCodecAudioTokenizer
from ...datamodule.diagnostic import (
    SampleRef,
    SampleSplit,
    source_item,
    target_item,
)
from ...datamodule.types import (
    ModelBatch,
    TrainInput,
)
from ...runtime.types import (
    CodecBackend,
    TextTokenizer,
    acoustic_codec,
    codec_sample_rate,
)
from ...task import Task
from .._oom import batch_report, generation_report, report_oom
from ..interval import TrainInterval
from ._sample_metrics import audio_metrics, text_metrics

_TOKEN_PREVIEW_LIMIT = 128


class _Module(Protocol):
    model: Any

    def materialize_batch(self, batch: TrainInput) -> ModelBatch: ...

    def generate(
        self,
        requests: Sequence[Request],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> list[Result]: ...


class _GenerationKwargs(TypedDict):
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    use_cache: bool


class _DataModule(Protocol):
    runtime: _LoggingRuntime

    def diagnostic_samples(
        self,
        indices: Sequence[int],
        *,
        split: SampleSplit,
        loader_name: str,
    ) -> list[types.Sample]: ...

    def diagnostic_collator(
        self,
        task: Task,
        *,
        split: SampleSplit,
        loader_name: str,
    ) -> Callable[[list[types.Sample]], TrainInput]: ...


class _LoggingRuntime(Protocol):
    @property
    def audio_view(self) -> types.AudioView: ...

    @property
    def codec(self) -> CodecBackend: ...

    @property
    def audio_tokenizer(self) -> object: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @property
    def text_tokenizer(self) -> TextTokenizer: ...

    @property
    def layout(self) -> Layout: ...


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
        datamodule = cast(_DataModule, _attached_datamodule(trainer))
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
        module = cast(_Module, cast(object, pl_module))
        datamodule = cast(_DataModule, _attached_datamodule(trainer))
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
            self._log_result_row(
                audio_writer,
                scalar_writer,
                text_writer,
                datamodule,
                module,
                sample_batch,
                row,
                dataset_index,
                sample,
                request,
                result,
                generation_metadata,
                step=trainer.global_step,
            )

    def _should_log(self, trainer: Trainer) -> bool:
        return self.interval.should_run(int(trainer.global_step)) and trainer.is_global_zero

    def _writers(self, trainer: Trainer) -> tuple[Any | None, Any | None, Any | None]:
        return experiment.audio(trainer), experiment.scalar(trainer), experiment.text(trainer)

    def _materialize_samples(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        module: _Module,
        datamodule: _DataModule,
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
        module: _Module,
        datamodule: _DataModule,
        sample_batch: ModelBatch,
        requests: Sequence[Request],
        generation: _GenerationKwargs,
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
        datamodule: _DataModule,
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
                _metadata_json(
                    _sample_log_record(
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

    def _log_result_row(
        self,
        audio_writer: Any | None,
        scalar_writer: Any | None,
        text_writer: Any | None,
        datamodule: _DataModule,
        module: _Module,
        sample_batch: ModelBatch,
        row: int,
        dataset_index: int,
        sample: types.Sample,
        request: Request,
        result: Result,
        generation_metadata: Mapping[str, Any],
        *,
        step: int,
    ) -> None:
        tag = self._tag(dataset_index)
        row_batch = sample_batch.row(row)
        metrics: dict[str, float] = {}
        prediction = _prediction_modality(request)
        decode_error = result.get("decode_error")
        status = "partial" if decode_error is not None else "ok"
        result_metadata = _result_metadata(
            result,
            max_new_tokens=self.max_new_tokens,
            prediction=prediction,
            runtime=datamodule.runtime,
        )
        metrics.update(
            {
                "generation/response_tokens": float(result_metadata["response_tokens"]),
                "generation/reached_max_new_tokens": float(
                    result_metadata["reached_max_new_tokens"]
                ),
            }
        )
        if audio_writer is not None:
            _log_source_audio(audio_writer, datamodule, sample, request["task"], tag, step)
        generated_text = self._log_generation_payload(
            audio_writer,
            datamodule,
            module,
            row_batch,
            sample,
            request,
            result,
            result_metadata,
            prediction,
            metrics,
            tag=tag,
            step=step,
        )
        self._write_row_outputs(
            audio_writer,
            scalar_writer,
            text_writer,
            datamodule,
            row_batch,
            dataset_index,
            sample,
            request,
            result,
            generation_metadata,
            result_metadata,
            metrics,
            status=status,
            generated_text=generated_text,
            tag=tag,
            step=step,
        )

    def _log_generation_payload(
        self,
        audio_writer: Any | None,
        datamodule: _DataModule,
        module: _Module,
        row_batch: ModelBatch,
        sample: types.Sample,
        request: Request,
        result: Result,
        result_metadata: Mapping[str, Any],
        prediction: PredictionModality,
        metrics: dict[str, float],
        *,
        tag: str,
        step: int,
    ) -> str | None:
        audio = result["audio"]
        decode_error = result.get("decode_error")
        if audio is None:
            return self._log_text_or_partial_audio(
                audio_writer,
                datamodule,
                module,
                row_batch,
                sample,
                request,
                result,
                result_metadata,
                prediction,
                metrics,
                decode_error,
                tag=tag,
                step=step,
            )
        metrics["generation/stopped_without_eoa"] = float(
            result_metadata["stopped_without_eoa"]
        )
        metrics["generation/audio_available"] = 1.0
        metrics["generation/audio_decode_failed"] = 0.0
        target_audio = _log_target_reference_audio(
            audio_writer,
            datamodule,
            module,
            row_batch,
            sample,
            request["task"],
            tag,
            step,
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
        return None

    def _log_text_or_partial_audio(
        self,
        audio_writer: Any | None,
        datamodule: _DataModule,
        module: _Module,
        row_batch: ModelBatch,
        sample: types.Sample,
        request: Request,
        result: Result,
        result_metadata: Mapping[str, Any],
        prediction: PredictionModality,
        metrics: dict[str, float],
        decode_error: Mapping[str, str] | None,
        *,
        tag: str,
        step: int,
    ) -> str | None:
        if prediction.supervises_audio:
            metrics["generation/stopped_without_eoa"] = float(
                result_metadata["stopped_without_eoa"]
            )
            metrics["generation/audio_available"] = 0.0
            if decode_error is not None:
                metrics["generation/audio_decode_failed"] = 1.0
            if audio_writer is not None:
                _log_target_reference_audio(
                    audio_writer,
                    datamodule,
                    module,
                    row_batch,
                    sample,
                    request["task"],
                    tag,
                    step,
                )
            if prediction.supervises_text:
                return _decode_chat_template(datamodule, result["response_ids"])
            return None
        if not prediction.supervises_text:
            return None
        generated_text = _generated_text(datamodule, result["response_ids"])
        target_text = _target_text(sample, request["task"])
        if generated_text is not None and target_text is not None:
            metrics.update(text_metrics(target_text, generated_text))
        metrics["generation/stopped_without_eos"] = float(
            result_metadata["stopped_without_eos"]
        )
        return generated_text

    def _write_row_outputs(
        self,
        audio_writer: Any | None,
        scalar_writer: Any | None,
        text_writer: Any | None,
        datamodule: _DataModule,
        row_batch: ModelBatch,
        dataset_index: int,
        sample: types.Sample,
        request: Request,
        result: Result,
        generation_metadata: Mapping[str, Any],
        result_metadata: Mapping[str, Any],
        metrics: Mapping[str, float],
        *,
        status: str,
        generated_text: str | None,
        tag: str,
        step: int,
    ) -> None:
        if scalar_writer is not None:
            for name, value in metrics.items():
                scalar_writer.add_scalar(f"{tag}/{name}", value, step)
        if text_writer is not None:
            text_writer.add_text(
                f"{tag}/metadata",
                _metadata_json(
                    _sample_log_record(
                        datamodule,
                        row_batch,
                        dataset_index,
                        sample,
                        request,
                        status=status,
                        generation_settings=generation_metadata,
                        result_metadata=result_metadata,
                        response_ids=result["response_ids"],
                        generated_text=generated_text,
                        metrics=metrics,
                    )
                ),
                step,
            )
        audio = result["audio"]
        if audio is not None and audio_writer is not None:
            audio_writer.add_audio(
                f"{tag}/generated",
                audio["waveform"].detach().cpu(),
                step,
                sample_rate=audio["sample_rate"],
            )
        elif text_writer is not None:
            self._write_text_fallback(
                text_writer,
                sample,
                request,
                result,
                generated_text,
                tag=tag,
                step=step,
            )

    def _write_text_fallback(
        self,
        text_writer: Any,
        sample: types.Sample,
        request: Request,
        result: Result,
        generated_text: str | None,
        *,
        tag: str,
        step: int,
    ) -> None:
        if _prediction_modality(request).supervises_text:
            target_text = _target_text(sample, request["task"])
            if target_text is not None:
                text_writer.add_text(f"{tag}/target", target_text, step)
        text_writer.add_text(
            f"{tag}/generated_ids",
            " ".join(str(value) for value in result["response_ids"].tolist()),
            step,
        )
        if generated_text is not None:
            text_writer.add_text(f"{tag}/generated", generated_text, step)

    def _tag(self, dataset_index: int) -> str:
        return f"sample/{self.task.value}/{dataset_index}"

    def _generation_kwargs(self) -> _GenerationKwargs:
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


def _request_metadata(
    dataset_index: int,
    sample: types.Sample,
    request: Request,
) -> dict[str, Any]:
    task = request["task"]
    reference_modality = (
        task.target_modality
        if task.target_modality is not None
        else types.Modality.AUDIO
    )
    return {
        "dataset_index": dataset_index,
        "task": task.value,
        "prompt_tokens": int(request["prompt_ids"].numel()),
        "source": _modality_metadata(source_item(sample, task), task.source_modality),
        "reference": _modality_metadata(target_item(sample, task), reference_modality),
    }


def _sample_log_record(
    datamodule: _DataModule,
    batch: ModelBatch,
    dataset_index: int,
    sample: types.Sample,
    request: Request,
    *,
    status: str,
    generation_settings: Mapping[str, Any],
    result_metadata: Mapping[str, Any] | None = None,
    response_ids: Tensor | None = None,
    generated_text: str | None = None,
    metrics: Mapping[str, float] | None = None,
    error: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request_metadata = _request_metadata(dataset_index, sample, request)
    token_labels = batch.token_labels[0].detach().cpu()
    supervised = token_labels[token_labels.ne(-100)]
    prompt_ids = request["prompt_ids"].detach().cpu()
    generation: dict[str, Any] = {
        "status": status,
        "settings": dict(generation_settings),
    }
    if result_metadata is not None:
        generation["result"] = dict(result_metadata)
    if response_ids is not None:
        response_ids = response_ids.detach().cpu()
        generation["response_ids"] = _token_sequence(response_ids)
        decoded = generated_text or _decode_text(datamodule, response_ids)
        if decoded is not None:
            generation["text"] = decoded
    if metrics is not None:
        generation["metrics"] = dict(metrics)
    if error is not None:
        generation["error"] = dict(error)

    chat_template: dict[str, Any] = {
        "dataset_index": dataset_index,
        "task": request_metadata["task"],
        "prompt_tokens": request_metadata["prompt_tokens"],
        "prompt_ids": _token_sequence(prompt_ids),
        "source": request_metadata["source"],
    }
    prediction = request.get("prediction")
    if prediction is not None:
        chat_template["prediction"] = prediction.value
    labels: dict[str, Any] = {
        "token_labels": _token_sequence(token_labels),
        "supervised_token_ids": _token_sequence(supervised),
        "reference": request_metadata["reference"],
    }
    if error is None:
        prompt_text = _decode_chat_template(datamodule, prompt_ids)
        if prompt_text is not None:
            chat_template["text"] = prompt_text
        label_text = _decode_text(datamodule, supervised)
        if label_text is not None:
            labels["text"] = label_text

    return {
        "chat_template": chat_template,
        "labels": labels,
        "generation": generation,
    }


def _prediction_modality(request: Request) -> PredictionModality:
    prediction = request.get("prediction")
    if prediction is None:
        return request["task"].prediction_modality
    if not isinstance(prediction, PredictionModality):
        raise TypeError("generation request prediction must be a PredictionModality.")
    return prediction


def _log_target_reference_audio(
    audio_writer: Any | None,
    datamodule: _DataModule,
    module: _Module,
    batch: Any,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> tuple[Tensor, int] | None:
    if not task.prediction_modality.supervises_audio:
        return None
    if batch.acoustic_target is None:
        target, sample_rate = _sample_audio(datamodule, sample, task, source=False)
        if audio_writer is not None:
            audio_writer.add_audio(
                f"{tag}/target", target, step, sample_rate=sample_rate
            )
        return target, sample_rate
    codec = acoustic_codec(datamodule.runtime.codec)
    target, reference = reference_audio(module.model, batch, codec, seed=0)
    sample_rate = codec_sample_rate(codec)
    target = target.detach().cpu()
    if audio_writer is not None:
        audio_writer.add_audio(
            f"{tag}/target", target, step, sample_rate=sample_rate
        )
        audio_writer.add_audio(
            f"{tag}/reference_generation",
            reference.detach().cpu(),
            step,
            sample_rate=sample_rate,
        )
    return target, sample_rate


def _log_source_audio(
    audio_writer: Any,
    datamodule: _DataModule,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> None:
    if task.source_modality is not types.Modality.AUDIO:
        return
    waveform, sample_rate = _sample_audio(datamodule, sample, task, source=True)
    audio_writer.add_audio(
        f"{tag}/source", waveform, step, sample_rate=sample_rate
    )


def _log_target_audio(
    audio_writer: Any,
    datamodule: _DataModule,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> None:
    if not task.prediction_modality.supervises_audio:
        return
    waveform, sample_rate = _sample_audio(datamodule, sample, task, source=False)
    audio_writer.add_audio(
        f"{tag}/target",
        waveform,
        step,
        sample_rate=sample_rate,
    )


def _sample_audio(
    datamodule: _DataModule,
    sample: types.Sample,
    task: Task,
    *,
    source: bool,
) -> tuple[Tensor, int]:
    ref = source_item(sample, task) if source else target_item(sample, task)
    if ref is None:
        raise ValueError("sample audio reference is missing.")
    _, item = ref
    if not isinstance(item, types.AudioItem):
        raise TypeError("sample audio reference must contain an AudioItem.")
    raw = item.views.get(types.AudioView.WAVEFORM)
    if raw is not None:
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
        waveform, sample_rate = raw
        if not isinstance(waveform, Tensor):
            raise TypeError("AudioView.WAVEFORM waveform must be a Tensor.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("AudioView.WAVEFORM sample_rate must be an integer.")
        if sample_rate <= 0:
            raise ValueError("AudioView.WAVEFORM sample_rate must be positive.")
        return waveform.detach().cpu(), sample_rate
    runtime = datamodule.runtime
    codes = item.views[runtime.audio_view]
    waveform = decode_reference_codes(codes, codec=runtime.codec)
    if waveform.dim() < 2:
        raise ValueError("codec sample decode must return a batched waveform.")
    return waveform[0].detach().cpu(), codec_sample_rate(runtime.codec)


def _generated_text(datamodule: _DataModule, response_ids: Tensor) -> str | None:
    return _decode_text(datamodule, response_ids)


def _decode_chat_template(
    datamodule: _DataModule,
    token_ids: Tensor,
) -> str | None:
    if token_ids.numel() == 0:
        return None
    try:
        runtime = datamodule.runtime
        tokenizer = runtime.text_tokenizer
        start, end = runtime.layout.block(types.Modality.TEXT.value)
    except (AttributeError, KeyError):
        return None
    pieces: list[str] = []
    text_ids: list[int] = []
    audio_span = False

    def flush_text() -> None:
        if not text_ids:
            return
        try:
            decoded = tokenizer.decode(text_ids, skip_special_tokens=False)
        except TypeError:
            decoded = tokenizer.decode(text_ids)
        pieces.append(decoded)
        text_ids.clear()

    for value in token_ids.detach().cpu().reshape(-1).tolist():
        token_id = int(value)
        if start <= token_id < end:
            if audio_span:
                pieces.append("<audio>")
                audio_span = False
            text_ids.append(token_id - start)
        else:
            flush_text()
            audio_span = True
    flush_text()
    if audio_span:
        pieces.append("<audio>")
    return "".join(pieces)


def _decode_text(datamodule: _DataModule, token_ids: Tensor) -> str | None:
    if token_ids.numel() == 0:
        return None
    try:
        runtime = datamodule.runtime
        tokenizer = runtime.text_tokenizer
        start, end = runtime.layout.block(types.Modality.TEXT.value)
    except (AttributeError, KeyError):
        return None
    ids = token_ids.detach().cpu()
    inside = (ids >= start) & (ids < end)
    if not bool(inside.all()):
        return None
    local_ids = (ids - start).tolist()
    try:
        return tokenizer.decode(local_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(local_ids)


def _token_sequence(token_ids: Tensor) -> dict[str, Any]:
    values = [int(value) for value in token_ids.reshape(-1).tolist()]
    return {
        "count": len(values),
        "ids": values[:_TOKEN_PREVIEW_LIMIT],
        "truncated": len(values) > _TOKEN_PREVIEW_LIMIT,
    }


def _target_text(sample: types.Sample, task: Task) -> str | None:
    if not task.prediction_modality.supervises_text:
        return None
    if task.prediction_modality.is_mixed:
        # Mixed targets are Speech items; read aligned text from the DEFAULT/TARGET
        # text view when present.
        role = _text_role(sample, task)
        try:
            item = sample[(role, types.Modality.TEXT)]
        except KeyError:
            return None
        if not isinstance(item, types.TextItem):
            raise TypeError("mixed-task text view must contain a TextItem.")
        return item.views[types.TextView.TEXT]
    _, item = target_item(sample, task)
    if not isinstance(item, types.TextItem):
        raise TypeError("text-target task sample must contain a TextItem.")
    return item.views[types.TextView.TEXT]


def _text_role(sample: types.Sample, task: Task) -> types.Role:
    roles = {role for role, _ in sample}
    if types.Role.DEFAULT in roles:
        return types.Role.DEFAULT
    return types.Role.TARGET if not task.uses_source_role else types.Role.TARGET


def _modality_metadata(
    ref: SampleRef | None,
    modality: types.Modality | None,
) -> dict[str, Any] | None:
    if modality is None:
        if ref is not None:
            raise ValueError("a modality-free task source must not resolve a sample item.")
        return None
    if ref is None:
        raise ValueError("task modality metadata requires a sample item.")
    role, item = ref
    if modality is types.Modality.TEXT:
        if not isinstance(item, types.TextItem):
            raise TypeError("text modality metadata requires a TextItem.")
        return {
            "modality": modality.value,
            "role": role.value,
            "language": item.meta[types.TextMeta.LANG],
            "text": item.views[types.TextView.TEXT],
        }
    if modality is types.Modality.AUDIO:
        if not isinstance(item, types.AudioItem):
            raise TypeError("audio modality metadata requires an AudioItem.")
        view, value = _diagnostic_audio_view(item)
        return {
            "modality": modality.value,
            "role": role.value,
            "view": view.value,
            **(
                _waveform_metadata(value)
                if view is types.AudioView.WAVEFORM
                else _codes_metadata(value)
            ),
        }
    raise AssertionError(f"unsupported sample modality: {modality.value}")


def _result_metadata(
    result: Result,
    *,
    max_new_tokens: int,
    prediction: PredictionModality,
    runtime: _LoggingRuntime,
) -> dict[str, Any]:
    response_ids = result["response_ids"]
    audio = result["audio"]
    response_tokens = int(response_ids.numel())
    # TEXT/AUDIO paths strip the stop token from response_ids. Hitting the budget
    # therefore means generation ended without emitting EOS/EOA.
    reached_max = response_tokens >= max_new_tokens
    metadata: dict[str, Any] = {
        "response_tokens": response_tokens,
        "reached_max_new_tokens": reached_max,
    }
    decode_error = result.get("decode_error")
    if decode_error is not None:
        metadata["audio_decode_failed"] = True
        metadata["audio_decode_error"] = dict(decode_error)
    if prediction.supervises_audio:
        metadata["stopped_without_eoa"] = reached_max
        if audio is None:
            bicodec = _partial_bicodec_metadata(runtime, response_ids)
            if bicodec is not None:
                metadata["bicodec_streams"] = bicodec
            return metadata
    if audio is None:
        metadata["stopped_without_eos"] = reached_max
        return metadata
    waveform = audio["waveform"]
    features = audio["features"]
    return {
        **metadata,
        "stopped_without_eoa": reached_max,
        "sample_rate": audio["sample_rate"],
        "waveform": _tensor_metadata(waveform),
        "waveform_samples": int(waveform.size(-1)),
        "duration_seconds": waveform.size(-1) / audio["sample_rate"],
        "waveform_finite": _finite(waveform),
        "features": None if features is None else _tensor_metadata(features),
    }


def _partial_bicodec_metadata(
    runtime: _LoggingRuntime,
    response_ids: Tensor,
) -> dict[str, Any] | None:
    try:
        tokenizer = runtime.audio_tokenizer
        audio_token_range = runtime.codec_audio_range
    except AttributeError:
        return None
    if not isinstance(tokenizer, BiCodecAudioTokenizer):
        return None
    ids = response_ids.detach().cpu().reshape(-1)
    start, end = audio_token_range
    audio_mask = ids.ge(start) & ids.lt(end)
    local = ids[audio_mask] - start
    values = [int(value) for value in local.tolist()]
    expected_acoustic = tokenizer.acoustic_unit_length * len(
        tokenizer.acoustic_codebook_sizes
    )
    semantic_start, semantic_end = tokenizer.semantic_token_range
    summary: dict[str, Any] = {
        "expected_acoustic_tokens": expected_acoustic,
        "acoustic_tokens": 0,
        "semantic_tokens": 0,
        "has_end_marker": tokenizer.end_token_id in values,
    }

    acoustic_marker_index = _first_index(values, tokenizer.acoustic_token_id)
    if acoustic_marker_index is not None:
        payload_start = acoustic_marker_index + 1
        payload_end = _first_marker_index(
            values,
            (
                tokenizer.semantic_token_id,
                tokenizer.end_token_id,
            ),
            start=payload_start,
        )
        if payload_end is None:
            payload_end = len(values)
        acoustic_payload = values[payload_start:payload_end]
        summary["acoustic_tokens"] = len(acoustic_payload)

    semantic_marker_index = _first_index(values, tokenizer.semantic_token_id)
    if semantic_marker_index is not None:
        payload_start = semantic_marker_index + 1
        payload_end = _first_marker_index(
            values,
            (tokenizer.end_token_id,),
            start=payload_start,
        )
        if payload_end is None:
            payload_end = len(values)
        semantic_payload = values[payload_start:payload_end]
        summary["semantic_tokens"] = sum(
            1
            for token_id in semantic_payload
            if semantic_start <= token_id < semantic_end
        )

    return summary


def _first_index(values: Sequence[int], target: int) -> int | None:
    try:
        return values.index(target)
    except ValueError:
        return None


def _first_marker_index(
    values: Sequence[int],
    markers: Sequence[int],
    *,
    start: int,
) -> int | None:
    marker_set = set(markers)
    for index in range(start, len(values)):
        if values[index] in marker_set:
            return index
    return None


def _codes_metadata(codes: object) -> dict[str, Any]:
    if isinstance(codes, SemanticAcousticCodes):
        semantic = codes.semantic
        acoustic = codes.acoustic
    elif isinstance(codes, Mapping):
        semantic = codes.get("semantic")
        acoustic = codes.get("acoustic")
        if not isinstance(semantic, Tensor) or not isinstance(acoustic, Tensor):
            raise TypeError(
                "structured audio sample codes require Tensor semantic/acoustic fields."
            )
    else:
        semantic = None
        acoustic = None
    if semantic is not None and acoustic is not None:
        return {
            "structured": True,
            "semantic": _code_tensor_metadata(semantic),
            "acoustic": _code_tensor_metadata(acoustic),
        }
    if not isinstance(codes, Tensor):
        raise TypeError("audio sample codes must be a Tensor or structured mapping.")
    return _code_tensor_metadata(codes)


def _diagnostic_audio_view(
    item: types.AudioItem,
) -> tuple[types.AudioView, object]:
    for view, value in item.views.items():
        if view is not types.AudioView.WAVEFORM:
            return view, value
    try:
        return types.AudioView.WAVEFORM, item.views[types.AudioView.WAVEFORM]
    except KeyError as error:
        raise ValueError("diagnostic audio item has no views.") from error


def _waveform_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
    waveform, sample_rate = value
    if not isinstance(waveform, Tensor):
        raise TypeError("AudioView.WAVEFORM waveform must be a Tensor.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("AudioView.WAVEFORM sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("AudioView.WAVEFORM sample_rate must be positive.")
    return {
        "waveform": _tensor_metadata(waveform),
        "sample_rate": sample_rate,
        "duration_seconds": waveform.size(-1) / sample_rate,
        "waveform_finite": _finite(waveform),
    }


def _code_tensor_metadata(codes: Tensor) -> dict[str, Any]:
    if codes.dim() != 2:
        raise ValueError("audio sample codes must have shape [frames, codebooks].")
    return {
        "frames": int(codes.size(0)),
        "codebooks": int(codes.size(1)),
        "codes_dtype": str(codes.dtype),
    }


def _tensor_metadata(tensor: Tensor) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
    }


def _finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def _metadata_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
