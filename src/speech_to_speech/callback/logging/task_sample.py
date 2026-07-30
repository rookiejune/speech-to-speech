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
from ...generation.evaluation import reference_audio
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
from ..interval import TrainInterval
from .._lightning import attached_datamodule
from ._sample_metrics import audio_metrics, text_metrics


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
    def text_tokenizer(self) -> TextTokenizer: ...

    @property
    def layout(self) -> Layout: ...


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
        every_audio_seconds: float | None = None,
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
        self.indices = list(indices)
        self.loader_name = loader_name
        self.split = split
        self.task = task
        self.seed = seed
        self.every_n_steps = every_n_steps
        self.interval = TrainInterval(
            every_n_steps=every_n_steps,
            every_audio_seconds=every_audio_seconds,
        )
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
            every_audio_seconds=self.interval.every_audio_seconds,
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        datamodule = cast(_DataModule, attached_datamodule(trainer))
        self.samples = datamodule.diagnostic_samples(
            self.indices,
            split=self.split,
            loader_name=self.loader_name,
        )

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        del batch_idx
        if not self.interval.should_run(trainer, pl_module, batch):
            return
        if not trainer.is_global_zero:
            return
        audio_writer = experiment.audio(trainer)
        scalar_writer = experiment.scalar(trainer)
        text_writer = experiment.text(trainer)
        if audio_writer is None and scalar_writer is None and text_writer is None:
            return
        module = cast(_Module, cast(object, pl_module))
        datamodule = cast(_DataModule, attached_datamodule(trainer))
        collator = datamodule.diagnostic_collator(
            self.task,
            split=self.split,
            loader_name=self.loader_name,
        )
        materialized = module.materialize_batch(collator(self.samples))
        if not isinstance(materialized, ModelBatch):
            raise TypeError("task sample logging requires one materialized ModelBatch.")
        sample_batch = materialized
        requests = requests_from_batch(sample_batch)
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        generation = self._generation_kwargs()
        generation_metadata = {**generation, "seed": self.seed}
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.set_rng_state(torch.Generator().manual_seed(self.seed).get_state())
                if cuda_devices:
                    torch.cuda.manual_seed(self.seed)
                results = module.generate(requests, **generation)
        except Exception as error:
            if text_writer is not None:
                for dataset_index, sample, request in zip(
                    self.indices, self.samples, requests
                ):
                    text_writer.add_text(
                        f"{self._tag(dataset_index)}/metadata",
                        _metadata_json(
                            {
                                **_request_metadata(dataset_index, sample, request),
                                "status": "failed",
                                "generation": generation_metadata,
                                "error": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                },
                            }
                        ),
                        trainer.global_step,
                    )
            raise
        if len(results) != len(requests):
            error = RuntimeError("task sample generation returned the wrong row count.")
            if text_writer is not None:
                for dataset_index, sample, request in zip(
                    self.indices, self.samples, requests
                ):
                    text_writer.add_text(
                        f"{self._tag(dataset_index)}/metadata",
                        _metadata_json(
                            {
                                **_request_metadata(dataset_index, sample, request),
                                "status": "failed",
                                "generation": generation_metadata,
                                "error": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                },
                            }
                        ),
                        trainer.global_step,
                    )
            raise error
        for row, (dataset_index, sample, request, result) in enumerate(
            zip(self.indices, self.samples, requests, results)
        ):
            tag = self._tag(dataset_index)
            metrics: dict[str, float] = {}
            result_metadata = _result_metadata(
                result,
                max_new_tokens=self.max_new_tokens,
            )
            metrics.update(
                {
                    "generation/response_tokens": float(
                        result_metadata["response_tokens"]
                    ),
                    "generation/reached_max_new_tokens": float(
                        result_metadata["reached_max_new_tokens"]
                    ),
                }
            )
            if audio_writer is not None:
                _log_source_audio(
                    audio_writer,
                    datamodule,
                    sample,
                    request["task"],
                    tag,
                    trainer.global_step,
                )
            audio = result["audio"]
            generated_text = None
            if audio is None:
                generated_text = _generated_text(datamodule, result["response_ids"])
                target_text = _target_text(sample, request["task"])
                if generated_text is not None and target_text is not None:
                    metrics.update(text_metrics(target_text, generated_text))
            else:
                target_audio = _log_target_reference_audio(
                    audio_writer,
                    datamodule,
                    module,
                    sample_batch.row(row),
                    sample,
                    request["task"],
                    tag,
                    trainer.global_step,
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
            if scalar_writer is not None:
                for name, value in metrics.items():
                    scalar_writer.add_scalar(
                        f"{tag}/{name}", value, trainer.global_step
                    )
            if text_writer is not None:
                text_writer.add_text(
                    f"{tag}/metadata",
                    _metadata_json(
                        {
                            **_request_metadata(dataset_index, sample, request),
                            "status": "ok",
                            "generation": generation_metadata,
                            "generated": result_metadata,
                            "metrics": metrics,
                        }
                    ),
                    trainer.global_step,
                )
            if audio is not None and audio_writer is not None:
                audio_writer.add_audio(
                    f"{tag}/generated",
                    audio["waveform"].detach().cpu(),
                    trainer.global_step,
                    sample_rate=audio["sample_rate"],
                )
            elif text_writer is not None:
                target_text = _target_text(sample, request["task"])
                if target_text is not None:
                    text_writer.add_text(
                        f"{tag}/target",
                        target_text,
                        trainer.global_step,
                    )
                text_writer.add_text(
                    f"{tag}/generated_ids",
                    " ".join(str(value) for value in result["response_ids"].tolist()),
                    trainer.global_step,
                )
                if generated_text is not None:
                    text_writer.add_text(
                        f"{tag}/generated",
                        generated_text,
                        trainer.global_step,
                    )

    def _tag(self, dataset_index: int) -> str:
        return (
            f"task_sample/{self.split.value}/{self.loader_name}/"
            f"{self.task.value}/{dataset_index}"
        )

    def _generation_kwargs(self) -> _GenerationKwargs:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, float]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))


def _request_metadata(
    dataset_index: int,
    sample: types.Sample,
    request: Request,
) -> dict[str, Any]:
    task = request["task"]
    return {
        "dataset_index": dataset_index,
        "task": task.value,
        "prompt_tokens": int(request["prompt_ids"].numel()),
        "source": _modality_metadata(source_item(sample, task), task.source_modality),
        "reference": _modality_metadata(target_item(sample, task), task.target_modality),
    }


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
    if task.target_modality is not types.Modality.AUDIO:
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
    if task.target_modality is not types.Modality.AUDIO:
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
    runtime = datamodule.runtime
    tokenizer = runtime.text_tokenizer
    layout = runtime.layout
    local_ids = layout.to_local(response_ids.detach().cpu())
    return tokenizer.decode(local_ids.tolist(), skip_special_tokens=True)


def _target_text(sample: types.Sample, task: Task) -> str | None:
    if task.target_modality is not types.Modality.TEXT:
        return None
    _, item = target_item(sample, task)
    if not isinstance(item, types.TextItem):
        raise TypeError("text-target task sample must contain a TextItem.")
    return item.views[types.TextView.TEXT]


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


def _result_metadata(result: Result, *, max_new_tokens: int) -> dict[str, Any]:
    response_ids = result["response_ids"]
    audio = result["audio"]
    metadata: dict[str, Any] = {
        "response_tokens": int(response_ids.numel()),
        "reached_max_new_tokens": bool(response_ids.numel() >= max_new_tokens),
    }
    if audio is None:
        return metadata
    waveform = audio["waveform"]
    features = audio["features"]
    return {
        **metadata,
        "sample_rate": audio["sample_rate"],
        "waveform": _tensor_metadata(waveform),
        "waveform_samples": int(waveform.size(-1)),
        "duration_seconds": waveform.size(-1) / audio["sample_rate"],
        "waveform_finite": _finite(waveform),
        "features": None if features is None else _tensor_metadata(features),
    }


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
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
