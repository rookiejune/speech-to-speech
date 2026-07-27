from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypedDict, cast

import torch
from anydataset import types
from anytrain.lightning import experiment
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...generation import Request, Result
from ...generation.batch import requests_from_batch
from ...generation.evaluation import reference_audio
from ...task import Task
from ..interval import TrainInterval
from .._lightning import attached_datamodule


class _Module(Protocol):
    model: Any

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
    collator: Callable[[list[types.Sample]], Any]
    runtime: Any

    def train_samples(
        self,
        indices: Sequence[int],
        *,
        loader_name: str | None = None,
    ) -> list[types.Sample]: ...

    def collator_for(
        self,
        loader_name: str | None = None,
    ) -> Callable[[list[types.Sample]], Any]: ...


class TaskSampleLogger(Callback):
    def __init__(
        self,
        indices: Sequence[int],
        every_n_steps: int | None,
        *,
        loader_name: str | None = None,
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
        self.indices = list(indices)
        self.loader_name = loader_name
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

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        datamodule = cast(_DataModule, attached_datamodule(trainer))
        self.samples = datamodule.train_samples(
            self.indices,
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
        text_writer = experiment.text(trainer)
        if audio_writer is None and text_writer is None:
            return
        module = cast(_Module, cast(object, pl_module))
        datamodule = cast(_DataModule, attached_datamodule(trainer))
        collator_for = getattr(datamodule, "collator_for", None)
        if callable(collator_for):
            collator = cast(
                Callable[
                    [str | None],
                    Callable[[list[types.Sample]], Any],
                ],
                collator_for,
            )(self.loader_name)
        else:
            collator = datamodule.collator
        sample_batch = collator(self.samples)
        requests = requests_from_batch(sample_batch)
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        generation = self._generation_kwargs()
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                results = module.generate(requests, **generation)
        except Exception as error:
            if text_writer is not None:
                for dataset_index, sample, request in zip(
                    self.indices, self.samples, requests
                ):
                    text_writer.add_text(
                        f"{_tag(dataset_index)}/metadata",
                        _metadata_json(
                            {
                                **_request_metadata(dataset_index, sample, request),
                                "status": "failed",
                                "generation": generation,
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
                        f"{_tag(dataset_index)}/metadata",
                        _metadata_json(
                            {
                                **_request_metadata(dataset_index, sample, request),
                                "status": "failed",
                                "generation": generation,
                                "error": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                },
                            }
                        ),
                        trainer.global_step,
                    )
            raise error
        for dataset_index, sample, request, result in zip(
            self.indices, self.samples, requests, results
        ):
            tag = _tag(dataset_index)
            if text_writer is not None:
                text_writer.add_text(
                    f"{tag}/metadata",
                    _metadata_json(
                        {
                            **_request_metadata(dataset_index, sample, request),
                            "status": "ok",
                            "generation": generation,
                            "generated": _result_metadata(
                                result,
                                max_new_tokens=self.max_new_tokens,
                            ),
                        }
                    ),
                    trainer.global_step,
                )
            audio = result["audio"]
            if audio is not None and audio_writer is not None:
                if hasattr(sample_batch, "acoustic_target") and sample_batch.acoustic_target is not None:
                    _log_reference_audio(
                        audio_writer,
                        datamodule,
                        module,
                        sample_batch,
                        tag,
                        trainer.global_step,
                    )
                else:
                    _log_target_audio(
                        audio_writer,
                        datamodule,
                        sample,
                        request["task"],
                        tag,
                        trainer.global_step,
                    )
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
                generated_text = _generated_text(datamodule, result["response_ids"])
                if generated_text is not None:
                    text_writer.add_text(
                        f"{tag}/generated",
                        generated_text,
                        trainer.global_step,
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
    source_role = types.Role.SOURCE if task.uses_source_role else types.Role.TARGET
    return {
        "dataset_index": dataset_index,
        "task": task.value,
        "prompt_tokens": int(request["prompt_ids"].numel()),
        "source": _modality_metadata(sample, source_role, task.source_modality),
        "reference": _modality_metadata(sample, types.Role.TARGET, task.target_modality),
    }


def _log_reference_audio(
    audio_writer: Any,
    datamodule: _DataModule,
    module: _Module,
    batch: Any,
    tag: str,
    step: int,
) -> None:
    if not hasattr(batch, "acoustic_target") or batch.acoustic_target is None:
        return
    runtime = getattr(datamodule, "runtime", None)
    if runtime is None:
        return
    codec = getattr(runtime, "codec", None)
    if codec is None or not hasattr(codec, "acoustic_codes_to_features"):
        return
    target, reference = reference_audio(module.model, batch, codec, seed=0)
    sample_rate = int(getattr(codec, "sample_rate"))
    audio_writer.add_audio(
        f"{tag}/target",
        target.detach().cpu(),
        step,
        sample_rate=sample_rate,
    )
    audio_writer.add_audio(
        f"{tag}/reference_generation",
        reference.detach().cpu(),
        step,
        sample_rate=sample_rate,
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
    runtime = getattr(datamodule, "runtime", None)
    if runtime is None:
        return
    codec = getattr(runtime, "codec", None)
    if codec is None or not hasattr(codec, "decode"):
        return
    item = cast(types.AudioItem, sample[(types.Role.TARGET, types.Modality.AUDIO)])
    view = getattr(runtime, "audio_view")
    codes = item.views[view]
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError("fixed target audio codes must have shape [frames, codebooks].")
    waveform = codec.decode(codes.unsqueeze(0))
    if waveform.dim() < 2:
        raise ValueError("codec target decode must return a batched waveform.")
    audio_writer.add_audio(
        f"{tag}/target",
        waveform[0].detach().cpu(),
        step,
        sample_rate=int(codec.sample_rate),
    )


def _generated_text(datamodule: _DataModule, response_ids: Tensor) -> str | None:
    runtime = getattr(datamodule, "runtime", None)
    if runtime is None:
        return None
    tokenizer = getattr(runtime, "text_tokenizer", None)
    layout = getattr(runtime, "layout", None)
    if tokenizer is None or layout is None:
        return None
    local_ids = layout.to_local(response_ids.detach().cpu())
    return tokenizer.decode(local_ids.tolist(), skip_special_tokens=True)


def _target_text(sample: types.Sample, task: Task) -> str | None:
    if task.target_modality is not types.Modality.TEXT:
        return None
    item = cast(types.TextItem, sample[(types.Role.TARGET, types.Modality.TEXT)])
    return item.views[types.TextView.TEXT]


def _tag(dataset_index: int) -> str:
    return f"task_sample/{dataset_index}"


def _modality_metadata(
    sample: types.Sample,
    role: types.Role,
    modality: types.Modality | None,
) -> dict[str, Any] | None:
    if modality is None:
        return None
    if modality is types.Modality.TEXT:
        item = cast(types.TextItem, sample[(role, types.Modality.TEXT)])
        return {
            "modality": modality.value,
            "role": role.value,
            "language": item.meta[types.TextMeta.LANG],
            "text": item.views[types.TextView.TEXT],
        }
    if modality is types.Modality.AUDIO:
        item = cast(types.AudioItem, sample[(role, types.Modality.AUDIO)])
        view, codes = next(iter(item.views.items()))
        return {
            "modality": modality.value,
            "role": role.value,
            "view": view.value,
            **_codes_metadata(codes),
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


def _codes_metadata(codes: Tensor) -> dict[str, Any]:
    if not isinstance(codes, Tensor):
        raise TypeError("audio sample codes must be a Tensor.")
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
