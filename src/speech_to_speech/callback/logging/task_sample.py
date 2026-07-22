from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict, cast

import torch
from anydataset import types
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...generation import Request, Result
from ...generation.batch import requests_from_batch
from .._lightning import attached_datamodule, audio_experiment, text_experiment


class _Module(Protocol):
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
    collator: Any

    def train_samples(self, indices: Sequence[int]) -> list[types.Sample]: ...


class TaskSampleLogger(Callback):
    def __init__(
        self,
        indices: Sequence[int],
        every_n_steps: int,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if not indices:
            raise ValueError("indices must contain at least one sample index.")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("indices must contain integer sample indices.")
        self.indices = list(indices)
        self.every_n_steps = every_n_steps
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
        self.samples = datamodule.train_samples(self.indices)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        del batch, batch_idx
        if not trainer.is_global_zero:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return
        audio_writer = audio_experiment(trainer)
        text_writer = text_experiment(trainer)
        if audio_writer is None and text_writer is None:
            return
        module = cast(_Module, cast(object, pl_module))
        datamodule = cast(_DataModule, attached_datamodule(trainer))
        sample_batch = datamodule.collator(self.samples)
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
                audio_writer.add_audio(
                    f"{tag}/generated",
                    audio["waveform"].detach().cpu(),
                    trainer.global_step,
                    sample_rate=audio["sample_rate"],
                )
            elif text_writer is not None:
                text_writer.add_text(
                    f"{tag}/generated_ids",
                    " ".join(str(value) for value in result["response_ids"].tolist()),
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
        "source_acoustic_frames": _acoustic_frames(request),
    }


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


def _acoustic_frames(request: Request) -> int:
    acoustic = request["acoustic_prompt"]
    if acoustic is None:
        return 0
    return int(acoustic["codes"].size(0))


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
