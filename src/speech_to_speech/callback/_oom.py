from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from .._oom import context as exception_context
from .._oom import is_oom, tensor_report
from ..datamodule.types import (
    ModelBatch,
    RawSpeech,
    RawSpeechBatch,
    Speech,
    Text,
)

if TYPE_CHECKING:
    from ..generation.eval.text import TextProbe
    from ..generation.types import Request


class _CallbackTrainer(Protocol):
    callbacks: list[Callback]


class OOMDiagnostics(Callback):
    """Report the active schema-aware input summary when PyTorch runs out of memory."""

    def __init__(self) -> None:
        super().__init__()
        self._context: dict[str, object] | None = None

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module
        self.capture(
            phase="train_step",
            inputs=batch_report(batch),
            batch_idx=batch_idx,
        )

    def on_before_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        loss: Tensor,
    ) -> None:
        del trainer, pl_module, loss
        self._phase("train_backward")

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del trainer, pl_module, optimizer
        self._phase("train_optimizer")

    def on_after_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module
        self._phase("train_post_backward")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        self._context = None

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module
        self.capture(
            phase="validation_step",
            inputs=batch_report(batch),
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        self._context = None

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        if not is_oom(exception):
            return
        write_report(
            trainer,
            pl_module,
            exception,
            exception_context(exception) or self._context,
        )

    def capture(
        self,
        *,
        phase: str,
        inputs: Mapping[str, object],
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ) -> None:
        context: dict[str, object] = {
            "phase": phase,
            "inputs": dict(inputs),
        }
        if batch_idx is not None:
            context["batch_idx"] = batch_idx
        if dataloader_idx is not None:
            context["dataloader_idx"] = dataloader_idx
        self._context = context

    def _phase(self, phase: str) -> None:
        if self._context is not None:
            self._context["phase"] = phase


def report_oom(
    trainer: Trainer,
    pl_module: LightningModule,
    exception: BaseException,
    *,
    phase: str,
    inputs: Mapping[str, object],
) -> bool:
    """Attach exact call-site context or report immediately without the callback."""

    if not is_oom(exception):
        return False
    attached = exception_context(exception)
    if attached is not None:
        phase = attached["phase"]
        inputs = attached["inputs"]
    callback_trainer = cast(_CallbackTrainer, cast(object, trainer))
    for callback in callback_trainer.callbacks:
        if isinstance(callback, OOMDiagnostics):
            callback.capture(phase=phase, inputs=inputs)
            return True
    write_report(
        trainer,
        pl_module,
        exception,
        {"phase": phase, "inputs": dict(inputs)},
    )
    return True


def batch_report(batch: object) -> dict[str, object]:
    if isinstance(batch, ModelBatch):
        target = batch.acoustic_target
        return {
            "type": type(batch).__name__,
            "tasks": [task.value for task in batch.tasks],
            "input_ids": tensor_report(batch.input_ids),
            "token_labels": tensor_report(batch.token_labels),
            "token_groups": tensor_report(batch.token_groups),
            "audio_seconds": tensor_report(batch.audio_seconds),
            "generation_prompt_lengths": tensor_report(batch.generation_prompt_lengths),
            "audio_input_positions": tensor_report(batch.audio_input_positions),
            "acoustic_target": (
                None
                if target is None
                else {
                    "semantic_codes": tensor_report(target["semantic_codes"]),
                    "codes": tensor_report(target["codes"]),
                    "token_positions": tensor_report(target["token_positions"]),
                }
            ),
            "audio_contexts": [
                (
                    None
                    if context is None
                    else {
                        "semantic": tensor_report(context.semantic),
                        "acoustic": tensor_report(context.acoustic),
                    }
                )
                for context in batch.audio_contexts or ()
            ],
        }
    if isinstance(batch, RawSpeechBatch):
        return {
            "type": type(batch).__name__,
            "samples": [
                {
                    "task": sample.task.value,
                    "source": item_report(sample.source),
                    "target": item_report(sample.target),
                    "audio_context": item_report(sample.audio_context),
                }
                for sample in batch.samples
            ],
        }
    return {"type": type(batch).__name__}


def generation_report(
    requests: Sequence[Request],
    *,
    max_new_tokens: int,
    do_sample: bool,
    use_cache: bool,
) -> dict[str, object]:
    prompt_width = max(
        (request["prompt_ids"].numel() for request in requests), default=0
    )
    audio_width: int | None = None
    for request in requests:
        positions = request.get("audio_input_positions")
        if positions is not None:
            audio_width = max(audio_width or 0, positions.numel())
    return {
        "type": "GenerationRequests",
        "tasks": [request["task"].value for request in requests],
        "prompt_ids": [tensor_report(request["prompt_ids"]) for request in requests],
        "padded_prompt_shape": [len(requests), prompt_width],
        "padded_audio_input_positions_shape": (
            None if audio_width is None else [len(requests), audio_width]
        ),
        "audio_input_positions": [
            tensor_report(request.get("audio_input_positions")) for request in requests
        ],
        "audio_contexts": [
            (
                None
                if (context := request.get("audio_context")) is None
                else {
                    "semantic": tensor_report(context.semantic),
                    "acoustic": tensor_report(context.acoustic),
                }
            )
            for request in requests
        ],
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": use_cache,
    }


def text_probe_report(
    probes: Mapping[str, TextProbe],
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    return {
        "type": "TextProbes",
        "count": len(probes),
        "instruction_characters": {
            name: len(probe["instruction"]) for name, probe in probes.items()
        },
        "reference_characters": {
            name: len(probe["reference"]) for name, probe in probes.items()
        },
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }


def item_report(value: Speech | Text | RawSpeech | None) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, RawSpeech):
        return {
            "type": type(value).__name__,
            "text_token_ids": tensor_report(value.text_token_ids),
            "waveform": tensor_report(value.waveform),
            "sample_rate": value.sample_rate,
        }
    if isinstance(value, Text):
        return {
            "type": type(value).__name__,
            "text_token_ids": tensor_report(value.text_token_ids),
        }
    return {
        "type": type(value).__name__,
        "semantic_codes": tensor_report(value.semantic_codes),
        "acoustic_codes": tensor_report(value.acoustic_codes),
        "text_token_ids": tensor_report(value.text_token_ids),
        "audio_token_ids": tensor_report(value.audio_token_ids),
        "audio_token_spans": tensor_report(value.audio_token_spans),
    }


def write_report(
    trainer: Trainer,
    pl_module: LightningModule,
    exception: BaseException,
    context: Mapping[str, object] | None,
) -> None:
    payload: dict[str, object] = {
        "event": "out_of_memory",
        "error": {
            "type": type(exception).__name__,
            "message": str(exception),
        },
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "global_rank": trainer.global_rank,
        "local_rank": trainer.local_rank,
        "context": (
            {"phase": "outside_batch", "inputs": None}
            if context is None
            else dict(context)
        ),
        "cuda": cuda_report(pl_module),
    }
    print(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def cuda_report(pl_module: LightningModule) -> dict[str, object] | None:
    device = pl_module.device
    if device.type != "cuda":
        return None
    try:
        return {
            "device": str(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    except RuntimeError as error:
        return {
            "device": str(device),
            "error": f"{type(error).__name__}: {error}",
        }


__all__ = ["OOMDiagnostics"]
