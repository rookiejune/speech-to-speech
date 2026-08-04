from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from anydataset import types

from ...datamodule.batch import ModelBatch
from ...generation import decode_response_text
from ...generation.result import Result
from ...task import PredictionModality, Request
from ._sample_audio import log_source_audio, log_target_reference_audio
from ._sample_metadata import (
    build_result_metadata,
    metadata_json,
    prediction_modality,
    reference_text,
    sample_log_record,
)
from ._sample_metrics import audio_metrics, text_metrics
from ._sample_protocol import DataModule, Module


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
                None if target_audio is None else target_audio[0].size(-1) / target_audio[1]
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

__all__ = ["RowLogContext", "log_result_row"]
