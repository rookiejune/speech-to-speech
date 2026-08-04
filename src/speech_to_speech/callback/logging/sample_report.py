from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import Any, Protocol, TypedDict

import torch
from anydataset import types
from anytrain.module.idspace import Layout
from torch import Tensor

from ...audio import AudioCodes
from ...datamodule.batch import ModelBatch, TrainInput
from ...datamodule.diagnostic import SampleRef, SampleSplit, source_item, target_item
from ...generation.contract import Result
from ...generation.decode import decode_reference_codes
from ...generation.evaluation import reference_audio
from ...runtime.audio_tokenizer import BiCodecAudioTokenizer
from ...runtime.backbone.contract import TextTokenizer
from ...runtime.codec_contract import (
    CodecBackend,
    acoustic_codec,
    codec_sample_rate,
)
from ...task import PredictionModality, Request, Task


_TOKEN_PREVIEW_LIMIT = 128


class Module(Protocol):
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


class GenerationKwargs(TypedDict):
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    use_cache: bool


class DataModule(Protocol):
    runtime: LoggingRuntime

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


class LoggingRuntime(Protocol):
    @property
    def audio_view(self) -> types.AudioView: ...

    @property
    def codec(self) -> CodecBackend: ...

    @property
    def audio_tokenizer(self) -> object: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def layout(self) -> Layout: ...


def build_request_metadata(
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
        "source": modality_metadata(source_item(sample, task), task.source_modality),
        "reference": modality_metadata(target_item(sample, task), reference_modality),
    }


def sample_log_record(
    datamodule: DataModule,
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
    request_metadata = build_request_metadata(dataset_index, sample, request)
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
        generation["response_ids"] = token_sequence(response_ids)
        decoded = (
            generated_text
            if generated_text is not None
            else decode_text(datamodule, response_ids)
        )
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
        "prompt_ids": token_sequence(prompt_ids),
        "source": request_metadata["source"],
    }
    prediction = request.get("prediction")
    if prediction is not None:
        chat_template["prediction"] = prediction.value
    labels: dict[str, Any] = {
        "token_labels": token_sequence(token_labels),
        "supervised_token_ids": token_sequence(supervised),
        "reference": request_metadata["reference"],
    }
    if error is None:
        prompt_text = decode_chat_template(datamodule, prompt_ids)
        if prompt_text is not None:
            chat_template["text"] = prompt_text
        label_text = decode_text(datamodule, supervised)
        if label_text is not None:
            labels["text"] = label_text

    return {
        "chat_template": chat_template,
        "labels": labels,
        "generation": generation,
    }


def prediction_modality(request: Request) -> PredictionModality:
    prediction = request.get("prediction")
    if prediction is None:
        return request["task"].prediction_modality
    if not isinstance(prediction, PredictionModality):
        raise TypeError("generation request prediction must be a PredictionModality.")
    return prediction




def decode_chat_template(
    datamodule: DataModule,
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


def decode_text(datamodule: DataModule, token_ids: Tensor) -> str | None:
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


def token_sequence(token_ids: Tensor) -> dict[str, Any]:
    values = [int(value) for value in token_ids.reshape(-1).tolist()]
    return {
        "count": len(values),
        "ids": values[:_TOKEN_PREVIEW_LIMIT],
        "truncated": len(values) > _TOKEN_PREVIEW_LIMIT,
    }


def reference_text(
    sample: types.Sample,
    task: Task,
    prediction: PredictionModality,
) -> str | None:
    if not prediction.supervises_text:
        return None
    if prediction.is_mixed:
        # Mixed targets are Speech items; read aligned text from the DEFAULT/TARGET
        # text view when present.
        role = text_role(sample, task)
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


def text_role(sample: types.Sample, task: Task) -> types.Role:
    roles = {role for role, _ in sample}
    if types.Role.DEFAULT in roles:
        return types.Role.DEFAULT
    return types.Role.TARGET if not task.uses_source_role else types.Role.TARGET


def modality_metadata(
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
        view, value = diagnostic_audio_view(item)
        return {
            "modality": modality.value,
            "role": role.value,
            "view": view.value,
            **(
                waveform_metadata(value)
                if view is types.AudioView.WAVEFORM
                else codes_metadata(value, view=view)
            ),
        }
    raise AssertionError(f"unsupported sample modality: {modality.value}")


def build_result_metadata(
    result: Result,
    *,
    max_new_tokens: int,
    prediction: PredictionModality,
    runtime: LoggingRuntime,
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
            bicodec = partial_bicodec_metadata(runtime, response_ids)
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
        "waveform": tensor_metadata(waveform),
        "waveform_samples": int(waveform.size(-1)),
        "duration_seconds": waveform.size(-1) / audio["sample_rate"],
        "waveform_finite": finite(waveform),
        "features": None if features is None else tensor_metadata(features),
    }


def partial_bicodec_metadata(
    runtime: LoggingRuntime,
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
    expected_global = tokenizer.global_unit_length * len(
        tokenizer.global_codebook_sizes
    )
    semantic_start, semantic_end = tokenizer.semantic_token_range
    summary: dict[str, Any] = {
        "expected_global_tokens": expected_global,
        "global_tokens": 0,
        "semantic_tokens": 0,
        "has_end_marker": tokenizer.end_token_id in values,
    }

    global_marker_index = first_index(values, tokenizer.global_token_id)
    if global_marker_index is not None:
        payload_start = global_marker_index + 1
        payload_end = first_marker_index(
            values,
            (
                tokenizer.semantic_token_id,
                tokenizer.end_token_id,
            ),
            start=payload_start,
        )
        if payload_end is None:
            payload_end = len(values)
        global_payload = values[payload_start:payload_end]
        summary["global_tokens"] = len(global_payload)

    semantic_marker_index = first_index(values, tokenizer.semantic_token_id)
    if semantic_marker_index is not None:
        payload_start = semantic_marker_index + 1
        payload_end = first_marker_index(
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


def first_index(values: Sequence[int], target: int) -> int | None:
    try:
        return values.index(target)
    except ValueError:
        return None


def first_marker_index(
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


def codes_metadata(
    codes: object,
    *,
    view: types.AudioView | None = None,
) -> dict[str, Any]:
    if isinstance(codes, AudioCodes):
        semantic = codes.semantic_codes
        global_codes = codes.global_codes
        acoustic = codes.acoustic_codes
    elif isinstance(codes, Mapping):
        semantic = codes.get("semantic")
        if view is types.AudioView.BICODEC:
            if set(codes) != {"semantic", "global"}:
                raise ValueError(
                    "anydataset BiCodec codes require exactly semantic and global."
                )
            global_codes = codes.get("global")
            if not isinstance(semantic, Tensor) or not isinstance(global_codes, Tensor):
                raise TypeError(
                    "anydataset BiCodec codes require Tensor semantic/global fields."
                )
            acoustic = None
        else:
            if set(codes) != {"semantic", "acoustic"}:
                raise ValueError(
                    "anydataset semantic-acoustic codes require exactly semantic and acoustic."
                )
            acoustic = codes.get("acoustic")
            if not isinstance(semantic, Tensor) or not isinstance(acoustic, Tensor):
                raise TypeError(
                    "anydataset semantic-acoustic codes require Tensor fields."
                )
            global_codes = None
    else:
        semantic = None
        global_codes = None
        acoustic = None
    if semantic is not None and (global_codes is not None or acoustic is not None):
        return {
            "structured": True,
            "semantic": codetensor_metadata(semantic),
            "global": (
                None
                if global_codes is None
                else codetensor_metadata(global_codes)
            ),
            "acoustic": (
                None if acoustic is None else codetensor_metadata(acoustic)
            ),
        }
    if not isinstance(codes, Tensor):
        raise TypeError("audio sample codes must be a Tensor or structured mapping.")
    return codetensor_metadata(codes)


def diagnostic_audio_view(
    item: types.AudioItem,
) -> tuple[types.AudioView, object]:
    for view, value in item.views.items():
        if view is not types.AudioView.WAVEFORM:
            return view, value
    try:
        return types.AudioView.WAVEFORM, item.views[types.AudioView.WAVEFORM]
    except KeyError as error:
        raise ValueError("diagnostic audio item has no views.") from error


def waveform_metadata(value: object) -> dict[str, Any]:
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
        "waveform": tensor_metadata(waveform),
        "sample_rate": sample_rate,
        "duration_seconds": waveform.size(-1) / sample_rate,
        "waveform_finite": finite(waveform),
    }


def codetensor_metadata(codes: Tensor) -> dict[str, Any]:
    if codes.dim() != 2:
        raise ValueError("audio sample codes must have shape [frames, codebooks].")
    return {
        "frames": int(codes.size(0)),
        "codebooks": int(codes.size(1)),
        "codes_dtype": str(codes.dtype),
    }


def tensor_metadata(tensor: Tensor) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
    }


def finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def metadata_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def text_metrics(reference: str, generated: str) -> dict[str, float]:
    expected = _normalized_text(reference)
    actual = _normalized_text(generated)
    return {
        "text/cer": _edit_distance(expected, actual) / max(1, len(expected)),
        "text/exact_match": float(actual == expected),
        "text/reference_chars": float(len(expected)),
        "text/generated_chars": float(len(actual)),
    }


def audio_metrics(
    waveform: Tensor,
    sample_rate: int,
    *,
    target_duration: float | None = None,
) -> dict[str, float]:
    if waveform.numel() == 0:
        raise ValueError("sample audio metrics require a non-empty waveform.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample audio metric sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("sample audio metric sample_rate must be positive.")
    value = waveform.detach().float()
    duration = value.size(-1) / sample_rate
    finite_value = bool(torch.isfinite(value).all().item())
    metrics = {
        "audio/duration_seconds": float(duration),
        "audio/finite": float(finite_value),
    }
    if finite_value:
        magnitude = value.abs()
        metrics.update(
            {
                "audio/rms": float(value.square().mean().sqrt().item()),
                "audio/peak": float(magnitude.max().item()),
                "audio/silence_ratio": float(magnitude.lt(1e-4).float().mean().item()),
                "audio/clipping_ratio": float(magnitude.ge(0.999).float().mean().item()),
            }
        )
    if target_duration is not None:
        if not math.isfinite(target_duration) or target_duration <= 0:
            raise ValueError("sample target duration must be finite and positive.")
        metrics["audio/duration_ratio"] = duration / target_duration
    return metrics


def _normalized_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("sample text metrics require strings.")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _edit_distance(reference: str, generated: str) -> int:
    previous = list(range(len(generated) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(generated, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(expected != actual),
                )
            )
        previous = current
    return previous[-1]


def log_target_reference_audio(
    audio_writer: Any | None,
    datamodule: DataModule,
    module: Module,
    batch: Any,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> tuple[Tensor, int] | None:
    if not task.prediction_modality.supervises_audio:
        return None
    if batch.acoustic_target is None:
        target, sample_rate = sample_audio(datamodule, sample, task, source=False)
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


def log_source_audio(
    audio_writer: Any,
    datamodule: DataModule,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> None:
    if task.source_modality is not types.Modality.AUDIO:
        return
    waveform, sample_rate = sample_audio(datamodule, sample, task, source=True)
    audio_writer.add_audio(
        f"{tag}/source", waveform, step, sample_rate=sample_rate
    )


def log_target_audio(
    audio_writer: Any,
    datamodule: DataModule,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> None:
    if not task.prediction_modality.supervises_audio:
        return
    waveform, sample_rate = sample_audio(datamodule, sample, task, source=False)
    audio_writer.add_audio(
        f"{tag}/target",
        waveform,
        step,
        sample_rate=sample_rate,
    )


def sample_audio(
    datamodule: DataModule,
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

__all__ = [
    "DataModule",
    "GenerationKwargs",
    "LoggingRuntime",
    "Module",
    "audio_metrics",
    "metadata_json",
    "prediction_modality",
    "build_request_metadata",
    "build_result_metadata",
    "log_source_audio",
    "log_target_audio",
    "log_target_reference_audio",
    "sample_audio",
    "sample_log_record",
    "reference_text",
    "text_metrics",
]
