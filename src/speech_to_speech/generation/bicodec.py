"""BiCodec TTS request builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from anydataset.types import Modality
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..audio_route import (
    BICODEC_GENERATE_GLOBAL,
    BICODEC_REUSE_PROMPT_GLOBAL,
)
from ..prediction import PredictionModality
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.protocol import GenerationRuntime
from ..task import Task
from .types import Request

def prepare_bicodec_tts_request(
    text: str,
    reference_codes: SemanticAcousticCodes,
    runtime: GenerationRuntime,
    *,
    language: str = "English",
    messages: Sequence[Mapping[str, str]] | None = None,
    task: Task = Task.TTS,
) -> Request:
    """Build a reference-conditioned text-to-speech generation request.

    ``reference_codes`` must be one unbatched BiCodec sample with semantic codes
    shaped ``[units, 1]`` and fixed-length acoustic codes shaped
    ``[slots, codebooks]``.  The returned request keeps those codes as its
    ``audio_context`` while serializing the route-owned prompt streams into
    layout-global ``prompt_ids``.

    The helper intentionally accepts pre-encoded codes only.  Waveform encoding
    belongs to the runtime/codec boundary and should happen before this API.
    """
    _validate_tts_arguments(text, language, task)

    if runtime.audio_route != BICODEC_REUSE_PROMPT_GLOBAL:
        raise ValueError(
            "BiCodec reference requests require the global-only reference route."
        )
    route = BICODEC_REUSE_PROMPT_GLOBAL

    tokenizer = _validate_bicodec_tokenizer(runtime)
    _validate_reference_codes(reference_codes)

    streams = route.prompt.canonical_streams
    local_audio_ids = tokenizer.encode_streams(reference_codes, streams)
    text_ids = _text_prompt_ids(text, language, task, runtime, messages=messages)
    global_audio_ids = runtime.layout.to_global(
        Modality.AUDIO.value,
        local_audio_ids,
    ).to(device=text_ids.device)
    prompt_ids = torch.cat(
        (
            text_ids,
            text_ids.new_tensor([runtime.boa_token_id]),
            global_audio_ids,
            text_ids.new_tensor([runtime.eoa_token_id]),
            text_ids.new_tensor([runtime.boa_token_id]),
        )
    )
    return Request(
        prompt_ids=prompt_ids,
        task=task,
        audio_input_positions=None,
        audio_context=reference_codes,
    )


def prepare_bicodec_global_tts_request(
    text: str,
    runtime: GenerationRuntime,
    *,
    language: str = "English",
    messages: Sequence[Mapping[str, str]] | None = None,
    task: Task = Task.TTS,
) -> Request:
    """Build an unconditioned BiCodec global-plus-semantic AR request.

    The selected route must own both output streams and must not require prompt
    audio.
    """
    _validate_tts_arguments(text, language, task)
    _validate_bicodec_tokenizer(runtime)
    if runtime.audio_route != BICODEC_GENERATE_GLOBAL:
        raise ValueError(
            "BiCodec global requests require the no-reference global route."
        )

    text_ids = _text_prompt_ids(text, language, task, runtime, messages=messages)
    prompt_ids = torch.cat(
        (text_ids, text_ids.new_tensor([runtime.boa_token_id]))
    )
    return Request(
        prompt_ids=prompt_ids,
        task=task,
        audio_input_positions=None,
        audio_context=None,
    )


def _validate_tts_arguments(text: str, language: str, task: Task) -> None:
    if not isinstance(text, str):
        raise TypeError("BiCodec TTS text must be a string.")
    if not text.strip():
        raise ValueError("BiCodec TTS text must not be empty.")
    if not isinstance(language, str):
        raise TypeError("BiCodec TTS language must be a string.")
    if not language.strip():
        raise ValueError("BiCodec TTS language must not be empty.")
    if not isinstance(task, Task):
        raise TypeError("BiCodec TTS task must be a Task.")
    if (
        task.source_modality is not Modality.TEXT
        or task.prediction_modality is not PredictionModality.AUDIO
    ):
        raise ValueError("BiCodec requests require a text-to-audio task.")


def _validate_bicodec_tokenizer(runtime: GenerationRuntime) -> BiCodecAudioTokenizer:
    tokenizer = runtime.audio_tokenizer
    if not isinstance(tokenizer, BiCodecAudioTokenizer):
        raise TypeError("BiCodec requests require BiCodecAudioTokenizer.")
    return tokenizer


def _validate_reference_codes(value: object) -> None:
    if not isinstance(value, SemanticAcousticCodes):
        raise TypeError("BiCodec reference codes must be SemanticAcousticCodes.")
    if value.semantic.dim() != 2 or value.semantic.size(1) != 1:
        raise ValueError("BiCodec semantic reference codes must have shape [units, 1].")
    if value.acoustic.dim() != 2:
        raise ValueError(
            "BiCodec acoustic reference codes must have shape [slots, codebooks]."
        )
    if value.semantic.numel() == 0 or value.acoustic.numel() == 0:
        raise ValueError("BiCodec reference codes must not be empty.")
    if value.semantic.device != value.acoustic.device:
        raise ValueError(
            "BiCodec semantic and acoustic reference codes must share a device."
        )
    for name, codes in (("semantic", value.semantic), ("acoustic", value.acoustic)):
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError(f"BiCodec {name} reference codes must use signed integers.")


def _text_prompt_ids(
    text: str,
    language: str,
    task: Task,
    runtime: GenerationRuntime,
    *,
    messages: Sequence[Mapping[str, str]] | None = None,
) -> Tensor:
    if messages is None:
        instruction = _task_instruction(text, language, task)
        messages = [{"role": "user", "content": instruction}]
    rendered = runtime.text_tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("text tokenizer chat template must return a string.")
    local_ids = _token_ids(rendered, runtime)
    if local_ids.numel() == 0:
        raise ValueError("text prompt must contain at least one token.")
    return runtime.layout.to_global(Modality.TEXT.value, local_ids)


def _task_instruction(text: str, language: str, task: Task) -> str:
    template = task.templates[0]
    if "{source}" not in template:
        raise ValueError(
            f"{task.value} chat template must include a {{source}} placeholder."
        )
    kwargs: dict[str, str] = {"source": text}
    if "{language}" in template:
        kwargs["language"] = language
    return template.format(**kwargs)


def _token_ids(text: str, runtime: GenerationRuntime) -> Tensor:
    values = torch.as_tensor(
        runtime.text_tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
    )
    if values.dim() != 1:
        raise ValueError("text tokenizer must return a 1D token sequence.")
    return values


__all__ = [
    "prepare_bicodec_global_tts_request",
    "prepare_bicodec_tts_request",
]
