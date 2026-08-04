"""BiCodec TTS request builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from anydataset.types import Modality
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..prediction import PredictionModality
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.protocol import GenerationRuntime
from ..task import Task
from ..templates import format_instruction, select_template
from .types import Request


def prepare_bicodec_tts_request(
    text: str,
    runtime: GenerationRuntime,
    *,
    reference_codes: SemanticAcousticCodes | None = None,
    language: str = "English",
    messages: Sequence[Mapping[str, str]] | None = None,
    task: Task = Task.TTS,
) -> Request:
    """Build a BiCodec request whose token sequence owns global generation.

    When ``reference_codes`` is present, its global stream is serialized into
    the prompt and generation starts from the semantic marker. Without a
    reference, generation starts after BOA and the model produces both global
    and semantic streams.

    The helper intentionally accepts pre-encoded codes only.  Waveform encoding
    belongs to the runtime/codec boundary and should happen before this API.
    """
    _validate_tts_arguments(text, language, task)
    tokenizer = _validate_bicodec_tokenizer(runtime)
    text_ids = _text_prompt_ids(text, language, task, runtime, messages=messages)
    values = [text_ids]
    if reference_codes is not None:
        _validate_reference_codes(reference_codes)
        local_audio_ids = tokenizer.encode_acoustic(reference_codes)
        global_audio_ids = runtime.layout.to_global(
            Modality.AUDIO.value,
            local_audio_ids,
        ).to(device=text_ids.device)
        values.extend(
            (
                text_ids.new_tensor([runtime.boa_token_id]),
                global_audio_ids,
                text_ids.new_tensor([runtime.eoa_token_id]),
            )
        )
    values.append(text_ids.new_tensor([runtime.boa_token_id]))
    prompt_ids = torch.cat(values)
    return Request(
        prompt_ids=prompt_ids,
        task=task,
        audio_input_positions=None,
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


def _task_instruction(
    text: str,
    language: str,
    task: Task,
    *,
    template_index: int = 0,
) -> str:
    instruction = select_template(task, template_index)
    if "{source}" not in instruction:
        raise ValueError(
            f"{task.value} chat template must include a {{source}} placeholder."
        )
    return format_instruction(
        task,
        source=text,
        language=language,
        index=template_index,
    )


def _token_ids(text: str, runtime: GenerationRuntime) -> Tensor:
    values = torch.as_tensor(
        runtime.text_tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
    )
    if values.dim() != 1:
        raise ValueError("text tokenizer must return a 1D token sequence.")
    return values


__all__ = [
    "prepare_bicodec_tts_request",
]
