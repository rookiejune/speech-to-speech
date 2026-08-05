"""OpenAI-style messages adapter: messages -> HF chat template -> tensor Request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, Union, cast

import torch
from anydataset.types import AudioView, Modality
from anytrain.codec import SemanticGlobalCodes
from torch import Tensor
from typing_extensions import NotRequired

from .._tensor import is_signed_integer_dtype
from ..audio import AudioCodes
from ..datamodule.parse import parse_audio_codes
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.codec import has_codec_loader
from ..runtime.codec_contract import frame_codec, global_codec, supports_global
from ..runtime.protocol import GenerationRuntime
from ..task import (
    FieldRole,
    PredictionModality,
    Request,
    ResponseSpec,
    Task,
    normalize_language_code,
    resolve_response,
)
from ..task.templates import (
    format_instruction,
    format_response_instruction,
    select_template,
)
from .bicodec import prepare_bicodec_tts_request
from .contract import AudioOutput, Result, TokenGenerator
from .service import generate_responses
from .text import ResponseStepRuntime, decode_response_text_steps

_AUDIO_SOURCE_PLACEHOLDER = "$$$AUDIO_SOURCE$$$"


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class AudioPart(TypedDict):
    type: Literal["audio"]
    waveform: Tensor
    sample_rate: int


class CodecCodesPart(TypedDict):
    type: Literal["codec_codes"]
    codec: str
    codes: AudioCodes | Tensor


ContentPart = Union[TextPart, AudioPart, CodecCodesPart]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class ChatRequest(TypedDict):
    messages: list[Message]
    task: Task
    language: NotRequired[str]
    trace: NotRequired[str | None]


class ChatTraceStep(TypedDict):
    index: int
    role: Literal["source", "target"]
    modality: Literal["text"]
    content: str


class ChatMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    audio: AudioOutput | None
    trace: NotRequired[list[ChatTraceStep]]
    decode_error: NotRequired[dict[str, str]]


class ChatChoice(TypedDict):
    index: int
    message: ChatMessage


class ChatCompletion(TypedDict):
    choices: list[ChatChoice]


@torch.no_grad()
def create(
    request: ChatRequest,
    model: TokenGenerator,
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    use_cache: bool = True,
) -> ChatCompletion:
    """OpenAI-style chat entry: messages -> HF template / codec -> generate."""
    private = to_request(request, model.runtime)
    results = generate_responses(
        [private],
        model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_cache=use_cache,
    )
    if len(results) != 1:
        raise RuntimeError("chat create expects exactly one generation result.")
    return completion_from_result(results[0], request, model.runtime)


def to_request(request: ChatRequest, runtime: GenerationRuntime) -> Request:
    """Lower ChatRequest to the private tensor Request."""
    task = _task(request)
    response = _response(request, task)
    language = _language(request)
    messages = _messages(request)
    media = _media(messages)
    prompt_messages, source_text = _prompt_messages(
        messages,
        task,
        language,
        response,
        allow_empty_user=task.source_modality is Modality.AUDIO and media is not None,
    )
    codes = (
        None
        if media is None
        else (
            _materialize_input_codes(media, runtime)
            if task.source_modality is Modality.AUDIO
            else materialize_codes(media, runtime)
        )
    )
    return _build_request(
        prompt_messages,
        source_text,
        codes,
        task=task,
        response=response,
        language=language,
        runtime=runtime,
    )


def materialize_codes(
    part: AudioPart | CodecCodesPart,
    runtime: GenerationRuntime,
) -> AudioCodes | Tensor:
    """Turn an audio or codec_codes part into runtime codes."""
    if part["type"] == "codec_codes":
        return _validated_codes(part, runtime)
    if part["type"] != "audio":
        raise TypeError(f"unsupported media part type: {part.get('type')!r}")
    return _encode_audio(part, runtime)


def _materialize_input_codes(
    part: AudioPart | CodecCodesPart,
    runtime: GenerationRuntime,
) -> AudioCodes | Tensor:
    if not runtime.input_audio_decoupled:
        return materialize_codes(part, runtime)
    if part["type"] == "codec_codes":
        return _validated_input_codes(part, runtime)
    if part["type"] != "audio":
        raise TypeError(f"unsupported media part type: {part.get('type')!r}")
    if not has_codec_loader(runtime.input_codec_name):
        raise ValueError(
            f"input audio tokenizer {runtime.input_codec_name!r} has no runtime "
            "codec backend; pass precomputed input codec_codes."
        )
    return _encode_audio(part, runtime, input_audio=True)


def completion_from_result(
    result: Result,
    request: ChatRequest,
    runtime: GenerationRuntime,
) -> ChatCompletion:
    task = _task(request)
    response = _response(request, task)
    text_steps = decode_response_text_steps(
        cast(ResponseStepRuntime, runtime),
        result["response_ids"],
        response,
        target_language=(
            normalize_language_code(_language(request))
            if response.requires_target_language
            else None
        ),
    )
    target_text_indices = [
        index
        for index, field in enumerate(response.fields)
        if field.role is FieldRole.TARGET and field.modality is Modality.TEXT
    ]
    content_index = target_text_indices[-1] if target_text_indices else None
    content = None if content_index is None else text_steps[content_index]
    trace_steps = [
        ChatTraceStep(
            index=index,
            role="source" if field.role is FieldRole.SOURCE else "target",
            modality="text",
            content=value,
        )
        for index, (field, value) in enumerate(zip(response.fields, text_steps))
        if field.modality is Modality.TEXT
        and index != content_index
        and value is not None
    ]
    message = ChatMessage(
        role="assistant",
        content=content,
        audio=result["audio"],
    )
    if trace_steps:
        message["trace"] = trace_steps
    if "decode_error" in result:
        message["decode_error"] = result["decode_error"]
    return ChatCompletion(choices=[ChatChoice(index=0, message=message)])


def _build_request(
    prompt_messages: Sequence[Mapping[str, str]],
    source_text: str,
    codes: AudioCodes | Tensor | None,
    *,
    task: Task,
    response: ResponseSpec,
    language: str,
    runtime: GenerationRuntime,
) -> Request:
    target_language = (
        normalize_language_code(language)
        if response.requires_target_language
        else None
    )
    if task.source_modality is Modality.AUDIO:
        if codes is None:
            raise ValueError("audio-source chat requests require audio or codec_codes.")
        return _build_audio_source_request(
            prompt_messages,
            codes,
            task=task,
            response=response,
            target_language=target_language,
            runtime=runtime,
        )
    if response.prediction is PredictionModality.TEXT:
        if codes is not None:
            raise ValueError("text prediction chat requests cannot include audio media.")
        prompt_ids = _prompt_ids(prompt_messages, runtime)
        private = Request(
            prompt_ids=prompt_ids,
            task=task,
            audio_input_positions=None,
            trace=response.name,
        )
        if target_language is not None:
            private["target_language"] = target_language
        return private

    tokenizer = runtime.audio_tokenizer
    if isinstance(tokenizer, BiCodecAudioTokenizer):
        if response.prediction.is_mixed:
            raise ValueError("BiCodec chat does not support mixed response traces.")
        if codes is not None and not isinstance(codes, AudioCodes):
            raise TypeError("BiCodec chat requests require AudioCodes.")
        private = prepare_bicodec_tts_request(
            source_text,
            runtime,
            reference_codes=codes,
            language=language,
            messages=prompt_messages,
            task=task,
        )
        private["trace"] = response.name
        if target_language is not None:
            private["target_language"] = target_language
        return private

    if codes is not None:
        raise ValueError(
            "chat audio/codec_codes are not supported for the current "
            "audio_sequence_layout."
        )
    prompt_ids = _prompt_ids(prompt_messages, runtime)
    private = Request(
        prompt_ids=prompt_ids,
        task=task,
        audio_input_positions=None,
        trace=response.name,
    )
    if target_language is not None:
        private["target_language"] = target_language
    return private


def _build_audio_source_request(
    prompt_messages: Sequence[Mapping[str, str]],
    codes: AudioCodes | Tensor,
    *,
    task: Task,
    response: ResponseSpec,
    target_language: str | None,
    runtime: GenerationRuntime,
) -> Request:
    if response.prediction.is_mixed and isinstance(
        runtime.audio_tokenizer,
        BiCodecAudioTokenizer,
    ):
        raise ValueError("BiCodec chat does not support mixed response traces.")
    parsed = parse_audio_codes(codes, runtime, input_audio=True)
    semantic = parsed.semantic_codes
    if semantic is None:
        raise ValueError("audio-source chat codes require semantic units.")
    tokenizer = runtime.input_audio_tokenizer
    if isinstance(tokenizer, BiCodecAudioTokenizer):
        if parsed.global_codes is None:
            raise ValueError("BiCodec audio input requires global codes.")
        local_ids = torch.as_tensor(tokenizer.encode_full(parsed), dtype=torch.long)
    else:
        local_ids = torch.as_tensor(tokenizer.encode(semantic), dtype=torch.long)
    if local_ids.dim() != 1 or local_ids.numel() == 0:
        raise ValueError("input audio tokenizer must return a non-empty 1D sequence.")
    payload = runtime.layout.to_global(runtime.input_audio_block_name, local_ids)
    rendered = _render_prompt(prompt_messages, runtime)
    pieces = rendered.split(_AUDIO_SOURCE_PLACEHOLDER)
    if len(pieces) != 2:
        raise ValueError(
            "audio-source chat prompt must contain exactly one source placeholder."
        )
    prefix = runtime.layout.to_global(
        Modality.TEXT.value,
        _token_ids(pieces[0], runtime),
    )
    suffix = runtime.layout.to_global(
        Modality.TEXT.value,
        _token_ids(pieces[1], runtime),
    )
    start = prefix.numel() + 2
    positions = torch.arange(
        start,
        start + payload.numel(),
        dtype=torch.long,
        device=payload.device,
    )
    prompt_ids = torch.cat(
        (
            prefix,
            payload.new_tensor([runtime.input_boa_token_id]),
            payload.new_tensor([runtime.input_audio_schema_token_id]),
            payload,
            payload.new_tensor([runtime.input_eoa_token_id]),
            suffix,
        )
    )
    private = Request(
        prompt_ids=prompt_ids,
        task=task,
        audio_input_positions=positions,
        trace=response.name,
    )
    if target_language is not None:
        private["target_language"] = target_language
    return private


def _prompt_ids(
    messages: Sequence[Mapping[str, str]],
    runtime: GenerationRuntime,
) -> Tensor:
    rendered = _render_prompt(messages, runtime)
    local_ids = _token_ids(rendered, runtime)
    if local_ids.numel() == 0:
        raise ValueError("chat text prompt must contain at least one token.")
    if local_ids.dim() != 1:
        raise ValueError("chat text prompt token ids must be one-dimensional.")
    return runtime.layout.to_global(Modality.TEXT.value, local_ids)


def _render_prompt(
    messages: Sequence[Mapping[str, str]],
    runtime: GenerationRuntime,
) -> str:
    rendered = runtime.text_tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("text tokenizer chat template must return a string.")
    return rendered


def _prompt_messages(
    messages: Sequence[Message],
    task: Task,
    language: str,
    response: ResponseSpec,
    *,
    allow_empty_user: bool = False,
) -> tuple[list[dict[str, str]], str]:
    prompt_messages: list[dict[str, str]] = []
    source_index: int | None = None
    for message in messages:
        content = _message_text(message)
        if not content and not (
            allow_empty_user and message["role"] == "user"
        ):
            continue
        prompt_messages.append(
            {
                "role": message["role"],
                "content": content,
            }
        )
        if message["role"] == "user":
            source_index = len(prompt_messages) - 1
    if source_index is None:
        raise ValueError("chat request requires user text content.")
    source_text = prompt_messages[source_index]["content"]
    instruction_source = (
        _AUDIO_SOURCE_PLACEHOLDER
        if task.source_modality is Modality.AUDIO
        else source_text
    )
    prompt_messages[source_index]["content"] = _task_instruction(
        instruction_source,
        task,
        language,
        response,
    )
    return prompt_messages, source_text


def _task_instruction(
    text: str,
    task: Task,
    language: str,
    response: ResponseSpec,
    *,
    template_index: int = 0,
) -> str:
    instruction = select_template(task, template_index)
    if "{source}" not in instruction:
        raise ValueError(
            f"{task.value} chat template must include a {{source}} placeholder."
        )
    base = format_instruction(
        task,
        source=text,
        language=language,
        index=template_index,
    )
    return format_response_instruction(base, response, language=language)


def _token_ids(text: str, runtime: GenerationRuntime) -> Tensor:
    values = torch.as_tensor(
        runtime.text_tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
    )
    if values.dim() != 1:
        raise ValueError("text tokenizer must return a 1D token sequence.")
    return values


def _encode_audio(
    part: AudioPart,
    runtime: GenerationRuntime,
    *,
    input_audio: bool = False,
) -> AudioCodes | Tensor:
    waveform = part["waveform"]
    sample_rate = part["sample_rate"]
    if not isinstance(waveform, Tensor):
        raise TypeError("audio waveform must be a Tensor.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("audio sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be positive.")
    waveform = waveform.to(dtype=torch.float32)
    batched = _batched_waveform(waveform)
    view = runtime.input_audio_view if input_audio else runtime.audio_view
    backend = runtime.input_codec if input_audio else runtime.codec
    with torch.autocast(device_type=batched.device.type, enabled=False):
        if view is AudioView.BICODEC:
            if not supports_global(backend):
                raise TypeError("BiCodec audio parts require a semantic-global codec.")
            encoded = global_codec(backend).tokenize(batched, sample_rate)
            if not isinstance(encoded, SemanticGlobalCodes):
                raise TypeError(
                    "BiCodec tokenize must return SemanticGlobalCodes."
                )
            if encoded.semantic.size(0) != 1 or encoded.global_codes.size(0) != 1:
                raise ValueError("audio encode expects one item.")
            return AudioCodes.from_semantic_global(
                SemanticGlobalCodes(
                    semantic=encoded.semantic[0].detach(),
                    global_codes=encoded.global_codes[0].detach(),
                )
            )
        codes = frame_codec(backend).encode(batched, sample_rate)
    if not isinstance(codes, Tensor):
        raise TypeError("codec encode must return a Tensor.")
    if codes.dim() == 3:
        if codes.size(0) != 1:
            raise ValueError("audio encode expects one item.")
        return codes[0].detach()
    if codes.dim() == 2:
        return codes.detach()
    raise ValueError("codec encode must return [frames, codebooks] or [1, frames, codebooks].")


def _batched_waveform(waveform: Tensor) -> Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError("audio waveform must have shape [time] or [channel, time].")
    return waveform.unsqueeze(0)


def _validated_codes(
    part: CodecCodesPart,
    runtime: GenerationRuntime,
) -> AudioCodes | Tensor:
    codec = part["codec"]
    if not isinstance(codec, str) or not codec.strip():
        raise TypeError("codec_codes codec must be a non-empty string.")
    if codec != runtime.codec_name:
        raise ValueError(
            f"codec_codes codec {codec!r} does not match runtime codec "
            f"{runtime.codec_name!r}."
        )
    codes = part["codes"]
    if isinstance(codes, AudioCodes):
        _validate_structured_codes(codes)
        return codes
    if isinstance(codes, Tensor):
        if codes.dim() != 2:
            raise ValueError("frame codec_codes must have shape [frames, codebooks].")
        if codes.numel() == 0:
            raise ValueError("codec_codes must not be empty.")
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError("frame codec_codes must use signed integers.")
        return codes
    raise TypeError(
        "codec_codes codes must be AudioCodes or a frame-code Tensor."
    )


def _validated_input_codes(
    part: CodecCodesPart,
    runtime: GenerationRuntime,
) -> AudioCodes | Tensor:
    codec = part["codec"]
    if not isinstance(codec, str) or not codec.strip():
        raise TypeError("codec_codes codec must be a non-empty string.")
    expected = getattr(runtime, "input_codec_name", runtime.codec_name)
    if codec != expected:
        raise ValueError(
            f"codec_codes codec {codec!r} does not match runtime input codec "
            f"{expected!r}."
        )
    codes = part["codes"]
    if isinstance(codes, AudioCodes):
        semantic = codes.semantic_codes
        if semantic is None or semantic.dim() != 2:
            raise ValueError(
                "input AudioCodes must contain 2D semantic_codes."
            )
        if runtime.input_audio_view is AudioView.BICODEC:
            if codes.global_codes is None or codes.acoustic_codes is not None:
                raise ValueError(
                    "BiCodec input AudioCodes require semantic_codes and "
                    "global_codes only."
                )
            _validate_structured_codes(codes)
            return codes
        if codes.acoustic_codes is not None or codes.global_codes is not None:
            raise ValueError(
                "decoupled input AudioCodes must contain semantic_codes only."
            )
        if not is_signed_integer_dtype(semantic.dtype):
            raise TypeError("input semantic codec_codes must use signed integers.")
        return codes
    if isinstance(codes, Tensor):
        if codes.dim() != 2 or codes.numel() == 0:
            raise ValueError(
                "input frame codec_codes must be non-empty [frames, codebooks]."
            )
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError("input frame codec_codes must use signed integers.")
        return codes
    raise TypeError(
        "input codec_codes codes must be AudioCodes or a frame-code Tensor."
    )


def _validate_structured_codes(value: AudioCodes) -> None:
    semantic = value.semantic_codes
    if semantic is not None and (semantic.dim() != 2 or semantic.size(1) != 1):
        raise ValueError("structured semantic codes must have shape [units, 1].")
    secondary = value.global_codes
    name = "global"
    if secondary is None:
        secondary = value.acoustic_codes
        name = "acoustic"
    if secondary is None or secondary.dim() != 2:
        raise ValueError(
            "structured codes must contain global or aligned acoustic codes."
        )
    if value.global_codes is not None and value.acoustic_codes is not None:
        raise ValueError("structured codec_codes must select one non-semantic layout.")
    if semantic is not None and semantic.device != secondary.device:
        raise ValueError(
            f"structured semantic and {name} codes must share a device."
        )
    fields = [(name, secondary)]
    if semantic is not None:
        fields.insert(0, ("semantic", semantic))
    for field, codes in fields:
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError(f"structured {field} codes must use signed integers.")

def _task(request: ChatRequest) -> Task:
    task = request["task"]
    if not isinstance(task, Task):
        raise TypeError("chat request task must be a Task.")
    return task


def _response(request: ChatRequest, task: Task) -> ResponseSpec:
    if "prediction" in request:
        raise ValueError(
            "chat request prediction override is not supported; "
            "select a task response with trace."
        )
    trace = request.get("trace")
    if trace is not None:
        if not isinstance(trace, str):
            raise TypeError("chat request trace must be a string or None.")
        if not trace:
            raise ValueError("chat request trace must be a non-empty string or None.")
    return resolve_response(task, trace=trace)


def _language(request: ChatRequest) -> str:
    language = request.get("language", "English")
    if not isinstance(language, str):
        raise TypeError("chat request language must be a string.")
    if not language.strip():
        raise ValueError("chat request language must not be empty.")
    return language


def _messages(request: ChatRequest) -> list[Message]:
    messages = request["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat request messages must be a non-empty list.")
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("chat message must be a mapping.")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat message role: {role!r}")
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise TypeError("chat message content must be a string or part list.")
        if isinstance(content, str) and not content.strip():
            raise ValueError("chat message text content must not be empty.")
        if isinstance(content, list):
            if not content:
                raise ValueError("chat message content parts must not be empty.")
            for part in content:
                _validate_part(part)
    return cast(list[Message], messages)


def _validate_part(part: object) -> None:
    if not isinstance(part, Mapping):
        raise TypeError("content part must be a mapping.")
    kind = part.get("type")
    if kind == "text":
        text = part.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text part text must be a non-empty string.")
        return
    if kind == "audio":
        if "waveform" not in part or "sample_rate" not in part:
            raise ValueError("audio part requires waveform and sample_rate.")
        return
    if kind == "codec_codes":
        if "codec" not in part or "codes" not in part:
            raise ValueError("codec_codes part requires codec and codes.")
        return
    raise ValueError(f"unsupported content part type: {kind!r}")


def _message_text(message: Message) -> str:
    chunks: list[str] = []
    content = message["content"]
    if isinstance(content, str):
        chunks.append(content)
    else:
        for part in content:
            if part["type"] == "text":
                chunks.append(part["text"])
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _media(messages: Sequence[Message]) -> AudioPart | CodecCodesPart | None:
    found: AudioPart | CodecCodesPart | None = None
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            continue
        for part in content:
            if part["type"] in {"audio", "codec_codes"}:
                if found is not None:
                    raise ValueError(
                        "chat request allows at most one audio or codec_codes part."
                    )
                found = cast(Union[AudioPart, CodecCodesPart], part)
    return found


__all__ = [
    "AudioPart",
    "ChatChoice",
    "ChatCompletion",
    "ChatMessage",
    "ChatRequest",
    "ChatTraceStep",
    "CodecCodesPart",
    "ContentPart",
    "Message",
    "TextPart",
    "completion_from_result",
    "create",
    "materialize_codes",
    "to_request",
]
