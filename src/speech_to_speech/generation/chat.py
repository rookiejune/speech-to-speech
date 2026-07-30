"""OpenAI-style messages adapter: messages -> HF chat template -> tensor Request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, Union, cast

import torch
from anydataset.types import AudioView, Modality
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor
from typing_extensions import NotRequired

from .._tensor import is_signed_integer_dtype
from ..audio_route import (
    BICODEC_GENERATE_GLOBAL,
    BICODEC_REUSE_PROMPT_GLOBAL,
    PromptSource,
)
from ..prediction import PredictionModality
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.protocol import GenerationRuntime
from ..runtime.types import frame_codec, structured_codec, supports_structured
from ..task import Task
from .bicodec import (
    prepare_bicodec_global_tts_request,
    prepare_bicodec_tts_request,
)
from .protocol import TokenGenerator
from .service import generate_responses
from .types import AudioOutput, Request, Result

_PLACEHOLDER = "$$$PLACEHOLDER$$$"


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
    codes: SemanticAcousticCodes | Tensor


ContentPart = Union[TextPart, AudioPart, CodecCodesPart]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class ChatRequest(TypedDict):
    messages: list[Message]
    task: Task
    language: NotRequired[str]


class ChatMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    audio: AudioOutput | None


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
    language = _language(request)
    messages = _messages(request)
    text = _text(messages)
    media = _media(messages)
    codes = None if media is None else materialize_codes(media, runtime)
    return _build_request(
        text,
        codes,
        task=task,
        language=language,
        runtime=runtime,
    )


def materialize_codes(
    part: AudioPart | CodecCodesPart,
    runtime: GenerationRuntime,
) -> SemanticAcousticCodes | Tensor:
    """Turn an audio or codec_codes part into runtime codes."""
    if part["type"] == "codec_codes":
        return _validated_codes(part, runtime)
    if part["type"] != "audio":
        raise TypeError(f"unsupported media part type: {part.get('type')!r}")
    return _encode_audio(part, runtime)


def completion_from_result(
    result: Result,
    request: ChatRequest,
    runtime: GenerationRuntime,
) -> ChatCompletion:
    task = _task(request)
    content: str | None = None
    if task.prediction_modality is PredictionModality.TEXT:
        content = _decode_text(runtime, result["response_ids"])
    message = ChatMessage(
        role="assistant",
        content=content,
        audio=result["audio"],
    )
    return ChatCompletion(choices=[ChatChoice(index=0, message=message)])


def _build_request(
    text: str,
    codes: SemanticAcousticCodes | Tensor | None,
    *,
    task: Task,
    language: str,
    runtime: GenerationRuntime,
) -> Request:
    if task.prediction_modality is PredictionModality.TEXT:
        if codes is not None:
            raise ValueError("text prediction chat requests cannot include audio media.")
        return Request(
            prompt_ids=_instruction_prompt_ids(text, task, language, runtime),
            task=task,
            audio_input_positions=None,
            audio_context=None,
        )

    tokenizer = runtime.audio_tokenizer
    route = runtime.audio_route
    if isinstance(tokenizer, BiCodecAudioTokenizer):
        if route == BICODEC_REUSE_PROMPT_GLOBAL:
            if codes is None:
                raise ValueError(
                    "reference BiCodec chat requests require audio or codec_codes."
                )
            if not isinstance(codes, SemanticAcousticCodes):
                raise TypeError(
                    "BiCodec reference chat requests require SemanticAcousticCodes."
                )
            return prepare_bicodec_tts_request(
                text,
                codes,
                runtime,
                language=language,
                task=task,
            )
        if route == BICODEC_GENERATE_GLOBAL:
            if codes is not None:
                raise ValueError(
                    "global BiCodec chat requests do not accept prompt audio or codes."
                )
            return prepare_bicodec_global_tts_request(
                text,
                runtime,
                language=language,
                task=task,
            )

    if codes is not None and _prompt_needs_context(route):
        if not isinstance(codes, SemanticAcousticCodes):
            raise TypeError(
                "prompt audio context requires SemanticAcousticCodes for this route."
            )
        prompt_ids = _instruction_prompt_ids(text, task, language, runtime)
        prompt_ids = torch.cat(
            (prompt_ids, prompt_ids.new_tensor([runtime.boa_token_id]))
        )
        return Request(
            prompt_ids=prompt_ids,
            task=task,
            audio_input_positions=None,
            audio_context=codes,
        )

    if codes is not None:
        raise ValueError(
            "chat audio/codec_codes are not supported for the current audio route."
        )
    prompt_ids = _instruction_prompt_ids(text, task, language, runtime)
    if task.prediction_modality is PredictionModality.AUDIO:
        prompt_ids = torch.cat(
            (prompt_ids, prompt_ids.new_tensor([runtime.boa_token_id]))
        )
    return Request(
        prompt_ids=prompt_ids,
        task=task,
        audio_input_positions=None,
        audio_context=None,
    )


def _prompt_needs_context(route: object) -> bool:
    if route is None:
        return False
    prompt = getattr(route, "prompt", None)
    if prompt is None:
        return False
    source = getattr(prompt, "source", None)
    streams = getattr(prompt, "streams", ())
    return source is PromptSource.REFERENCE and bool(streams)


def _instruction_prompt_ids(
    text: str,
    task: Task,
    language: str,
    runtime: GenerationRuntime,
) -> Tensor:
    template = task.templates[0]
    if "{source}" not in template:
        raise ValueError(
            f"{task.value} chat template must include a {{source}} placeholder."
        )
    kwargs: dict[str, str] = {"source": _PLACEHOLDER}
    if "{language}" in template:
        kwargs["language"] = language
    instruction = template.format(**kwargs)
    rendered = runtime.text_tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("text tokenizer chat template must return a string.")
    parts = rendered.split(_PLACEHOLDER)
    if len(parts) != 2:
        raise ValueError("source placeholder must occur exactly once in chat template.")
    local_ids = torch.cat(
        (
            _token_ids(parts[0], runtime),
            _token_ids(text, runtime),
            _token_ids(parts[1], runtime),
        )
    )
    if local_ids.numel() == 0:
        raise ValueError("chat text prompt must contain at least one token.")
    if local_ids.dim() != 1:
        raise ValueError("chat text prompt token ids must be one-dimensional.")
    return runtime.layout.to_global(Modality.TEXT.value, local_ids)


def _token_ids(text: str, runtime: GenerationRuntime) -> Tensor:
    values = torch.as_tensor(
        runtime.text_tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
    )
    if values.dim() != 1:
        raise ValueError("text tokenizer must return a 1D token sequence.")
    return values


def _decode_text(runtime: GenerationRuntime, token_ids: Tensor) -> str:
    if token_ids.dim() != 1:
        raise ValueError("response token ids must be one-dimensional.")
    if token_ids.numel():
        local_ids = runtime.layout.to_local(token_ids).detach().cpu().tolist()
    else:
        local_ids = []
    return runtime.text_tokenizer.decode(local_ids, skip_special_tokens=True)


def _encode_audio(part: AudioPart, runtime: GenerationRuntime) -> SemanticAcousticCodes | Tensor:
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
    with torch.autocast(device_type=batched.device.type, enabled=False):
        if runtime.audio_view is AudioView.BICODEC:
            if not supports_structured(runtime.codec):
                raise TypeError("BiCodec audio parts require a structured codec.")
            encoded = structured_codec(runtime.codec).tokenize(batched, sample_rate)
            if not isinstance(encoded, SemanticAcousticCodes):
                raise TypeError(
                    "structured codec tokenize must return SemanticAcousticCodes."
                )
            if encoded.semantic.size(0) != 1 or encoded.acoustic.size(0) != 1:
                raise ValueError("audio encode expects one item.")
            return SemanticAcousticCodes(
                semantic=encoded.semantic[0].detach(),
                acoustic=encoded.acoustic[0].detach(),
            )
        codes = frame_codec(runtime.codec).encode(batched, sample_rate)
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
) -> SemanticAcousticCodes | Tensor:
    codec = part["codec"]
    if not isinstance(codec, str) or not codec.strip():
        raise TypeError("codec_codes codec must be a non-empty string.")
    if codec != runtime.codec_name:
        raise ValueError(
            f"codec_codes codec {codec!r} does not match runtime codec "
            f"{runtime.codec_name!r}."
        )
    codes = part["codes"]
    if isinstance(codes, SemanticAcousticCodes):
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
        "codec_codes codes must be SemanticAcousticCodes or a frame-code Tensor."
    )


def _validate_structured_codes(value: SemanticAcousticCodes) -> None:
    if value.semantic.dim() != 2 or value.semantic.size(1) != 1:
        raise ValueError("structured semantic codes must have shape [units, 1].")
    if value.acoustic.dim() != 2:
        raise ValueError("structured acoustic codes must have shape [slots, codebooks].")
    if value.semantic.numel() == 0 or value.acoustic.numel() == 0:
        raise ValueError("structured codec_codes must not be empty.")
    if value.semantic.device != value.acoustic.device:
        raise ValueError("structured semantic and acoustic codes must share a device.")
    for name, codes in (("semantic", value.semantic), ("acoustic", value.acoustic)):
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError(f"structured {name} codes must use signed integers.")


def _task(request: ChatRequest) -> Task:
    task = request["task"]
    if not isinstance(task, Task):
        raise TypeError("chat request task must be a Task.")
    return task


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


def _text(messages: Sequence[Message]) -> str:
    chunks: list[str] = []
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            chunks.append(content)
            continue
        for part in content:
            if part["type"] == "text":
                chunks.append(part["text"])
    text = "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    if not text:
        raise ValueError("chat request requires user text content.")
    return text


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
    "CodecCodesPart",
    "ContentPart",
    "Message",
    "TextPart",
    "completion_from_result",
    "create",
    "materialize_codes",
    "to_request",
]
