from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
from anydataset import types
from torch import Tensor

from .builder import token_ids
from .contract import DataRuntime, TextRuntime
from .sample import (
    Language,
    AudioContextSample,
    RawSpeech,
    Speech,
    SpeechPair,
    SpeechTaskSample,
    Text,
    TextPair,
    seconds,
)
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer, FlattenedAudioTokenizer
from ..runtime.codec import has_codec_loader
from ..audio import AudioCodes
from ..task import SourceLayout, Task, resolve_response


def from_frames(
    value: object,
    *,
    frames: int,
    frame_rate: float,
) -> float:
    duration = seconds(value, name="AudioMeta.DURATION")
    if duration is not None:
        return duration
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise TypeError("audio frame count must be an integer.")
    if frames < 0:
        raise ValueError("audio frame count must be non-negative.")
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)):
        raise TypeError("codec frame_rate must be a number.")
    rate = float(frame_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("codec frame_rate must be finite and positive.")
    return frames / rate


def from_samples(
    value: object,
    *,
    samples: int,
    sample_rate: int,
) -> float:
    duration = seconds(value, name="AudioMeta.DURATION")
    if duration is not None:
        return duration
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("audio sample count must be an integer.")
    if samples < 0:
        raise ValueError("audio sample count must be non-negative.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("audio sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be positive.")
    return samples / sample_rate


def parse_sample(sample: types.Sample, runtime: DataRuntime) -> SpeechPair:
    return SpeechPair(
        _parse_role(sample, types.Role.SOURCE, runtime, input_audio=True),
        _parse_role(sample, types.Role.TARGET, runtime, input_audio=False),
    )


def parse_task_sample(
    sample: types.Sample,
    task: Task,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool = False,
    trace: str | None = None,
) -> SpeechTaskSample:
    response = resolve_response(task, trace=trace)
    if task.source_layout is SourceLayout.TEXT_AUDIO and runtime.input_audio_decoupled:
        raise ValueError(
            "MASKED_AR does not support different input and output audio tokenizers."
        )
    source = None
    if task.source_layout is SourceLayout.TEXT_AUDIO:
        source = _parse_task_item(
            sample,
            types.Role.TARGET,
            types.Modality.AUDIO,
            runtime,
            encode_missing_codes=encode_missing_codes,
            input_audio=True,
        )
    elif task.source_modality is not None:
        source_role = types.Role.SOURCE if task.uses_source_role else types.Role.TARGET
        source = _parse_task_item(
            sample,
            source_role,
            task.source_modality,
            runtime,
            encode_missing_codes=encode_missing_codes,
            input_audio=task.source_modality is types.Modality.AUDIO,
        )
    target_modality = (
        types.Modality.AUDIO
        if response.prediction.supervises_audio
        else types.Modality.TEXT
    )
    if task.source_layout is SourceLayout.TEXT_AUDIO:
        if not isinstance(source, (Speech, RawSpeech)):
            raise TypeError("MASKED_AR source must be Speech or RawSpeech.")
        target: Speech | Text | RawSpeech = source
    else:
        target = _parse_task_item(
            sample,
            types.Role.TARGET,
            target_modality,
            runtime,
            encode_missing_codes=encode_missing_codes,
            input_audio=False,
        )
    audio_context = _parse_audio_context(
        sample,
        runtime,
        encode_missing_codes=encode_missing_codes,
    )
    return SpeechTaskSample(
        source=source,
        target=target,
        task=task,
        trace=response.name,
        audio_context=audio_context,
    )


def _parse_audio_context(
    sample: types.Sample,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool,
) -> Speech | RawSpeech | None:
    """Resolve an explicitly supplied reference audio sample."""
    if not isinstance(sample, AudioContextSample):
        return None
    context = _parse_task_item(
        sample.audio_context,
        types.Role.DEFAULT,
        types.Modality.AUDIO,
        runtime,
        encode_missing_codes=encode_missing_codes,
        input_audio=False,
    )
    if isinstance(context, Text):
        raise AssertionError("audio context parser returned text.")
    return context


def parse_text_sample(sample: types.Sample, runtime: TextRuntime) -> TextPair:
    return TextPair(
        _parse_text_role(sample, types.Role.SOURCE, runtime),
        _parse_text_role(sample, types.Role.TARGET, runtime),
    )


def _parse_audio_item(
    audio_item: types.AudioItem,
    view: types.AudioView,
) -> AudioCodes:
    codes = audio_item.views[view]
    return _split_audio_codes(codes, view)


def parse_audio_codes(
    codes: object,
    runtime: DataRuntime,
    *,
    input_audio: bool = False,
) -> AudioCodes:
    view = runtime.input_audio_view if input_audio else runtime.audio_view
    parsed = _split_audio_codes(codes, view)
    tokenizer = runtime.input_audio_tokenizer if input_audio else runtime.audio_tokenizer
    if (
        isinstance(tokenizer, FlattenedAudioTokenizer)
        and view is not types.AudioView.BICODEC
    ):
        if not isinstance(codes, Tensor):
            raise ValueError("frame codec views must contain a Tensor.")
        return AudioCodes(semantic_codes=_frame_codes(codes))
    return parsed


def speech_from_codes(
    codes: object,
    *,
    text_token_ids: Tensor,
    language: Language,
    duration_seconds: float | None,
    runtime: DataRuntime,
    input_audio: bool = False,
) -> Speech:
    parsed = parse_audio_codes(codes, runtime, input_audio=input_audio)
    semantic_codes = parsed.semantic_codes
    if semantic_codes is None:
        raise ValueError("prepared speech codes require semantic_codes.")
    tokenizer = runtime.input_audio_tokenizer if input_audio else runtime.audio_tokenizer
    if isinstance(tokenizer, BiCodecAudioTokenizer):
        if parsed.global_codes is None:
            raise ValueError("BiCodec full sequence requires global codes.")
        audio_token_ids = tokenizer.encode_full(parsed)
    else:
        audio_token_ids = _as_tensor(tokenizer.encode(semantic_codes))
    audio_token_spans = _as_tensor(
        tokenizer.frame_spans(audio_token_ids)
    ).to(dtype=torch.long)
    return Speech(
        semantic_codes=semantic_codes,
        acoustic_codes=parsed.acoustic_codes,
        text_token_ids=text_token_ids,
        audio_token_ids=audio_token_ids,
        audio_token_spans=audio_token_spans,
        language=language,
        duration_seconds=duration_seconds,
        global_codes=parsed.global_codes,
    )


def _split_audio_codes(
    codes: object,
    view: types.AudioView,
) -> AudioCodes:
    if view is types.AudioView.BICODEC:
        if isinstance(codes, AudioCodes):
            semantic = codes.semantic_codes
            global_codes = codes.global_codes
            if (
                semantic is None
                or global_codes is None
                or codes.acoustic_codes is not None
            ):
                raise ValueError(
                    "BiCodec prepared AudioCodes require semantic_codes and "
                    "global_codes only."
                )
            return AudioCodes(
                semantic_codes=_frame_codes(semantic),
                global_codes=_frame_codes(global_codes),
            )
        if not isinstance(codes, Mapping):
            raise ValueError(
                "BiCodec view must be AudioCodes or an anydataset structured mapping."
            )
        semantic = codes.get("semantic")
        global_codes = codes.get("global")
        if not isinstance(semantic, Tensor) or not isinstance(global_codes, Tensor):
            raise ValueError(
                "anydataset BiCodec views must contain Tensor semantic/global fields."
            )
        return AudioCodes(
            semantic_codes=_frame_codes(semantic),
            global_codes=_frame_codes(global_codes),
        )
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError("codec view must have shape [frame, codebook].")
    if view is types.AudioView.LONGCAT:
        if codes.size(1) < 2:
            raise ValueError(
                "LongCat view must contain semantic and acoustic codebooks."
            )
        return AudioCodes(
            semantic_codes=_frame_codes(codes[:, :1]),
            acoustic_codes=_frame_codes(codes[:, 1:]),
        )
    if view in {
        types.AudioView.GLM4,
        types.AudioView.STABLE,
        types.AudioView.UNICODEC,
    }:
        return AudioCodes(semantic_codes=_frame_codes(codes))
    raise ValueError(f"unsupported codec audio view: {view.value}")


def _parse_role(
    sample: types.Sample,
    role: types.Role,
    runtime: DataRuntime,
    *,
    input_audio: bool,
) -> Speech:
    audio_item = _audio_item(sample, role)
    text_item = _text_item(sample, role)
    return _speech(audio_item, text_item, runtime, input_audio=input_audio)


def _speech(
    audio_item: types.AudioItem,
    text_item: types.TextItem,
    runtime: DataRuntime,
    *,
    input_audio: bool,
) -> Speech:
    text = text_item.views[types.TextView.TEXT]
    view = runtime.input_audio_view if input_audio else runtime.audio_view
    codes = audio_item.views[view]
    semantic_codes = _split_audio_codes(codes, view).semantic_codes
    if semantic_codes is None:
        raise ValueError("prepared speech codes require semantic_codes.")
    return speech_from_codes(
        codes,
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        language=Language(text_item.meta[types.TextMeta.LANG]),
        duration_seconds=from_frames(
            audio_item.meta.get(types.AudioMeta.DURATION),
            frames=semantic_codes.size(0),
            frame_rate=(
                runtime.input_codec_frame_rate
                if input_audio
                else runtime.codec_frame_rate
            ),
        ),
        runtime=runtime,
        input_audio=input_audio,
    )


def raw_speech(
    audio_item: types.AudioItem,
    text_item: types.TextItem,
    runtime: DataRuntime,
) -> RawSpeech:
    value = audio_item.views.get(types.AudioView.WAVEFORM)
    if value is None:
        raise ValueError(
            "waveform fallback requires AudioView.WAVEFORM when codec codes are missing."
        )
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
    waveform, sample_rate = value
    if not isinstance(waveform, Tensor):
        raise TypeError("AudioView.WAVEFORM waveform must be a Tensor.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("AudioView.WAVEFORM sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("AudioView.WAVEFORM sample_rate must be positive.")
    text = text_item.views[types.TextView.TEXT]
    return RawSpeech(
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        waveform=waveform,
        sample_rate=sample_rate,
        language=Language(text_item.meta[types.TextMeta.LANG]),
        duration_seconds=from_samples(
            audio_item.meta.get(types.AudioMeta.DURATION),
            samples=waveform.size(-1),
            sample_rate=sample_rate,
        ),
    )


def _parse_text_role(
    sample: types.Sample,
    role: types.Role,
    runtime: TextRuntime,
) -> Text:
    text_item = cast(types.TextItem, sample[(role, types.Modality.TEXT)])
    text = text_item.views[types.TextView.TEXT]
    return Text(
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        language=Language(text_item.meta[types.TextMeta.LANG]),
    )


def _parse_task_item(
    sample: types.Sample,
    role: types.Role,
    modality: types.Modality,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool,
    input_audio: bool,
) -> Speech | Text | RawSpeech:
    if modality is types.Modality.TEXT:
        return _parse_text_role(sample, role, runtime)
    if modality is not types.Modality.AUDIO:
        raise ValueError(f"unsupported speech task modality: {modality.value}")
    audio_item = _audio_item(sample, role)
    text_item = _text_item(sample, role)
    view = runtime.input_audio_view if input_audio else runtime.audio_view
    if view in audio_item.views:
        return _speech(
            audio_item,
            text_item,
            runtime,
            input_audio=input_audio,
        )
    if (
        input_audio
        and runtime.input_audio_decoupled
        and runtime.input_codec_name != runtime.codec_name
        and encode_missing_codes
        and not has_codec_loader(runtime.input_codec_name)
    ):
        raise ValueError(
            f"input audio tokenizer {runtime.input_codec_name!r} has no runtime "
            "codec backend for waveform fallback; "
            f"materialize the {view.value!r} input view before training."
        )
    if not encode_missing_codes:
        raise ValueError(
            f"{role.value} audio sample is missing {view.value!r} codec "
            "codes; materialize codec views before training or enable explicit "
            "waveform fallback."
        )
    return raw_speech(audio_item, text_item, runtime)


def _audio_item(sample: types.Sample, role: types.Role) -> types.AudioItem:
    try:
        item = sample[(role, types.Modality.AUDIO)]
    except KeyError as error:
        raise ValueError(f"sample is missing {role.value}/audio.") from error
    if not isinstance(item, types.AudioItem):
        raise TypeError(f"sample {role.value}/audio must be an AudioItem.")
    return item


def _text_item(sample: types.Sample, role: types.Role) -> types.TextItem:
    try:
        item = sample[(role, types.Modality.TEXT)]
    except KeyError as error:
        raise ValueError(f"sample is missing {role.value}/text.") from error
    if not isinstance(item, types.TextItem):
        raise TypeError(f"sample {role.value}/text must be a TextItem.")
    return item

def _frame_codes(codes: Tensor) -> Tensor:
    if codes.dim() == 1:
        return codes.unsqueeze(-1)
    if codes.dim() != 2:
        raise ValueError("audio codes must have shape [frames, codebooks].")
    return codes


def _as_tensor(value: Tensor | list[int]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.long)
