from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
from anydataset import types
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from ._tokenization import token_ids
from .protocol import DataRuntime, TextRuntime
from .types import (
    Language,
    RawSpeech,
    Speech,
    SpeechPair,
    SpeechTaskSample,
    Text,
    TextPair,
)
from ..runtime import AudioRepresentation
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..task import Task


def parse_sample(sample: types.Sample, runtime: DataRuntime) -> SpeechPair:
    return SpeechPair(
        _parse_role(sample, types.Role.SOURCE, runtime),
        _parse_role(sample, types.Role.TARGET, runtime),
    )


def parse_task_sample(
    sample: types.Sample,
    task: Task,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool = False,
) -> SpeechTaskSample:
    source = None
    if task.source_modality is not None:
        source_role = types.Role.SOURCE if task.uses_source_role else types.Role.TARGET
        source = _parse_task_item(
            sample,
            source_role,
            task.source_modality,
            runtime,
            encode_missing_codes=encode_missing_codes,
        )
    target = _parse_task_item(
        sample,
        types.Role.TARGET,
        task.target_modality,
        runtime,
        encode_missing_codes=encode_missing_codes,
    )
    return SpeechTaskSample(source=source, target=target, task=task)


def parse_text_sample(sample: types.Sample, runtime: TextRuntime) -> TextPair:
    return TextPair(
        _parse_text_role(sample, types.Role.SOURCE, runtime),
        _parse_text_role(sample, types.Role.TARGET, runtime),
    )


def _parse_audio_item(
    audio_item: types.AudioItem,
    view: types.AudioView,
) -> tuple[Tensor, Tensor | None]:
    codes = audio_item.views[view]
    return _split_audio_codes(codes, view)


def parse_audio_codes(
    codes: object,
    runtime: DataRuntime,
) -> tuple[Tensor, Tensor | None]:
    semantic_codes, acoustic_codes = _split_audio_codes(codes, runtime.audio_view)
    if (
        runtime.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE
        and runtime.audio_view is not types.AudioView.BICODEC
    ):
        if not isinstance(codes, Tensor):
            raise ValueError("frame codec views must contain a Tensor.")
        semantic_codes, acoustic_codes = _frame_codes(codes), None
    return _frame_codes(semantic_codes), (
        None if acoustic_codes is None else _frame_codes(acoustic_codes)
    )


def speech_from_codes(
    codes: object,
    *,
    text_token_ids: Tensor,
    language: Language,
    duration_seconds: float | None,
    runtime: DataRuntime,
) -> Speech:
    semantic_codes, acoustic_codes = parse_audio_codes(codes, runtime)
    if (
        runtime.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE
        and runtime.audio_view is types.AudioView.BICODEC
    ):
        tokenizer = runtime.audio_tokenizer
        if not isinstance(tokenizer, BiCodecAudioTokenizer):
            raise TypeError("BiCodec full sequence requires BiCodecAudioTokenizer.")
        audio_token_ids = tokenizer.encode_full(codes)
    else:
        audio_token_ids = _as_tensor(runtime.audio_tokenizer.encode(semantic_codes))
    audio_token_spans = _as_tensor(
        runtime.audio_tokenizer.frame_spans(audio_token_ids)
    ).to(dtype=torch.long)
    return Speech(
        semantic_codes=semantic_codes,
        acoustic_codes=acoustic_codes,
        acoustic_layout=runtime.acoustic_layout,
        acoustic_unit_length=runtime.acoustic_unit_length,
        text_token_ids=text_token_ids,
        audio_token_ids=audio_token_ids,
        audio_token_spans=audio_token_spans,
        language=language,
        duration_seconds=duration_seconds,
    )


def _split_audio_codes(
    codes: object,
    view: types.AudioView,
) -> tuple[Tensor, Tensor | None]:
    if view is types.AudioView.BICODEC:
        if isinstance(codes, SemanticAcousticCodes):
            return _frame_codes(codes.semantic), _frame_codes(codes.acoustic)
        if not isinstance(codes, Mapping):
            raise ValueError("BiCodec view must be a semantic/acoustic mapping.")
        semantic = codes.get("semantic")
        acoustic = codes.get("acoustic")
        if not isinstance(semantic, Tensor) or not isinstance(acoustic, Tensor):
            raise ValueError("BiCodec view must contain Tensor semantic/acoustic fields.")
        return _frame_codes(semantic), _frame_codes(acoustic)
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError("codec view must have shape [frame, codebook].")
    if view is types.AudioView.LONGCAT:
        if codes.size(1) < 2:
            raise ValueError(
                "LongCat view must contain semantic and acoustic codebooks."
            )
        return codes[:, :1], codes[:, 1:]
    if view in {types.AudioView.STABLE, types.AudioView.UNICODEC}:
        return codes, None
    raise ValueError(f"unsupported codec audio view: {view.value}")


def _parse_role(
    sample: types.Sample,
    role: types.Role,
    runtime: DataRuntime,
) -> Speech:
    audio_item = _audio_item(sample, role)
    text_item = _text_item(sample, role)
    return _speech(audio_item, text_item, runtime)


def _speech(
    audio_item: types.AudioItem,
    text_item: types.TextItem,
    runtime: DataRuntime,
) -> Speech:
    text = text_item.views[types.TextView.TEXT]
    codes = audio_item.views[runtime.audio_view]
    return speech_from_codes(
        codes,
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        language=Language(text_item.meta[types.TextMeta.LANG]),
        duration_seconds=_duration_seconds(
            audio_item,
            frames=_frame_codes(_split_audio_codes(codes, runtime.audio_view)[0]).size(0),
            frame_rate=runtime.codec_frame_rate,
        ),
        runtime=runtime,
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
        duration_seconds=_raw_duration_seconds(audio_item, waveform, sample_rate),
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
) -> Speech | Text | RawSpeech:
    if modality is types.Modality.TEXT:
        return _parse_text_role(sample, role, runtime)
    if modality is not types.Modality.AUDIO:
        raise ValueError(f"unsupported speech task modality: {modality.value}")
    audio_item = _audio_item(sample, role)
    text_item = _text_item(sample, role)
    if runtime.audio_view in audio_item.views:
        return _speech(audio_item, text_item, runtime)
    if not encode_missing_codes:
        raise ValueError(
            f"{role.value} audio sample is missing {runtime.audio_view.value!r} codec "
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


def _duration_seconds(
    audio_item: types.AudioItem,
    *,
    frames: int,
    frame_rate: float,
) -> float:
    value = audio_item.meta.get(types.AudioMeta.DURATION)
    if value is None:
        return _frames_to_seconds(frames, frame_rate)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("AudioMeta.DURATION must be a number of seconds.")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("AudioMeta.DURATION must be finite and non-negative.")
    return duration


def _frames_to_seconds(frames: int, frame_rate: float) -> float:
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise TypeError("audio frame count must be an integer.")
    if frames < 0:
        raise ValueError("audio frame count must be non-negative.")
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)):
        raise TypeError("codec frame_rate must be a number.")
    rate = float(frame_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("codec frame_rate must be finite and positive.")
    return float(frames) / rate


def _raw_duration_seconds(
    audio_item: types.AudioItem,
    waveform: Tensor,
    sample_rate: int,
) -> float:
    value = audio_item.meta.get(types.AudioMeta.DURATION)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("AudioMeta.DURATION must be a number of seconds.")
        duration = float(value)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("AudioMeta.DURATION must be finite and non-negative.")
        return duration
    return float(waveform.size(-1)) / float(sample_rate)


def _as_tensor(value: Tensor | list[int]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.long)
