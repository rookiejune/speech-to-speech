from __future__ import annotations

import math
from typing import cast

import torch
from anydataset import types
from torch import Tensor

from ._tokenization import token_ids
from .protocol import DataRuntime, TextRuntime
from .types import Language, Speech, SpeechPair, Text, TextPair
from ..runtime import AudioRepresentation


def parse_sample(sample: types.Sample, runtime: DataRuntime) -> SpeechPair:
    return SpeechPair(
        _parse_role(sample, types.Role.SOURCE, runtime),
        _parse_role(sample, types.Role.TARGET, runtime),
    )


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
    codes: Tensor,
    runtime: DataRuntime,
) -> tuple[Tensor, Tensor | None]:
    semantic_codes, acoustic_codes = _split_audio_codes(codes, runtime.audio_view)
    if runtime.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE:
        semantic_codes, acoustic_codes = _frame_codes(codes), None
    return _frame_codes(semantic_codes), (
        None if acoustic_codes is None else _frame_codes(acoustic_codes)
    )


def speech_from_codes(
    codes: Tensor,
    *,
    text_token_ids: Tensor,
    language: Language,
    duration_seconds: float | None,
    runtime: DataRuntime,
) -> Speech:
    semantic_codes, acoustic_codes = parse_audio_codes(codes, runtime)
    audio_token_ids = _as_tensor(runtime.audio_tokenizer.encode(semantic_codes))
    audio_token_spans = _as_tensor(
        runtime.audio_tokenizer.frame_spans(audio_token_ids)
    ).to(dtype=torch.long)
    return Speech(
        semantic_codes=semantic_codes,
        acoustic_codes=acoustic_codes,
        text_token_ids=text_token_ids,
        audio_token_ids=audio_token_ids,
        audio_token_spans=audio_token_spans,
        language=language,
        duration_seconds=duration_seconds,
    )


def _split_audio_codes(
    codes: Tensor,
    view: types.AudioView,
) -> tuple[Tensor, Tensor | None]:
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError("codec view must have shape [frame, codebook].")
    if view is types.AudioView.LONGCAT:
        if codes.size(1) < 2:
            raise ValueError(
                "LongCat view must contain semantic and acoustic codebooks."
            )
        return codes[:, :1], codes[:, 1:]
    if view is types.AudioView.UNICODEC:
        return codes, None
    if view is types.AudioView.BICODEC:
        return codes, None
    raise ValueError(f"unsupported codec audio view: {view.value}")


def _parse_role(
    sample: types.Sample,
    role: types.Role,
    runtime: DataRuntime,
) -> Speech:
    audio_item = cast(types.AudioItem, sample[(role, types.Modality.AUDIO)])
    text_item = cast(types.TextItem, sample[(role, types.Modality.TEXT)])
    text = text_item.views[types.TextView.TEXT]
    codes = cast(Tensor, audio_item.views[runtime.audio_view])
    return speech_from_codes(
        codes,
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        language=Language(text_item.meta[types.TextMeta.LANG]),
        duration_seconds=_duration_seconds(
            audio_item,
            frames=_frame_codes(codes).size(0),
            frame_rate=runtime.codec_frame_rate,
        ),
        runtime=runtime,
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


def _as_tensor(value: Tensor | list[int]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.long)
