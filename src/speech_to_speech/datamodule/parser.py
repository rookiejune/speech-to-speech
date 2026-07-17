from __future__ import annotations

from typing import cast

import torch
from anydataset import types
from torch import Tensor

from ._tokenization import token_ids
from .protocol import DataRuntime
from .types import Language, Speech, SpeechPair


def parse_sample(sample: types.Sample, runtime: DataRuntime) -> SpeechPair:
    return SpeechPair(
        _parse_role(sample, types.Role.SOURCE, runtime),
        _parse_role(sample, types.Role.TARGET, runtime),
    )


def _parse_audio_item(
    audio_item: types.AudioItem,
    view: types.AudioView,
) -> tuple[Tensor, Tensor | None]:
    codes = audio_item.views[view]
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
    raise ValueError(f"unsupported codec audio view: {view.value}")


def _parse_role(
    sample: types.Sample,
    role: types.Role,
    runtime: DataRuntime,
) -> Speech:
    audio_item = cast(types.AudioItem, sample[(role, types.Modality.AUDIO)])
    semantic_codes, acoustic_codes = _parse_audio_item(
        audio_item,
        runtime.audio_view,
    )
    semantic_codes = _frame_codes(semantic_codes)
    acoustic_codes = (
        None if acoustic_codes is None else _frame_codes(acoustic_codes)
    )
    audio_token_ids = _as_tensor(runtime.audio_tokenizer.encode(semantic_codes))
    audio_token_spans = _as_tensor(
        runtime.audio_tokenizer.frame_spans(audio_token_ids)
    ).to(dtype=torch.long)

    text_item = cast(types.TextItem, sample[(role, types.Modality.TEXT)])
    text = text_item.views[types.TextView.TEXT]
    return Speech(
        semantic_codes=semantic_codes,
        acoustic_codes=acoustic_codes,
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        audio_token_ids=audio_token_ids,
        audio_token_spans=audio_token_spans,
        language=Language(text_item.meta[types.TextMeta.LANG]),
    )


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
