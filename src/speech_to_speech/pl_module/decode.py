from __future__ import annotations

import torch
from torch import Tensor

from ..runtime.audio_tokenizer import semantic_ids_from_audio_tokens
from ..runtime.types import AudioTokenizer, Codec


def decode_generated_audio(
    audio_token_ids: Tensor,
    acoustic_features: Tensor,
    *,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode generated audio tokens and acoustic features into waveforms."""

    local_start, local_end = audio_token_range
    if bool((audio_token_ids < local_start).any()) or bool(
        (audio_token_ids >= local_end).any()
    ):
        raise ValueError("audio token ids must be codec-decodable global audio ids.")
    local_ids = audio_token_ids - local_start
    rows = [
        semantic_ids_from_audio_tokens(audio_tokenizer, row)
        for row in local_ids
    ]
    if not rows or len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError(
            "audio token rows must expand to the same frame and codebook shape."
        )
    semantic_ids = torch.stack(rows)
    if semantic_ids.shape[:2] != acoustic_features.shape[:2]:
        raise ValueError(
            "semantic ids and acoustic features must align on [batch, frame]."
        )
    return codec.decode_features(semantic_ids, acoustic_features)


def decode_generated_codes(
    audio_token_ids: Tensor,
    acoustic_codes: Tensor,
    *,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode generated audio tokens and acoustic codes into waveforms."""
    return decode_generated_audio(
        audio_token_ids,
        codec.acoustic_codes_to_features(acoustic_codes),
        codec=codec,
        audio_tokenizer=audio_tokenizer,
        audio_token_range=audio_token_range,
    )
