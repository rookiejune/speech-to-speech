from __future__ import annotations

import torch
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..runtime.audio_tokenizer import semantic_codes_from_audio_tokens
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
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    return _decode_audio(
        local_ids,
        acoustic_features,
        codec=codec,
        audio_tokenizer=audio_tokenizer,
    )


def _decode_audio(
    local_ids: Tensor,
    acoustic_features: Tensor,
    *,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
) -> Tensor:
    rows = [semantic_codes_from_audio_tokens(audio_tokenizer, row) for row in local_ids]
    if not rows or len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError(
            "audio token rows must expand to the same frame and codebook shape."
        )
    semantic_codes = torch.stack(rows)
    if semantic_codes.shape[:2] != acoustic_features.shape[:2]:
        raise ValueError(
            "semantic codes and acoustic features must align on [batch, frame]."
        )
    return codec.decode_features(semantic_codes, acoustic_features)


def decode_generated_semantic(
    audio_token_ids: Tensor,
    *,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode semantic-only codec tokens directly into waveforms."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    semantic_codes = torch.stack(
        [semantic_codes_from_audio_tokens(audio_tokenizer, row) for row in local_ids]
    )
    return codec.decode(semantic_codes)


def decode_generated_codes(
    audio_token_ids: Tensor,
    acoustic_codes: Tensor,
    *,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode generated audio tokens and acoustic codes into waveforms."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    return _decode_audio(
        local_ids,
        codec.acoustic_codes_to_features(acoustic_codes),
        codec=codec,
        audio_tokenizer=audio_tokenizer,
    )


def _local_ids(audio_token_ids: Tensor, audio_token_range: tuple[int, int]) -> Tensor:
    if not isinstance(audio_token_ids, Tensor):
        raise TypeError("audio token ids must be a Tensor.")
    if not is_signed_integer_dtype(audio_token_ids.dtype):
        raise TypeError(
            "audio token ids must contain integer ids using a signed dtype."
        )
    if audio_token_ids.dim() != 2:
        raise ValueError("audio token ids must have shape [batch, tokens].")
    if audio_token_ids.size(0) < 1 or audio_token_ids.size(1) < 1:
        raise ValueError("audio token ids must contain at least one token row.")

    global_start, global_end = audio_token_range
    if bool((audio_token_ids < global_start).any()) or bool(
        (audio_token_ids >= global_end).any()
    ):
        raise ValueError("audio token ids must be codec-decodable global audio ids.")
    return audio_token_ids.to(dtype=torch.long) - global_start
