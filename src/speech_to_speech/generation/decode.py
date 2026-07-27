from __future__ import annotations

import torch
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer, semantic_codes_from_audio_tokens
from ..runtime.types import AcousticCodec, AudioTokenizer, Codec, SemanticCodec, StructuredCodec


def decode_generated_audio(
    audio_token_ids: Tensor,
    acoustic_features: Tensor,
    *,
    codec: AcousticCodec,
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
    codec: AcousticCodec,
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
    codec: SemanticCodec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode semantic-only codec tokens directly into waveforms."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    semantic_codes = torch.stack(
        [semantic_codes_from_audio_tokens(audio_tokenizer, row) for row in local_ids]
    )
    return codec.decode(semantic_codes)


def decode_generated_frame_codes(
    audio_token_ids: Tensor,
    *,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode generated full frame-code tokens with a ``FrameCodec`` backend."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    rows = [_decoded_frames(audio_tokenizer, row) for row in local_ids]
    if not rows or len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError(
            "full codec token rows must expand to the same frame and codebook shape."
        )
    return codec.decode(torch.stack(rows))


def decode_generated_bicodec_full(
    audio_token_ids: Tensor,
    *,
    codec: StructuredCodec,
    audio_tokenizer: BiCodecAudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode a structured BiCodec full sequence without collapsing its axes."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    rows = [audio_tokenizer.decode_full(row) for row in local_ids]
    if not rows or len({tuple(row.semantic.shape) for row in rows}) != 1:
        raise ValueError("BiCodec semantic token rows must have the same shape.")
    if len({tuple(row.acoustic.shape) for row in rows}) != 1:
        raise ValueError("BiCodec acoustic token rows must have the same shape.")
    codes = SemanticAcousticCodes(
        semantic=torch.stack([row.semantic for row in rows]),
        acoustic=torch.stack([row.acoustic for row in rows]),
    )
    return codec.detokenize(codes)


def decode_generated_codes(
    audio_token_ids: Tensor,
    acoustic_codes: Tensor,
    *,
    codec: AcousticCodec,
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


def _decoded_frames(tokenizer: AudioTokenizer, token_ids: Tensor) -> Tensor:
    decoded = tokenizer.decode(token_ids)
    if isinstance(decoded, Tensor):
        frames = decoded
    else:
        frames = torch.as_tensor(decoded, device=token_ids.device)
    if frames.dim() != 2:
        raise ValueError("full codec tokens must decode to [frames, codebooks].")
    if not is_signed_integer_dtype(frames.dtype):
        raise TypeError("decoded full codec codes must use a signed integer dtype.")
    return frames.to(device=token_ids.device, dtype=torch.long)
