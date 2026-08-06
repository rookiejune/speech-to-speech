from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .contract import AudioTokenizer
from .bicodec import BiCodecAudioTokenizer, SemanticGlobalAudioTokenizer
from .bpe import TorchCodecBPE
from .flattened import FlattenedAudioTokenizer
from .native import NativeAudioTokenizer

__all__ = [
    "BiCodecAudioTokenizer",
    "SemanticGlobalAudioTokenizer",
    "AudioTokenizer",
    "FlattenedAudioTokenizer",
    "NativeAudioTokenizer",
    "TorchCodecBPE",
    "semantic_codes_from_audio_tokens",
]


def semantic_codes_from_audio_tokens(
    audio_tokenizer: AudioTokenizer,
    audio_token_ids: Sequence[int] | Tensor,
) -> Tensor:
    """Decode one BPE audio sequence to ``[frames, semantic_codebooks]``."""
    decoded = audio_tokenizer.decode(audio_token_ids)
    device = audio_token_ids.device if isinstance(audio_token_ids, Tensor) else None
    if isinstance(decoded, Tensor):
        if decoded.dim() != 2:
            raise ValueError(
                "decoded semantic codes must have shape [frames, codebooks]."
            )
        return decoded.to(device=device, dtype=torch.long)

    frames: list[Tensor] = []
    for frame in decoded:
        values = (
            frame.reshape(-1).tolist() if isinstance(frame, Tensor) else list(frame)
        )
        if not values:
            raise ValueError(
                "audio tokenizer decoded a token to no semantic codebooks."
            )
        frames.append(torch.tensor(values, device=device, dtype=torch.long))
    if not frames:
        raise ValueError("audio tokenizer decoded audio tokens to no frames.")
    try:
        return torch.stack(frames)
    except RuntimeError as error:
        raise ValueError(
            "decoded semantic frames must have the same codebook count."
        ) from error
