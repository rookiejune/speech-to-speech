from __future__ import annotations

import torch
from anytrain.idspace import Layout
from torch import Tensor

from ..runtime.audio_tokenizer import semantic_ids_from_audio_tokens
from ..runtime.types import AudioTokenizer, Codec


def decode_generated_audio(
    audio_token_ids: Tensor,
    acoustic_features: Tensor | None = None,
    *,
    acoustic_codes: Tensor | None = None,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    layout: Layout,
) -> Tensor:
    """Decode generated audio tokens and acoustic output into waveforms."""
    if (acoustic_features is None) == (acoustic_codes is None):
        raise ValueError("provide exactly one of acoustic_features or acoustic_codes.")
    if acoustic_codes is not None:
        acoustic_features = codec.acoustic_codes_to_features(acoustic_codes)
    if acoustic_features is None:
        raise RuntimeError("acoustic features were not created.")

    local_start, local_end = layout.blocks["audio"]
    if bool((audio_token_ids < local_start).any()) or bool(
        (audio_token_ids >= local_end).any()
    ):
        raise ValueError("audio token ids must be global ids from the audio layout block.")
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
