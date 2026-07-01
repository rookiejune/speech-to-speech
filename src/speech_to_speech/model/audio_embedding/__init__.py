"""Audio token embedding implementations hidden behind token-space config."""

from __future__ import annotations

from .builder import audio_embedding
from .lookup import lookup_audio_embedding
from .semantic import (
    BPEAudioEmbedding,
    SemanticEmbeddingConfig,
    semantic_audio_embedding,
)

__all__ = [
    "BPEAudioEmbedding",
    "SemanticEmbeddingConfig",
    "audio_embedding",
    "lookup_audio_embedding",
    "semantic_audio_embedding",
]
