from __future__ import annotations

from .dummy import DummyAudioTokenizer, semantic_ids_from_audio_tokens

NativeAudioTokenizer = DummyAudioTokenizer

__all__ = [
    "DummyAudioTokenizer",
    "NativeAudioTokenizer",
    "semantic_ids_from_audio_tokens",
]
