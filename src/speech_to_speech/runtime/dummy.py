from __future__ import annotations

from .audio_tokenizer import NativeAudioTokenizer, semantic_ids_from_audio_tokens

DummyAudioTokenizer = NativeAudioTokenizer

__all__ = [
    "DummyAudioTokenizer",
    "NativeAudioTokenizer",
    "semantic_ids_from_audio_tokens",
]
