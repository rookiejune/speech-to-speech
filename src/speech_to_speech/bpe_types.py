"""LongCat BPE artifact metadata contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BPEArtifactMeta:
    codec_name: str
    requested_vocab_size: int
    actual_vocab_size: int
    min_frequency: int = 0
    max_token_length: int | None = None
    codebook_sizes: tuple[int, ...] = (8192,)
    datasets: tuple[Mapping[str, object], ...] = ()
