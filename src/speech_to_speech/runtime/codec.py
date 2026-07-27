from __future__ import annotations

from typing import Protocol, cast

import torch
from anytrain.codec import load_frame, load_semantic_acoustic
from torch import Tensor

from .types import CodecBackend, StructuredCodec


class UnifiedCodecModel(Protocol):
    frame_rate: float


class UnifiedCodecSource(Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def model(self) -> UnifiedCodecModel: ...

    @property
    def sample_rate(self) -> int: ...

    def codes_to_features(self, codes: Tensor) -> Tensor: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...


class StableCodecSource(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...


class UnifiedCodec:
    """Adapt a unified-token codec with no independent acoustic stream."""

    def __init__(self, codec: UnifiedCodecSource) -> None:
        self.codec = codec
        vocab_size = int(codec.codebook_sizes[0])
        ids = torch.arange(vocab_size, device=codec.device).view(1, vocab_size, 1)
        self._semantic_codebook = codec.codes_to_features(ids)[0].detach()

    @property
    def sample_rate(self) -> int:
        return int(self.codec.sample_rate)

    @property
    def frame_rate(self) -> float:
        return float(self.codec.model.frame_rate)

    @property
    def semantic_feature_dim(self) -> int:
        return int(self._semantic_codebook.size(-1))

    @property
    def semantic_codebook(self) -> Tensor:
        return self._semantic_codebook

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.codec.codebook_sizes)

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        return self.codec.encode(audio, sample_rate)

    def decode(self, codes: Tensor) -> Tensor:
        return self.codec.decode(codes)


class StableCodec:
    """Frame-code adapter for Stable Codec's full-sequence training path."""

    def __init__(self, codec: StableCodecSource) -> None:
        self.codec = codec

    @property
    def name(self) -> str:
        return "stable_codec"

    @property
    def sample_rate(self) -> int:
        return int(self.codec.sample_rate)

    @property
    def frame_rate(self) -> float:
        return float(self.codec.frame_rate)

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.codec.codebook_sizes)

    @property
    def semantic_feature_dim(self) -> int:
        return 1

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        return self.codec.encode(audio, sample_rate)

    def decode(self, codes: Tensor) -> Tensor:
        return self.codec.decode(codes)


def load_codec(name: str, device: str | None) -> CodecBackend:
    if name == "longcat":
        return cast(
            CodecBackend,
            cast(object, load_semantic_acoustic("longcat", device=device)),
        )
    if name == "bicodec":
        return cast(
            StructuredCodec,
            cast(object, load_semantic_acoustic("bicodec", device=device)),
        )
    if name == "unicodec":
        source = cast(
            UnifiedCodecSource,
            cast(object, load_frame("unicodec", device=device)),
        )
        return cast(CodecBackend, UnifiedCodec(source))
    if name == "stable_codec":
        return cast(
            CodecBackend,
            StableCodec(cast(StableCodecSource, load_frame("stable_codec", device=device))),
        )
    raise NotImplementedError(f"unsupported codec: {name}")
