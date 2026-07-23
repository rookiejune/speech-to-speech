from __future__ import annotations

from typing import Protocol, cast

import torch
from anytrain.codec.longcat import LongCat
from torch import Tensor

from .types import Codec


class LongCatCodec:
    """Adapt LongCat to the model-facing codec contract."""

    def __init__(self, codec: LongCat) -> None:
        self.codec = codec
        decoders = list(codec.decoders.values())
        latent_dim = None if not decoders else getattr(decoders[0], "latent_dim", None)
        if not isinstance(latent_dim, int):
            raise TypeError("LongCat decoder must expose an integer latent_dim.")
        self._acoustic_feature_dim = latent_dim

    @property
    def sample_rate(self) -> int:
        return self.codec.sample_rate

    @property
    def frame_rate(self) -> float:
        return float(self.codec.encoder.input_sample_rate / self.codec.encoder.hop_length)

    @property
    def acoustic_feature_dim(self) -> int:
        return self._acoustic_feature_dim

    @property
    def semantic_codebook(self) -> Tensor:
        return self.codec.semantic_codebook

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.codec.codebook_sizes)

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.codec.codebook_sizes[1:])

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        return self.codec.encode(audio, sample_rate)

    def decode(self, codes: Tensor) -> Tensor:
        return self.codec.decode(codes)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return self.codec.acoustic_codes_to_features(acoustic_codes)

    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor:
        return self.codec.decode_features(semantic_codes, acoustic_features)


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
    def acoustic_feature_dim(self) -> int:
        raise RuntimeError("unified-token codec has no acoustic feature representation.")

    @property
    def semantic_codebook(self) -> Tensor:
        return self._semantic_codebook

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.codec.codebook_sizes)

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return ()

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        return self.codec.encode(audio, sample_rate)

    def decode(self, codes: Tensor) -> Tensor:
        return self.codec.decode(codes)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        raise RuntimeError("unified-token codec has no acoustic codes.")

    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor:
        raise RuntimeError("unified-token codec has no acoustic features.")


def load_codec(name: str, device: str | None) -> Codec:
    if name == "longcat":
        return cast(Codec, LongCatCodec(LongCat.from_pretrained(device=device)))
    if name == "unicodec":
        from anytrain.codec.unicodec import UniCodec

        source = cast(
            UnifiedCodecSource,
            cast(object, UniCodec.from_pretrained(device=device)),
        )
        return cast(Codec, UnifiedCodec(source))
    raise NotImplementedError(f"unsupported codec: {name}")
