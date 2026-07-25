from __future__ import annotations

import math
from typing import Any, Protocol, cast

import torch
from anytrain.codec.longcat import LongCat
from torch import Tensor

from .types import Codec


class LongCatCodec:
    """Adapt LongCat to the model-facing codec contract."""

    name = "longcat"

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
    def semantic_feature_dim(self) -> int:
        return int(self.semantic_codebook.size(-1))

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


class BiCodecTokens(Protocol):
    semantic: Tensor
    global_tokens: Tensor


class BiCodecSource(Protocol):
    config: dict[str, Any]
    global_codebook_sizes: tuple[int, ...]
    sample_rate: int
    semantic_codebook_sizes: tuple[int, ...]

    def encode(self, audio: Tensor, sample_rate: int) -> BiCodecTokens: ...

    def detokenize(self, semantic: Tensor, global_tokens: Tensor) -> Tensor: ...


class BiCodecCodec:
    """Adapt Spark-TTS BiCodec to the full codec sequence contract."""

    name = "bicodec"
    _semantic_feature_dim = 1024
    _default_global_codebooks = 3

    def __init__(self, codec: BiCodecSource) -> None:
        self.codec = codec
        self._codebook_sizes = _bicodec_codebook_sizes(codec)
        self._frame_rate = _bicodec_frame_rate(codec)

    @property
    def sample_rate(self) -> int:
        return int(self.codec.sample_rate)

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def semantic_feature_dim(self) -> int:
        return self._semantic_feature_dim

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return self._codebook_sizes

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        tokens = self.codec.encode(audio, sample_rate)
        return _bicodec_codes(tokens.semantic, tokens.global_tokens)

    def decode(self, codes: Tensor) -> Tensor:
        semantic, global_tokens = _bicodec_tokens(codes, self.codebook_sizes)
        return self.codec.detokenize(semantic, global_tokens)


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
    if name == "bicodec":
        from anytrain.codec.bicodec import BiCodec

        source = cast(
            BiCodecSource,
            cast(object, BiCodec.from_pretrained(device=device)),
        )
        return cast(Codec, BiCodecCodec(source))
    raise NotImplementedError(f"unsupported codec: {name}")


def _bicodec_codebook_sizes(codec: BiCodecSource) -> tuple[int, ...]:
    semantic = _sizes(codec.semantic_codebook_sizes, name="BiCodec semantic codebooks")
    if len(semantic) != 1:
        raise ValueError("BiCodec adapter expects exactly one semantic codebook.")
    global_sizes = _sizes(codec.global_codebook_sizes, name="BiCodec global codebooks")
    if len(global_sizes) == 1:
        global_sizes = global_sizes * _global_codebook_count(codec)
    return semantic + global_sizes


def _global_codebook_count(codec: BiCodecSource) -> int:
    config = codec.config
    for key in (
        "global_codebooks",
        "global_num_codebooks",
        "speaker_codebooks",
        "speaker_num_codebooks",
        "num_global_tokens",
        "num_speaker_tokens",
    ):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    speaker_encoder = getattr(getattr(codec, "model", None), "speaker_encoder", None)
    count = getattr(speaker_encoder, "num_codebooks", None)
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return count
    return BiCodecCodec._default_global_codebooks


def _sizes(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    output = tuple(int(value) for value in values)
    if any(value <= 0 for value in output):
        raise ValueError(f"{name} must be positive.")
    return output


def _bicodec_frame_rate(codec: BiCodecSource) -> float:
    hop = codec.config.get("latent_hop_length")
    if isinstance(hop, bool) or not isinstance(hop, int):
        raise ValueError("BiCodec config must expose integer latent_hop_length.")
    if hop <= 0:
        raise ValueError("BiCodec latent_hop_length must be positive.")
    rate = float(codec.sample_rate) / float(hop)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("BiCodec frame_rate must be finite and positive.")
    return rate


def _bicodec_codes(semantic: Tensor, global_tokens: Tensor) -> Tensor:
    semantic = _semantic_tokens(semantic)
    global_tokens = _global_tokens(global_tokens)
    return torch.cat(
        [semantic.unsqueeze(-1), global_tokens.expand(-1, semantic.size(1), -1)],
        dim=-1,
    ).to(dtype=torch.long)


def _semantic_tokens(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("BiCodec semantic tokens must be a Tensor.")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError("BiCodec semantic tokens must contain integer ids.")
    if value.dim() == 1:
        value = value.unsqueeze(0)
    if value.dim() != 2:
        raise ValueError("BiCodec semantic tokens must have shape [batch, time].")
    if value.size(1) <= 0:
        raise ValueError("BiCodec semantic tokens must not be empty.")
    return value.to(dtype=torch.long)


def _global_tokens(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("BiCodec global tokens must be a Tensor.")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError("BiCodec global tokens must contain integer ids.")
    if value.dim() == 2:
        value = value.unsqueeze(1)
    if value.dim() != 3 or value.size(1) != 1:
        raise ValueError("BiCodec global tokens must have shape [batch, 1, codebooks].")
    if value.size(-1) <= 0:
        raise ValueError("BiCodec global tokens must not be empty.")
    return value.to(dtype=torch.long)


def _bicodec_tokens(
    codes: Tensor,
    codebook_sizes: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    if not isinstance(codes, Tensor):
        raise TypeError("BiCodec codes must be a Tensor.")
    if codes.dtype == torch.bool or torch.is_floating_point(codes) or torch.is_complex(codes):
        raise TypeError("BiCodec codes must contain integer ids.")
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)
    if codes.dim() != 3:
        raise ValueError("BiCodec codes must have shape [batch, frame, codebook].")
    if codes.size(1) <= 0:
        raise ValueError("BiCodec codes must contain at least one frame.")
    if codes.size(-1) != len(codebook_sizes):
        raise ValueError(
            "BiCodec codes do not match runtime codebook count: "
            f"{codes.size(-1)} != {len(codebook_sizes)}."
        )
    for index, size in enumerate(codebook_sizes):
        values = codes[..., index]
        if bool(((values < 0) | (values >= size)).any()):
            raise ValueError(f"BiCodec codebook {index} ids are outside [0, {size}).")
    return codes[..., 0].to(dtype=torch.long), codes[:, :1, 1:].to(dtype=torch.long)
