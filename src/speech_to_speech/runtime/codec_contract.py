from __future__ import annotations

import math
from typing import Protocol, Union, cast, runtime_checkable

from anytrain.codec import AcousticLayout
from torch import Generator, Tensor


class SemanticCodec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    def decode(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: Generator | None = None,
    ) -> Tensor: ...


class SemanticCodebookCodec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def semantic_codebook(self) -> Tensor: ...


class Codec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...


class StructuredCodec(SemanticCodebookCodec, Protocol):
    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_layout(self) -> AcousticLayout: ...

    @property
    def acoustic_unit_length(self) -> int | None: ...

    @property
    def acoustic_feature_dim(self) -> int: ...

    def tokenize(self, audio: Tensor, sample_rate: int) -> object: ...

    def detokenize(self, codes: object) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


CodecBackend = Union[Codec, StructuredCodec]


class CodebookCodec(SemanticCodebookCodec, Protocol):
    pass


class AcousticCodec(CodebookCodec, Protocol):
    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


@runtime_checkable
class _CodebookCapability(Protocol):
    @property
    def semantic_codebook(self) -> Tensor: ...


@runtime_checkable
class _SemanticFeatureCapability(Protocol):
    @property
    def semantic_feature_dim(self) -> int: ...


@runtime_checkable
class _SampleRateCapability(Protocol):
    @property
    def sample_rate(self) -> int: ...


@runtime_checkable
class _FrameRateCapability(Protocol):
    @property
    def frame_rate(self) -> float: ...


@runtime_checkable
class _FrameCapability(_SampleRateCapability, _FrameRateCapability, Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...


@runtime_checkable
class _FrameCodebookCapability(Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...


@runtime_checkable
class _AcousticCapability(
    _SampleRateCapability,
    _FrameRateCapability,
    _CodebookCapability,
    Protocol,
):
    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


@runtime_checkable
class _StructuredCapability(
    _SampleRateCapability,
    _FrameRateCapability,
    _CodebookCapability,
    Protocol,
):
    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_layout(self) -> AcousticLayout: ...

    @property
    def acoustic_unit_length(self) -> int | None: ...

    @property
    def acoustic_feature_dim(self) -> int: ...

    def tokenize(self, audio: Tensor, sample_rate: int) -> object: ...

    def detokenize(self, codes: object) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


def codebook_codec(codec: object) -> CodebookCodec:
    if not isinstance(codec, _CodebookCapability):
        raise TypeError("codec-initialized audio embeddings require a semantic codebook.")
    _semantic_codebook(codec.semantic_codebook)
    return cast(CodebookCodec, codec)


def semantic_feature_dim(codec: object) -> int:
    if isinstance(codec, _CodebookCapability):
        return int(_semantic_codebook(codec.semantic_codebook).size(-1))
    if not isinstance(codec, _SemanticFeatureCapability):
        raise TypeError(
            "random audio embeddings require a semantic codebook or feature dimension."
        )
    value = codec.semantic_feature_dim
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("codec semantic feature dimension must be an integer.")
    if value <= 0:
        raise ValueError("codec semantic feature dimension must be positive.")
    return value


@runtime_checkable
class _FsqLevelsCapability(Protocol):
    @property
    def fsq_levels(self) -> tuple[tuple[int, ...], ...]: ...


def fsq_levels(codec: object) -> tuple[tuple[int, ...], ...] | None:
    """Return FSQ levels when the codec is a dim-1 FSQ source; otherwise None."""
    if not isinstance(codec, _SemanticFeatureCapability):
        return None
    if semantic_feature_dim(codec) != 1:
        return None
    if not isinstance(codec, _FsqLevelsCapability):
        return None
    levels = tuple(
        tuple(int(level) for level in stage) for stage in codec.fsq_levels
    )
    if not levels:
        raise ValueError("fsq_levels must be a non-empty tuple of stages.")
    for stage in levels:
        if not stage:
            raise ValueError("each FSQ stage must declare at least one level.")
        if any(level < 2 for level in stage):
            raise ValueError("FSQ levels must be at least 2.")
    if isinstance(codec, _FrameCodebookCapability):
        sizes = _codebook_sizes(codec.codebook_sizes, "FSQ codec")
        if len(sizes) != len(levels):
            raise ValueError("fsq_levels must align with codebook_sizes.")
        for size, stage in zip(sizes, levels):
            product = 1
            for level in stage:
                product *= level
            if product != size:
                raise ValueError(
                    f"FSQ levels {stage} must multiply to codebook size {size}."
                )
    return levels


@runtime_checkable
class _FsqLevelValuesCapability(Protocol):
    @property
    def fsq_level_values(
        self,
    ) -> tuple[tuple[tuple[float, ...], ...], ...] | None: ...


def fsq_level_values(
    codec: object,
) -> tuple[tuple[tuple[float, ...], ...], ...] | None:
    """Return codec-canonical FSQ values when the backend exposes them."""
    levels = fsq_levels(codec)
    if levels is None or not isinstance(codec, _FsqLevelValuesCapability):
        return None
    raw = codec.fsq_level_values
    if raw is None:
        return None
    if len(raw) != len(levels):
        raise ValueError("fsq_level_values must align with FSQ stages.")

    result: list[tuple[tuple[float, ...], ...]] = []
    for stage_levels, stage_values in zip(levels, raw):
        if len(stage_values) != len(stage_levels):
            raise ValueError("fsq_level_values must align with FSQ digits.")
        digits: list[tuple[float, ...]] = []
        for level, values in zip(stage_levels, stage_values):
            digit = tuple(float(value) for value in values)
            if len(digit) != level:
                raise ValueError("FSQ digit values must match their level count.")
            if any(not math.isfinite(value) for value in digit):
                raise ValueError("FSQ digit values must be finite.")
            if any(left >= right for left, right in zip(digit, digit[1:])):
                raise ValueError("FSQ digit values must be strictly increasing.")
            digits.append(digit)
        result.append(tuple(digits))
    return tuple(result)


@runtime_checkable
class _FsqRadixCapability(Protocol):
    @property
    def fsq_radix_order(self) -> str: ...


def fsq_radix_order(codec: object) -> str | None:
    """Return the codec's packed-product digit order when declared."""
    if fsq_levels(codec) is None or not isinstance(codec, _FsqRadixCapability):
        return None
    value = codec.fsq_radix_order
    if not isinstance(value, str) or not value:
        raise TypeError("FSQ radix order must be a non-empty string.")
    return value


def frame_codec(codec: object) -> Codec:
    if not isinstance(codec, _FrameCapability):
        raise TypeError("full frame-code encoding and decoding require a frame codec capability.")
    codec_sample_rate(codec)
    codec_frame_rate(codec)
    _codebook_sizes(codec.codebook_sizes, "frame codec")
    return cast(Codec, codec)


def frame_codebook_sizes(codec: object) -> tuple[int, ...]:
    if not isinstance(codec, _FrameCodebookCapability):
        raise TypeError("frame codec codebook metadata is required.")
    return _codebook_sizes(codec.codebook_sizes, "frame codec")


def acoustic_codec(codec: object) -> AcousticCodec:
    if not isinstance(codec, _AcousticCapability):
        raise TypeError("acoustic decoding requires an acoustic codec capability.")
    codec_sample_rate(codec)
    codec_frame_rate(codec)
    _codebook_sizes(codec.acoustic_codebook_sizes, "acoustic codec")
    _positive_int(codec.acoustic_feature_dim, "acoustic codec feature dimension")
    return cast(AcousticCodec, codec)


def supports_acoustic(codec: object) -> bool:
    if not isinstance(codec, _AcousticCapability):
        return False
    acoustic_codec(codec)
    return True


def structured_codec(codec: object) -> StructuredCodec:
    if not isinstance(codec, _StructuredCapability):
        raise TypeError("structured codec capability is required.")
    codec_sample_rate(codec)
    codec_frame_rate(codec)
    _codebook_sizes(codec.semantic_codebook_sizes, "structured semantic codec")
    _codebook_sizes(codec.acoustic_codebook_sizes, "structured acoustic codec")
    _positive_int(codec.acoustic_feature_dim, "structured codec acoustic feature dimension")
    layout = codec.acoustic_layout
    if not isinstance(layout, AcousticLayout):
        raise TypeError("structured codec acoustic layout must be an AcousticLayout.")
    unit_length = codec.acoustic_unit_length
    if layout is AcousticLayout.FIXED_LENGTH:
        _positive_int(unit_length, "fixed-length structured codec acoustic unit length")
    elif unit_length is not None:
        raise ValueError(
            "frame-aligned structured codec acoustic unit length must be None."
        )
    return cast(StructuredCodec, codec)


def supports_structured(codec: object) -> bool:
    if not isinstance(codec, _StructuredCapability):
        return False
    structured_codec(codec)
    return True


def codec_sample_rate(codec: object) -> int:
    if not isinstance(codec, _SampleRateCapability):
        raise TypeError("codec sample rate metadata is required.")
    return _positive_int(codec.sample_rate, "codec sample rate")


def codec_frame_rate(codec: object) -> float:
    if not isinstance(codec, _FrameRateCapability):
        raise TypeError("codec frame rate metadata is required.")
    value = codec.frame_rate
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("codec frame rate must be a number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("codec frame rate must be finite and positive.")
    return result


def _codebook_sizes(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} codebook sizes must be a tuple of integers.")
    if not value:
        raise ValueError(f"{name} codebook sizes must be non-empty.")
    for size in value:
        _positive_int(size, f"{name} codebook size")
    return cast(tuple[int, ...], value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _semantic_codebook(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("codec semantic codebook must be a Tensor.")
    if value.dim() not in {2, 3}:
        raise ValueError(
            "codec semantic codebook must have shape [vocab, dim] or "
            "[codebooks, vocab, dim]."
        )
    if any(size <= 0 for size in value.shape):
        raise ValueError("codec semantic codebook dimensions must be positive.")
    return value

__all__ = [
    "AcousticCodec",
    "CodebookCodec",
    "Codec",
    "CodecBackend",
    "SemanticCodebookCodec",
    "SemanticCodec",
    "StructuredCodec",
    "acoustic_codec",
    "codebook_codec",
    "codec_frame_rate",
    "codec_sample_rate",
    "frame_codebook_sizes",
    "frame_codec",
    "fsq_level_values",
    "fsq_levels",
    "fsq_radix_order",
    "semantic_feature_dim",
    "structured_codec",
    "supports_acoustic",
    "supports_structured",
]
