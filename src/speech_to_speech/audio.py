"""Canonical semantic, global, and frame-aligned acoustic code representation."""

from __future__ import annotations

from dataclasses import dataclass

from anytrain.codec import SemanticAcousticCodes, SemanticGlobalCodes
from torch import Tensor

from ._compat import StrEnum, auto
from ._tensor import is_signed_integer_dtype


class AudioStream(StrEnum):
    """Non-text AR audio streams used by structured codecs."""

    GLOBAL = auto()
    SEMANTIC = auto()


@dataclass(frozen=True)
class AudioCodes:
    """Codec codes split by modeling meaning rather than storage schema."""

    semantic_codes: Tensor | None = None
    global_codes: Tensor | None = None
    acoustic_codes: Tensor | None = None

    def __post_init__(self) -> None:
        if (
            self.semantic_codes is None
            and self.global_codes is None
            and self.acoustic_codes is None
        ):
            raise ValueError("AudioCodes requires at least one code stream.")
        if self.semantic_codes is not None:
            _validate_codes(self.semantic_codes, "semantic_codes")
        if self.global_codes is not None:
            _validate_codes(self.global_codes, "global_codes")
        if self.acoustic_codes is not None:
            _validate_codes(self.acoustic_codes, "acoustic_codes")
        if self.global_codes is not None and self.acoustic_codes is not None:
            raise ValueError(
                "AudioCodes cannot contain global and frame-aligned acoustic codes together."
            )

    @classmethod
    def from_semantic_acoustic(
        cls,
        value: SemanticAcousticCodes,
    ) -> AudioCodes:
        if not isinstance(value, SemanticAcousticCodes):
            raise TypeError("codes must be SemanticAcousticCodes.")
        return cls(semantic_codes=value.semantic, acoustic_codes=value.acoustic)

    @classmethod
    def from_semantic_global(
        cls,
        value: SemanticGlobalCodes,
    ) -> AudioCodes:
        if not isinstance(value, SemanticGlobalCodes):
            raise TypeError("codes must be SemanticGlobalCodes.")
        return cls(semantic_codes=value.semantic, global_codes=value.global_codes)

    def to_semantic_acoustic(self) -> SemanticAcousticCodes:
        if self.semantic_codes is None or self.acoustic_codes is None:
            raise ValueError(
                "semantic-acoustic export requires semantic_codes and acoustic_codes."
            )
        if self.global_codes is not None:
            raise ValueError("semantic-acoustic export does not accept global_codes.")
        return SemanticAcousticCodes(
            semantic=self.semantic_codes,
            acoustic=self.acoustic_codes,
        )

    def to_semantic_global(self) -> SemanticGlobalCodes:
        if self.semantic_codes is None or self.global_codes is None:
            raise ValueError(
                "semantic-global export requires semantic_codes and global_codes."
            )
        if self.acoustic_codes is not None:
            raise ValueError("semantic-global export does not accept acoustic_codes.")
        return SemanticGlobalCodes(
            semantic=self.semantic_codes,
            global_codes=self.global_codes,
        )


def _validate_codes(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.dim() != 2:
        raise ValueError(f"{name} must have shape [units, codebooks].")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must use a signed integer dtype.")


__all__ = ["AudioCodes", "AudioStream"]
