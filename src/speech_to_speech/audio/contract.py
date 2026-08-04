"""Canonical semantic/global/aligned-acoustic code representation.

The anycodec contract stores both non-semantic layouts in its ``acoustic``
field.  This module gives the speech-to-speech layers an explicit vocabulary
without changing that external contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from torch import Tensor

from .._compat import StrEnum, auto
from .._tensor import is_signed_integer_dtype


class AudioStream(StrEnum):
    """Non-text AR audio streams used by structured codecs."""

    GLOBAL = auto()
    SEMANTIC = auto()


@dataclass(frozen=True)
class AudioCodes:
    """Codec codes split by modeling meaning rather than external storage schema.

    ``global_codes`` are utterance-level fixed slots. ``acoustic_codes`` are
    frame-aligned non-semantic codes. Instances may hold only the streams needed
    by the current operation; at least one stream must be present. The anycodec
    adapter exposes one non-semantic layout at a time.
    """

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

    @classmethod
    def from_anycodec(
        cls,
        value: SemanticAcousticCodes,
        layout: AcousticLayout,
    ) -> AudioCodes:
        """Translate anycodec's overloaded ``acoustic`` field at the boundary."""
        if not isinstance(value, SemanticAcousticCodes):
            raise TypeError("anycodec codes must be SemanticAcousticCodes.")
        if layout is AcousticLayout.FIXED_LENGTH:
            return cls(semantic_codes=value.semantic, global_codes=value.acoustic)
        if layout is AcousticLayout.FRAME_ALIGNED:
            return cls(semantic_codes=value.semantic, acoustic_codes=value.acoustic)
        raise ValueError(f"unsupported acoustic layout: {layout!r}.")

    def to_anycodec(self, layout: AcousticLayout) -> SemanticAcousticCodes:
        """Translate canonical codes back to anycodec's structured contract."""
        if self.semantic_codes is None:
            raise ValueError("anycodec export requires semantic_codes.")
        if layout is AcousticLayout.FIXED_LENGTH:
            if self.global_codes is None or self.acoustic_codes is not None:
                raise ValueError(
                    "fixed-length anycodec codes require global_codes only."
                )
            return SemanticAcousticCodes(
                semantic=self.semantic_codes,
                acoustic=self.global_codes,
            )
        if layout is AcousticLayout.FRAME_ALIGNED:
            if self.acoustic_codes is None or self.global_codes is not None:
                raise ValueError(
                    "frame-aligned anycodec codes require acoustic_codes only."
                )
            return SemanticAcousticCodes(
                semantic=self.semantic_codes,
                acoustic=self.acoustic_codes,
            )
        raise ValueError(f"unsupported acoustic layout: {layout!r}.")


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
