from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor

from ._common import validate_ids, validate_range


class NativeAudioTokenizer:
    """Identity tokenizer for native single-codebook semantic IDs."""

    embedding_initialization = "codec"

    def __init__(self, *, vocab_size: int) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral):
            raise TypeError("native audio tokenizer vocab size must be an integer.")
        if vocab_size < 1:
            raise ValueError("native audio tokenizer vocab size must be positive.")
        self._vocab_size = int(vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def contract_state(self) -> dict[str, object]:
        """Return the effective token-ID grammar used by checkpoints."""
        return {
            "grammar": "native-v1",
            "vocab_size": self.vocab_size,
        }

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        if isinstance(frames, Tensor):
            validate_ids(frames, "frames")
            if frames.dim() != 2 or frames.size(1) != 1:
                raise ValueError("native audio tokenizer expects [frames, codebooks].")
            validate_range(frames, "frames", self.vocab_size)
            return frames[:, 0].to(dtype=torch.long)
        return torch.tensor(
            [_single_code(frame, self.vocab_size) for frame in frames],
            dtype=torch.long,
        )

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        if isinstance(token_ids, Tensor):
            _validate_native_token_ids(token_ids, self.vocab_size)
            return token_ids.to(dtype=torch.long).unsqueeze(-1)
        values = _native_token_ids(token_ids, self.vocab_size)
        return [(token_id,) for token_id in values]

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor:
        if not isinstance(token_ids, Tensor):
            _native_token_ids(token_ids, self.vocab_size)
            return [1] * len(token_ids)
        _validate_native_token_ids(token_ids, self.vocab_size)
        return torch.ones_like(token_ids, dtype=torch.long)


def _single_code(frame: Sequence[int], vocab_size: int) -> int:
    if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
        raise ValueError("native audio tokenizer expects [frames, codebooks].")
    if len(frame) != 1:
        raise ValueError(
            "identity audio tokenizer requires one semantic code per frame; "
            "configure a CodecBPE tokenizer for multi-codebook semantic codes."
        )
    return _native_id(frame[0], "frames", vocab_size)


def _native_token_ids(token_ids: Sequence[int], vocab_size: int) -> list[int]:
    return [_native_id(token_id, "token ids", vocab_size) for token_id in token_ids]


def _native_id(value: object, name: str, vocab_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must contain integer ids.")
    token_id = int(value)
    if not 0 <= token_id < vocab_size:
        raise ValueError(f"{name} must contain ids in [0, {vocab_size}).")
    return token_id


def _validate_native_token_ids(token_ids: Tensor, vocab_size: int) -> None:
    validate_ids(token_ids, "token ids")
    if token_ids.dim() != 1:
        raise ValueError("token id tensor must have shape [tokens].")
    validate_range(token_ids, "token ids", vocab_size)
