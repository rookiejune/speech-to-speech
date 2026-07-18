"""Audio tokenizer implementations used by the runtime singleton."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import cast, overload

import torch
from anytrain.tokenizer import CodecBPE
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from .types import AudioTokenizer


class NativeAudioTokenizer:
    """Identity tokenizer for native single-codebook semantic IDs."""

    def __init__(self, *, vocab_size: int) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral):
            raise TypeError("native audio tokenizer vocab size must be an integer.")
        if vocab_size < 1:
            raise ValueError("native audio tokenizer vocab size must be positive.")
        self._vocab_size = int(vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        if isinstance(frames, Tensor):
            _validate_ids(frames, "frames")
            if frames.dim() != 2 or frames.size(1) != 1:
                raise ValueError("native audio tokenizer expects [frames, codebooks].")
            _validate_range(frames, "frames", self.vocab_size)
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


class TorchCodecBPE(CodecBPE):
    """CodecBPE with tensor conveniences for model/runtime integration."""

    @classmethod
    def wrap(cls, tokenizer: CodecBPE) -> TorchCodecBPE:
        if isinstance(tokenizer, cls):
            return tokenizer
        return cls(tokenizer._core, tokenizer._codec)

    @overload
    def encode(self, frames: Sequence[Sequence[int]]) -> list[int]: ...

    @overload
    def encode(self, frames: Tensor) -> Tensor: ...

    def encode(
        self,
        frames: Sequence[Sequence[int]] | Tensor,
    ) -> list[int] | Tensor:
        if not isinstance(frames, Tensor):
            return super().encode(frames)
        token_ids = super().encode(_frames(frames, self.codebook_sizes))
        return torch.tensor(token_ids, dtype=torch.long, device=frames.device)

    @overload
    def decode(self, token_ids: Sequence[int]) -> list[tuple[int, ...]]: ...

    @overload
    def decode(self, token_ids: Tensor) -> Tensor: ...

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        if not isinstance(token_ids, Tensor):
            return super().decode(token_ids)
        frames = super().decode(_ids(token_ids))
        return torch.tensor(frames, dtype=torch.long, device=token_ids.device)

    @overload
    def frame_spans(self, token_ids: Sequence[int]) -> list[int]: ...

    @overload
    def frame_spans(self, token_ids: Tensor) -> Tensor: ...

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor:
        values = _ids(token_ids) if isinstance(token_ids, Tensor) else token_ids
        spans = [len(self._core.tokens[int(token_id)]) for token_id in values]
        if isinstance(token_ids, Tensor):
            return token_ids.new_tensor(spans, dtype=torch.long)
        return spans


def semantic_codes_from_audio_tokens(
    audio_tokenizer: AudioTokenizer,
    audio_token_ids: Sequence[int] | Tensor,
) -> Tensor:
    """Decode one BPE audio sequence to ``[frames, semantic_codebooks]``."""
    decoded = audio_tokenizer.decode(audio_token_ids)
    device = audio_token_ids.device if isinstance(audio_token_ids, Tensor) else None
    if isinstance(decoded, Tensor):
        if decoded.dim() != 2:
            raise ValueError(
                "decoded semantic codes must have shape [frames, codebooks]."
            )
        return decoded.to(device=device, dtype=torch.long)

    frames: list[Tensor] = []
    for frame in decoded:
        values = (
            frame.reshape(-1).tolist() if isinstance(frame, Tensor) else list(frame)
        )
        if not values:
            raise ValueError(
                "audio tokenizer decoded a token to no semantic codebooks."
            )
        frames.append(torch.tensor(values, device=device, dtype=torch.long))
    if not frames:
        raise ValueError("audio tokenizer decoded audio tokens to no frames.")
    try:
        return torch.stack(frames)
    except RuntimeError as error:
        raise ValueError(
            "decoded semantic frames must have the same codebook count."
        ) from error


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
    _validate_ids(token_ids, "token ids")
    if token_ids.dim() != 1:
        raise ValueError("token id tensor must have shape [tokens].")
    _validate_range(token_ids, "token ids", vocab_size)


def _frames(frames: Tensor, codebook_sizes: Sequence[int]) -> list[list[int]]:
    _validate_ids(frames, "frames")
    if frames.dim() == 1:
        if len(codebook_sizes) != 1:
            raise ValueError(
                "1D frame tensors are only valid for single-codebook tokenizers."
            )
        return [[value] for value in frames.detach().cpu().tolist()]
    if frames.dim() != 2:
        raise ValueError("frame tensor must have shape [frames, codebooks].")
    if frames.size(-1) != len(codebook_sizes):
        raise ValueError("frame tensor must match tokenizer codebook count.")
    return cast(list[list[int]], frames.detach().cpu().tolist())


def _ids(ids: Tensor) -> list[int]:
    _validate_ids(ids, "token ids")
    if ids.dim() != 1:
        raise ValueError("token id tensor must have shape [tokens].")
    return ids.detach().cpu().tolist()


def _validate_ids(ids: Tensor, name: str) -> None:
    if not is_signed_integer_dtype(ids.dtype):
        raise TypeError(f"{name} must contain integer ids using a signed dtype.")


def _validate_range(ids: Tensor, name: str, vocab_size: int) -> None:
    if bool(((ids < 0) | (ids >= vocab_size)).any()):
        raise ValueError(f"{name} must contain ids in [0, {vocab_size}).")


__all__ = [
    "NativeAudioTokenizer",
    "TorchCodecBPE",
    "semantic_codes_from_audio_tokens",
]
