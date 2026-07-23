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


class FlattenedAudioTokenizer:
    """Flatten fixed-width codec codebooks into one audio token sequence."""

    embedding_initialization = "random"

    def __init__(self, *, codebook_sizes: Sequence[int], codec_name: str) -> None:
        if not codebook_sizes:
            raise ValueError("flattened audio tokenizer requires codebook sizes.")
        if not codec_name:
            raise ValueError("flattened audio tokenizer requires a codec name.")
        sizes = [_codebook_size(size) for size in codebook_sizes]
        self._codec_name = codec_name
        self._codebook_sizes = tuple(sizes)
        offsets = [0]
        for size in sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self._offsets = tuple(offsets)
        self._code_vocab_size = sum(sizes)
        self._codec_token_id = self._code_vocab_size
        self._codebook_token_ids = tuple(
            self._codec_token_id + 1 + index for index in range(len(sizes))
        )
        self._vocab_size = self._code_vocab_size + 1 + len(sizes)

    @property
    def codec_name(self) -> str:
        return self._codec_name

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return self._codebook_sizes

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def codec_token_id(self) -> int:
        return self._codec_token_id

    @property
    def codebook_token_ids(self) -> tuple[int, ...]:
        return self._codebook_token_ids

    @property
    def special_tokens(self) -> dict[str, int]:
        tokens = {f"codec:{self.codec_name}": self.codec_token_id}
        tokens.update(
            {
                f"codec:{self.codec_name}:codebook:{index}": token_id
                for index, token_id in enumerate(self.codebook_token_ids)
            }
        )
        return tokens

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        tensor = _frame_tensor(frames, self.codebook_sizes)
        _validate_frame_ranges(tensor, self.codebook_sizes)
        values = [tensor.new_tensor([self.codec_token_id])]
        for index, offset in enumerate(self._offsets):
            values.append(tensor.new_tensor([self.codebook_token_ids[index]]))
            values.append(tensor[:, index] + offset)
        return torch.cat(values).to(dtype=torch.long)

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        tensor = _token_tensor(token_ids)
        payload = _flattened_payload(tensor, self)
        if payload.numel() % len(self.codebook_sizes) != 0:
            raise ValueError(
                "flattened token sequence length must be divisible by codebook count."
            )
        frames = payload.reshape(len(self.codebook_sizes), -1).transpose(0, 1).clone()
        offsets = frames.new_tensor(self._offsets)
        frames -= offsets
        _validate_frame_ranges(frames, self.codebook_sizes)
        if isinstance(token_ids, Tensor):
            return frames.to(device=token_ids.device, dtype=torch.long)
        return [tuple(int(value) for value in row) for row in frames.tolist()]

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor:
        tensor = _token_tensor(token_ids)
        _validate_flattened_sequence(tensor, self)
        spans = _flattened_frame_spans(tensor, self)
        if isinstance(token_ids, Tensor):
            return spans.to(device=token_ids.device, dtype=torch.long)
        return [int(value) for value in spans.tolist()]


class TorchCodecBPE(CodecBPE):
    """CodecBPE with tensor conveniences for model/runtime integration."""

    embedding_initialization = "codec"

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


def _codebook_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("codebook sizes must be integers.")
    size = int(value)
    if size < 1:
        raise ValueError("codebook sizes must be positive.")
    return size


def _frame_tensor(
    frames: Sequence[Sequence[int]] | Tensor,
    codebook_sizes: Sequence[int],
) -> Tensor:
    if isinstance(frames, Tensor):
        _validate_ids(frames, "frames")
        tensor = frames.to(dtype=torch.long)
    else:
        tensor = torch.tensor(frames, dtype=torch.long)
    if tensor.dim() != 2 or tensor.size(1) != len(codebook_sizes):
        raise ValueError("frames must have shape [frames, codebooks].")
    return tensor


def _token_tensor(token_ids: Sequence[int] | Tensor) -> Tensor:
    if isinstance(token_ids, Tensor):
        _validate_ids(token_ids, "token ids")
        tensor = token_ids.to(dtype=torch.long)
    else:
        tensor = torch.tensor(token_ids, dtype=torch.long)
    if tensor.dim() != 1:
        raise ValueError("token id tensor must have shape [tokens].")
    return tensor


def _validate_frame_ranges(frames: Tensor, codebook_sizes: Sequence[int]) -> None:
    for index, size in enumerate(codebook_sizes):
        _validate_range(frames[:, index], f"codebook {index} frames", size)


def _flattened_payload(token_ids: Tensor, tokenizer: FlattenedAudioTokenizer) -> Tensor:
    if token_ids.numel() < 1 + len(tokenizer.codebook_sizes):
        raise ValueError("flattened token sequence is missing codec codebook markers.")
    if int(token_ids[0].item()) != tokenizer.codec_token_id:
        raise ValueError("flattened token sequence must start with a codec marker.")
    payloads = []
    index = 1
    expected_frames: int | None = None
    for codebook, marker in enumerate(tokenizer.codebook_token_ids):
        if index >= token_ids.numel() or int(token_ids[index].item()) != marker:
            raise ValueError(f"flattened token sequence is missing codebook {codebook} marker.")
        index += 1
        next_markers = set(tokenizer.codebook_token_ids[codebook + 1 :])
        end = index
        while end < token_ids.numel() and int(token_ids[end].item()) not in next_markers:
            end += 1
        values = token_ids[index:end]
        if values.numel() == 0:
            raise ValueError("flattened codebook blocks must not be empty.")
        if expected_frames is None:
            expected_frames = values.numel()
        elif values.numel() != expected_frames:
            raise ValueError("flattened codebook blocks must have equal lengths.")
        _validate_range(
            values - tokenizer._offsets[codebook],
            f"codebook {codebook} token ids",
            tokenizer.codebook_sizes[codebook],
        )
        payloads.append(values)
        index = end
    if index != token_ids.numel():
        raise ValueError("flattened token sequence has trailing unknown markers.")
    return torch.cat(payloads)


def _validate_flattened_sequence(
    token_ids: Tensor,
    tokenizer: FlattenedAudioTokenizer,
) -> None:
    if _is_flattened_vocab_range(token_ids, tokenizer):
        return
    _flattened_payload(token_ids, tokenizer)


def _is_flattened_vocab_range(
    token_ids: Tensor,
    tokenizer: FlattenedAudioTokenizer,
) -> bool:
    return token_ids.numel() == tokenizer.vocab_size and torch.equal(
        token_ids.cpu(),
        torch.arange(tokenizer.vocab_size, dtype=token_ids.dtype),
    )


def _flattened_frame_spans(
    token_ids: Tensor,
    tokenizer: FlattenedAudioTokenizer,
) -> Tensor:
    spans = torch.zeros_like(token_ids, dtype=torch.long)
    first_start = tokenizer._offsets[0]
    first_end = first_start + tokenizer.codebook_sizes[0]
    spans[(token_ids >= first_start) & (token_ids < first_end)] = 1
    return spans


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
    "FlattenedAudioTokenizer",
    "NativeAudioTokenizer",
    "TorchCodecBPE",
    "semantic_codes_from_audio_tokens",
]
