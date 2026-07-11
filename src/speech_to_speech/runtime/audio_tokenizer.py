"""Audio tokenizer implementations used by the runtime singleton."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from anytrain.tokenizer import CodecBPE
from torch import Tensor

from .types import AudioTokenizer


class NativeAudioTokenizer:
    """Identity tokenizer for native single-codebook semantic IDs."""

    def __init__(self, *, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        if isinstance(frames, Tensor):
            if frames.dim() != 2:
                raise ValueError("native audio tokenizer expects [frames, codebooks].")
            frames = frames.tolist()
        return torch.tensor([_single_code(frame) for frame in frames], dtype=torch.long)

    def expand(
        self,
        ids: Sequence[int],
        *,
        strict: bool | None = None,
    ) -> list[tuple[int, ...]]:
        del strict
        return [(int(token_id),) for token_id in ids]

    def expand_with_counts(
        self,
        ids: Sequence[int],
        *,
        strict: bool | None = None,
    ) -> tuple[list[tuple[int, ...]], list[int]]:
        del strict
        return self.expand(ids), [1 for _ in ids]

    def repeat_interleave(
        self,
        x: Tensor,
        spans: Tensor,
        mask: Tensor | None = None,
        *,
        dim: int = -2,
        strict: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        del strict
        if spans.dim() != 2:
            raise ValueError("audio spans must have shape [batch, bpe_frames].")
        if mask is None:
            mask = torch.ones_like(spans, dtype=torch.bool)
        if mask.shape != spans.shape:
            raise ValueError("audio span mask must align with spans.")
        if dim < 0:
            dim += x.dim()
        if dim != 1:
            raise ValueError("native audio tokenizer expects sequence dim 1.")
        if x.size(0) != spans.size(0) or x.size(dim) != spans.size(1):
            raise ValueError("values and audio spans must align on batch and sequence.")

        counts = spans.masked_fill(~mask, 0).to(dtype=torch.long)
        lengths = counts.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        expanded = x.new_zeros((spans.size(0), max_len, *x.shape[2:]))
        expanded_mask = torch.zeros(
            (spans.size(0), max_len),
            dtype=torch.bool,
            device=spans.device,
        )
        for row, length in enumerate(lengths.tolist()):
            if length == 0:
                continue
            positions = mask[row].nonzero(as_tuple=False).flatten()
            expanded[row, :length] = torch.repeat_interleave(
                x[row].index_select(0, positions),
                counts[row].index_select(0, positions),
                dim=0,
            )
            expanded_mask[row, :length] = True
        return expanded, expanded_mask


class TorchCodecBPE(CodecBPE):
    """CodecBPE with tensor conveniences for model/runtime integration."""

    @classmethod
    def wrap(cls, tokenizer: CodecBPE) -> TorchCodecBPE:
        if isinstance(tokenizer, cls):
            return tokenizer
        return cls(tokenizer._core, tokenizer._codec)

    def encode(
        self,
        frames: Sequence[Sequence[int]] | Tensor,
    ) -> list[int] | Tensor:
        if not isinstance(frames, Tensor):
            return super().encode(frames)
        token_ids = super().encode(_frames(frames, self.codebook_sizes))
        return torch.tensor(token_ids, dtype=torch.long, device=frames.device)

    def expand(
        self,
        ids: Sequence[int] | Tensor,
        *,
        strict: bool | None = None,
    ) -> list[tuple[int, ...]] | Tensor:
        del strict
        if not isinstance(ids, Tensor):
            return super().expand(ids)
        frames = super().expand(_ids(ids))
        return torch.tensor(frames, dtype=torch.long, device=ids.device)

    def expand_with_counts(
        self,
        ids: Sequence[int] | Tensor,
        *,
        strict: bool | None = None,
    ) -> tuple[list[tuple[int, ...]] | Tensor, list[int] | Tensor]:
        del strict
        if not isinstance(ids, Tensor):
            return super().expand_with_counts(ids)
        frames, counts = super().expand_with_counts(_ids(ids))
        return (
            torch.tensor(frames, dtype=torch.long, device=ids.device),
            torch.tensor(counts, dtype=torch.long, device=ids.device),
        )


def semantic_ids_from_audio_tokens(
    audio_tokenizer: AudioTokenizer,
    audio_token_ids: Sequence[int] | Tensor,
) -> Tensor:
    """Expand one BPE audio sequence to ``[frames, semantic_codebooks]``."""
    expanded = audio_tokenizer.expand(audio_token_ids)
    device = audio_token_ids.device if isinstance(audio_token_ids, Tensor) else None
    if isinstance(expanded, Tensor):
        if expanded.dim() != 2:
            raise ValueError("expanded semantic ids must have shape [frames, codebooks].")
        return expanded.to(device=device, dtype=torch.long)

    frames: list[Tensor] = []
    for frame in expanded:
        values = (
            frame.reshape(-1).tolist() if isinstance(frame, Tensor) else list(frame)
        )
        if not values:
            raise ValueError(
                "audio tokenizer expanded a token to no semantic codebooks."
            )
        frames.append(torch.tensor(values, device=device, dtype=torch.long))
    if not frames:
        raise ValueError("audio tokenizer expanded audio tokens to no frames.")
    try:
        return torch.stack(frames)
    except RuntimeError as error:
        raise ValueError(
            "expanded semantic frames must have the same codebook count."
        ) from error


def _single_code(frame: Sequence[int]) -> int:
    if len(frame) != 1:
        raise ValueError(
            "identity audio tokenizer requires one semantic code per frame; "
            "configure a CodecBPE tokenizer for multi-codebook semantic ids."
        )
    return int(frame[0])


def _frames(frames: Tensor, codebook_sizes: Sequence[int]) -> list[list[int]]:
    _check_ids(frames, "frames")
    if frames.dim() == 1:
        if len(codebook_sizes) != 1:
            raise ValueError(
                "1D frame tensors are only valid for single-codebook tokenizers."
            )
        return [[int(value)] for value in frames.detach().cpu().tolist()]
    if frames.dim() != 2:
        raise ValueError("frame tensor must have shape [frames, codebooks].")
    if frames.size(-1) != len(codebook_sizes):
        raise ValueError("frame tensor must match tokenizer codebook count.")
    return cast(list[list[int]], frames.detach().cpu().tolist())


def _ids(ids: Tensor) -> list[int]:
    _check_ids(ids, "token ids")
    if ids.dim() != 1:
        raise ValueError("token id tensor must have shape [tokens].")
    return [int(value) for value in ids.detach().cpu().tolist()]


def _check_ids(ids: Tensor, name: str) -> None:
    if ids.dtype == torch.bool or torch.is_floating_point(ids) or torch.is_complex(ids):
        raise TypeError(f"{name} must contain integer ids.")


__all__ = ["NativeAudioTokenizer", "TorchCodecBPE", "semantic_ids_from_audio_tokens"]
