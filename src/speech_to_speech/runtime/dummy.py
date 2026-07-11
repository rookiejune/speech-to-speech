"""Audio tokenizer implementations used by the runtime singleton."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .types import AudioTokenizer


class DummyAudioTokenizer:
    """Identity Audio Tokenizer."""

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


def semantic_ids_from_audio_tokens(
    audio_tokenizer: AudioTokenizer,
    audio_token_ids: Sequence[int] | Tensor,
) -> Tensor:
    """Expand one BPE audio sequence to ``[frames, semantic_codebooks]``."""
    expanded = audio_tokenizer.expand(audio_token_ids)
    device = audio_token_ids.device if isinstance(audio_token_ids, Tensor) else None
    if isinstance(expanded, Tensor):
        if expanded.dim() != 2:
            raise ValueError(
                "expanded semantic ids must have shape [frames, codebooks]."
            )
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
        raise ValueError("native audio tokenizer requires one semantic code per frame.")
    return int(frame[0])
