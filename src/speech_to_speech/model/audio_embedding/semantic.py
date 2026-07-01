"""Semantic-code-composed audio BPE embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SemanticEmbeddingConfig:
    codebook_size: int = 8192
    rope_base: float = 10_000.0
    shift_rank: int = 16


class BPEAudioEmbedding(nn.Module):
    """Builds BPE embeddings from codec semantic embeddings plus learnable shift."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        codec_embedding: nn.Embedding,
        expansion: Tensor,
        expansion_mask: Tensor,
        config: SemanticEmbeddingConfig | None = None,
        like: Tensor | None = None,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        config = config or SemanticEmbeddingConfig()
        _validate_shift_rank(config.shift_rank)
        self.num_embeddings = vocab_size
        self.embedding_dim = hidden_size
        self.config = config
        self.shift_down = nn.Embedding(
            vocab_size,
            config.shift_rank,
            device=_device(like),
            dtype=_dtype(like),
        )
        self.shift_up = nn.Linear(
            config.shift_rank,
            hidden_size,
            bias=False,
            device=_device(like),
            dtype=_dtype(like),
        )
        nn.init.normal_(self.shift_down.weight, std=std)
        nn.init.zeros_(self.shift_up.weight)

        self.register_buffer("expansion", _validate_expansion(expansion, vocab_size))
        self.register_buffer(
            "expansion_mask",
            _validate_expansion_mask(expansion_mask, self.expansion),
        )
        self.register_buffer("base_weight", self._base_weight(codec_embedding))

    @property
    def shift_rank(self) -> int:
        return self.config.shift_rank

    def forward(self, input_ids: Tensor) -> Tensor:
        local_ids = _validate_input_ids(input_ids, self.num_embeddings)
        return self.weight[local_ids]

    def base_embedding(self, input_ids: Tensor) -> Tensor:
        local_ids = _validate_input_ids(input_ids, self.num_embeddings)
        return self.base_weight[local_ids]

    def _base_weight(self, codec_embedding: nn.Embedding) -> Tensor:
        local_ids = torch.arange(
            self.num_embeddings,
            device=self.shift_down.weight.device,
            dtype=torch.long,
        )
        expanded = self.expansion.to(device=local_ids.device)[local_ids]
        mask = self.expansion_mask.to(device=local_ids.device)[local_ids]
        with torch.no_grad():
            semantic = codec_embedding(expanded)
            semantic = _apply_rope(semantic, mask, base=self.config.rope_base)
            weights = mask.to(device=semantic.device, dtype=semantic.dtype).unsqueeze(-1)
            summed = (semantic * weights).sum(dim=-2)
            lengths = weights.sum(dim=-2).clamp_min(1.0)
            return (summed / torch.sqrt(lengths)).detach()

    def shift(self, input_ids: Tensor) -> Tensor:
        local_ids = _validate_input_ids(input_ids, self.num_embeddings)
        return self.shift_up(self.shift_down(local_ids))

    @property
    def weight(self) -> Tensor:
        return self.base_weight + self.shift_up(self.shift_down.weight)

    def requires_grad_(self, requires_grad: bool = True) -> BPEAudioEmbedding:
        self.shift_down.requires_grad_(requires_grad)
        self.shift_up.requires_grad_(requires_grad)
        return self


def semantic_audio_embedding(
    *,
    vocab_size: int,
    hidden_size: int,
    bpe: object,
    like: Tensor,
    std: float,
    config: SemanticEmbeddingConfig | None = None,
) -> BPEAudioEmbedding:
    codec = nn.Embedding(
        (config or SemanticEmbeddingConfig()).codebook_size,
        hidden_size,
        device=like.device,
        dtype=like.dtype,
    )
    nn.init.normal_(codec.weight, std=std)
    expansion, mask = _bpe_expansion_table(bpe, vocab_size)
    return BPEAudioEmbedding(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        codec_embedding=codec,
        expansion=expansion,
        expansion_mask=mask,
        config=config,
        like=like,
        std=std,
    )


def _bpe_expansion_table(bpe: object, vocab_size: int) -> tuple[Tensor, Tensor]:
    expand = getattr(bpe, "expand_ids", None)
    if not callable(expand):
        raise TypeError("LongCat BPE tokenizer must provide expand_ids().")

    rows: list[list[int]] = []
    max_length = 0
    for token_id in range(vocab_size):
        frames = expand([token_id])
        ids = _single_codebook_frame_ids(frames)
        if not ids:
            raise ValueError("LongCat BPE tokens must expand to at least one semantic id.")
        rows.append(ids)
        max_length = max(max_length, len(ids))

    expansion = torch.zeros((vocab_size, max_length), dtype=torch.long)
    mask = torch.zeros((vocab_size, max_length), dtype=torch.bool)
    for row_index, ids in enumerate(rows):
        expansion[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row_index, : len(ids)] = True
    return expansion, mask


def _single_codebook_frame_ids(frames: object) -> list[int]:
    if not isinstance(frames, list | tuple):
        raise TypeError("LongCat BPE expand_ids() must return a sequence of frames.")
    ids: list[int] = []
    for frame in frames:
        if not isinstance(frame, list | tuple):
            raise TypeError("LongCat BPE expand_ids() must return frame sequences.")
        if len(frame) != 1:
            raise ValueError("LongCat semantic BPE must use exactly one codebook.")
        ids.append(int(frame[0]))
    return ids


def _apply_rope(values: Tensor, mask: Tensor, *, base: float) -> Tensor:
    dim = values.size(-1)
    if dim % 2 != 0:
        return values
    positions = torch.arange(values.size(-2), device=values.device, dtype=torch.float32)
    frequencies = torch.arange(0, dim, 2, device=values.device, dtype=torch.float32)
    inv_freq = base ** (-frequencies / dim)
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos().to(dtype=values.dtype)
    sin = angles.sin().to(dtype=values.dtype)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.empty_like(values)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return torch.where(mask.unsqueeze(-1), rotated, values)


def _validate_expansion(expansion: Tensor, vocab_size: int) -> Tensor:
    if expansion.dim() != 2 or expansion.size(0) != vocab_size:
        raise ValueError("expansion must have shape [vocab_size, max_expanded_length].")
    if (
        expansion.dtype == torch.bool
        or torch.is_floating_point(expansion)
        or torch.is_complex(expansion)
    ):
        raise TypeError("expansion must contain integer semantic ids.")
    return expansion.to(dtype=torch.long)


def _validate_expansion_mask(mask: Tensor, expansion: Tensor) -> Tensor:
    if mask.shape != expansion.shape:
        raise ValueError("expansion_mask must match expansion shape.")
    return mask.to(dtype=torch.bool)


def _validate_input_ids(input_ids: Tensor, vocab_size: int) -> Tensor:
    if (
        input_ids.dtype == torch.bool
        or torch.is_floating_point(input_ids)
        or torch.is_complex(input_ids)
    ):
        raise TypeError("BPE audio ids must contain integer ids.")
    local_ids = input_ids.to(dtype=torch.long)
    if bool(local_ids.lt(0).any()) or bool(local_ids.ge(vocab_size).any()):
        raise ValueError("BPE audio ids must be inside the audio vocabulary.")
    return local_ids


def _validate_shift_rank(rank: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("semantic_shift_rank must be an integer.")
    if rank <= 0:
        raise ValueError("semantic_shift_rank must be positive.")


def _device(like: Tensor | None) -> torch.device | None:
    if like is None:
        return None
    return like.device


def _dtype(like: Tensor | None) -> torch.dtype | None:
    if like is None:
        return None
    return like.dtype


__all__ = [
    "BPEAudioEmbedding",
    "SemanticEmbeddingConfig",
    "semantic_audio_embedding",
]
