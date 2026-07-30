from __future__ import annotations

from functools import cached_property
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from ...runtime.types import (
    AudioTokenizer,
    CodebookCodec,
    codebook_codec,
    fsq_levels,
    semantic_feature_dim,
)
from .fsq import FsqAffineEmbedding

_ROPE_THETA = 10000.0
_EMBEDDING_CHUNK_SIZE = 2_048


class _Runtime(Protocol):
    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer: ...

    @cached_property
    def codec(self) -> object: ...


@runtime_checkable
class _CodebookTokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...


def create_semantic_audio_embedding(
    runtime: _Runtime,
    *,
    reference: Tensor,
    embedding_dim: int | None = None,
) -> nn.Module:
    levels = fsq_levels(runtime.codec)
    if levels is not None:
        if embedding_dim is None:
            raise ValueError("FSQ affine audio embedding requires embedding_dim.")
        tokenizer = runtime.audio_tokenizer
        if not isinstance(tokenizer, _CodebookTokenizer):
            raise TypeError(
                "FSQ affine embedding requires a flattened codebook tokenizer."
            )
        return FsqAffineEmbedding(
            codebook_sizes=tuple(int(size) for size in tokenizer.codebook_sizes),
            fsq_levels=levels,
            num_embeddings=tokenizer.vocab_size + 3,
            embedding_dim=embedding_dim,
        )
    return embedding(
        runtime.codec,
        runtime.audio_tokenizer,
        reference=reference,
    )


def _merge(embeddings: Tensor) -> Tensor:
    """Apply one-dimensional RoPE over expanded units, then mean-pool them."""
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [units, dim].")
    positions = torch.arange(embeddings.size(0), device=embeddings.device)
    return _rotate(embeddings, positions).mean(0)


def _rotate(embeddings: Tensor, positions: Tensor) -> Tensor:
    if embeddings.size(-1) % 2 != 0:
        raise ValueError("embedding dimension must be even for RoPE.")
    if positions.shape != embeddings.shape[:1]:
        raise ValueError("RoPE positions must align with embedding units.")
    positions = positions.to(dtype=torch.float32)
    dimensions = torch.arange(
        0,
        embeddings.size(-1),
        2,
        device=embeddings.device,
        dtype=torch.float32,
    )
    inverse_frequencies = _ROPE_THETA ** (-dimensions / embeddings.size(-1))
    angles = positions[:, None] * inverse_frequencies[None, :]
    cosines = angles.cos().to(dtype=embeddings.dtype)
    sines = angles.sin().to(dtype=embeddings.dtype)

    even = embeddings[..., 0::2]
    odd = embeddings[..., 1::2]
    rotated = torch.stack(
        [even * cosines - odd * sines, even * sines + odd * cosines],
        dim=-1,
    )
    return rotated.flatten(-2)


def merge_by_positions(
    features: Tensor,
    positions: Tensor,
    sequence_length: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Merge frame features into the token positions they belong to.

    ``features`` stays frame-level while the returned tensor is aligned to the
    BPE-level input sequence. Each occupied position uses the same RoPE plus
    mean-pooling rule as semantic audio-token embedding. The second output marks
    positions occupied by at least one active frame.
    """
    if features.dim() != 3 or positions.dim() != 2:
        raise ValueError(
            "features and positions must have shapes [B, F, D] and [B, F]."
        )
    if features.shape[:2] != positions.shape:
        raise ValueError(
            "features and positions must align on batch and frame dimensions."
        )
    if mask is None:
        mask = positions >= 0
    if mask.shape != positions.shape:
        raise ValueError("frame mask must align with positions.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")

    active = mask & positions.ge(0)
    active_positions = positions[active]
    if active_positions.numel() == 0:
        output = features.new_zeros(
            features.size(0), sequence_length, features.size(-1)
        )
        return output, positions.new_zeros(
            features.size(0), sequence_length, dtype=torch.bool
        )

    rows = torch.arange(features.size(0), device=features.device)[:, None]
    groups = (rows * sequence_length + positions)[active]
    order = groups.argsort(stable=True)
    groups = groups[order]
    values = features[active][order]

    indices = torch.arange(groups.numel(), device=groups.device)
    starts = torch.zeros_like(indices)
    new_group = torch.ones_like(groups, dtype=torch.bool)
    new_group[1:] = groups[1:] != groups[:-1]
    starts[new_group] = indices[new_group]
    offsets = indices - starts.cummax(dim=0).values
    values = _rotate(values, offsets)

    size = features.size(0) * sequence_length
    output = features.new_zeros(size, features.size(-1))
    output.index_add_(0, groups, values)
    counts = features.new_zeros(size, 1)
    counts.index_add_(0, groups, features.new_ones(groups.numel(), 1))
    output = (output / counts.clamp_min(1)).view(
        features.size(0), sequence_length, features.size(-1)
    )
    occupied = counts.view(features.size(0), sequence_length).gt(0)
    return output, occupied


def base_weight(codec: CodebookCodec, tokenizer: AudioTokenizer) -> Tensor:
    """Create one fixed feature vector for every audio-tokenizer ID."""
    codebook = codec.semantic_codebook.detach()
    if codebook.dim() not in {2, 3}:
        raise ValueError(
            "codec semantic_codebook must have shape [vocab, dim] or "
            "[codebooks, vocab, dim]."
        )

    output = codebook.new_empty(tokenizer.vocab_size, codebook.size(-1))
    for start in range(0, tokenizer.vocab_size, _EMBEDDING_CHUNK_SIZE):
        end = min(start + _EMBEDDING_CHUNK_SIZE, tokenizer.vocab_size)
        token_ids = list(range(start, end))
        unit_ids = _unit_ids(tokenizer.decode(token_ids), codebook)
        spans = torch.as_tensor(
            tokenizer.frame_spans(token_ids),
            dtype=torch.long,
        )
        if spans.shape != (end - start,) or bool((spans <= 0).any()):
            raise ValueError("audio tokenizer tokens must have positive frame spans.")
        if int(spans.sum()) != unit_ids.size(0):
            raise ValueError(
                "audio tokenizer spans must align with decoded codec units."
            )
        spans = spans.to(device=codebook.device)

        groups = torch.repeat_interleave(
            torch.arange(end - start, device=codebook.device),
            spans,
        )
        starts = torch.repeat_interleave(spans.cumsum(0) - spans, spans)
        positions = torch.arange(unit_ids.size(0), device=codebook.device) - starts
        values = _rotate(_unit_embeddings(codebook, unit_ids), positions)
        rows = values.new_zeros(end - start, values.size(-1))
        rows.index_add_(0, groups, values)
        output[start:end] = rows / spans[:, None]

    return output


def random_weight(
    feature_dim: int,
    tokenizer: AudioTokenizer,
    *,
    reference: Tensor,
) -> Tensor:
    """Create randomly initialized audio-token weights in codec feature space."""
    if feature_dim <= 0:
        raise ValueError("audio embedding feature dimension must be positive.")
    output = reference.new_empty(
        tokenizer.vocab_size,
        feature_dim,
    )
    nn.init.normal_(output, std=feature_dim**-0.5)
    return output


def _unit_ids(units: list[tuple[int, ...]] | Tensor, codebook: Tensor) -> Tensor:
    if isinstance(units, Tensor):
        unit_ids = units.to(device=codebook.device, dtype=torch.long)
    else:
        unit_ids = torch.tensor(units, device=codebook.device, dtype=torch.long)
    if unit_ids.dim() != 2:
        raise ValueError(
            "audio tokenizer expand must return [frames, semantic_codebooks]."
        )
    return unit_ids


def _unit_embeddings(codebook: Tensor, unit_ids: Tensor) -> Tensor:
    if codebook.dim() == 2:
        if unit_ids.size(-1) != 1:
            raise ValueError(
                "single semantic codebook cannot initialize multi-codebook units."
            )
        ids = unit_ids.flatten()
        if bool((ids < 0).any()) or bool((ids >= codebook.size(0)).any()):
            raise ValueError("semantic unit id is outside the codec codebook.")
        return codebook.index_select(0, ids)

    if unit_ids.size(-1) != codebook.size(0):
        raise ValueError("semantic units must match the codec semantic codebook count.")
    frames = []
    for index in range(codebook.size(0)):
        ids = unit_ids[:, index]
        table = codebook[index]
        if bool((ids < 0).any()) or bool((ids >= table.size(0)).any()):
            raise ValueError("semantic unit id is outside the codec codebook.")
        frames.append(table.index_select(0, ids))
    return torch.stack(frames, dim=1).mean(dim=1)


def embedding(
    codec: object,
    tokenizer: AudioTokenizer,
    *,
    reference: Tensor,
) -> nn.Embedding:
    """Build a lookup initialized from the codec codebook.

    The final three rows are reserved for BOA, EOA, and MASK.
    """
    initialization = tokenizer.embedding_initialization
    if initialization == "codec":
        base = base_weight(codebook_codec(codec), tokenizer)
    elif initialization == "random":
        base = random_weight(
            semantic_feature_dim(codec),
            tokenizer,
            reference=reference,
        )
    else:
        raise ValueError(f"unsupported audio embedding initialization: {initialization}")
    special = torch.empty(
        (3, base.size(1)),
        device=base.device,
        dtype=base.dtype,
    )
    nn.init.normal_(special, std=base.size(1) ** -0.5)
    weight = torch.cat([base, special], dim=0)
    output = nn.Embedding.from_pretrained(weight, freeze=False)
    return output
