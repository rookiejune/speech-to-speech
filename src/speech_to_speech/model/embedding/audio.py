from __future__ import annotations

import torch
from torch import Tensor, nn

from ...runtime.types import AudioTokenizer, Codec

_ROPE_THETA = 10000.0


def _merge(embeddings: Tensor) -> Tensor:
    """Apply one-dimensional RoPE over expanded units, then mean-pool them."""
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [units, dim].")
    if embeddings.size(-1) % 2 != 0:
        raise ValueError("embedding dimension must be even for RoPE.")

    positions = torch.arange(
        embeddings.size(0),
        device=embeddings.device,
        dtype=torch.float32,
    )
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
    return rotated.flatten(-2).mean(0)


def merge_by_positions(
    features: Tensor,
    positions: Tensor,
    sequence_length: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Merge frame features into the token positions they belong to.

    ``features`` stays frame-level while the returned tensor is aligned to the
    BPE-level input sequence. Each occupied position uses the same RoPE plus
    mean-pooling rule as semantic audio-token embedding.
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

    output = features.new_zeros(features.size(0), sequence_length, features.size(-1))
    for row in range(features.size(0)):
        active = mask[row] & positions[row].ge(0)
        if not bool(active.any()):
            continue
        row_positions = positions[row][active]
        if bool((row_positions >= sequence_length).any()):
            raise ValueError("frame position exceeds sequence length.")
        for position in row_positions.unique(sorted=True).tolist():
            output[row, position] = _merge(
                features[row][active & positions[row].eq(position)]
            )
    return output


def base_weight(codec: Codec, tokenizer: AudioTokenizer) -> Tensor:
    """Create one fixed feature vector for every audio-tokenizer ID."""
    codebook = codec.semantic_codebook
    if codebook.dim() != 2:
        raise ValueError("codec semantic_codebook must have shape [vocab, dim].")

    rows = []
    for token_id in range(tokenizer.vocab_size):
        units, counts = tokenizer.expand_with_counts([token_id])
        if isinstance(units, Tensor):
            if units.dim() != 2 or units.size(0) != 1:
                raise ValueError("audio tokenizer expand returned an invalid shape.")
            unit_ids = units[0].to(device=codebook.device, dtype=torch.long)
        else:
            if len(units) != 1:
                raise ValueError("audio tokenizer expand returned an invalid length.")
            unit_ids = torch.tensor(units[0], device=codebook.device, dtype=torch.long)
        count = int(counts[0].item()) if isinstance(counts, Tensor) else int(counts[0])
        if count != unit_ids.numel():
            raise ValueError(
                "audio tokenizer expansion count does not match expansion."
            )
        if unit_ids.numel() == 0:
            raise ValueError(
                f"audio tokenizer token {token_id} expands to no codec units."
            )
        rows.append(_merge(codebook.index_select(0, unit_ids)))

    return torch.stack(rows)


def embedding(codec: Codec, tokenizer: AudioTokenizer) -> nn.Embedding:
    """Build a lookup initialized from the codec codebook.

    The final two rows are reserved for BOA and EOA.
    """
    base = base_weight(codec, tokenizer)
    special = torch.empty(
        (2, base.size(1)),
        device=base.device,
        dtype=base.dtype,
    )
    nn.init.normal_(special, std=base.size(1) ** -0.5)
    weight = torch.cat([base, special], dim=0)
    output = nn.Embedding.from_pretrained(weight, freeze=False)
    return output
