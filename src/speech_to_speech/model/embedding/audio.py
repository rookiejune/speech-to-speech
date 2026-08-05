from __future__ import annotations

from functools import cached_property
from inspect import getattr_static
from typing import Any, Protocol, cast, runtime_checkable

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from ...runtime.codec_contract import (
    CodebookCodec,
    codebook_codec,
    fsq_level_values,
    fsq_levels,
    fsq_radix_order,
    semantic_feature_dim,
)
from ...runtime.audio_tokenizer.contract import AudioTokenizer
from .fsq import FsqEmbedding, FsqEmbeddingConfig, reference_rms

_MISSING = object()


class SemanticAudioEmbedding(Protocol):
    @property
    def weight(self) -> Tensor: ...

    @property
    def num_embeddings(self) -> int: ...

    @property
    def embedding_dim(self) -> int: ...

    def forward(self, input_ids: Tensor, /) -> Tensor: ...

    def __call__(self, input_ids: Tensor, /) -> Tensor: ...

    def to(self, *args: Any, **kwargs: Any) -> "SemanticAudioEmbedding": ...

_ROPE_THETA = 10000.0
_EMBEDDING_CHUNK_SIZE = 2_048


class _Runtime(Protocol):
    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer: ...

    @cached_property
    def codec(self) -> object: ...

    @cached_property
    def layout(self) -> Layout: ...


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
    fsq: FsqEmbeddingConfig | None = None,
) -> SemanticAudioEmbedding:
    audio_start, audio_end = runtime.layout.blocks[Modality.AUDIO.value]
    audio_rows = audio_end - audio_start
    levels = fsq_levels(runtime.codec)
    if levels is not None:
        if embedding_dim is None:
            raise ValueError("FSQ audio embedding requires embedding_dim.")
        tokenizer = runtime.audio_tokenizer
        if not isinstance(tokenizer, _CodebookTokenizer):
            raise TypeError(
                "FSQ embedding requires a flattened codebook tokenizer."
            )
        radix_order = fsq_radix_order(runtime.codec)
        if radix_order not in {None, FsqEmbedding.radix_order}:
            raise ValueError(
                "FSQ embedding only supports first_fastest mixed-radix IDs."
            )
        return FsqEmbedding(
            codebook_sizes=tuple(int(size) for size in tokenizer.codebook_sizes),
            fsq_levels=levels,
            num_embeddings=audio_rows,
            embedding_dim=embedding_dim,
            target_rms=reference_rms(reference),
            config=fsq,
            level_values=fsq_level_values(runtime.codec),
        )
    return embedding(
        runtime.codec,
        runtime.audio_tokenizer,
        reference=reference,
        num_embeddings=audio_rows,
    )


def require_semantic_audio_embedding(
    value: object,
    name: str,
) -> SemanticAudioEmbedding:
    required = ("weight", "num_embeddings", "embedding_dim", "forward", "__call__", "to")
    missing = [
        attribute
        for attribute in required
        if not _has_semantic_audio_embedding_attribute(value, attribute)
    ]
    if missing:
        raise TypeError(
            f"{name} must implement the semantic audio embedding interface "
            f"(missing {', '.join(missing)})."
        )
    return cast(SemanticAudioEmbedding, value)


def _has_semantic_audio_embedding_attribute(value: object, attribute: str) -> bool:
    if getattr_static(value, attribute, _MISSING) is not _MISSING:
        return True
    if isinstance(value, nn.Module):
        return (
            attribute in value._parameters
            or attribute in value._buffers
            or attribute in value._modules
        )
    return False


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
    num_embeddings: int,
) -> nn.Embedding:
    """Build a lookup initialized from the codec codebook.

    Rows after the codec tokenizer vocabulary are runtime-owned audio controls.
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
    control_rows = num_embeddings - tokenizer.vocab_size
    if control_rows < 1:
        raise ValueError("audio token block must reserve runtime control rows.")
    special = torch.empty(
        (control_rows, base.size(1)),
        device=base.device,
        dtype=base.dtype,
    )
    nn.init.normal_(special, std=base.size(1) ** -0.5)
    weight = torch.cat([base, special], dim=0)
    output = nn.Embedding.from_pretrained(weight, freeze=False)
    return output
