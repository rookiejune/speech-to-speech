from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from .._sampling import top_p_filter
from ..base import SemanticModel


class AcousticRVQDecoder(nn.Module):
    """Frame-parallel, codebook-autoregressive acoustic code predictor.

    ``codebook_embeddings`` uses the codec's local codebook order and has
    contains one ``[size_q, embedding_dim]`` tensor per codebook. The
    embeddings are copied into trainable model embeddings; the codec remains
    unchanged.
    """

    def __init__(
        self,
        condition_dim: int,
        codebooks: int,
        codebook_size: int | Sequence[int],
        *,
        codebook_embeddings: Sequence[Tensor] | None = None,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or codebooks <= 0:
            raise ValueError("condition_dim and codebooks must be positive.")
        sizes = (
            (codebook_size,) * codebooks
            if isinstance(codebook_size, int)
            else tuple(codebook_size)
        )
        if len(sizes) != codebooks or any(size <= 0 for size in sizes):
            raise ValueError(
                "codebook_size must provide one positive size per codebook."
            )
        if codebook_embeddings is not None:
            if len(codebook_embeddings) != codebooks:
                raise ValueError(
                    "codebook_embeddings must provide one tensor per codebook."
                )
            if any(not torch.is_floating_point(value) for value in codebook_embeddings):
                raise TypeError("codebook_embeddings must be floating point.")
            if any(value.dim() != 2 for value in codebook_embeddings):
                raise ValueError(
                    "each codebook embedding must have shape [size_q, dim]."
                )
            if any(
                value.size(0) != size for value, size in zip(codebook_embeddings, sizes)
            ):
                raise ValueError("codebook embeddings must match codebook sizes.")
            embedding_dim = codebook_embeddings[0].size(-1)
            if any(value.size(-1) != embedding_dim for value in codebook_embeddings):
                raise ValueError(
                    "all codebook embeddings must have the same dimension."
                )
        else:
            embedding_dim = condition_dim

        self.condition_dim = condition_dim
        self.codebooks = codebooks
        self.codebook_sizes = sizes
        self.embedding_dim = embedding_dim
        self.codebook_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in sizes
        )
        if codebook_embeddings is None:
            for module in self.codebook_embeddings:
                embedding = cast(nn.Embedding, cast(object, module))
                nn.init.normal_(embedding.weight, std=embedding_dim**-0.5)
        else:
            with torch.no_grad():
                for index, module in enumerate(self.codebook_embeddings):
                    embedding = cast(nn.Embedding, cast(object, module))
                    embedding.weight.copy_(codebook_embeddings[index])

        self.embedding_projections = nn.ModuleList(
            nn.Identity()
            if embedding_dim == condition_dim
            else nn.Linear(embedding_dim, condition_dim)
            for _ in range(codebooks)
        )
        self.codebook_bos = nn.Parameter(torch.zeros(codebooks, condition_dim))
        self.heads = nn.ModuleList(nn.Linear(condition_dim, size) for size in sizes)

    def _validate_condition(self, condition: Tensor) -> None:
        if condition.dim() != 3 or condition.size(-1) != self.condition_dim:
            raise ValueError("condition must have shape [batch, frame, condition_dim].")

    def _embedding(self, codebook: int, ids: Tensor) -> Tensor:
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise TypeError("acoustic code labels must contain integer ids.")
        if bool((ids < 0).any()) or bool((ids >= self.codebook_sizes[codebook]).any()):
            raise ValueError("acoustic code label is outside the codec codebook.")
        embedding = cast(
            nn.Embedding,
            cast(object, self.codebook_embeddings[codebook]),
        )
        projection = cast(
            nn.Module,
            cast(object, self.embedding_projections[codebook]),
        )
        value = embedding(ids)
        return projection(value)

    def forward(
        self,
        condition: Tensor,
        acoustic_labels: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Return one teacher-forced ``[B, F, K_q]`` tensor per codebook."""
        self._validate_condition(condition)
        if acoustic_labels is not None:
            if acoustic_labels.shape != (
                condition.size(0),
                condition.size(1),
                self.codebooks,
            ):
                raise ValueError("acoustic_labels must have shape [B, F, codebooks].")
            if bool((acoustic_labels < 0).any()):
                raise ValueError("acoustic_labels cannot contain padding values.")

        state = condition
        logits: list[Tensor] = []
        for codebook in range(self.codebooks):
            state = condition + self.codebook_bos[codebook]
            if acoustic_labels is not None and codebook:
                state = state + sum(
                    (
                        self._embedding(previous, acoustic_labels[..., previous])
                        for previous in range(codebook)
                    ),
                    start=torch.zeros_like(condition),
                )
            head = cast(nn.Linear, cast(object, self.heads[codebook]))
            logits.append(head(state))
        return tuple(logits)

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Tensor:
        """Sample codebooks autoregressively while keeping frames parallel."""
        self._validate_condition(condition)
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError(
                "temperature must be positive and top_p must be in (0, 1]."
            )

        previous: list[Tensor] = []
        output: list[Tensor] = []
        for codebook in range(self.codebooks):
            state = condition + self.codebook_bos[codebook]
            if previous:
                state = state + sum(
                    (
                        self._embedding(index, value)
                        for index, value in enumerate(previous)
                    ),
                    start=torch.zeros_like(condition),
                )
            head = cast(nn.Linear, cast(object, self.heads[codebook]))
            logits = head(state) / temperature
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)
            value = torch.distributions.Categorical(logits=logits).sample()
            previous.append(value)
            output.append(value)
        return torch.stack(output, dim=-1)


class SpeechToSpeechRVQModel(SemanticModel):
    """Semantic model composition using a discrete RVQ acoustic decoder."""

    def __init__(
        self,
        config=None,
        runtime_snapshot=None,
        *,
        codebook_embeddings: Sequence[Tensor] | None = None,
    ) -> None:
        super().__init__(config=config, runtime_snapshot=runtime_snapshot)
        sizes = self.runtime.codec.acoustic_codebook_sizes
        backbone_weight = self.backbone.get_input_embeddings().weight
        self.acoustic_decoder = AcousticRVQDecoder(
            self.backbone.config.hidden_size,
            len(sizes),
            sizes,
            codebook_embeddings=codebook_embeddings,
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        acoustic_labels: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        condition = self.target_frame_condition(hidden_states, target_positions)
        return self.acoustic_decoder(condition, acoustic_labels)

    @torch.no_grad()
    def sample_acoustic(self, condition: Tensor, **kwargs: float) -> Tensor:
        return self.acoustic_decoder.generate(condition, **kwargs)
