from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn


def _heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    raise RuntimeError(
        "a positive hidden dimension must have an attention head divisor"
    )


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        half = self.hidden_dim // 2
        frequency = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = time.float()[:, None] * frequency[None]
        embedding = torch.cat((angle.cos(), angle.sin()), dim=-1)
        if self.hidden_dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        projection = cast(nn.Linear, self.projection[0])
        return self.projection(embedding.to(dtype=projection.weight.dtype))


def _position(length: int, hidden_dim: int, reference: Tensor) -> Tensor:
    half = hidden_dim // 2
    frequency = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=reference.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angle = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None]
    embedding = torch.cat(
        ((angle * frequency).cos(), (angle * frequency).sin()), dim=-1
    )
    if hidden_dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding.to(dtype=reference.dtype)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_ratio: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
        )
        self.film = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, hidden: Tensor, film: Tensor, mask: Tensor) -> Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = self.film(film).chunk(6, dim=-1)
        normalized = self.attention_norm(hidden)
        normalized = normalized * (1 + attention_scale) + attention_shift
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )[0]
        hidden = hidden + attention_gate * attended
        normalized = self.ffn_norm(hidden)
        normalized = normalized * (1 + ffn_scale) + ffn_shift
        hidden = hidden + ffn_gate * self.ffn(normalized)
        return hidden.masked_fill(~mask[..., None], 0)


class AcousticDiT(nn.Module):
    """Frame transformer with timestep and frame-aligned FiLM conditioning."""

    def __init__(
        self,
        condition_dim: int,
        latent_dim: int,
        *,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
        repa_dim: int | None = None,
        repa_layer: int | None = None,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or latent_dim <= 0:
            raise ValueError("condition_dim and latent_dim must be positive")
        if layers <= 0 or heads <= 0 or ffn_ratio <= 0:
            raise ValueError("DiT depth, heads, and FFN ratio must be positive")
        hidden_dim = condition_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("DiT hidden dimension must be positive")
        repa_dim = condition_dim if repa_dim is None else repa_dim
        repa_layer = (layers + 1) // 2 if repa_layer is None else repa_layer
        if repa_dim <= 0 or not 1 <= repa_layer <= layers:
            raise ValueError("REPA dimension must be positive and layer must exist")

        self.latent_dim = latent_dim
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.time = TimeEmbedding(hidden_dim)
        self.condition = nn.Linear(condition_dim, hidden_dim)
        attention_heads = _heads(hidden_dim, heads)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, attention_heads, ffn_ratio) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)
        self.repa = nn.Linear(hidden_dim, repa_dim)
        self.repa_layer = repa_layer

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        velocity, _ = self._forward(
            x_t,
            t,
            condition=condition,
            mask=mask,
        )
        return velocity

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        velocity, representation = self._forward(
            x_t,
            t,
            condition=condition,
            mask=mask,
        )
        return velocity, self.repa(representation)

    def _forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if x_t.shape[:2] != condition.shape[:2] or x_t.size(-1) != self.latent_dim:
            raise ValueError("acoustic latent and condition shapes do not align")
        if t.shape != (x_t.size(0),):
            raise ValueError("flow time must have shape [batch]")
        if mask is None:
            mask = torch.ones(x_t.shape[:2], dtype=torch.bool, device=x_t.device)
        if mask.shape != x_t.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("acoustic frame mask must be boolean with shape [B, F]")
        if not bool(mask.any(dim=1).all()):
            raise ValueError(
                "each acoustic sequence must contain at least one valid frame"
            )

        hidden = self.input(x_t)
        hidden = hidden + _position(hidden.size(1), hidden.size(2), hidden)[None]
        hidden = hidden.masked_fill(~mask[..., None], 0)
        film = self.condition(condition) + self.time(t)[:, None]
        representation = hidden
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, film, mask)
            if index + 1 == self.repa_layer:
                representation = hidden
        velocity = self.output(self.output_norm(hidden))
        velocity = velocity.masked_fill(~mask[..., None], 0)
        return velocity, representation
