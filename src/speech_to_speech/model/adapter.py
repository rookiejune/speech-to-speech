"""Configurable hidden-state adapters around audio token interfaces."""

from __future__ import annotations

from torch import Tensor, nn

from ..config import AdapterConfig, AdapterType
from .qwen3 import Qwen3Config, Qwen3MLP


class HiddenAdapter(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states


class LinearAdapter(HiddenAdapter):
    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        *,
        bias: bool,
        like: Tensor,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(
            input_hidden_size,
            output_hidden_size,
            bias=bias,
            device=like.device,
            dtype=like.dtype,
        )
        _init_linear(self.proj)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states)


class QwenMLPAdapter(HiddenAdapter):
    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        *,
        intermediate_size: int | None,
        like: Tensor,
    ) -> None:
        super().__init__()
        hidden_size = output_hidden_size
        self.input_proj = (
            HiddenAdapter()
            if input_hidden_size == hidden_size
            else LinearAdapter(input_hidden_size, hidden_size, bias=False, like=like)
        )
        config = Qwen3Config()
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size or _default_intermediate_size(hidden_size)
        self.mlp = Qwen3MLP(config).to(device=like.device, dtype=like.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.input_proj(hidden_states)
        return self.mlp(hidden_states)


class ResidualHiddenAdapter(HiddenAdapter):
    def __init__(self, adapter: HiddenAdapter) -> None:
        super().__init__()
        self.adapter = adapter

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.adapter(hidden_states)


class AdaptedEmbedding(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        adapter: HiddenAdapter,
        *,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.adapter = adapter
        self.num_embeddings = _num_embeddings(embedding)
        self.embedding_dim = embedding_dim

    @property
    def weight(self) -> Tensor:
        return self.adapter(_embedding_weight(self.embedding))

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.adapter(self.embedding(input_ids))


def hidden_adapter(
    *,
    config: AdapterConfig,
    like: Tensor,
    std: float,
) -> HiddenAdapter:
    del std
    if config.in_features is None or config.out_features is None:
        raise ValueError("adapter in_features and out_features must be set before build.")
    match config.type:
        case AdapterType.IDENTITY:
            if config.in_features != config.out_features:
                raise ValueError("identity adapter requires matching hidden sizes.")
            return HiddenAdapter()
        case AdapterType.LINEAR:
            return LinearAdapter(
                config.in_features,
                config.out_features,
                bias=False,
                like=like,
            )
        case AdapterType.QWEN_MLP:
            adapter = QwenMLPAdapter(
                config.in_features,
                config.out_features,
                intermediate_size=None,
                like=like,
            )
            if config.in_features == config.out_features:
                return ResidualHiddenAdapter(adapter)
            return adapter


def adapted_embedding(
    embedding: nn.Module,
    adapter: HiddenAdapter,
    *,
    embedding_dim: int,
) -> AdaptedEmbedding:
    return AdaptedEmbedding(
        embedding=embedding,
        adapter=adapter,
        embedding_dim=embedding_dim,
    )


def _num_embeddings(embedding: nn.Module) -> int:
    value = getattr(embedding, "num_embeddings", None)
    if not isinstance(value, int):
        raise TypeError("embedding.num_embeddings must be an integer.")
    return value


def _embedding_weight(embedding: nn.Module) -> Tensor:
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError("embedding must expose a tensor weight.")
    return weight


def _init_linear(layer: nn.Linear) -> None:
    if layer.weight.size(0) == layer.weight.size(1):
        nn.init.eye_(layer.weight)
        return
    nn.init.xavier_uniform_(layer.weight)


def _default_intermediate_size(hidden_size: int) -> int:
    return max(1, int(round((8.0 / 3.0) * hidden_size)))


__all__ = [
    "AdaptedEmbedding",
    "HiddenAdapter",
    "LinearAdapter",
    "QwenMLPAdapter",
    "ResidualHiddenAdapter",
    "adapted_embedding",
    "hidden_adapter",
]
