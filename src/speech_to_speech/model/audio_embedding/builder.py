"""Build audio token embeddings from token-space config."""

from __future__ import annotations

from torch import Tensor, nn

from ...config import AudioEmbeddingType, ModelConfig
from .lookup import lookup_audio_embedding
from .semantic import SemanticEmbeddingConfig, semantic_audio_embedding


def audio_embedding(
    vocab_size: int,
    hidden_size: int,
    *,
    like: Tensor,
    std: float,
    bpe: object | None = None,
    config: ModelConfig | None = None,
) -> nn.Module:
    model_config = config or ModelConfig()
    match model_config.token_space.audio_embedding_type:
        case AudioEmbeddingType.LOOKUP:
            return lookup_audio_embedding(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                like=like,
                std=std,
            )
        case AudioEmbeddingType.SEMANTIC_COMPOSITION:
            if bpe is None:
                raise ValueError("semantic audio embedding requires a LongCat BPE tokenizer.")
            return semantic_audio_embedding(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                bpe=bpe,
                like=like,
                std=std,
                config=SemanticEmbeddingConfig(
                    codebook_size=model_config.token_space.semantic_codebook_size,
                    rope_base=model_config.token_space.semantic_rope_base,
                    shift_rank=model_config.token_space.semantic_shift_rank,
                ),
            )


__all__ = ["audio_embedding"]
