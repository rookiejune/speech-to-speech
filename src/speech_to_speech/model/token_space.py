from __future__ import annotations

from typing import cast

import torch
from anytrain.idspace import IdSpaceEmbedding, Modality
from torch import Tensor, nn

from ..config import ModelConfig
from ..types import AudioBoundary, SpecialToken


def special_token_ids(tokenizer: object) -> dict[str, int]:
    return {member.value: _token_id(tokenizer, member.value) for member in SpecialToken}


def audio_special_embeddings(
    hidden_size: int,
    *,
    like: Tensor,
) -> nn.ParameterDict:
    embeddings = nn.ParameterDict()
    for name in (AudioBoundary.BOA.value, AudioBoundary.EOA.value):
        parameter = nn.Parameter(torch.empty(hidden_size, device=like.device, dtype=like.dtype))
        nn.init.normal_(parameter)
        embeddings[name] = parameter
    return embeddings


def text_embedding(model: nn.Module) -> nn.Embedding | IdSpaceEmbedding:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embed = get_input_embeddings()
        if isinstance(embed, nn.Embedding | IdSpaceEmbedding):
            return embed

    embed = getattr(model, "embed_tokens", None)
    if isinstance(embed, nn.Embedding | IdSpaceEmbedding):
        return embed

    child = getattr(model, "model", None)
    if isinstance(child, nn.Module):
        embed = getattr(child, "embed_tokens", None)
        if isinstance(embed, nn.Embedding | IdSpaceEmbedding):
            return embed

    raise AttributeError("qwen3 model must expose embed_tokens.")


def hidden_size(module: nn.Module | None) -> int | None:
    if module is None:
        return None
    config = getattr(module, "config", None)
    value = getattr(config, "hidden_size", None)
    if isinstance(value, int):
        return value
    return None


def set_text_embedding(model: nn.Module, embedding: IdSpaceEmbedding) -> None:
    set_input_embeddings = getattr(model, "set_input_embeddings", None)
    if callable(set_input_embeddings):
        set_input_embeddings(embedding)
        return

    if hasattr(model, "embed_tokens"):
        model.embed_tokens = embedding
        return

    child = getattr(model, "model", None)
    if isinstance(child, nn.Module) and hasattr(child, "embed_tokens"):
        child.embed_tokens = embedding
        return

    raise AttributeError("qwen3 model must expose embed_tokens.")


def configure_trainable(
    *,
    qwen3: nn.Module,
    embedding: nn.Module,
    dit: nn.Module | None,
    acoustic_condition_proj: nn.Module,
    config: ModelConfig,
    peft_applied: bool,
) -> None:
    if config.train_backbone:
        _set_trainable(qwen3, True)
    else:
        _set_trainable(qwen3, False)
    if peft_applied and config.lora.enabled:
        _set_lora_trainable(qwen3)
    _set_embedding_trainable(cast(IdSpaceEmbedding, embedding), config)
    if dit is not None:
        _set_trainable(dit, config.train_dit)
    _set_trainable(acoustic_condition_proj, config.train_dit)


def _token_id(tokenizer: object, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = convert(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id

    encode = getattr(tokenizer, "encode")
    ids = encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"special token {token!r} must map to exactly one token id.")
    return int(ids[0])


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _set_lora_trainable(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True


def _set_embedding_trainable(embedding: IdSpaceEmbedding, config: ModelConfig) -> None:
    embedding.modality_embeddings[Modality.TEXT.value].requires_grad_(
        config.train_text_embedding
    )
    embedding.modality_embeddings[Modality.AUDIO.value].requires_grad_(
        config.train_audio_embedding
    )
    for name, parameter in embedding.special_embeddings.items():
        parameter.requires_grad = name in {
            AudioBoundary.BOA.value,
            AudioBoundary.EOA.value,
        } and config.train_audio_special_tokens
