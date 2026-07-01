from __future__ import annotations

from typing import cast

from anytrain.idspace import IdSpaceEmbedding, Modality
from torch import nn

from ..config import ModelConfig, ModelTrainMode
from .types import AudioBoundary


def configure_trainable(
    *,
    qwen3: nn.Module,
    embedding: nn.Module,
    output_adapter: nn.Module,
    dit: nn.Module | None,
    acoustic_condition_adapter: nn.Module,
    acoustic_condition_encoder: nn.Module | None,
    config: ModelConfig,
    peft_applied: bool,
) -> None:
    if config.train_mode is ModelTrainMode.ACOUSTIC_ONLY:
        _set_trainable(qwen3, False)
        _set_acoustic_only_embedding_trainable(cast(IdSpaceEmbedding, embedding))
        _set_trainable(output_adapter, True)
        if dit is None:
            raise ValueError("model.train_mode=acoustic_only requires an acoustic decoder.")
        _set_trainable(dit, config.acoustic.train)
        _set_trainable(acoustic_condition_adapter, config.acoustic.train)
        if acoustic_condition_encoder is not None:
            _set_trainable(acoustic_condition_encoder, config.acoustic.train)
        return

    if config.backbone.train:
        _set_trainable(qwen3, True)
    else:
        _set_trainable(qwen3, False)
    if peft_applied and config.backbone.lora.enabled:
        _set_lora_trainable(qwen3)
    _set_embedding_trainable(cast(IdSpaceEmbedding, embedding), config)
    _set_trainable(output_adapter, True)
    if dit is not None:
        _set_trainable(dit, config.acoustic.train)
    _set_trainable(acoustic_condition_adapter, config.acoustic.train)
    if acoustic_condition_encoder is not None:
        _set_trainable(acoustic_condition_encoder, config.acoustic.train)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _set_lora_trainable(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True


def _set_embedding_trainable(embedding: IdSpaceEmbedding, config: ModelConfig) -> None:
    embedding.modality_embeddings[Modality.TEXT.value].requires_grad_(
        config.token_space.train_text_embedding
    )
    embedding.modality_embeddings[Modality.AUDIO.value].requires_grad_(
        config.token_space.train_audio_embedding
    )
    for name, parameter in embedding.special_embeddings.items():
        parameter.requires_grad = name in {
            AudioBoundary.BOA.value,
            AudioBoundary.EOA.value,
        } and config.token_space.train_audio_special_tokens


def _set_acoustic_only_embedding_trainable(embedding: IdSpaceEmbedding) -> None:
    embedding.modality_embeddings[Modality.TEXT.value].requires_grad_(False)
    embedding.modality_embeddings[Modality.AUDIO.value].requires_grad_(True)
    for parameter in embedding.special_embeddings.values():
        parameter.requires_grad = False
