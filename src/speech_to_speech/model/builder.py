"""Constructs speech-to-speech model submodules from runtime configuration.

This module owns loading/configuring Qwen3, optional DiT acoustic decoder,
shared text/audio token space, adapters, and trainability flags.
The `Orchestrator` consumes the resulting modules and keeps the runtime
forward/generation/acoustic contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import warnings

import torch
from anytrain.idspace import IdSpace, IdSpaceEmbedding, Modality
from torch import nn
from transformers import BitsAndBytesConfig

from ..config import (
    AcousticAttentionMode,
    AdapterConfig,
    BPEConfig,
    DiTModelConfig,
    LoRAConfig,
    ModelConfig,
)
from ..runtime import longcat_tokenizer, qwen3_longcat_idspace, qwen3_tokenizer
from .adapter import HiddenAdapter, adapted_embedding, hidden_adapter
from .acoustic import ConditionEncoder, condition_encoder_config
from .DiT.model import DiT
from .audio_embedding import audio_embedding
from .qwen3 import Qwen3Config, Qwen3Model
from .token_space import (
    audio_special_embeddings,
    hidden_size,
    set_text_embedding,
    text_embedding,
)
from .trainable import configure_trainable
from ..types.model import AudioBoundary


@dataclass(frozen=True)
class OrchestratorComponents:
    qwen3: nn.Module
    dit: nn.Module | None
    output_adapter: HiddenAdapter
    acoustic_condition_adapter: HiddenAdapter
    acoustic_condition_encoder: ConditionEncoder | None


def build_orchestrator_components(
    *,
    qwen3: nn.Module | None = None,
    dit: nn.Module | None = None,
    qwen3_config: Qwen3Config | None = None,
    bnb_config: BitsAndBytesConfig | None = None,
    lora_config: object | None = None,
    model_config: ModelConfig | None = None,
    bpe_config: BPEConfig | None = None,
    bpe: object | None = None,
    tokenizer: object | None = None,
    bpe_vocab_size: int | None = None,
    space: IdSpace | None = None,
    qwen3_pretrained: bool = True,
) -> OrchestratorComponents:
    model_config = model_config or ModelConfig()
    qwen3, peft_applied = _build_qwen3(
        qwen3=qwen3,
        qwen3_config=qwen3_config,
        bnb_config=bnb_config,
        lora_config=lora_config,
        model_config=model_config,
        qwen3_pretrained=qwen3_pretrained,
    )
    dit = _build_dit(dit=dit, model_config=model_config)
    token_space = _build_token_space(
        qwen3=qwen3,
        dit=dit,
        model_config=model_config,
        bpe_config=bpe_config,
        bpe=bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe_vocab_size,
        space=space,
    )
    configure_trainable(
        qwen3=qwen3,
        embedding=token_space.embedding,
        output_adapter=token_space.output_adapter,
        dit=dit,
        acoustic_condition_adapter=token_space.acoustic_condition_adapter,
        acoustic_condition_encoder=token_space.acoustic_condition_encoder,
        config=model_config,
        peft_applied=peft_applied,
    )
    return OrchestratorComponents(
        qwen3=qwen3,
        dit=dit,
        output_adapter=token_space.output_adapter,
        acoustic_condition_adapter=token_space.acoustic_condition_adapter,
        acoustic_condition_encoder=token_space.acoustic_condition_encoder,
    )


def dit_config(
    config: DiTModelConfig | None = None,
    *,
    attention_mode: AcousticAttentionMode = AcousticAttentionMode.CAUSAL,
) -> Qwen3Config:
    model_config = config or DiTModelConfig()
    qwen_config = Qwen3Config()
    qwen_config.num_hidden_layers = model_config.num_hidden_layers
    qwen_config.hidden_size = model_config.hidden_size
    qwen_config.intermediate_size = model_config.intermediate_size
    if model_config.num_attention_heads is not None:
        qwen_config.num_attention_heads = model_config.num_attention_heads
    if model_config.num_key_value_heads is not None:
        qwen_config.num_key_value_heads = model_config.num_key_value_heads
    qwen_config.attention_mode = attention_mode
    qwen_config.norm_time = model_config.norm_time
    qwen_config.norm_hidden = model_config.norm_hidden
    qwen_config.norm_acoustic = model_config.norm_acoustic
    return qwen_config


@dataclass(frozen=True)
class _TokenSpaceComponents:
    embedding: IdSpaceEmbedding
    output_adapter: HiddenAdapter
    acoustic_condition_adapter: HiddenAdapter
    acoustic_condition_encoder: ConditionEncoder | None


def _build_qwen3(
    *,
    qwen3: nn.Module | None,
    qwen3_config: Qwen3Config | None,
    bnb_config: BitsAndBytesConfig | None,
    lora_config: object | None,
    model_config: ModelConfig,
    qwen3_pretrained: bool,
) -> tuple[nn.Module, bool]:
    peft_applied = False
    if qwen3 is not None:
        return qwen3, peft_applied

    if model_config.backbone.lora.enabled and not qwen3_pretrained:
        warnings.warn(
            "model.backbone.lora.enabled requires qwen3_pretrained=True; "
            "randomly initialized Qwen3 will be built without LoRA.",
            UserWarning,
            stacklevel=2,
        )

    if qwen3_pretrained:
        quantization_config = bnb_config
        if quantization_config is None and model_config.backbone.load_in_4bit:
            quantization_config = _bnb_config()
        model = Qwen3Model.from_pretrained(
            model_config.backbone.model_name_or_path,
            trust_remote_code=model_config.backbone.trust_remote_code,
            quantization_config=quantization_config,
        )
    else:
        model = Qwen3Model(qwen3_config or _qwen3_config())

    if qwen3_pretrained and model_config.backbone.lora.enabled:
        from peft import get_peft_model

        model = get_peft_model(
            model,
            lora_config or _lora_config(model_config.backbone.lora),
        )
        peft_applied = True
        model.print_trainable_parameters()

    return model, peft_applied


def _build_dit(
    *,
    dit: nn.Module | None,
    model_config: ModelConfig,
) -> nn.Module | None:
    if dit is not None:
        return dit
    if not model_config.acoustic.enabled:
        return None
    return DiT(
        dit_config(
            model_config.acoustic.dit,
            attention_mode=model_config.acoustic.attention_mode,
        )
    )


def _build_token_space(
    *,
    qwen3: nn.Module,
    dit: nn.Module | None,
    model_config: ModelConfig,
    bpe_config: BPEConfig | None,
    bpe: object | None,
    tokenizer: object | None,
    bpe_vocab_size: int | None,
    space: IdSpace | None,
) -> _TokenSpaceComponents:
    tokenizer = tokenizer or qwen3_tokenizer(model_config)
    qwen_vocab_size = cast(int, qwen3.config.vocab_size)  # type: ignore[attr-defined]
    qwen_hidden_size = cast(int, qwen3.config.hidden_size)  # type: ignore[attr-defined]
    if bpe_vocab_size is None or model_config.token_space.audio_embedding_type.requires_bpe:
        bpe = bpe or longcat_tokenizer(bpe_config or BPEConfig())
    audio_modality_vocab_size = bpe_vocab_size or cast(object, bpe).vocab_size
    space = space or qwen3_longcat_idspace(
        tokenizer=tokenizer,
        bpe_vocab_size=audio_modality_vocab_size,
        config=model_config,
        qwen_vocab_size=qwen_vocab_size,
    )
    _validate_model_idspace(
        space,
        qwen_vocab_size=qwen_vocab_size,
        audio_vocab_size=audio_modality_vocab_size,
    )
    text_embed = text_embedding(qwen3)
    audio_init_std = _embedding_init_std(qwen3.config)  # type: ignore[attr-defined]
    audio_embed = audio_embedding(
        audio_modality_vocab_size,
        qwen_hidden_size,
        like=text_embed.weight,
        std=audio_init_std,
        bpe=bpe,
        config=model_config,
    )
    input_adapter_config = adapter_config(
        model_config.token_space.input_adapter,
        in_features=qwen_hidden_size,
        out_features=qwen_hidden_size,
    )
    input_adapter = hidden_adapter(
        config=input_adapter_config,
        like=text_embed.weight,
        std=audio_init_std,
    )
    if type(input_adapter) is not HiddenAdapter:
        audio_embed = adapted_embedding(
            audio_embed,
            input_adapter,
            embedding_dim=qwen_hidden_size,
        )

    embedding = IdSpaceEmbedding(
        space=space,
        dim=qwen_hidden_size,
        special_embeddings=audio_special_embeddings(
            qwen_hidden_size,
            like=text_embed.weight,
            std=audio_init_std,
        ),
        modality_embeddings={
            Modality.TEXT: text_embed,
            Modality.AUDIO: audio_embed,
        },
        init_missing_special_embeddings=False,
    )
    set_text_embedding(qwen3, embedding)
    output_adapter_config = adapter_config(
        model_config.token_space.output_adapter,
        in_features=qwen_hidden_size,
        out_features=qwen_hidden_size,
    )
    output_adapter = hidden_adapter(
        config=output_adapter_config,
        like=text_embed.weight,
        std=audio_init_std,
    )
    dit_hidden_size = hidden_size(dit) or qwen_hidden_size
    acoustic_condition_adapter_config = adapter_config(
        model_config.acoustic.condition_adapter,
        in_features=qwen_hidden_size,
        out_features=dit_hidden_size,
    )
    acoustic_condition_adapter = hidden_adapter(
        config=acoustic_condition_adapter_config,
        like=text_embed.weight,
        std=audio_init_std,
    )
    acoustic_condition_encoder = (
        ConditionEncoder(
            condition_encoder_config(
                model_config.acoustic.condition_encoder,
                hidden_size=dit_hidden_size,
                attention_mode=model_config.acoustic.attention_mode,
            )
        )
        if model_config.acoustic.condition_encoder.enabled
        else None
    )
    return _TokenSpaceComponents(
        embedding=embedding,
        output_adapter=output_adapter,
        acoustic_condition_adapter=acoustic_condition_adapter,
        acoustic_condition_encoder=acoustic_condition_encoder,
    )


def adapter_config(
    config: AdapterConfig,
    *,
    in_features: int,
    out_features: int,
) -> AdapterConfig:
    return AdapterConfig(
        type=config.type,
        in_features=in_features if config.in_features is None else config.in_features,
        out_features=out_features if config.out_features is None else config.out_features,
    )


def _qwen3_config() -> Qwen3Config:
    config = Qwen3Config()
    config.num_hidden_layers = 36
    config.num_key_value_heads = 8
    config.intermediate_size = 12288
    return config


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _lora_config(config: LoRAConfig | None = None) -> object:
    from peft import LoraConfig, TaskType

    config = config or LoRAConfig()
    kwargs = {
        "r": config.rank,
        "lora_alpha": config.alpha,
        "target_modules": list(config.targets),
        "lora_dropout": config.dropout,
        "bias": "none",
    }
    task_type = getattr(TaskType, "FEATURE_EXTRACTION", None)
    if task_type is not None:
        kwargs["task_type"] = task_type
    return LoraConfig(**kwargs)


def _embedding_init_std(config: object) -> float:
    value = getattr(config, "initializer_range", None)
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0.0:
        return float(value)
    return 0.02


def _validate_model_idspace(
    space: IdSpace,
    *,
    qwen_vocab_size: int,
    audio_vocab_size: int,
) -> None:
    text_block = space.modality_block(Modality.TEXT)
    if text_block.start != 0 or text_block.vocab_size != qwen_vocab_size:
        raise ValueError("idspace text block must match qwen vocab size.")
    audio_block = space.modality_block(Modality.AUDIO)
    if audio_block.start != qwen_vocab_size + len(AudioBoundary):
        raise ValueError("idspace audio block must start after qwen and audio boundary ids.")
    if audio_block.vocab_size != audio_vocab_size:
        raise ValueError("idspace audio block must match LongCat BPE vocab size.")
    if space.special_token_id(AudioBoundary.BOA) != qwen_vocab_size:
        raise ValueError("idspace BOA id must match qwen vocab size.")
    if space.special_token_id(AudioBoundary.EOA) != qwen_vocab_size + 1:
        raise ValueError("idspace EOA id must follow BOA.")
