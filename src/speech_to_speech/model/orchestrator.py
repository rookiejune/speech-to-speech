from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from anytrain.idspace import IdSpace, IdSpaceEmbedding, Modality, ModalityBlock
from anytrain.tokenizer import IntBPE
from torch import Tensor, nn
from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import BPEConfig, LoRAConfig, ModelConfig
from ..runtime import longcat_tokenizer, qwen3_tokenizer
from ..types import (
    AcousticCondition,
    AcousticConditionGeneration,
    AcousticFeatureGenerator,
    AudioBoundary,
    CausalLMBatch,
    GenerationBatch,
    IGNORE_INDEX,
    SemanticBPE,
    WaveformCodec,
    WaveformGeneration,
)
from .acoustic import (
    acoustic_condition,
    continuous_flow_loss,
    null_acoustic_condition,
    validate_acoustic_features,
)
from .diagonal import DiagonalSample, diagonal_flow_sample
from .generation import Generator
from .qwen3 import Qwen3Config, Qwen3Model
from .token_space import (
    audio_special_embeddings,
    configure_trainable,
    hidden_size,
    set_text_embedding,
    special_token_ids,
    text_embedding,
)


def _qwen3_config():
    config = Qwen3Config()
    config.num_hidden_layers = 36
    config.num_key_value_heads = 8
    config.intermediate_size = 12288
    return config


def _dit_config():
    config = Qwen3Config()
    config.num_hidden_layers = 8
    config.hidden_size = 1024  # LongCat Acoustic dim
    config.intermediate_size = 3072  # 3 x hidden_size
    return config


def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # QLoRA 常用 NF4
        bnb_4bit_use_double_quant=True,  # double quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # 如果显卡不支持 bf16，可改 fp16
    )


def _lora_config(config: LoRAConfig | None = None):
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


class Orchestrator(nn.Module):
    """Holds decoder-only semantic and optional acoustic models for task-side training."""

    def __init__(
        self,
        *,
        qwen3: nn.Module | None = None,
        dit: nn.Module | None = None,
        qwen3_config: Qwen3Config | None = None,
        bnb_config: BitsAndBytesConfig | None = None,
        lora_config: object | None = None,
        model_config: ModelConfig | None = None,
        bpe_config: BPEConfig | None = None,
        tokenizer: object | None = None,
        bpe_vocab_size: int | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        model_config = model_config or ModelConfig()

        peft_applied = False
        if qwen3 is not None:
            self.qwen3 = qwen3
        elif pretrained:
            quantization_config = bnb_config
            if quantization_config is None and model_config.load_in_4bit:
                quantization_config = _bnb_config()
            self.qwen3 = Qwen3Model.from_pretrained(
                model_config.model_name_or_path,
                trust_remote_code=model_config.trust_remote_code,
                quantization_config=quantization_config,
            )
        else:
            self.qwen3 = Qwen3Model(qwen3_config or _qwen3_config())

        if pretrained and model_config.lora.enabled:
            from peft import get_peft_model

            self.qwen3 = get_peft_model(
                self.qwen3,
                lora_config or _lora_config(model_config.lora),
            )
            peft_applied = True
            self.qwen3.print_trainable_parameters()

        tokenizer = tokenizer or qwen3_tokenizer(model_config)
        special_ids = special_token_ids(tokenizer)
        qwen_vocab_size = cast(int, self.qwen3.config.vocab_size)  # type: ignore[attr-defined]
        qwen_hidden_size = cast(int, self.qwen3.config.hidden_size)  # type: ignore[attr-defined]
        audio_modality_vocab_size = bpe_vocab_size or longcat_tokenizer(
            bpe_config or BPEConfig()
        ).vocab_size
        audio_start = qwen_vocab_size + len(AudioBoundary)
        audio_special_ids = {
            AudioBoundary.BOA.value: qwen_vocab_size,
            AudioBoundary.EOA.value: qwen_vocab_size + 1,
        }
        all_special_token_ids = {**special_ids, **audio_special_ids}
        text_embed = text_embedding(self.qwen3)

        embedding = IdSpaceEmbedding(
            space=IdSpace(
                special_token_ids=all_special_token_ids,
                modality_blocks=(
                    ModalityBlock(
                        modality=Modality.TEXT,
                        start=0,
                        vocab_size=qwen_vocab_size,
                    ),
                    ModalityBlock(
                        modality=Modality.AUDIO,
                        start=audio_start,
                        vocab_size=audio_modality_vocab_size,
                    ),
                ),
            ),
            dim=qwen_hidden_size,
            special_embeddings=audio_special_embeddings(
                qwen_hidden_size,
                like=text_embed.weight,
            ),
            modality_embeddings={
                Modality.TEXT: text_embed,
            },
            init_missing_special_embeddings=False,
        )
        set_text_embedding(self.qwen3, embedding)
        self.dit = dit
        dit_hidden_size = hidden_size(dit)
        self.acoustic_condition_proj = (
            nn.Identity()
            if dit_hidden_size is None or dit_hidden_size == qwen_hidden_size
            else nn.Linear(qwen_hidden_size, dit_hidden_size)
        )

        self.lm_head = embedding.as_head(
            special_tokens=(
                AudioBoundary.BOA.value,
                AudioBoundary.EOA.value,
            ),
            modalities=(Modality.AUDIO,),
        )
        configure_trainable(
            qwen3=self.qwen3,
            embedding=self.embed_tokens,
            dit=self.dit,
            acoustic_condition_proj=self.acoustic_condition_proj,
            config=model_config,
            peft_applied=peft_applied,
        )

    @property
    def embed_tokens(self) -> IdSpaceEmbedding:
        return cast(IdSpaceEmbedding, text_embedding(self.qwen3))

    def forward(self, batch: CausalLMBatch) -> CausalLMOutputWithPast:
        inputs_embeds = self.embed_tokens(batch.input_ids)
        outputs = self.qwen3(
            attention_mask=batch.attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        positions = _loss_positions(batch)
        selected_hidden = hidden_states[positions[:, 0], positions[:, 1]]
        labels = batch.labels[positions[:, 0], positions[:, 1]]
        target = self.lm_head.to_head_ids(labels)
        logits = self.lm_head(selected_hidden)
        loss = F.cross_entropy(logits.float(), target)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def acoustic_condition(
        self,
        batch: CausalLMBatch,
        bpe: IntBPE,
        *,
        hidden_states: Tensor | None = None,
    ) -> AcousticCondition:
        if hidden_states is None:
            inputs_embeds = self.embed_tokens(batch.input_ids)
            outputs = self.qwen3(
                attention_mask=batch.attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            hidden_states = outputs.last_hidden_state
        return acoustic_condition(
            batch=batch,
            hidden_states=hidden_states,
            embedding=self.embed_tokens,
            bpe=bpe,
        )

    def acoustic_flow_loss(
        self,
        batch: CausalLMBatch,
        bpe: IntBPE,
        target_features: Tensor,
        *,
        hidden_states: Tensor | None = None,
        target_mask: Tensor | None = None,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
        acoustic_condition: Tensor | None = None,
    ) -> Tensor:
        if self.dit is None:
            raise RuntimeError("acoustic_flow_loss requires a DiT acoustic decoder.")

        condition = self.acoustic_condition(batch, bpe, hidden_states=hidden_states)
        target_features = target_features.to(
            device=condition.hidden_states.device,
            dtype=condition.hidden_states.dtype,
        )
        validate_acoustic_features(
            target_features,
            condition.mask,
            target_mask=target_mask,
        )

        last_hidden_state = self.acoustic_condition_proj(condition.hidden_states)
        if target_features.size(-1) != last_hidden_state.size(-1):
            raise ValueError("target_features last dimension must match DiT hidden size.")
        target_features = target_features.to(dtype=last_hidden_state.dtype)
        if noise is None:
            noise = torch.randn_like(target_features)
        else:
            noise = noise.to(device=target_features.device, dtype=target_features.dtype)
        if noise.shape != target_features.shape:
            raise ValueError("noise and target_features must have the same shape.")

        if acoustic_condition is None:
            acoustic_condition = null_acoustic_condition(self.dit, target_features)
        else:
            acoustic_condition = acoustic_condition.to(
                device=target_features.device,
                dtype=target_features.dtype,
            )

        return continuous_flow_loss(
            self.dit,
            target_features,
            x_0=noise,
            timesteps=timesteps,
            last_hidden_state=last_hidden_state,
            acoustic_condition=acoustic_condition,
            mask=condition.mask,
        )

    @torch.no_grad()
    def generate_acoustic_condition(
        self,
        batch: GenerationBatch,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_token_ids: bool = False,
    ) -> AcousticConditionGeneration:
        return Generator(
            qwen3=self.qwen3,
            embed_tokens=self.embed_tokens,
            lm_head=self.lm_head,
        ).acoustic_condition(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_token_ids=return_token_ids,
        )

    @torch.no_grad()
    def generate_waveform(
        self,
        batch: GenerationBatch,
        *,
        bpe: SemanticBPE,
        codec: WaveformCodec,
        acoustic_generator: AcousticFeatureGenerator | None,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> WaveformGeneration:
        return Generator(
            qwen3=self.qwen3,
            embed_tokens=self.embed_tokens,
            lm_head=self.lm_head,
        ).waveform(
            batch,
            bpe=bpe,
            codec=codec,
            acoustic_generator=acoustic_generator,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @torch.no_grad()
    def diagonal_acoustic_sample(
        self,
        x_0: Tensor,
        *,
        last_hidden_state: Tensor,
        acoustic_condition: Tensor,
        mask: Tensor,
        num_steps: int,
        chunk_size: int,
        wave_stride: int = 1,
    ) -> DiagonalSample:
        if self.dit is None:
            raise RuntimeError("diagonal_acoustic_sample requires a DiT acoustic decoder.")
        return diagonal_flow_sample(
            self.dit,
            x_0,
            last_hidden_state=last_hidden_state,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=num_steps,
            chunk_size=chunk_size,
            wave_stride=wave_stride,
        )


def _loss_positions(batch: CausalLMBatch) -> Tensor:
    if isinstance(batch.logits_to_keep, Tensor):
        positions = batch.logits_to_keep.to(device=batch.labels.device, dtype=torch.long)
        if positions.dim() != 2 or positions.size(-1) != 2:
            raise ValueError("logits_to_keep tensor must have shape (n, 2).")
        return positions
    if batch.logits_to_keep <= 0:
        raise ValueError("logits_to_keep must be positive.")
    mask = batch.labels.ne(IGNORE_INDEX)
    if not bool(mask.any()):
        raise ValueError("labels must contain at least one supervised token.")
    if batch.logits_to_keep >= mask.size(1):
        return mask.nonzero(as_tuple=False)
    keep = mask.cumsum(dim=1) > (mask.sum(dim=1, keepdim=True) - batch.logits_to_keep).clamp_min(0)
    return (mask & keep).nonzero(as_tuple=False)
