from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import cast
import warnings

import torch
import torch.nn.functional as F
from anytrain.idspace import IdSpace, IdSpaceEmbedding, Modality, ModalityBlock
from anytrain.tokenizer import CodecBPE
from torch import Tensor, nn
from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import AcousticConditionSource, BPEConfig, DiTModelConfig, LoRAConfig, ModelConfig
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
    SemanticGeneration,
    TeacherForcedWaveformGeneration,
    WaveformCodec,
    WaveformGeneration,
)
from .acoustic import (
    AcousticFlowLossStats,
    acoustic_condition,
    acoustic_condition_from_target_audio_embedding,
    continuous_flow_loss,
    continuous_flow_loss_stats,
    null_acoustic_condition,
    pooled_acoustic_condition_from_batch_side,
    validate_acoustic_features,
)
from .diagonal import (
    DiagonalSample,
    causal_window_flow_sample,
    diagonal_flow_sample,
    diagonal_flow_sample_chunks,
    full_sequence_flow_sample,
)
from .DiT.model import DiT, DiTConditionTensors
from .generation import Generator
from .qwen3 import Qwen3Config, Qwen3Model
from .token_space import (
    audio_embedding,
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


def dit_config(config: DiTModelConfig | None = None):
    model_config = config or DiTModelConfig()
    config = Qwen3Config()
    config.num_hidden_layers = model_config.num_hidden_layers
    config.hidden_size = model_config.hidden_size
    config.intermediate_size = model_config.intermediate_size
    if model_config.num_attention_heads is not None:
        config.num_attention_heads = model_config.num_attention_heads
    if model_config.num_key_value_heads is not None:
        config.num_key_value_heads = model_config.num_key_value_heads
    config.attention_mode = model_config.attention_mode
    config.norm_time = model_config.norm_time
    config.norm_hidden = model_config.norm_hidden
    config.norm_acoustic = model_config.norm_acoustic
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


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return fallback


def _embedding_init_std(config: object) -> float:
    value = getattr(config, "initializer_range", None)
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0.0:
        return float(value)
    return 0.02


class AcousticSampler(StrEnum):
    SERIAL = auto()
    DIAGONAL = auto()
    DIAGONAL_BPE = auto()
    CAUSAL_WINDOW = auto()


@dataclass(frozen=True)
class AcousticFlowInputs:
    target_features: Tensor
    noise: Tensor
    last_hidden_state: Tensor
    acoustic_condition: Tensor
    mask: Tensor


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
        qwen3_pretrained: bool = True,
    ) -> None:
        super().__init__()

        model_config = model_config or ModelConfig()
        self.acoustic_condition_drop = model_config.acoustic.condition_dropout
        self.acoustic_condition_source = model_config.acoustic.condition_source

        peft_applied = False
        if qwen3 is not None:
            self.qwen3 = qwen3
        elif qwen3_pretrained:
            quantization_config = bnb_config
            if quantization_config is None and model_config.backbone.load_in_4bit:
                quantization_config = _bnb_config()
            self.qwen3 = Qwen3Model.from_pretrained(
                model_config.backbone.model_name_or_path,
                trust_remote_code=model_config.backbone.trust_remote_code,
                quantization_config=quantization_config,
            )
        else:
            self.qwen3 = Qwen3Model(qwen3_config or _qwen3_config())

        if qwen3_pretrained and model_config.backbone.lora.enabled:
            from peft import get_peft_model

            self.qwen3 = get_peft_model(
                self.qwen3,
                lora_config or _lora_config(model_config.backbone.lora),
            )
            peft_applied = True
            self.qwen3.print_trainable_parameters()

        if dit is not None:
            self.dit = dit
        elif model_config.acoustic.enabled:
            self.dit = DiT(dit_config(model_config.acoustic.dit))
        else:
            self.dit = None

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
        audio_init_std = _embedding_init_std(self.qwen3.config)

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
                std=audio_init_std,
            ),
            modality_embeddings={
                Modality.TEXT: text_embed,
                Modality.AUDIO: audio_embedding(
                    audio_modality_vocab_size,
                    qwen_hidden_size,
                    like=text_embed.weight,
                    std=audio_init_std,
                ),
            },
            init_missing_special_embeddings=False,
        )
        set_text_embedding(self.qwen3, embedding)
        dit_hidden_size = hidden_size(self.dit)
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

    def forward(
        self,
        batch: CausalLMBatch,
        *,
        return_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = self.embed_tokens(batch.input_ids)
        outputs = self.qwen3(
            attention_mask=batch.attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=return_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        positions = _loss_positions(batch)
        selected_hidden = hidden_states[positions[:, 0], positions[:, 1]]
        labels = batch.labels[positions[:, 0], positions[:, 1]]
        target = self.lm_head.to_head_ids(labels)
        logits = self.lm_head(selected_hidden)
        token_loss = F.cross_entropy(logits.float(), target, reduction="none")
        loss = _batch_loss(token_loss, positions[:, 0], batch.input_ids.size(0))
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast | None = None,
    ) -> Tensor:
        """Returns token accuracy over the same supervised positions as semantic loss."""
        if output is None:
            output = self(batch)
        logits = output.logits
        if logits is None:
            raise RuntimeError("model output must include logits.")
        positions = _loss_positions(batch)
        if logits.dim() != 2 or logits.size(0) != positions.size(0):
            raise RuntimeError("model logits must have one row per supervised token.")
        labels = batch.labels[positions[:, 0], positions[:, 1]]
        target = self.lm_head.to_head_ids(labels)
        return logits.detach().argmax(dim=-1).eq(target).float().mean()

    def acoustic_condition(
        self,
        batch: CausalLMBatch,
        bpe: CodecBPE,
        *,
        hidden_states: Tensor | None = None,
    ) -> AcousticCondition:
        if self.acoustic_condition_source is AcousticConditionSource.TARGET_AUDIO_EMBEDDING:
            return acoustic_condition_from_target_audio_embedding(
                batch=batch,
                embedding=self.embed_tokens,
                bpe=bpe,
            )

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
        bpe: CodecBPE,
        target_features: Tensor,
        *,
        hidden_states: Tensor | None = None,
        target_mask: Tensor | None = None,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
        acoustic_condition: Tensor | None = None,
        source_feature_extractor: object | None = None,
    ) -> Tensor:
        inputs = self.acoustic_flow_inputs(
            batch,
            bpe,
            target_features,
            hidden_states=hidden_states,
            target_mask=target_mask,
            noise=noise,
            acoustic_condition=acoustic_condition,
            source_feature_extractor=source_feature_extractor,
        )
        return continuous_flow_loss(
            self._require_dit(),
            inputs.target_features,
            x_0=inputs.noise,
            timesteps=timesteps,
            last_hidden_state=inputs.last_hidden_state,
            acoustic_condition=inputs.acoustic_condition,
            mask=inputs.mask,
        )

    def acoustic_flow_loss_stats(
        self,
        batch: CausalLMBatch,
        bpe: CodecBPE,
        target_features: Tensor,
        *,
        hidden_states: Tensor | None = None,
        target_mask: Tensor | None = None,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
        acoustic_condition: Tensor | None = None,
        source_feature_extractor: object | None = None,
    ) -> AcousticFlowLossStats:
        inputs = self.acoustic_flow_inputs(
            batch,
            bpe,
            target_features,
            hidden_states=hidden_states,
            target_mask=target_mask,
            noise=noise,
            acoustic_condition=acoustic_condition,
            source_feature_extractor=source_feature_extractor,
        )
        return self.acoustic_flow_loss_stats_from_inputs(inputs, timesteps=timesteps)

    def acoustic_flow_loss_stats_from_inputs(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: Tensor | None = None,
    ) -> AcousticFlowLossStats:
        return continuous_flow_loss_stats(
            self._require_dit(),
            inputs.target_features,
            x_0=inputs.noise,
            timesteps=timesteps,
            last_hidden_state=inputs.last_hidden_state,
            acoustic_condition=inputs.acoustic_condition,
            mask=inputs.mask,
        )

    def acoustic_condition_tensors(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: Tensor,
    ) -> DiTConditionTensors:
        dit = self._require_dit()
        condition_tensors = getattr(dit, "condition_tensors", None)
        if not callable(condition_tensors):
            raise TypeError("DiT acoustic decoder must provide condition_tensors().")
        tensors = condition_tensors(
            last_hidden_state=inputs.last_hidden_state,
            timesteps=timesteps,
            acoustic_condition=inputs.acoustic_condition,
        )
        if not isinstance(tensors, DiTConditionTensors):
            raise TypeError("DiT condition_tensors() must return DiTConditionTensors.")
        return tensors

    def _require_dit(self) -> nn.Module:
        if self.dit is None:
            raise RuntimeError("acoustic flow requires a DiT acoustic decoder.")
        return self.dit

    def acoustic_flow_inputs(
        self,
        batch: CausalLMBatch,
        bpe: CodecBPE,
        target_features: Tensor,
        *,
        hidden_states: Tensor | None = None,
        target_mask: Tensor | None = None,
        noise: Tensor | None = None,
        acoustic_condition: Tensor | None = None,
        source_feature_extractor: object | None = None,
    ) -> AcousticFlowInputs:
        if self.dit is None:
            raise RuntimeError("acoustic_flow_loss requires a DiT acoustic decoder.")

        condition = self.acoustic_condition(batch, bpe, hidden_states=hidden_states)
        flow_dtype = _module_dtype(self.dit, condition.hidden_states.dtype)
        target_features = target_features.to(
            device=condition.hidden_states.device,
            dtype=flow_dtype,
        )
        validate_acoustic_features(
            target_features,
            condition.mask,
            target_mask=target_mask,
        )

        condition_hidden = condition.hidden_states
        proj_dtype = _module_dtype(self.acoustic_condition_proj, condition_hidden.dtype)
        condition_hidden = condition_hidden.to(dtype=proj_dtype)
        last_hidden_state = self.acoustic_condition_proj(condition_hidden).to(dtype=flow_dtype)
        if target_features.size(-1) != last_hidden_state.size(-1):
            raise ValueError("target_features last dimension must match DiT hidden size.")
        if noise is None:
            noise = torch.randn_like(target_features)
        else:
            noise = noise.to(device=target_features.device, dtype=target_features.dtype)
        if noise.shape != target_features.shape:
            raise ValueError("noise and target_features must have the same shape.")

        null_condition = null_acoustic_condition(self.dit, target_features)
        source_condition = False
        if acoustic_condition is None and source_feature_extractor is not None:
            if batch.source_audio is None:
                acoustic_condition = null_condition
            else:
                acoustic_condition = pooled_acoustic_condition_from_batch_side(
                    batch.source_audio,
                    feature_extractor=source_feature_extractor,
                    empty_condition=null_condition,
                )
                source_condition = True
        elif acoustic_condition is None:
            acoustic_condition = null_condition
        acoustic_condition = acoustic_condition.to(
            device=target_features.device,
            dtype=target_features.dtype,
        )
        if source_condition:
            acoustic_condition = self._maybe_drop_acoustic_condition(
                acoustic_condition,
                null_condition,
            )
        if acoustic_condition.shape != (target_features.size(0), target_features.size(-1)):
            raise ValueError(
                "acoustic_condition must have shape [batch, target_features dim]."
            )

        return AcousticFlowInputs(
            target_features=target_features,
            noise=noise,
            last_hidden_state=last_hidden_state,
            acoustic_condition=acoustic_condition,
            mask=condition.mask,
        )

    def _maybe_drop_acoustic_condition(
        self,
        condition: Tensor,
        null_condition: Tensor,
    ) -> Tensor:
        probability = self.acoustic_condition_drop
        if not self.training or probability <= 0.0:
            return condition
        if probability >= 1.0:
            drop_mask = torch.ones(
                (condition.size(0), 1),
                device=condition.device,
                dtype=torch.bool,
            )
        else:
            drop_mask = torch.rand(
                (condition.size(0), 1),
                device=condition.device,
            ) < probability
        return torch.where(drop_mask, null_condition, condition)

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
    def generate_semantic(
        self,
        batch: GenerationBatch,
        *,
        bpe: SemanticBPE,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> SemanticGeneration:
        return Generator(
            qwen3=self.qwen3,
            embed_tokens=self.embed_tokens,
            lm_head=self.lm_head,
        ).semantic(
            batch,
            bpe=bpe,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
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
        guidance_scale: float = 1.0,
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
            guidance_scale=guidance_scale,
        )

    def acoustic_feature_generator(
        self,
        *,
        num_steps: int,
        chunk_size: int | None = None,
        left_context_chunks: int | None = None,
        guidance_scale: float = 1.0,
        sampler: AcousticSampler = AcousticSampler.SERIAL,
        acoustic_condition: Tensor | None = None,
    ) -> AcousticFeatureGenerator:
        if self.dit is None:
            raise RuntimeError("acoustic_feature_generator requires a DiT acoustic decoder.")
        if sampler is AcousticSampler.DIAGONAL:
            warnings.warn(
                "AcousticSampler.DIAGONAL is deprecated; use "
                "AcousticSampler.DIAGONAL_BPE for BPE-boundary diagonal generation "
                "or AcousticSampler.SERIAL for full-sequence acoustic generation.",
                DeprecationWarning,
                stacklevel=2,
            )
        return DiTAcousticFeatureGenerator(
            model=self,
            num_steps=num_steps,
            chunk_size=chunk_size,
            left_context_chunks=left_context_chunks,
            guidance_scale=guidance_scale,
            sampler=sampler,
            acoustic_condition=acoustic_condition,
        )

    @torch.no_grad()
    def teacher_forced_waveform(
        self,
        batch: CausalLMBatch,
        *,
        bpe: CodecBPE,
        codec: WaveformCodec,
        acoustic_generator: AcousticFeatureGenerator,
    ) -> TeacherForcedWaveformGeneration:
        condition = self.acoustic_condition(batch, bpe)
        acoustic_features = acoustic_generator(condition)
        validate_acoustic_features(acoustic_features, condition.mask, target_mask=None)
        if not torch.is_floating_point(acoustic_features) or torch.is_complex(acoustic_features):
            raise TypeError("acoustic generator must return floating point features.")
        acoustic_features = acoustic_features.to(device=condition.mask.device)
        audio = _decode_features(codec, condition.semantic_ids, acoustic_features)
        return TeacherForcedWaveformGeneration(
            audio=audio,
            audio_mask=torch.ones(
                audio.shape[:2],
                dtype=torch.bool,
                device=audio.device,
            ),
            semantic_ids=condition.semantic_ids,
            semantic_mask=condition.mask,
            acoustic_features=acoustic_features,
            condition_hidden_states=condition.hidden_states,
        )


def _decode_features(codec: object, semantic_ids: Tensor, acoustic_features: Tensor) -> Tensor:
    decode = getattr(codec, "decode_features", None)
    if not callable(decode):
        raise TypeError("LongCat codec must provide decode_features().")
    audio = decode(semantic_ids, acoustic_features)
    if not isinstance(audio, Tensor):
        raise TypeError("LongCat codec decode_features() must return a Tensor.")
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    if audio.dim() != 3:
        raise ValueError("decoded waveform must have shape [batch, channels, time].")
    return audio.detach().float()


@dataclass(frozen=True)
class DiTAcousticFeatureGenerator:
    model: Orchestrator
    num_steps: int
    chunk_size: int | None = None
    left_context_chunks: int | None = None
    guidance_scale: float = 1.0
    sampler: AcousticSampler = AcousticSampler.SERIAL
    acoustic_condition: Tensor | None = None

    @torch.no_grad()
    def __call__(self, condition: AcousticCondition) -> Tensor:
        if self.model.dit is None:
            raise RuntimeError("DiT acoustic feature generation requires a DiT decoder.")
        hidden = condition.hidden_states
        flow_dtype = _module_dtype(self.model.dit, hidden.dtype)
        projection = self.model.acoustic_condition_proj
        projection_dtype = _module_dtype(projection, hidden.dtype)
        hidden = projection(hidden.to(dtype=projection_dtype)).to(dtype=flow_dtype)
        initial = hidden.new_zeros(hidden.shape)
        if self.acoustic_condition is None:
            acoustic_condition = null_acoustic_condition(self.model.dit, initial)
        else:
            acoustic_condition = self.acoustic_condition.to(
                device=hidden.device,
                dtype=projection_dtype,
            )
            if acoustic_condition.size(-1) != hidden.size(-1):
                acoustic_condition = projection(acoustic_condition).to(dtype=flow_dtype)
            else:
                acoustic_condition = acoustic_condition.to(dtype=flow_dtype)
            if acoustic_condition.shape != (hidden.size(0), hidden.size(-1)):
                raise ValueError(
                    "acoustic_condition must have shape [batch, acoustic feature dim]."
                )
        chunk_size = self.chunk_size or hidden.size(1)
        sample_kwargs = {
            "last_hidden_state": hidden,
            "acoustic_condition": acoustic_condition,
            "mask": condition.mask.to(device=hidden.device, dtype=torch.bool),
            "num_steps": self.num_steps,
            "chunk_size": chunk_size,
            "guidance_scale": self.guidance_scale,
        }
        match self.sampler:
            case AcousticSampler.SERIAL:
                sample = full_sequence_flow_sample(
                    self.model.dit,
                    initial,
                    last_hidden_state=hidden,
                    acoustic_condition=acoustic_condition,
                    mask=condition.mask.to(device=hidden.device, dtype=torch.bool),
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
            case AcousticSampler.DIAGONAL:
                sample = diagonal_flow_sample(self.model.dit, initial, **sample_kwargs)
            case AcousticSampler.DIAGONAL_BPE:
                chunk_lengths = _single_chunk_lengths(condition)
                active_frames = sum(chunk_lengths)
                sample = diagonal_flow_sample_chunks(
                    self.model.dit,
                    initial[:, :active_frames],
                    chunk_lengths=chunk_lengths,
                    last_hidden_state=hidden[:, :active_frames],
                    acoustic_condition=acoustic_condition,
                    mask=condition.mask[:, :active_frames].to(
                        device=hidden.device,
                        dtype=torch.bool,
                    ),
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
                final = initial.clone()
                final[:, :active_frames] = sample.final
                return final
            case AcousticSampler.CAUSAL_WINDOW:
                sample = causal_window_flow_sample(
                    self.model.dit,
                    initial,
                    left_context_chunks=_left_context_chunks(
                        self.left_context_chunks,
                        frame_count=hidden.size(1),
                        chunk_size=chunk_size,
                    ),
                    **sample_kwargs,
                )
            case _:
                raise ValueError(f"unsupported acoustic sampler: {self.sampler}")
        return sample.final


def _left_context_chunks(
    value: int | None,
    *,
    frame_count: int,
    chunk_size: int,
) -> int:
    if value is not None:
        return value
    return max(0, (frame_count + chunk_size - 1) // chunk_size - 1)


def _single_chunk_lengths(condition: AcousticCondition) -> tuple[int, ...]:
    lengths = condition.chunk_lengths
    if lengths is None:
        raise ValueError("BPE diagonal acoustic generation requires condition chunk_lengths.")
    if len(lengths) != 1:
        raise ValueError("BPE diagonal acoustic generation currently requires batch size 1.")
    if sum(lengths[0]) != int(condition.mask[0].sum().detach().cpu()):
        raise ValueError("condition chunk_lengths must sum to active frame count.")
    if bool(condition.mask[0, : sum(lengths[0])].logical_not().any()):
        raise ValueError("BPE diagonal acoustic generation requires contiguous active frames.")
    return lengths[0]


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


def _batch_loss(token_loss: Tensor, batch_index: Tensor, batch_size: int) -> Tensor:
    loss = token_loss.new_zeros(batch_size)
    counts = token_loss.new_zeros(batch_size)
    loss.scatter_add_(0, batch_index, token_loss)
    counts.scatter_add_(0, batch_index, torch.ones_like(token_loss))
    if not bool(counts.gt(0).all()):
        raise ValueError("each batch row must contain at least one supervised token.")
    return loss / counts
