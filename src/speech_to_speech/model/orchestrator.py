from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import warnings

import torch
import torch.nn.functional as F
from anytrain.idspace import IdSpace, IdSpaceEmbedding
from anytrain.tokenizer import CodecBPE
from torch import Tensor, nn
from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import (
    AcousticConditionSource,
    BPEConfig,
    ModelConfig,
)
from ..datamodule.types import CausalLMBatch, GenerationBatch, IGNORE_INDEX
from .types import (
    AcousticCondition,
    AcousticConditionGeneration,
    AcousticFeatureGenerator,
    SemanticBPE,
    SemanticGeneration,
    TeacherForcedWaveformGeneration,
    WaveformCodec,
    WaveformGeneration,
)
from ._module import module_dtype
from .acoustic import (
    AcousticFlowLossStats,
    AcousticSampler,
    DiagonalSample,
    DiTAcousticFeatureGenerator,
    acoustic_condition,
    acoustic_condition_from_target_audio_embedding,
    continuous_flow_loss,
    continuous_flow_loss_stats,
    diagonal_flow_sample,
    null_acoustic_condition,
    pooled_acoustic_condition_from_batch_side,
    validate_acoustic_features,
)
from .builder import build_orchestrator_components
from .DiT.model import DiTConditionTensors
from .qwen3 import Qwen3Config
from .semantic import Generator, batch_loss, loss_positions, loss_weights
from .token_space import text_embedding


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
        bpe: object | None = None,
        tokenizer: object | None = None,
        bpe_vocab_size: int | None = None,
        space: IdSpace | None = None,
        qwen3_pretrained: bool = True,
    ) -> None:
        super().__init__()

        model_config = model_config or ModelConfig()
        self.acoustic_condition_drop = model_config.acoustic.condition_dropout
        self.acoustic_condition_source = model_config.acoustic.condition_source

        components = build_orchestrator_components(
            qwen3=qwen3,
            dit=dit,
            qwen3_config=qwen3_config,
            bnb_config=bnb_config,
            lora_config=lora_config,
            model_config=model_config,
            bpe_config=bpe_config,
            bpe=bpe,
            tokenizer=tokenizer,
            bpe_vocab_size=bpe_vocab_size,
            space=space,
            qwen3_pretrained=qwen3_pretrained,
        )
        self.qwen3 = components.qwen3
        self.dit = components.dit
        self.output_adapter = components.output_adapter
        self.acoustic_condition_adapter = components.acoustic_condition_adapter
        self.acoustic_condition_encoder = components.acoustic_condition_encoder
        self.lm_head = components.lm_head

    @property
    def idspace(self) -> IdSpace:
        return self.embed_tokens.space

    @property
    def embed_tokens(self) -> IdSpaceEmbedding:
        return cast(IdSpaceEmbedding, text_embedding(self.qwen3))

    def forward(
        self,
        batch: CausalLMBatch,
        *,
        return_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        outputs = self.qwen3(
            attention_mask=batch.attention_mask,
            inputs_embeds=self.embed_tokens(batch.input_ids),
            use_cache=False,
            output_hidden_states=return_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        positions = loss_positions(batch)
        selected_hidden = self.output_adapter(
            hidden_states[positions[:, 0], positions[:, 1]]
        )
        labels = batch.labels[positions[:, 0], positions[:, 1]]
        target = self.lm_head.to_head_ids(labels)
        logits = self.lm_head(selected_hidden)
        token_loss = F.cross_entropy(logits.float(), target, reduction="none")
        token_weights = loss_weights(batch, positions, dtype=token_loss.dtype)
        loss = batch_loss(
            token_loss,
            positions[:, 0],
            batch.input_ids.size(0),
            token_weights,
        )
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
        positions = loss_positions(batch)
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
        flow_dtype = module_dtype(self.dit, condition.hidden_states.dtype)
        target_features = target_features.to(
            device=condition.hidden_states.device,
            dtype=flow_dtype,
        )
        validate_acoustic_features(
            target_features,
            condition.mask,
            target_mask=target_mask,
        )

        last_hidden_state = self.acoustic_condition_hidden(condition, dtype=flow_dtype)
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

    def acoustic_condition_hidden(
        self,
        condition: AcousticCondition,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if self.dit is None:
            raise RuntimeError("acoustic condition hidden requires a DiT acoustic decoder.")
        flow_dtype = dtype or module_dtype(self.dit, condition.hidden_states.dtype)
        return self._adapt_acoustic_condition_hidden(
            condition.hidden_states,
            mask=condition.mask,
            dtype=flow_dtype,
        )

    def _adapt_acoustic_condition_hidden(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        adapter_dtype = module_dtype(self.acoustic_condition_adapter, hidden_states.dtype)
        hidden_states = self.acoustic_condition_adapter(
            hidden_states.to(dtype=adapter_dtype)
        )
        if self.acoustic_condition_encoder is not None:
            encoder_dtype = module_dtype(self.acoustic_condition_encoder, hidden_states.dtype)
            hidden_states = self.acoustic_condition_encoder(
                hidden_states.to(dtype=encoder_dtype),
                attention_mask=mask.to(device=hidden_states.device, dtype=torch.long),
            )
        return hidden_states.to(dtype=dtype)

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
            output_adapter=self.output_adapter,
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
            output_adapter=self.output_adapter,
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
            output_adapter=self.output_adapter,
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
