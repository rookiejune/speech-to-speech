from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Any, Protocol

import torch
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from lightning import pytorch as pl
from torch import Tensor, nn

from ..loss.causal_lm import CausalAcousticLoss
from ..loss.flow_matching import AcousticFlowLoss
from ..model import (
    AcousticDiT,
    AcousticFlow,
    AcousticRVQDecoder,
    AdapterType,
    DecoderConfig,
)
from ..model.adapter import create_adapter
from ..model.embedding.audio import base_weight
from ..runtime.audio_tokenizer import NativeAudioTokenizer
from ..runtime.types import AudioTokenizer, Codec
from .trace import timed
from .types import Initialization


class _Runtime(Protocol):
    @cached_property
    def codec(self) -> Codec: ...

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer: ...


class _AcousticOracleModel(nn.Module):
    """The trainable audio-side subset shared by oracle objectives.

    This deliberately has no reference to ``runtime.backbone``.  The semantic
    embedding and adapter are built with the same helpers used by ``TokenModel``;
    the acoustic decoder classes are the production implementations.  Keeping
    this subset under a nested ``model`` attribute gives checkpoints a stable
    ``model.*`` prefix while making it impossible to accidentally save Qwen.
    """

    def __init__(
        self,
        *,
        config_adapter: AdapterType | None,
        runtime: _Runtime,
        condition_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("oracle condition dimension must be positive.")
        self.runtime = runtime
        self.semantic_audio_embedding = _semantic_embedding(
            runtime,
            device=device,
            dtype=dtype,
        )
        self.semantic_audio_adapter = create_adapter(
            config_adapter,
            self.semantic_audio_embedding.embedding_dim,
            condition_dim,
        ).to(device=device, dtype=dtype)

    def semantic_condition(
        self,
        semantic_tokens: Tensor,
        spans: Tensor | None = None,
        *,
        frames: int | None = None,
    ) -> Tensor:
        if semantic_tokens.dim() != 2:
            raise ValueError("semantic tokens must have shape [batch, token].")
        condition = self.semantic_audio_adapter(
            self.semantic_audio_embedding(semantic_tokens)
        )
        if spans is None:
            return condition
        return _repeat_condition(condition, spans, frames=frames)

    def acoustic_code_features(self, codes: Tensor) -> Tensor:
        """Convert codec-local codes to the decoder's configured dtype/device."""
        reference = self.semantic_audio_embedding.weight
        return self.runtime.codec.acoustic_codes_to_features(codes).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def acoustic_target_latent(self, codes: Tensor) -> Tensor:
        if codes.dim() != 3:
            raise ValueError("target acoustic codes must have shape [B, F, N].")
        safe_codes = codes.clamp_min(0)
        features = self.acoustic_code_features(safe_codes)
        return features.masked_fill((codes < 0).all(dim=-1, keepdim=True), 0)


class AcousticFlowModel(_AcousticOracleModel):
    """Lightweight flow oracle model with the formal acoustic decoder."""

    def __init__(
        self,
        *,
        adapter: AdapterType | None,
        runtime: _Runtime,
        condition_dim: int,
        flow_runtime: ContinuousFlowRuntime,
        decoder: DecoderConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            config_adapter=adapter,
            runtime=runtime,
            condition_dim=condition_dim,
            device=device,
            dtype=dtype,
        )
        self.acoustic_flow = AcousticFlow(
            condition_dim,
            runtime.codec.acoustic_feature_dim,
            flow_runtime,
            hidden_dim=decoder.hidden_dim,
            layers=decoder.layers,
            heads=decoder.heads,
            ffn_ratio=decoder.ffn_ratio,
        ).to(device=device, dtype=dtype)

    @property
    def acoustic_decoder(self) -> AcousticDiT:
        return self.acoustic_flow.decoder


class AcousticRVQModel(_AcousticOracleModel):
    """Lightweight RVQ oracle model with the formal acoustic decoder."""

    def __init__(
        self,
        *,
        adapter: AdapterType | None,
        runtime: _Runtime,
        condition_dim: int,
        decoder: DecoderConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            config_adapter=adapter,
            runtime=runtime,
            condition_dim=condition_dim,
            device=device,
            dtype=dtype,
        )
        sizes = runtime.codec.acoustic_codebook_sizes
        if not sizes:
            raise ValueError("RVQ oracle requires at least one acoustic codebook.")
        self.acoustic_decoder = AcousticRVQDecoder(
            condition_dim,
            len(sizes),
            sizes,
            hidden_dim=decoder.hidden_dim,
            layers=decoder.layers,
            heads=decoder.heads,
            ffn_ratio=decoder.ffn_ratio,
        ).to(device=device, dtype=dtype)


class AcousticFlowScreening(pl.LightningModule):
    def __init__(
        self,
        model: AcousticFlowModel,
        *,
        initialization: Initialization,
        seed: int,
        flow_runtime: ContinuousFlowRuntime,
        learning_rate: float,
        weight_decay: float,
        target_mean: Tensor,
        target_std: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        _train_only(
            self.model,
            self.model.semantic_audio_embedding,
            self.model.semantic_audio_adapter,
            self.model.acoustic_flow,
        )
        weight = initialization.weight(
            self.model.semantic_audio_embedding.weight.detach(),
            seed=seed,
        )
        with torch.no_grad():
            self.model.semantic_audio_embedding.weight.copy_(weight)
        self.flow_runtime = flow_runtime
        self.objective = AcousticFlowLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_mean = nn.Buffer(target_mean)
        self.target_std = nn.Buffer(target_std)
        self._logged_dequantize = False

    def condition(
        self,
        semantic_tokens: Tensor,
        spans: Tensor | None = None,
        *,
        frames: int | None = None,
    ) -> Tensor:
        return self.model.semantic_condition(semantic_tokens, spans, frames=frames)

    def features(self, acoustic_codes: Tensor) -> Tensor:
        return self.model.acoustic_target_latent(acoustic_codes).float()

    def normalized_features(self, acoustic_codes: Tensor) -> Tensor:
        return (self.features(acoustic_codes) - self.target_mean) / self.target_std

    def training_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> dict[str, Any]:
        del batch_idx
        codes = batch["codes"]
        mask = batch["mask"]
        safe_codes = codes.masked_fill(~mask[..., None], 0)
        acoustic_codes = safe_codes[..., 1:]
        semantic_tokens = batch.get("semantic_tokens")
        spans = batch.get("semantic_token_spans")
        if semantic_tokens is None:
            semantic_tokens = safe_codes[..., 0]
            spans = None
        condition = self.condition(semantic_tokens, spans, frames=codes.size(1))
        if not self._logged_dequantize:
            with timed(
                "train.first_dequantize",
                code_shape=list(acoustic_codes.shape),
            ):
                target = self.normalized_features(acoustic_codes)
            self._logged_dequantize = True
        else:
            target = self.normalized_features(acoustic_codes)
        target = target.masked_fill(~mask[..., None], 0)
        item = self.objective(
            self.model.acoustic_decoder,
            condition,
            target,
            mask,
            self.flow_runtime,
        )
        loss = item.loss.mean()
        self.log("train/flow_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/batch_size", float(codes.size(0)), on_step=True, sync_dist=True)
        self.log("train/valid_frames", mask.sum().float(), on_step=True, sync_dist=True)
        return {"loss": loss, "flow_matching": item}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(
        self,
        semantic_tokens: Tensor,
        *,
        seed: int,
        spans: Tensor | None = None,
        frames: int | None = None,
    ) -> Tensor:
        condition = self.condition(semantic_tokens, spans, frames=frames)
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        normalized = self.model.acoustic_flow.sample(condition, generator=generator)
        return normalized * self.target_std + self.target_mean


class AcousticRVQScreening(pl.LightningModule):
    """Acoustic RVQ code prediction wrapper for codec oracle screening."""

    def __init__(
        self,
        model: AcousticRVQModel,
        *,
        initialization: Initialization,
        seed: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.model = model
        _train_only(
            self.model,
            self.model.semantic_audio_embedding,
            self.model.semantic_audio_adapter,
            self.model.acoustic_decoder,
        )
        weight = initialization.weight(
            self.model.semantic_audio_embedding.weight.detach(),
            seed=seed,
        )
        with torch.no_grad():
            self.model.semantic_audio_embedding.weight.copy_(weight)
        self.objective = CausalAcousticLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def condition(
        self,
        semantic_tokens: Tensor,
        spans: Tensor | None = None,
        *,
        frames: int | None = None,
    ) -> Tensor:
        return self.model.semantic_condition(semantic_tokens, spans, frames=frames)

    def features(self, acoustic_codes: Tensor) -> Tensor:
        return self.model.acoustic_code_features(acoustic_codes).float()

    def _logits(
        self,
        semantic_tokens: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
        spans: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        return self.model.acoustic_decoder(
            self.condition(semantic_tokens, spans, frames=acoustic_codes.size(1)),
            acoustic_codes,
            mask=mask,
        )

    def training_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> dict[str, Any]:
        del batch_idx
        codes = batch["codes"]
        mask = batch["mask"]
        safe_codes = codes.masked_fill(~mask[..., None], 0)
        acoustic_codes = safe_codes[..., 1:]
        semantic_tokens = batch.get("semantic_tokens")
        spans = batch.get("semantic_token_spans")
        if semantic_tokens is None:
            semantic_tokens = safe_codes[..., 0]
            spans = None
        logits = self._logits(semantic_tokens, acoustic_codes, mask, spans)
        item = self.objective(logits, acoustic_codes, mask, validate=False)
        loss = item.loss.mean()
        self.log("train/rvq_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        if item.details is not None:
            for name, value in item.details.items():
                if name == "frames":
                    continue
                self.log(
                    f"train/rvq_{name}_loss",
                    value.mean(),
                    on_step=True,
                    sync_dist=True,
                )
        self.log("train/batch_size", float(codes.size(0)), on_step=True, sync_dist=True)
        self.log("train/valid_frames", mask.sum().float(), on_step=True, sync_dist=True)
        return {"loss": loss, "rvq": item}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(
        self,
        semantic_tokens: Tensor,
        *,
        seed: int,
        spans: Tensor | None = None,
        frames: int | None = None,
    ) -> Tensor:
        condition = self.condition(semantic_tokens, spans, frames=frames)
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        return self.model.acoustic_decoder.generate(
            condition,
            generator=generator,
        )


def _semantic_embedding(
    runtime: _Runtime,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Embedding:
    codec = runtime.codec
    codebook = codec.semantic_codebook
    if not isinstance(codebook, Tensor):
        raise TypeError("codec semantic codebook must be a tensor.")
    if codebook.dim() != 2:
        raise ValueError(
            "codec oracle requires one semantic codebook with shape [vocab, dim]."
        )
    if codebook.size(0) < 1 or codebook.size(1) < 1:
        raise ValueError("codec semantic codebook must be non-empty.")
    if not codebook.is_floating_point():
        raise TypeError("codec semantic codebook must be floating point.")
    if not bool(torch.isfinite(codebook).all()):
        raise ValueError("codec semantic codebook must contain finite values.")
    if isinstance(runtime.audio_tokenizer, NativeAudioTokenizer):
        weight = codebook.detach()
    else:
        weight = base_weight(codec, runtime.audio_tokenizer)
    weight = weight.to(device=device, dtype=dtype).clone()
    return nn.Embedding.from_pretrained(weight, freeze=False)


def _repeat_condition(
    condition: Tensor,
    spans: Tensor,
    *,
    frames: int | None,
) -> Tensor:
    if condition.dim() != 3 or spans.dim() != 2:
        raise ValueError("condition and spans must have shapes [B, T, D] and [B, T].")
    if condition.shape[:2] != spans.shape:
        raise ValueError("condition and spans must align on batch and token axes.")
    if spans.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("semantic token spans must use an integer dtype.")
    if bool((spans < 0).any()):
        raise ValueError("semantic token spans must be non-negative.")

    frame_count = int(spans.sum(dim=1).max()) if frames is None else frames
    if frame_count < 1:
        raise ValueError("semantic token spans must cover at least one frame.")

    output = condition.new_zeros(condition.size(0), frame_count, condition.size(-1))
    span_rows = spans.to(device=condition.device, dtype=torch.long)
    for row_index in range(condition.size(0)):
        row_spans = span_rows[row_index]
        valid = row_spans > 0
        repeated = torch.repeat_interleave(
            condition[row_index, valid],
            row_spans[valid],
            dim=0,
        )
        if repeated.size(0) == 0:
            raise ValueError("semantic token spans must cover at least one frame.")
        if repeated.size(0) > frame_count:
            raise ValueError("semantic token spans exceed the target frame count.")
        output[row_index, : repeated.size(0)] = repeated
    return output


def _train_only(model: nn.Module, *modules: nn.Module) -> None:
    trainable = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    model.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)


__all__ = [
    "AcousticFlowModel",
    "AcousticFlowScreening",
    "AcousticRVQModel",
    "AcousticRVQScreening",
]
