from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Optional, Protocol, cast

import torch
from torch import Tensor
from transformers import AutoConfig

from ..model import Config as ModelConfig
from ..runtime import Config as RuntimeConfig
from ..runtime import Runtime
from .config import Config as OracleConfig
from .model import (
    AcousticFlowModel,
    AcousticFlowScreening,
    AcousticRVQModel,
    AcousticRVQScreening,
)
from .trace import timed
from .types import Initialization


class TrainConfig(Protocol):
    @property
    def seed(self) -> int: ...


class FactoryConfig(Protocol):
    @property
    def runtime(self) -> RuntimeConfig: ...

    @property
    def model(self) -> ModelConfig: ...

    @property
    def codec_oracle(self) -> OracleConfig: ...

    @property
    def train(self) -> TrainConfig: ...


def build_runtime(config: FactoryConfig, device: torch.device) -> Runtime:
    return Runtime(replace(config.runtime, device=str(device)))


@torch.no_grad()
def build_flow(
    config: FactoryConfig,
    codes: Tensor,
    initialization: Initialization,
    runtime: Runtime,
    device: torch.device,
) -> tuple[AcousticFlowScreening, dict[str, Any]]:
    if codes.size(-1) < 2:
        raise ValueError("flow screening requires semantic and acoustic codebooks.")
    model = AcousticFlowModel(
        adapter=config.model.semantic_audio_adapter,
        runtime=runtime,
        condition_dim=condition_dim(config),
        flow_runtime=runtime.flow_matching,
        decoder=config.codec_oracle.decoder,
        device=device,
        dtype=model_dtype(config.runtime.dtype),
    )
    codec = runtime.codec
    semantic_codes = codes[:, 0]
    acoustic_codes = codes[:, 1:]
    with timed(
        "codec.dequantize_probe",
        codec=config.runtime.codec,
        code_shape=list(acoustic_codes.shape),
    ):
        target = codec.acoustic_codes_to_features(
            acoustic_codes.unsqueeze(0).to(device)
        ).float()
    mean, std = feature_stats(
        target,
        enabled=config.codec_oracle.normalize_features,
    )
    codebook = codec.semantic_codebook.detach().float()
    module = AcousticFlowScreening(
        model,
        initialization=initialization,
        seed=config.train.seed,
        flow_runtime=runtime.flow_matching,
        learning_rate=config.codec_oracle.learning_rate,
        weight_decay=config.codec_oracle.weight_decay,
        target_mean=mean.cpu(),
        target_std=std.cpu(),
    )
    metadata = common_metadata(
        config,
        codes,
        codebook,
        frame_rate=codec.frame_rate,
    ) | {
        "semantic_frames": int(semantic_codes.size(0)),
        "feature_dim": int(target.size(-1)),
        "feature_mean": float(target.mean()),
        "feature_std": float(target.std(correction=0)),
    }
    return module, metadata


@torch.no_grad()
def build_rvq(
    config: FactoryConfig,
    codes: Tensor,
    initialization: Initialization,
    runtime: Runtime,
    device: torch.device,
) -> tuple[AcousticRVQScreening, dict[str, Any]]:
    codec = runtime.codec
    acoustic_sizes = codec.acoustic_codebook_sizes
    expected_codebooks = 1 + len(acoustic_sizes)
    if not acoustic_sizes:
        raise ValueError("RVQ screening requires acoustic codebooks.")
    if codes.size(-1) != expected_codebooks:
        raise ValueError(
            "RVQ screening prepared codes must match the runtime codec: "
            f"{codes.size(-1)} != {expected_codebooks}."
        )
    model = AcousticRVQModel(
        adapter=config.model.semantic_audio_adapter,
        runtime=runtime,
        condition_dim=condition_dim(config),
        decoder=config.codec_oracle.decoder,
        device=device,
        dtype=model_dtype(config.runtime.dtype),
    )
    codebook = codec.semantic_codebook.detach().float()
    module = AcousticRVQScreening(
        model,
        initialization=initialization,
        seed=config.train.seed,
        learning_rate=config.codec_oracle.learning_rate,
        weight_decay=config.codec_oracle.weight_decay,
    )
    metadata = common_metadata(
        config,
        codes,
        codebook,
        frame_rate=codec.frame_rate,
    ) | {
        "semantic_frames": int(codes.size(0)),
        "acoustic_codebooks": len(acoustic_sizes),
        "acoustic_codebook_sizes": list(acoustic_sizes),
    }
    return module, metadata


def common_metadata(
    config: FactoryConfig,
    codes: Tensor,
    codebook: Tensor,
    *,
    frame_rate: float,
) -> dict[str, Any]:
    return {
        "codec": config.runtime.codec,
        "audio_tokenizer": (
            "native" if config.runtime.audio_tokenizer is None else "artifact"
        ),
        "semantic_condition": (
            "native_frame"
            if config.runtime.audio_tokenizer is None
            else "audio_tokenizer_span_repeat"
        ),
        "objective": config.codec_oracle.objective.value,
        "initialization": config.codec_oracle.initialization.value,
        "code_shape": list(codes.shape),
        "codebook_shape": list(codebook.shape),
        "semantic_codebook_rows": codebook.size(0),
        "codebook_mean": float(codebook.mean()),
        "codebook_std": float(codebook.std(correction=0)),
        "frame_rate": frame_rate,
        "max_seconds": config.codec_oracle.data.max_seconds,
    }


def feature_stats(target: Tensor, *, enabled: bool) -> tuple[Tensor, Tensor]:
    if not enabled:
        shape = (1, 1, target.size(-1))
        return target.new_zeros(shape), target.new_ones(shape)
    mean = target.mean(dim=(0, 1), keepdim=True)
    std = target.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-5)
    return mean, std


class _BackboneConfig(Protocol):
    hidden_size: int


def condition_dim(config: FactoryConfig) -> int:
    if config.model.toy is not None:
        return config.model.toy.hidden_size
    loaded = AutoConfig.from_pretrained(config.runtime.backbone)
    backbone = cast(_BackboneConfig, cast(object, loaded))
    hidden_size = backbone.hidden_size
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
        raise TypeError("backbone hidden_size must be an integer.")
    if hidden_size <= 0:
        raise ValueError("backbone hidden_size must be positive.")
    return hidden_size


def model_dtype(value: Optional[str]) -> torch.dtype:
    if value is None:
        return torch.get_default_dtype()
    try:
        result = getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {value}") from error
    if not isinstance(result, torch.dtype) or not result.is_floating_point:
        raise ValueError(f"oracle model dtype must be floating point: {value}")
    return result


def process_device(configured: Optional[str]) -> torch.device:
    requested = torch.device("cuda" if configured is None else configured)
    if requested.type != "cuda":
        raise ValueError("codec oracle requires runtime.device to be cuda.")
    device = (
        requested
        if requested.index is not None
        else torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    )
    torch.cuda.set_device(device)
    return device
