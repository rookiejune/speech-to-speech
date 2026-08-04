from __future__ import annotations

from pathlib import Path

import torch
from anytrain.codec import AcousticLayout
from semantic_acoustic_generator.config import Route
from semantic_acoustic_generator.model import (
    AcousticRVQDecoder,
    FMFeatureGenerator,
    RVQCodeGenerator,
)
from semantic_acoustic_generator.runtime.artifact import (
    AcousticGeneratorBackend,
    AcousticGeneratorArtifact,
    load_generator_artifact,
)

from ._config import DecoderConfig, FlowRepaConfig


def load_acoustic_initialization(
    path: str | Path,
    *,
    codec: object,
    route: Route,
    device: torch.device,
) -> AcousticGeneratorArtifact:
    """Load and validate an external generator used to initialize joint training."""
    if not isinstance(codec, AcousticGeneratorBackend):
        raise TypeError(
            "acoustic initialization requires codec feature, codebook, and layout metadata."
        )
    artifact = load_generator_artifact(path, device=device)
    if artifact.spec.route is not route:
        raise ValueError(
            f"acoustic initialization route is {artifact.spec.route.value!r}, "
            f"expected {route.value!r}."
        )
    if artifact.spec.acoustic_layout is not AcousticLayout.FRAME_ALIGNED:
        raise ValueError(
            "joint S2S acoustic initialization currently requires frame-aligned units."
        )
    artifact.spec.validate_acoustic_backend(codec)
    return artifact


def flow_generator(
    options: DecoderConfig,
    repa: FlowRepaConfig | None,
    initialization: AcousticGeneratorArtifact | None,
) -> FMFeatureGenerator | None:
    if initialization is None:
        return None
    generator = initialization.generator
    if not isinstance(generator, FMFeatureGenerator):
        raise TypeError("Flow initialization requires an FMFeatureGenerator artifact.")
    _validate_decoder_options(options, initialization)
    _validate_repa(repa, initialization)
    return generator


def rvq_generator(
    options: DecoderConfig,
    initialization: AcousticGeneratorArtifact | None,
) -> RVQCodeGenerator | None:
    if initialization is None:
        return None
    generator = initialization.generator
    if not isinstance(generator, RVQCodeGenerator):
        raise TypeError("RVQ initialization requires an RVQCodeGenerator artifact.")
    if not isinstance(generator.core, AcousticRVQDecoder):
        raise ValueError(
            "joint S2S RVQ initialization currently requires the codebook_ar predictor."
        )
    _validate_decoder_options(options, initialization)
    return generator


def _validate_decoder_options(
    options: DecoderConfig,
    initialization: AcousticGeneratorArtifact,
) -> None:
    decoder = initialization.spec.decoder
    expected = (options.hidden_dim, options.layers, options.heads, options.ffn_ratio)
    actual = (decoder.hidden_dim, decoder.layers, decoder.heads, decoder.ffn_ratio)
    if expected != actual:
        raise ValueError(
            "acoustic decoder config does not match initialization artifact: "
            f"{expected!r} != {actual!r}."
        )


def _validate_repa(
    repa: FlowRepaConfig | None,
    initialization: AcousticGeneratorArtifact,
) -> None:
    decoder = initialization.spec.decoder
    expected = (
        None if repa is None else repa["feature_dim"],
        None if repa is None else repa["student_layer"],
    )
    actual = (decoder.repa_feature_dim, decoder.repa_student_layer)
    if expected != actual:
        raise ValueError(
            "Flow REPA config does not match initialization artifact: "
            f"{expected!r} != {actual!r}."
        )


__all__ = ["load_acoustic_initialization"]
