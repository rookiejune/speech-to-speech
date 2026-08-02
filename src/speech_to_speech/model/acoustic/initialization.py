from __future__ import annotations

from pathlib import Path

import torch
from anytrain.codec import AcousticLayout
from semantic_acoustic_codec.config import Route
from semantic_acoustic_codec.runtime.artifact import (
    AcousticGeneratorBackend,
    AcousticGeneratorArtifact,
    load_generator_artifact,
)


def load_acoustic_initialization(
    path: str | Path,
    *,
    codec: object,
    route: Route,
    device: torch.device,
) -> AcousticGeneratorArtifact:
    """Load and validate a SAC generator used to initialize joint training."""
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
        raise ValueError("joint S2S acoustic initialization currently requires frame-aligned units.")
    artifact.spec.validate_acoustic_backend(codec)
    return artifact


__all__ = ["load_acoustic_initialization"]
