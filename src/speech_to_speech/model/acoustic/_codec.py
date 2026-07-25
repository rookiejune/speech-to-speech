from __future__ import annotations

from torch import Tensor

from ...runtime.types import AcousticCodec, Backbone


def code_features(
    codec: AcousticCodec,
    backbone: Backbone,
    codes: Tensor,
) -> Tensor:
    features = codec.acoustic_codes_to_features(codes)
    weight = backbone.get_input_embeddings().weight
    return features.to(device=weight.device, dtype=weight.dtype)
