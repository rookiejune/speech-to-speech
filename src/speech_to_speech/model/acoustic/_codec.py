from __future__ import annotations

from torch import Tensor

from ...runtime.codec_contract import AcousticCodec
from ...runtime.backbone.contract import Backbone


def code_features(
    codec: AcousticCodec,
    backbone: Backbone,
    codes: Tensor,
    *,
    like: Tensor | None = None,
) -> Tensor:
    features = codec.acoustic_codes_to_features(codes)
    reference = backbone.get_input_embeddings().weight if like is None else like
    return features.to(device=reference.device, dtype=reference.dtype)
