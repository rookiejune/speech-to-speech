from __future__ import annotations

from ...runtime import runtime
from ..adapter import create_adapter
from .audio import embedding


def create_embedding(
    adapter_type: str | None,
    runtime_snapshot=None,
):
    rt = runtime() if runtime_snapshot is None else runtime_snapshot
    backbone_weight = rt.backbone.get_input_embeddings().weight
    adapter = create_adapter(
        adapter_type,
        rt.codec.semantic_codebook.size(-1),
        rt.backbone.config.hidden_size,
    ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    audio = embedding(rt.codec, rt.audio_tokenizer).to(
        device=backbone_weight.device,
        dtype=backbone_weight.dtype,
    )
    return audio, adapter
