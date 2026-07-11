from __future__ import annotations

from anytrain.idspace import Embedding

from ...runtime import runtime
from ..adapter import create_adapter
from .audio import embedding


def create_embedding(
    adapter_type: str | None,
    runtime_snapshot=None,
):
    rt = runtime() if runtime_snapshot is None else runtime_snapshot
    backbone_weight = rt.backbone.embed_tokens.weight
    adapter = create_adapter(
        adapter_type,
        rt.codec.semantic_codebook.size(-1),
        rt.backbone.config.hidden_size,
    ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    return Embedding(
        layout=rt.layout,
        adapters={"audio": adapter},
        audio=embedding(rt.codec, rt.audio_tokenizer),
        text=rt.backbone.embed_tokens,
    )
