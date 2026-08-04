from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn

if TYPE_CHECKING:
    from semantic_acoustic_generator.model import AcousticRVQDecoder

from ..acoustic.condition import HiddenConditionAdapter
from ._protocol import ConfigOwner, Flow
from ._value import (
    backbone_config_state,
    config_positive_int,
    positive_int,
    positive_sizes,
)
from .core import canonical_value, contract_sha256


def flow_acoustic_contract(
    condition: HiddenConditionAdapter,
    acoustic_flow: Flow,
) -> dict[str, object]:
    from anytrain.module.dit import DiTBlock

    core = acoustic_flow.decoder.decoder
    blocks = core.blocks
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("Flow decoder must expose a non-empty DiT block list.")
    first = cast(DiTBlock, blocks[0])
    ffn_input = first.ffn[0]
    if not isinstance(ffn_input, nn.Linear):
        raise TypeError("Flow decoder FFN must start with a linear projection.")
    feature_projection = core.feature_projection
    if feature_projection is not None and not isinstance(
        feature_projection,
        nn.Linear,
    ):
        raise TypeError("Flow REPA projection must be linear.")
    feature_readout = (
        None
        if feature_projection is None
        else {
            "feature_dim": feature_projection.out_features,
            "student_layer": positive_int(
                core.feature_layer,
                "Flow REPA student layer",
            ),
        }
    )
    return {
        "type": "flow",
        "condition": condition_adapter_contract(condition),
        "decoder": {
            "family": "dit-frame-film-v1",
            "input_dim": positive_int(core.input_dim, "Flow input dim"),
            "output_dim": positive_int(core.output_dim, "Flow output dim"),
            "hidden_dim": positive_int(core.hidden_dim, "Flow hidden dim"),
            "condition_dim": positive_int(
                core.condition_dim,
                "Flow condition dim",
            ),
            "layers": len(blocks),
            "heads": positive_int(first.attention.heads, "Flow attention heads"),
            "ffn_dim": ffn_input.out_features,
            "condition_type": canonical_value(core.condition_type),
            "feature_readout": feature_readout,
        },
    }


def rvq_acoustic_contract(
    condition: HiddenConditionAdapter,
    decoder: AcousticRVQDecoder,
) -> dict[str, object]:
    backbone = decoder.decoder
    if not isinstance(backbone, ConfigOwner):
        raise TypeError("RVQ decoder backbone must expose its resolved config.")
    config = backbone_config_state(backbone.config)
    sizes = positive_sizes(decoder.codebook_sizes, "RVQ acoustic codebooks")
    return {
        "type": "rvq",
        "condition": condition_adapter_contract(condition),
        "decoder": {
            "family": "qwen3-codebook-ar-v1",
            "condition_dim": positive_int(
                decoder.condition_dim,
                "RVQ condition dim",
            ),
            "hidden_dim": positive_int(decoder.hidden_dim, "RVQ hidden dim"),
            "embedding_dim": positive_int(
                decoder.embedding_dim,
                "RVQ embedding dim",
            ),
            "codebook_sizes": list(sizes),
            "layers": config_positive_int(
                config,
                "num_hidden_layers",
                "RVQ decoder layers",
            ),
            "heads": config_positive_int(
                config,
                "num_attention_heads",
                "RVQ decoder attention heads",
            ),
            "kv_heads": config_positive_int(
                config,
                "num_key_value_heads",
                "RVQ decoder key/value heads",
            ),
            "head_dim": config_positive_int(
                config,
                "head_dim",
                "RVQ decoder head dim",
            ),
            "ffn_dim": config_positive_int(
                config,
                "intermediate_size",
                "RVQ decoder FFN dim",
            ),
            "architecture_sha256": contract_sha256(config),
        },
    }


def condition_adapter_contract(
    value: HiddenConditionAdapter,
) -> dict[str, object]:
    return {
        "family": "layernorm-linear-v1",
        "hidden_dim": positive_int(
            value.hidden_dim,
            "acoustic condition hidden dim",
        ),
        "condition_dim": positive_int(
            value.condition_dim,
            "acoustic condition output dim",
        ),
        "norm_eps": float(value.norm.eps),
        "projection_bias": value.projection.bias is not None,
    }

__all__ = [
    "condition_adapter_contract",
    "flow_acoustic_contract",
    "rvq_acoustic_contract",
]
