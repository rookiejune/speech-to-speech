from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from ..ctc import CTCDecoderRoutes

from ..adapter import MLPAdapter
from .._helper import CastOutput
from ..audio_input import AudioInputAdapterType, AudioInputTower
from ..audio_output import AudioOutputAdapter, AudioOutputAdapterType
from ..embedding.audio import SemanticAudioEmbedding
from ..embedding.fsq import FsqEmbedding, FsqFeature
from ._protocol import ContractModel
from ._value import contract_state
from .core import contract_sha256


def state_dict_contract(model: nn.Module) -> dict[str, object]:
    state = model.state_dict(keep_vars=True)
    parameter_names = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    buffer_names = {
        name for name, _ in model.named_buffers(remove_duplicate=False)
    }
    entries: list[dict[str, object]] = []
    parameter_entries = 0
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"model state entry {name!r} must be a tensor for checkpoint contracts."
            )
        is_parameter = name in parameter_names
        is_buffer = name in buffer_names
        if is_parameter == is_buffer:
            raise TypeError(
                f"model state entry {name!r} must resolve to one parameter or buffer."
            )
        kind = "parameter" if is_parameter else "buffer"
        if is_parameter:
            parameter_entries += 1
        entries.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(value.shape),
            }
        )
    return {
        "grammar": "torch-state-dict-schema-v1",
        "entry_count": len(entries),
        "parameter_entries": parameter_entries,
        "buffer_entries": len(entries) - parameter_entries,
        "schema_sha256": contract_sha256(entries),
    }


def interface_contract(model: ContractModel) -> dict[str, object]:
    tokens = model.tokens
    return {
        "audio_embedding": audio_embedding_contract(tokens.audio_embedding),
        "audio_projection": adapter_contract(tokens.audio_projection),
        "audio_head": audio_head_contract(tokens.audio_head),
        "source_audio_encoder": source_audio_contract(model.source_audio_encoder),
        "ctc_decoders": ctc_decoders_contract(model.ctc_decoders),
    }


def ctc_decoders_contract(value: CTCDecoderRoutes) -> dict[str, object]:
    return {
        "source": contract_state(value.source),
        "target": contract_state(value.target),
    }


def audio_embedding_contract(
    value: SemanticAudioEmbedding,
) -> dict[str, object]:
    if isinstance(value, FsqEmbedding):
        result: dict[str, object] = {
            "family": "fsq-factorized-v1",
            "feature": value.config.feature.value,
            "rows": value.num_embeddings,
            "dim": value.embedding_dim,
            "codebook_sizes": list(value.codebook_sizes),
            "fsq_levels": value.fsq_levels,
            "radix_order": value.radix_order,
        }
        if value.config.feature is FsqFeature.DIGIT_VALUE:
            result["level_values_sha256"] = contract_sha256(
                value.level_values.detach().cpu().tolist()
            )
        return result
    if isinstance(value, nn.Embedding):
        return {
            "family": "dense-v1",
            "rows": value.num_embeddings,
            "dim": value.embedding_dim,
        }
    raise TypeError("audio embedding must be dense or factorized FSQ.")


def adapter_contract(value: object) -> dict[str, object]:
    module = value.module if isinstance(value, CastOutput) else value
    if isinstance(module, nn.Identity):
        return {"family": "identity-v1"}
    if isinstance(module, nn.Linear):
        return {
            "family": "linear-v1",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
    if isinstance(module, MLPAdapter):
        return {
            "family": "swiglu-mlp-v1",
            "in_features": module.gate_proj.in_features,
            "hidden_features": module.gate_proj.out_features,
            "out_features": module.down_proj.out_features,
            "bias": any(
                projection.bias is not None
                for projection in (
                    module.gate_proj,
                    module.up_proj,
                    module.down_proj,
                )
            ),
        }
    raise TypeError(f"unsupported model adapter: {type(module).__name__}.")


def audio_head_contract(value: AudioOutputAdapter) -> dict[str, object]:
    result: dict[str, object] = {
        "type": value.config.type.value,
        "in_features": value.in_features,
        "out_features": value.out_features,
    }
    if value.config.type is AudioOutputAdapterType.TRANSFORMER:
        result.update(
            {
                "family": "causal-transformer-v1",
                "layers": value.config.layers,
                "heads": value.config.heads,
                "ffn_dim": max(
                    1,
                    int(round(value.config.ffn_ratio * value.out_features)),
                ),
                "dropout": value.config.dropout,
            }
        )
    else:
        result["adapter"] = adapter_contract(value.adapter)
    return result


def source_audio_contract(
    value: AudioInputTower | None,
) -> dict[str, object]:
    if value is None:
        return {"type": "none"}
    result: dict[str, object] = {
        "type": value.config.type.value,
        "in_features": value.in_features,
        "out_features": value.out_features,
    }
    if value.config.type is AudioInputAdapterType.TRANSFORMER:
        result.update(
            {
                "family": (
                    "causal-transformer-encoder-v1"
                    if value.config.causal
                    else "bidirectional-transformer-encoder-v1"
                ),
                "causal": value.config.causal,
                "layers": value.config.layers,
                "heads": value.config.heads,
                "ffn_dim": max(
                    1,
                    int(round(value.config.ffn_ratio * value.out_features)),
                ),
                "dropout": value.config.dropout,
            }
        )
    elif value.config.type is AudioInputAdapterType.MLP:
        result["adapter"] = adapter_contract(value.adapter)
    else:
        raise TypeError("source audio tower has an unsupported effective type.")
    return result

__all__ = ["interface_contract", "state_dict_contract"]
