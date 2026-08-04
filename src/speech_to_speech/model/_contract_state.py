from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from anydataset.types import Modality
from anytrain.codec import AcousticLayout
from torch import nn

if TYPE_CHECKING:
    from semantic_acoustic_codec.model import AcousticRVQDecoder
    from semantic_acoustic_codec.model.dit import DiTDecoder

from ..runtime import AudioSequenceLayout
from ..runtime.backbone import BackboneEncoder
from ..runtime.protocol import TokenModelRuntime
from ..runtime.types import (
    Backbone,
    TextTokenizer,
    acoustic_codec,
    codec_frame_rate,
    codec_sample_rate,
    fsq_level_values,
    fsq_levels,
    fsq_radix_order,
    semantic_feature_dim,
    supports_acoustic,
)
from ._contract import (
    ModelCheckpointContract,
    canonical_value,
    contract_sha256,
)
from ._helper import CastOutput, MLPAdapter
from .acoustic.condition import HiddenConditionAdapter
from .audio_input import AudioInputAdapterType, AudioInputTower
from .audio_output import AudioOutputAdapter, AudioOutputAdapterType
from .embedding.audio import SemanticAudioEmbedding
from .embedding.fsq import FsqEmbedding, FsqFeature
from .token import TokenInterface


_HF_EXECUTION_CONFIG_FIELDS = frozenset(
    {
        "_attn_implementation",
        "_attn_implementation_internal",
        "_commit_hash",
        "_name_or_path",
        "architectures",
        "attn_implementation",
        "auto_map",
        "dtype",
        "finetuning_task",
        "gradient_checkpointing",
        "id2label",
        "label2id",
        "output_attentions",
        "output_hidden_states",
        "problem_type",
        "return_dict",
        "return_dict_in_generate",
        "task_specific_params",
        "tokenizer_class",
        "torch_dtype",
        "transformers_version",
        "use_cache",
    }
)


class _ContractModel(Protocol):
    @property
    def runtime(self) -> TokenModelRuntime: ...

    @property
    def backbone(self) -> Backbone: ...

    @property
    def tokens(self) -> TokenInterface: ...

    @property
    def source_audio_encoder(self) -> AudioInputTower | None: ...

    @property
    def _encoder(self) -> BackboneEncoder: ...


class _Flow(Protocol):
    @property
    def decoder(self) -> DiTDecoder: ...


@runtime_checkable
class _ContractStateProvider(Protocol):
    def contract_state(self) -> Mapping[str, object]: ...


@runtime_checkable
class _VocabularyProvider(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


@runtime_checkable
class _ChatTemplateProvider(Protocol):
    @property
    def chat_template(self) -> str | None: ...


@runtime_checkable
class _ConfigStateProvider(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


@runtime_checkable
class _ConfigOwner(Protocol):
    @property
    def config(self) -> object: ...


def build_model_contract(
    model: _ContractModel,
    acoustic: Mapping[str, object],
) -> ModelCheckpointContract:
    """Build a contract from resolved runtime resources and realized modules."""
    return ModelCheckpointContract.from_components(
        {
            "runtime": _runtime_contract(model),
            "interface": _interface_contract(model),
            "acoustic": canonical_value(acoustic),
        }
    )


def flow_acoustic_contract(
    condition: HiddenConditionAdapter,
    acoustic_flow: _Flow,
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
            "student_layer": _positive_int(
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
            "input_dim": _positive_int(core.input_dim, "Flow input dim"),
            "output_dim": _positive_int(core.output_dim, "Flow output dim"),
            "hidden_dim": _positive_int(core.hidden_dim, "Flow hidden dim"),
            "condition_dim": _positive_int(
                core.condition_dim,
                "Flow condition dim",
            ),
            "layers": len(blocks),
            "heads": _positive_int(first.attention.heads, "Flow attention heads"),
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
    if not isinstance(backbone, _ConfigOwner):
        raise TypeError("RVQ decoder backbone must expose its resolved config.")
    config = _backbone_config_state(backbone.config)
    sizes = _positive_sizes(decoder.codebook_sizes, "RVQ acoustic codebooks")
    return {
        "type": "rvq",
        "condition": condition_adapter_contract(condition),
        "decoder": {
            "family": "qwen3-codebook-ar-v1",
            "condition_dim": _positive_int(
                decoder.condition_dim,
                "RVQ condition dim",
            ),
            "hidden_dim": _positive_int(decoder.hidden_dim, "RVQ hidden dim"),
            "embedding_dim": _positive_int(
                decoder.embedding_dim,
                "RVQ embedding dim",
            ),
            "codebook_sizes": list(sizes),
            "layers": _config_positive_int(
                config,
                "num_hidden_layers",
                "RVQ decoder layers",
            ),
            "heads": _config_positive_int(
                config,
                "num_attention_heads",
                "RVQ decoder attention heads",
            ),
            "kv_heads": _config_positive_int(
                config,
                "num_key_value_heads",
                "RVQ decoder key/value heads",
            ),
            "head_dim": _config_positive_int(
                config,
                "head_dim",
                "RVQ decoder head dim",
            ),
            "ffn_dim": _config_positive_int(
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
        "hidden_dim": _positive_int(
            value.hidden_dim,
            "acoustic condition hidden dim",
        ),
        "condition_dim": _positive_int(
            value.condition_dim,
            "acoustic condition output dim",
        ),
        "norm_eps": float(value.norm.eps),
        "projection_bias": value.projection.bias is not None,
    }


def _runtime_contract(model: _ContractModel) -> dict[str, object]:
    return {
        "codec": _codec_contract(model.runtime),
        "token_space": _token_space_contract(model.runtime),
        "backbone": _backbone_contract(model),
    }


def _codec_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    codec = runtime.codec
    semantic_sizes = _positive_sizes(
        runtime.semantic_codebook_sizes,
        "semantic codec codebooks",
    )
    acoustic: dict[str, object] | None = None
    if supports_acoustic(codec):
        resolved = acoustic_codec(codec)
        layout = runtime.acoustic_layout
        if not isinstance(layout, AcousticLayout):
            raise TypeError("runtime acoustic layout must be an AcousticLayout.")
        acoustic = {
            "feature_dim": resolved.acoustic_feature_dim,
            "codebook_sizes": list(resolved.acoustic_codebook_sizes),
            "layout": layout.value,
            "unit_length": runtime.acoustic_unit_length,
        }
    name = runtime.codec_name
    if not isinstance(name, str) or not name:
        raise TypeError("runtime codec_name must be a non-empty string.")
    return {
        "name": name,
        "implementation": _qualified_name(codec),
        "sample_rate": codec_sample_rate(codec),
        "frame_rate": codec_frame_rate(codec),
        "semantic_codebook_sizes": list(semantic_sizes),
        "semantic_feature_dim": semantic_feature_dim(codec),
        "fsq_levels": fsq_levels(codec),
        "fsq_level_values": fsq_level_values(codec),
        "fsq_radix_order": fsq_radix_order(codec),
        "acoustic": acoustic,
    }


def _token_space_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    blocks = runtime.layout.blocks
    if not isinstance(blocks, Mapping):
        raise TypeError("runtime token layout blocks must be a mapping.")
    token_blocks = {
        modality.value: list(_token_block(blocks.get(modality.value), modality))
        for modality in (Modality.TEXT, Modality.AUDIO)
    }
    sequence_layout = runtime.audio_sequence_layout
    if not isinstance(sequence_layout, AudioSequenceLayout):
        raise TypeError("runtime audio sequence layout must be an AudioSequenceLayout.")
    return {
        "audio_sequence_layout": sequence_layout.value,
        "blocks": token_blocks,
        "special_ids": {
            "pad": _non_negative_int(runtime.pad_token_id, "runtime pad_token_id"),
            "bos": _non_negative_int(runtime.bos_token_id, "runtime bos_token_id"),
            "eos": _non_negative_int(runtime.eos_token_id, "runtime eos_token_id"),
            "boa": _non_negative_int(runtime.boa_token_id, "runtime boa_token_id"),
            "eoa": _non_negative_int(runtime.eoa_token_id, "runtime eoa_token_id"),
            "mask": _non_negative_int(
                runtime.mask_token_id,
                "runtime mask_token_id",
            ),
        },
        "text_tokenizer": _text_tokenizer_contract(
            runtime.text_tokenizer,
            token_blocks["text"],
            configured_chat_template=runtime.backbone_chat_template,
        ),
        "audio_tokenizer": _audio_tokenizer_contract(runtime),
    }


def _text_tokenizer_contract(
    tokenizer: TextTokenizer,
    text_block: Sequence[int],
    *,
    configured_chat_template: str | None,
) -> dict[str, object]:
    state = _text_tokenizer_state(tokenizer)
    chat_template = configured_chat_template
    if chat_template is None and isinstance(tokenizer, _ChatTemplateProvider):
        chat_template = tokenizer.chat_template
    if chat_template is not None and not isinstance(chat_template, str):
        raise TypeError("text tokenizer chat template must be a string or None.")
    return {
        "implementation": _qualified_name(tokenizer),
        "vocab_size": text_block[1] - text_block[0],
        "state_grammar": _state_grammar(state),
        "state_sha256": contract_sha256(state),
        "special_tokens_sha256": contract_sha256(tokenizer.special_tokens_map),
        "chat_template_sha256": (
            None if chat_template is None else contract_sha256(chat_template)
        ),
    }


def _text_tokenizer_state(tokenizer: TextTokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, _ContractStateProvider):
        return _contract_state(tokenizer)
    if isinstance(tokenizer, _VocabularyProvider):
        vocab = tokenizer.get_vocab()
        if not isinstance(vocab, Mapping):
            raise TypeError("tokenizer get_vocab() must return a mapping.")
        return {
            "grammar": "get-vocab-v1",
            "vocab": vocab,
        }
    return {
        "grammar": "text-tokenizer-interface-v1",
        "implementation": _qualified_name(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
    }


def _audio_tokenizer_contract(
    runtime: TokenModelRuntime,
) -> dict[str, object]:
    tokenizer = runtime.audio_tokenizer
    state = _contract_state(tokenizer)
    result: dict[str, object] = {
        "implementation": _qualified_name(tokenizer),
        "grammar": _state_grammar(state),
        "vocab_size": _positive_int(
            tokenizer.vocab_size,
            "audio tokenizer vocabulary size",
        ),
        "state_sha256": contract_sha256(state),
    }
    for key in (
        "codebook_sizes",
        "semantic_codebook_size",
        "semantic_vocab_size",
        "acoustic_codebook_sizes",
        "acoustic_unit_length",
    ):
        if key in state:
            result[key] = canonical_value(state[key])
    return result


def _contract_state(value: _ContractStateProvider) -> dict[str, Any]:
    state = canonical_value(value.contract_state())
    if not isinstance(state, dict):
        raise TypeError("contract_state() must return a mapping.")
    _state_grammar(state)
    return state


def _state_grammar(state: Mapping[str, Any]) -> str:
    grammar = state.get("grammar")
    if not isinstance(grammar, str) or not grammar:
        raise ValueError("contract state requires a non-empty grammar.")
    return grammar


def _backbone_contract(model: _ContractModel) -> dict[str, object]:
    encoder_state = canonical_value(model._encoder.contract_state())
    if not isinstance(encoder_state, dict):
        raise TypeError("backbone encoder contract_state() must return a mapping.")
    config = _backbone_config_state(model.backbone.config)
    return {
        "implementation": _qualified_name(model.backbone),
        "encoder": encoder_state,
        "hidden_size": _positive_int(
            model.backbone.config.hidden_size,
            "backbone hidden size",
        ),
        "architecture_sha256": contract_sha256(config),
    }


def _backbone_config_state(config: object) -> dict[str, Any]:
    if isinstance(config, _ConfigStateProvider):
        state = config.to_dict()
    else:
        try:
            state = vars(config)
        except TypeError as error:
            raise TypeError(
                "backbone config must expose to_dict() or attributes."
            ) from error
    if not isinstance(state, Mapping):
        raise TypeError("backbone config state must be a mapping.")
    canonical = _semantic_config_value(state)
    if not isinstance(canonical, dict):
        raise TypeError("backbone config state must resolve to a mapping.")
    return canonical


def _semantic_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("backbone config mapping keys must be strings.")
        return {
            key: _semantic_config_value(item)
            for key, item in sorted(value.items())
            if key not in _HF_EXECUTION_CONFIG_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_config_value(item) for item in value]
    return canonical_value(value)


def _interface_contract(model: _ContractModel) -> dict[str, object]:
    tokens = model.tokens
    return {
        "audio_embedding": _audio_embedding_contract(tokens.audio_embedding),
        "audio_projection": _adapter_contract(tokens.audio_projection),
        "audio_head": _audio_head_contract(tokens.audio_head),
        "source_audio_encoder": _source_audio_contract(model.source_audio_encoder),
    }


def _audio_embedding_contract(
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


def _adapter_contract(value: object) -> dict[str, object]:
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


def _audio_head_contract(value: AudioOutputAdapter) -> dict[str, object]:
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
        result["adapter"] = _adapter_contract(value.adapter)
    return result


def _source_audio_contract(
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
        result["adapter"] = _adapter_contract(value.adapter)
    else:
        raise TypeError("source audio tower has an unsupported effective type.")
    return result


def _token_block(
    value: object,
    modality: Modality,
) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"token layout block {modality.value!r} must contain two ids.")
    start = _non_negative_int(value[0], f"{modality.value} token block start")
    end = _positive_int(value[1], f"{modality.value} token block end")
    if end <= start:
        raise ValueError(f"{modality.value} token block must be non-empty.")
    return start, end


def _config_positive_int(
    config: Mapping[str, Any],
    key: str,
    name: str,
) -> int:
    return _positive_int(config.get(key), name)


def _positive_sizes(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    sizes = tuple(_positive_int(item, f"{name} size") for item in value)
    if not sizes:
        raise ValueError(f"{name} must not be empty.")
    return sizes


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _qualified_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


__all__ = [
    "build_model_contract",
    "condition_adapter_contract",
    "flow_acoustic_contract",
    "rvq_acoustic_contract",
]
