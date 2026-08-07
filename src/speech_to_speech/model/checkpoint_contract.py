"""Canonical checkpoint-contract payloads and validation."""

from __future__ import annotations


import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast, runtime_checkable

import torch
from anydataset.types import AudioView, Modality
from anytrain.codec import AudioBackendIdentity, AudioCodeSpec
from torch import Tensor, nn

from ..runtime import AudioSequenceLayout
from ..runtime.audio_schema import AudioTokenSpec
from ..task import ControlToken
from ..audio import AudioStream
from ..runtime.audio_tokenizer.contract import AudioTokenizer
from ..runtime.backbone import BackboneEncoder
from ..runtime.backbone.contract import Backbone, TextTokenizer
from ..runtime.codec import (
    audio_backend_identity,
    audio_code_spec,
)
from ..runtime.codec_contract import (
    acoustic_codec,
    codec_frame_rate,
    codec_sample_rate,
    fsq_level_values,
    fsq_levels,
    fsq_radix_order,
    global_codec,
    semantic_feature_dim,
    supports_acoustic,
    supports_global,
)
from ..runtime.protocol import TokenModelRuntime
from .tower import CastOutput
from .adapter import MLPAdapter
from .audio_input import AudioInputAdapterType, AudioInputTower
from .audio_output import AudioOutputAdapter, AudioOutputAdapterType
from .embedding.audio import SemanticAudioEmbedding
from .embedding.fsq import FsqEmbedding, FsqFeature
from .token import TokenInterface

if TYPE_CHECKING:
    from semantic_acoustic_generator.model import AcousticRVQDecoder
    from semantic_acoustic_generator.model.dit import DiTDecoder

    from .ctc import CTCDecoderRoutes


class _HiddenConditionAdapter(Protocol):
    hidden_dim: int
    condition_dim: int
    norm: nn.LayerNorm
    projection: nn.Linear


MODEL_CONTRACT_GRAMMAR = "s2s-model-v4-contract-v7"
_MODEL_CONTRACT_FIELDS = frozenset({"grammar", "components", "sha256"})
_MISSING = "<missing>"
_DIFFERENCE_KEY_ORDER = {
    "components": ("runtime", "interface", "acoustic", "state_dict"),
    "components.runtime": ("token_space", "audio_codecs", "backbone"),
    "components.runtime.token_space": (
        "audio_sequence_layout",
        "blocks",
        "special_ids",
        "text_controls",
        "text_tokenizer",
        "audio_schemas",
    ),
    "components.runtime.token_space.audio_schemas": (
        "sharing",
        "input",
        "output",
    ),
    "components.runtime.token_space.audio_schemas.input": (
        "selector_id",
        "payload_range",
        "tokenizer",
        "private_grammar",
        "schema_id",
        "selector",
        "spec",
    ),
    "components.runtime.token_space.audio_schemas.output": (
        "selector_id",
        "payload_range",
        "tokenizer",
        "private_grammar",
        "schema_id",
        "selector",
        "spec",
    ),
    "components.runtime.audio_codecs": (
        "sharing",
        "input",
        "output",
        "output_detokenizer",
    ),
    "components.interface": (
        "control_embedding",
        "audio_embeddings",
        "input_audio_projection",
        "audio_projection",
        "audio_head",
        "source_audio_encoder",
        "ctc_decoders",
    ),
    "components.interface.audio_embeddings": (
        "sharing",
        "input",
        "output",
    ),
    "components.interface.ctc_decoders": ("source", "target"),
    "components.state_dict": (
        "grammar",
        "schema_sha256",
        "entry_count",
        "parameter_entries",
        "buffer_entries",
    ),
}


class ModelCheckpointContractPayload(TypedDict):
    grammar: str
    components: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class ModelCheckpointContract:
    """Canonical semantic and topology identity for one S2S model."""

    _components_json: str
    sha256: str

    @classmethod
    def from_components(
        cls,
        components: Mapping[str, Any],
    ) -> ModelCheckpointContract:
        canonical = canonical_value(components)
        if not isinstance(canonical, dict):
            raise TypeError("model contract components must be a mapping.")
        serialized = _canonical_json(canonical)
        return cls(
            _components_json=serialized,
            sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    @property
    def components(self) -> Mapping[str, Any]:
        value = json.loads(self._components_json)
        if not isinstance(value, dict):
            raise RuntimeError("stored model contract components are invalid.")
        return value

    def checkpoint_payload(self) -> ModelCheckpointContractPayload:
        components = dict(self.components)
        return {
            "grammar": MODEL_CONTRACT_GRAMMAR,
            "components": components,
            "sha256": self.sha256,
        }


def validate_checkpoint_contract(
    actual: object,
    expected: ModelCheckpointContract,
) -> None:
    """Validate the payload digest, then compare its canonical components."""
    if not isinstance(actual, Mapping):
        raise TypeError("checkpoint model contract must be a mapping.")
    if set(actual) != _MODEL_CONTRACT_FIELDS:
        raise ValueError("checkpoint model contract fields do not match its grammar.")
    grammar = actual.get("grammar")
    if grammar != MODEL_CONTRACT_GRAMMAR:
        raise ValueError(
            "checkpoint model contract grammar is incompatible: "
            f"expected {MODEL_CONTRACT_GRAMMAR!r}, got {grammar!r}."
        )
    components = canonical_value(actual.get("components"))
    if not isinstance(components, dict):
        raise TypeError("checkpoint model contract components must be a mapping.")
    digest = actual.get("sha256")
    if not _is_sha256(digest) or digest != contract_sha256(components):
        raise ValueError("checkpoint model contract digest is invalid.")

    expected_components = canonical_value(expected.components)
    if not isinstance(expected_components, dict):
        raise TypeError("expected model contract components must be a mapping.")
    difference = _first_difference(
        components,
        expected_components,
        path="components",
    )
    if difference is None and digest == expected.sha256:
        return
    if difference is None:
        difference = ("sha256", digest, expected.sha256)
    path, checkpoint_value, model_value = difference
    raise ValueError(
        "checkpoint model contract does not match model at "
        f"{path}: {checkpoint_value!r} != {model_value!r}."
    )


def canonical_value(value: Any) -> Any:
    """Convert a contract value into deterministic JSON-safe data."""
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("model contract mapping keys must be strings.")
        return {key: canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item) for item in value]
        return sorted(items, key=_canonical_json)
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model contract floats must be finite.")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported model contract value: {type(value).__name__}.")


def contract_sha256(value: object) -> str:
    canonical = canonical_value(value)
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _first_difference(
    actual: Any,
    expected: Any,
    *,
    path: str,
) -> tuple[str, Any, Any] | None:
    if type(actual) is not type(expected):
        return path, actual, expected
    if isinstance(actual, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        order = {key: index for index, key in enumerate(_DIFFERENCE_KEY_ORDER.get(path, ()))}
        for key in sorted(
            actual_keys | expected_keys,
            key=lambda item: (order.get(item, len(order)), item),
        ):
            if key not in actual:
                return f"{path}.{key}", _MISSING, expected[key]
            if key not in expected:
                return f"{path}.{key}", actual[key], _MISSING
            difference = _first_difference(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}.length", len(actual), len(expected)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            difference = _first_difference(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if actual != expected:
        return path, actual, expected
    return None


class ContractModel(Protocol):
    @property
    def runtime(self) -> TokenModelRuntime: ...

    @property
    def backbone(self) -> Backbone: ...

    @property
    def tokens(self) -> TokenInterface: ...

    @property
    def source_audio_encoder(self) -> AudioInputTower | None: ...

    @property
    def ctc_decoders(self) -> CTCDecoderRoutes: ...

    @property
    def _encoder(self) -> BackboneEncoder: ...


class Flow(Protocol):
    @property
    def decoder(self) -> DiTDecoder: ...


@runtime_checkable
class ContractStateProvider(Protocol):
    def contract_state(self) -> Mapping[str, object]: ...


@runtime_checkable
class VocabularyProvider(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


@runtime_checkable
class BackendTokenizerProvider(Protocol):
    @property
    def backend_tokenizer(self) -> object: ...


@runtime_checkable
class TokenizerBackendSerializer(Protocol):
    def to_str(self) -> str: ...


@runtime_checkable
class ChatTemplateProvider(Protocol):
    @property
    def chat_template(self) -> str | None: ...


@runtime_checkable
class ConfigStateProvider(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


@runtime_checkable
class ConfigOwner(Protocol):
    @property
    def config(self) -> object: ...


HF_NON_ARCHITECTURE_CONFIG_FIELDS = frozenset(
    {
        "_attn_implementation",
        "_attn_implementation_internal",
        "_commit_hash",
        "_name_or_path",
        "architectures",
        "attn_implementation",
        "auto_map",
        "bad_words_ids",
        "begin_suppress_tokens",
        "diversity_penalty",
        "do_sample",
        "dtype",
        "early_stopping",
        "encoder_no_repeat_ngram_size",
        "exponential_decay_length_penalty",
        "finetuning_task",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "gradient_checkpointing",
        "id2label",
        "initializer_range",
        "label2id",
        "length_penalty",
        "max_length",
        "min_length",
        "no_repeat_ngram_size",
        "num_beam_groups",
        "num_beams",
        "num_return_sequences",
        "output_attentions",
        "output_hidden_states",
        "output_scores",
        "problem_type",
        "remove_invalid_values",
        "repetition_penalty",
        "return_dict",
        "return_dict_in_generate",
        "suppress_tokens",
        "task_specific_params",
        "temperature",
        "tokenizer_class",
        "top_k",
        "top_p",
        "torch_dtype",
        "transformers_version",
        "typical_p",
        "use_cache",
    }
)


def contract_state(value: ContractStateProvider) -> dict[str, Any]:
    state = canonical_value(value.contract_state())
    if not isinstance(state, dict):
        raise TypeError("contract_state() must return a mapping.")
    state_grammar(state)
    return state


def state_grammar(state: Mapping[str, Any]) -> str:
    grammar = state.get("grammar")
    if not isinstance(grammar, str) or not grammar:
        raise ValueError("contract state requires a non-empty grammar.")
    return grammar


def backbone_config_state(config: object) -> dict[str, Any]:
    if isinstance(config, ConfigStateProvider):
        state = config.to_dict()
    else:
        try:
            state = vars(config)
        except TypeError as error:
            raise TypeError("backbone config must expose to_dict() or attributes.") from error
    if not isinstance(state, Mapping):
        raise TypeError("backbone config state must be a mapping.")
    canonical = semantic_config_value(state)
    if not isinstance(canonical, dict):
        raise TypeError("backbone config state must resolve to a mapping.")
    return canonical


def semantic_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {
            semantic_config_key(key): semantic_config_value(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key in HF_NON_ARCHITECTURE_CONFIG_FIELDS)
        }
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [semantic_config_value(item) for item in value]
    return canonical_value(value)


def semantic_config_key(value: object) -> str:
    if isinstance(value, str):
        if value.startswith(("int:", "str:")):
            return f"str:{value}"
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("backbone config mapping keys must be strings or integers.")
    return f"int:{value}"


def token_block(
    value: object,
    name: str,
) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"token layout block {name!r} must contain two ids.")
    start = non_negative_int(value[0], f"{name} token block start")
    end = positive_int(value[1], f"{name} token block end")
    if end <= start:
        raise ValueError(f"{name} token block must be non-empty.")
    return start, end


def config_positive_int(
    config: Mapping[str, Any],
    key: str,
    name: str,
) -> int:
    return positive_int(config.get(key), name)


def positive_sizes(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    sizes = tuple(positive_int(item, f"{name} size") for item in value)
    if not sizes:
        raise ValueError(f"{name} must not be empty.")
    return sizes


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def qualified_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def state_dict_contract(model: nn.Module) -> dict[str, object]:
    state = model.state_dict(keep_vars=True)
    parameter_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    buffer_names = {name for name, _ in model.named_buffers(remove_duplicate=False)}
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
            raise TypeError(f"model state entry {name!r} must resolve to one parameter or buffer.")
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


def audio_token_space_sharing(runtime: TokenModelRuntime) -> str:
    decoupled = runtime.input_audio_decoupled
    if not isinstance(decoupled, bool):
        raise TypeError("runtime input_audio_decoupled must be a bool.")
    return "independent" if decoupled else "shared"


def audio_backend_sharing(runtime: TokenModelRuntime) -> str:
    """Describe encoder/decoder ownership independently of model token IDs."""

    input_identity = getattr(runtime, "input_audio_backend_identity", None)
    output_identity = getattr(runtime, "output_audio_backend_identity", None)
    if isinstance(input_identity, AudioBackendIdentity) and isinstance(
        output_identity,
        AudioBackendIdentity,
    ):
        return "shared" if input_identity == output_identity else "independent"
    return (
        "shared"
        if runtime.input_codec_name == runtime.codec_name
        and runtime.input_audio_view is runtime.audio_view
        else "independent"
    )


def interface_contract(model: ContractModel) -> dict[str, object]:
    tokens = model.tokens
    return {
        "control_embedding": control_embedding_contract(tokens),
        "audio_embeddings": audio_embeddings_contract(model),
        "input_audio_projection": input_audio_projection_contract(model),
        "audio_projection": adapter_contract(tokens.audio_projection),
        "audio_head": audio_head_contract(tokens.audio_head),
        "source_audio_encoder": source_audio_contract(model.source_audio_encoder),
        "ctc_decoders": ctc_decoders_contract(model.ctc_decoders),
    }


def control_embedding_contract(tokens: TokenInterface) -> dict[str, object]:
    text_start, text_end = token_block(
        tokens.layout.blocks.get(Modality.TEXT.value),
        Modality.TEXT.value,
    )
    lexical_size = positive_int(
        tokens.lexical_text_vocab_size,
        "lexical text vocabulary size",
    )
    control_rows = text_end - text_start - lexical_size
    if control_rows < 0:
        raise ValueError("lexical text vocabulary exceeds the text layout block.")
    embedding = tokens.control_embedding
    if control_rows == 0:
        if embedding is not None:
            raise ValueError("a lexical-only text block must not register a control embedding.")
        return {"family": "none"}
    if embedding is None:
        raise ValueError("a control-extended text block requires a control embedding.")
    if embedding.num_embeddings != control_rows:
        raise ValueError("control embedding rows must match the runtime control vocabulary.")
    return {
        "family": "dense-v1",
        "rows": embedding.num_embeddings,
        "dim": embedding.embedding_dim,
        "trainable": embedding.weight.requires_grad,
    }


def input_audio_projection_contract(model: ContractModel) -> dict[str, object]:
    sharing = audio_token_space_sharing(model.runtime)
    projection = model.tokens.input_audio_projection
    if sharing == "shared":
        if projection is not None:
            raise ValueError(
                "shared input/output audio token spaces must not register a separate "
                "input audio projection."
            )
        return {"sharing": "shared"}
    if projection is None:
        raise ValueError(
            "independent input/output audio token spaces require an input audio projection."
        )
    return {
        "sharing": "independent",
        "adapter": adapter_contract(projection),
    }


def audio_embeddings_contract(model: ContractModel) -> dict[str, object]:
    sharing = audio_token_space_sharing(model.runtime)
    output = audio_embedding_contract(model.tokens.audio_embedding)
    input_embedding = model.tokens.input_audio_embedding
    if sharing == "shared":
        if input_embedding is not None:
            raise ValueError(
                "shared input/output audio token spaces must not register a separate "
                "input audio embedding."
            )
        input_ = output
    else:
        if input_embedding is None:
            raise ValueError(
                "independent input/output audio token spaces require an input audio embedding."
            )
        input_ = audio_embedding_contract(input_embedding)
    return {
        "sharing": sharing,
        "input": input_,
        "output": output,
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


def runtime_contract(model: ContractModel) -> dict[str, object]:
    return {
        "token_space": token_space_contract(model.runtime),
        "audio_codecs": audio_codecs_contract(model.runtime),
        "backbone": backbone_contract(model),
    }


def audio_codecs_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    sharing = audio_backend_sharing(runtime)
    input_ = input_codec_contract(runtime)
    output = codec_contract(runtime)
    if sharing == "shared":
        shared_fields = {
            "name": output["name"],
            "audio_view": output["audio_view"],
            "frame_rate": output["frame_rate"],
        }
        if input_ != shared_fields:
            raise ValueError("shared input/output audio codec metadata must resolve identically.")
    return {
        "sharing": sharing,
        "input": input_,
        "output": output,
        "output_detokenizer": output_detokenizer_contract(runtime),
    }


def output_detokenizer_contract(
    runtime: TokenModelRuntime,
) -> dict[str, object] | None:
    name = getattr(runtime, "output_audio_detokenizer_name", runtime.codec_name)
    if name is None:
        return None
    if not isinstance(name, str) or not name:
        raise TypeError(
            "runtime output_audio_detokenizer_name must be a non-empty string or None."
        )
    identity = getattr(runtime, "output_audio_detokenizer_identity", None)
    if not isinstance(identity, AudioBackendIdentity):
        identity = audio_backend_identity(name)
    output_identity = getattr(runtime, "output_audio_backend_identity", None)
    if not isinstance(output_identity, AudioBackendIdentity):
        output_identity = audio_backend_identity(runtime.codec_name)
    input_identity = getattr(runtime, "input_audio_backend_identity", None)
    if not isinstance(input_identity, AudioBackendIdentity):
        input_identity = audio_backend_identity(runtime.input_codec_name)
    sharing = (
        "output"
        if identity == output_identity
        else "input"
        if identity == input_identity
        else "independent"
    )
    if sharing == "output":
        spec = runtime.output_audio_code_spec
    elif sharing == "input":
        spec = runtime.input_audio_code_spec
    else:
        spec = audio_code_spec(name)
    return {
        "name": name,
        "sharing": sharing,
        "backend_identity": audio_backend_identity_contract(identity),
        "code_spec_sha256": contract_sha256(audio_code_spec_contract(spec)),
    }


def audio_backend_identity_contract(
    identity: AudioBackendIdentity,
) -> dict[str, object]:
    return {
        "preset": identity.preset,
        "artifact": identity.artifact,
        "revision": identity.revision,
    }


def audio_code_spec_contract(spec: AudioCodeSpec) -> dict[str, object]:
    return {
        "view": spec.view,
        "schema": spec.schema.value,
        "sample_rate": spec.sample_rate,
        "frame_rate": spec.frame_rate,
        "frame_codebook_sizes": list(spec.frame_codebook_sizes),
        "semantic_codebook_sizes": list(spec.semantic_codebook_sizes),
        "acoustic_codebook_sizes": list(spec.acoustic_codebook_sizes),
        "global_codebook_sizes": list(spec.global_codebook_sizes),
        "acoustic_layout": (
            None if spec.acoustic_layout is None else spec.acoustic_layout.value
        ),
        "acoustic_unit_length": spec.acoustic_unit_length,
        "global_unit_length": spec.global_unit_length,
    }


def input_codec_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    names = getattr(runtime, "input_audio_stream_tokenizer_names", None)
    if names is not None and len(names) > 1:
        return composed_input_codec_contract(runtime, tuple(names))
    name = runtime.input_codec_name
    if not isinstance(name, str) or not name:
        raise TypeError("runtime input_codec_name must be a non-empty string.")
    view = runtime.input_audio_view
    if not isinstance(view, AudioView):
        raise TypeError("runtime input_audio_view must be an AudioView.")
    return {
        "name": name,
        "audio_view": view.value,
        "frame_rate": positive_float(
            runtime.input_codec_frame_rate,
            "input audio codec frame rate",
        ),
    }


def composed_input_codec_contract(
    runtime: TokenModelRuntime,
    names: tuple[tuple[AudioStream, str], ...],
) -> dict[str, object]:
    views = dict(runtime.input_audio_stream_views)
    specs = dict(runtime.input_audio_stream_code_specs)
    identities = dict(runtime.input_audio_stream_backend_identities)
    tokenizer_names = dict(names)
    expected = {AudioStream.SEMANTIC, AudioStream.GLOBAL}
    if set(views) != expected or set(specs) != expected or set(identities) != expected:
        raise ValueError(
            "composed input codec contract requires semantic and global stream metadata."
        )
    if set(tokenizer_names) != expected:
        raise ValueError(
            "composed input codec contract requires semantic and global tokenizer names."
        )
    streams: dict[str, object] = {}
    for stream in (AudioStream.SEMANTIC, AudioStream.GLOBAL):
        view = views[stream]
        if not isinstance(view, AudioView):
            raise TypeError("composed input codec views must be AudioView values.")
        streams[stream.value] = {
            "name": tokenizer_names[stream],
            "audio_view": view.value,
            "backend_identity": audio_backend_identity_contract(identities[stream]),
            "code_spec": audio_code_spec_contract(specs[stream]),
        }
    name = getattr(runtime, "input_audio_schema_codec_name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("composed input audio schema name must be a non-empty string.")
    return {
        "name": name,
        "audio_view": views[AudioStream.SEMANTIC].value,
        "frame_rate": positive_float(
            specs[AudioStream.SEMANTIC].frame_rate,
            "input semantic codec frame rate",
        ),
        "composition": "semantic-global-v1",
        "streams": streams,
    }


def codec_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    initialization = getattr(
        runtime,
        "output_audio_embedding_initialization",
        None,
    )
    if initialization == "model_random":
        return model_random_audio_tokenizer_contract(runtime)

    codec = runtime.codec
    semantic_sizes = positive_sizes(
        runtime.semantic_codebook_sizes,
        "semantic codec codebooks",
    )
    acoustic: dict[str, object] | None = None
    if supports_acoustic(codec):
        resolved = acoustic_codec(codec)
        acoustic = {
            "feature_dim": resolved.acoustic_feature_dim,
            "codebook_sizes": list(resolved.acoustic_codebook_sizes),
            "layout": "frame_aligned",
            "unit_length": None,
        }
    global_: dict[str, object] | None = None
    if supports_global(codec):
        resolved_global = global_codec(codec)
        global_ = {
            "feature_dim": resolved_global.global_feature_dim,
            "codebook_sizes": list(resolved_global.global_codebook_sizes),
            "unit_length": resolved_global.global_unit_length,
        }
    name = runtime.codec_name
    if not isinstance(name, str) or not name:
        raise TypeError("runtime codec_name must be a non-empty string.")
    view = runtime.audio_view
    if not isinstance(view, AudioView):
        raise TypeError("runtime audio_view must be an AudioView.")
    return {
        "name": name,
        "audio_view": view.value,
        "implementation": qualified_name(codec),
        "semantic_artifact_sha256": semantic_artifact_sha256(runtime),
        "sample_rate": codec_sample_rate(codec),
        "frame_rate": codec_frame_rate(codec),
        "semantic_codebook_sizes": list(semantic_sizes),
        "semantic_feature_dim": semantic_feature_dim(codec),
        "fsq_levels": fsq_levels(codec),
        "fsq_level_values": fsq_level_values(codec),
        "fsq_radix_order": fsq_radix_order(codec),
        "acoustic": acoustic,
        "global": global_,
    }


def model_random_audio_tokenizer_contract(
    runtime: TokenModelRuntime,
) -> dict[str, object]:
    name = runtime.codec_name
    if not isinstance(name, str) or not name:
        raise TypeError("runtime codec_name must be a non-empty string.")
    view = runtime.audio_view
    if not isinstance(view, AudioView):
        raise TypeError("runtime audio_view must be an AudioView.")
    identity = getattr(runtime, "output_audio_backend_identity", None)
    if not isinstance(identity, AudioBackendIdentity):
        identity = audio_backend_identity(name)
    spec = runtime.output_audio_code_spec
    return {
        "name": name,
        "audio_view": view.value,
        "frame_rate": positive_float(
            spec.frame_rate,
            "output audio tokenizer frame rate",
        ),
        "initialization": "model_random",
        "backend_identity": audio_backend_identity_contract(identity),
        "code_spec": audio_code_spec_contract(spec),
        "semantic_artifact_sha256": semantic_artifact_sha256(runtime),
    }


def token_space_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    blocks = runtime.layout.blocks
    if not isinstance(blocks, Mapping):
        raise TypeError("runtime token layout blocks must be a mapping.")
    sharing = audio_token_space_sharing(runtime)
    input_block_name = runtime.input_audio_block_name
    if not isinstance(input_block_name, str) or not input_block_name:
        raise TypeError("runtime input_audio_block_name must be a non-empty string.")
    expected_input_block = "audio" if sharing == "shared" else "audio_input"
    if input_block_name != expected_input_block:
        raise ValueError(
            "runtime input audio block does not match audio token-space sharing: "
            f"expected {expected_input_block!r}, got {input_block_name!r}."
        )
    block_names = (
        (Modality.TEXT.value, Modality.AUDIO.value)
        if sharing == "shared"
        else (Modality.TEXT.value, input_block_name, Modality.AUDIO.value)
    )
    if set(blocks) != set(block_names):
        raise ValueError(
            "runtime token layout blocks do not match the input/output audio contract."
        )
    token_blocks = {name: list(token_block(blocks.get(name), name)) for name in block_names}
    text_controls, lexical_text_vocab_size = text_control_contract(
        runtime,
        token_blocks[Modality.TEXT.value],
    )
    sequence_layout = runtime.audio_sequence_layout
    if not isinstance(sequence_layout, AudioSequenceLayout):
        raise TypeError("runtime audio sequence layout must be an AudioSequenceLayout.")
    input_schema = input_audio_schema_contract(
        runtime,
        token_blocks[input_block_name],
    )
    output_schema = output_audio_schema_contract(
        runtime,
        token_blocks[Modality.AUDIO.value],
    )
    if sharing == "shared" and input_schema != output_schema:
        raise ValueError("shared input/output audio schemas must resolve identically.")
    return {
        "audio_sequence_layout": sequence_layout.value,
        "blocks": token_blocks,
        "special_ids": {
            "text": {
                "pad": non_negative_int(
                    runtime.pad_token_id,
                    "runtime pad_token_id",
                ),
                "bos": non_negative_int(
                    runtime.bos_token_id,
                    "runtime bos_token_id",
                ),
                "eos": non_negative_int(
                    runtime.eos_token_id,
                    "runtime eos_token_id",
                ),
            },
            "input_audio": {
                "boa": non_negative_int(
                    runtime.input_boa_token_id,
                    "runtime input_boa_token_id",
                ),
                "eoa": non_negative_int(
                    runtime.input_eoa_token_id,
                    "runtime input_eoa_token_id",
                ),
            },
            "output_audio": {
                "boa": non_negative_int(
                    runtime.boa_token_id,
                    "runtime boa_token_id",
                ),
                "eoa": non_negative_int(
                    runtime.eoa_token_id,
                    "runtime eoa_token_id",
                ),
                "mask": non_negative_int(
                    runtime.mask_token_id,
                    "runtime mask_token_id",
                ),
            },
        },
        "text_controls": text_controls,
        "text_tokenizer": text_tokenizer_contract(
            runtime.text_tokenizer,
            lexical_text_vocab_size,
            configured_chat_template=runtime.backbone_chat_template,
        ),
        "audio_schemas": {
            "sharing": sharing,
            "input": input_schema,
            "output": output_schema,
        },
    }


def input_audio_schema_contract(
    runtime: TokenModelRuntime,
    audio_block: Sequence[int],
) -> dict[str, object]:
    reserved_ids = (
        (
            runtime.input_boa_token_id,
            runtime.input_eoa_token_id,
        )
        if runtime.input_audio_decoupled
        else (
            runtime.input_boa_token_id,
            runtime.input_eoa_token_id,
            runtime.mask_token_id,
        )
    )
    return audio_schema_contract(
        spec=runtime.input_audio_token_spec,
        tokenizer=runtime.input_audio_tokenizer,
        expected_schema_id=runtime.input_audio_schema_id,
        selector_id=runtime.input_audio_schema_token_id,
        payload_range=runtime.input_codec_audio_range,
        audio_block=audio_block,
        reserved_ids=reserved_ids,
        side="input",
        codec_name=getattr(
            runtime,
            "input_audio_schema_codec_name",
            runtime.input_codec_name,
        ),
        sequence_layout=runtime.audio_sequence_layout,
    )


def output_audio_schema_contract(
    runtime: TokenModelRuntime,
    audio_block: Sequence[int],
) -> dict[str, object]:
    return audio_schema_contract(
        spec=runtime.output_audio_token_spec,
        tokenizer=runtime.output_audio_tokenizer,
        expected_schema_id=runtime.output_audio_schema_id,
        selector_id=runtime.output_audio_schema_token_id,
        payload_range=runtime.output_codec_audio_range,
        audio_block=audio_block,
        reserved_ids=(
            runtime.output_boa_token_id,
            runtime.output_eoa_token_id,
            runtime.output_mask_token_id,
        ),
        side="output",
        codec_name=runtime.output_codec_name,
        sequence_layout=runtime.audio_sequence_layout,
    )


def audio_schema_contract(
    *,
    spec: AudioTokenSpec,
    tokenizer: AudioTokenizer,
    expected_schema_id: str,
    selector_id: int,
    payload_range: Sequence[int],
    audio_block: Sequence[int],
    reserved_ids: Sequence[int],
    side: str,
    codec_name: str,
    sequence_layout: AudioSequenceLayout,
) -> dict[str, object]:
    if not isinstance(spec, AudioTokenSpec):
        raise TypeError(f"runtime {side} audio token spec must be an AudioTokenSpec.")
    if spec.tokenizer is not tokenizer:
        raise ValueError(
            f"runtime {side} audio token spec must own its configured tokenizer."
        )
    if not isinstance(expected_schema_id, str) or not expected_schema_id:
        raise ValueError(f"runtime {side} audio schema id must be non-empty.")
    if spec.schema_id != expected_schema_id:
        raise ValueError(
            f"runtime {side} audio schema id does not match its token spec."
        )
    if spec.codec_name != codec_name:
        raise ValueError(
            f"runtime {side} audio token spec does not match its codec name."
        )
    if spec.sequence_layout != sequence_layout.value:
        raise ValueError(
            f"runtime {side} audio token spec does not match its sequence layout."
        )

    block_start, block_end = token_block(audio_block, f"{side} audio")
    payload_start, payload_end = token_block(
        payload_range,
        f"{side} audio payload",
    )
    if payload_start != block_start:
        raise ValueError(
            f"runtime {side} audio payload must begin at its layout block start."
        )
    tokenizer_contract = audio_tokenizer_contract(tokenizer)
    tokenizer_vocab_size = positive_int(
        tokenizer.vocab_size,
        f"runtime {side} audio tokenizer vocabulary size",
    )
    if payload_end - payload_start != tokenizer_vocab_size:
        raise ValueError(
            f"runtime {side} audio payload range must match its tokenizer vocabulary."
        )
    controls = tuple(
        non_negative_int(token_id, f"runtime {side} audio reserved token id")
        for token_id in reserved_ids
    )
    expected_controls = tuple(range(payload_end, payload_end + len(controls)))
    if controls != expected_controls:
        raise ValueError(
            f"runtime {side} audio reserved tokens must immediately follow its payload."
        )
    resolved_selector_id = non_negative_int(
        selector_id,
        f"runtime {side} audio schema selector id",
    )
    if resolved_selector_id != payload_end + len(controls):
        raise ValueError(
            f"runtime {side} audio schema selector must follow its reserved tokens."
        )
    if block_end != resolved_selector_id + 1:
        raise ValueError(
            f"runtime {side} audio schema selector must end its layout block."
        )

    spec_state = contract_state(spec)
    if spec_state.get("schema_id") != expected_schema_id:
        raise ValueError(
            f"runtime {side} audio token spec contract has an inconsistent schema id."
        )
    selector = spec_state.get("selector")
    if not isinstance(selector, str) or not selector:
        raise ValueError(
            f"runtime {side} audio token spec contract requires a selector."
        )
    tokenizer_state = contract_state(tokenizer)
    private_grammar = audio_private_grammar_contract(spec)
    if spec_state.get("tokenizer_grammar") != state_grammar(tokenizer_state):
        raise ValueError(
            f"runtime {side} audio token spec tokenizer grammar is inconsistent."
        )
    if spec_state.get("codec_grammar") != private_grammar:
        raise ValueError(
            f"runtime {side} audio token spec private grammar is inconsistent."
        )
    combined_digest = contract_sha256(
        {
            "tokenizer": tokenizer_state,
            "codec_grammar": private_grammar,
        }
    )
    if spec_state.get("tokenizer_state_sha256") != combined_digest:
        raise ValueError(
            f"runtime {side} audio token spec digest does not match its contracts."
        )
    return {
        "schema_id": expected_schema_id,
        "selector": selector,
        "selector_id": resolved_selector_id,
        "payload_range": [payload_start, payload_end],
        "spec": spec_state,
        "tokenizer": tokenizer_contract,
        "private_grammar": private_grammar,
    }


def text_control_contract(
    runtime: TokenModelRuntime,
    text_block: Sequence[int],
) -> tuple[dict[str, object], int]:
    if len(text_block) != 2:
        raise ValueError("text token block must contain start and end bounds.")
    text_start, text_end = text_block
    text_size = text_end - text_start
    lexical_value = getattr(runtime, "lexical_text_vocab_size", text_size)
    lexical_size = positive_int(
        lexical_value,
        "runtime lexical_text_vocab_size",
    )
    if lexical_size > text_size:
        raise ValueError("runtime lexical text vocabulary exceeds its text block.")
    control_rows = text_size - lexical_size
    if control_rows == 0:
        return {"grammar": "typed-text-control-v1", "tokens": {}}, lexical_size
    if control_rows != len(ControlToken):
        raise ValueError(
            "runtime text control rows must exactly cover the fixed control vocabulary."
        )
    ids = runtime.control_token_ids
    expected_ids = tuple(range(text_start + lexical_size, text_end))
    if tuple(ids) != expected_ids:
        raise ValueError("runtime control token ids must occupy the tail of the text block.")
    tokens: dict[str, int] = {}
    for token, token_id in zip(ControlToken, ids):
        resolved = runtime.control_token_id(token)
        if resolved != token_id:
            raise ValueError("runtime control token lookup is inconsistent.")
        tokens[token.value] = non_negative_int(
            token_id,
            f"runtime control token {token.value}",
        )
    return {
        "grammar": "typed-text-control-v1",
        "tokens": tokens,
    }, lexical_size


def text_tokenizer_contract(
    tokenizer: TextTokenizer,
    vocab_size: int,
    *,
    configured_chat_template: str | None,
) -> dict[str, object]:
    state = text_tokenizer_state(tokenizer)
    chat_template = configured_chat_template
    if chat_template is None and isinstance(tokenizer, ChatTemplateProvider):
        chat_template = tokenizer.chat_template
    if chat_template is not None and not isinstance(chat_template, str):
        raise TypeError("text tokenizer chat template must be a string or None.")
    return {
        "implementation": qualified_name(tokenizer),
        "vocab_size": positive_int(vocab_size, "text tokenizer vocabulary size"),
        "state_grammar": state_grammar(state),
        "state_sha256": contract_sha256(state),
        "special_tokens_sha256": contract_sha256(tokenizer.special_tokens_map),
        "chat_template_sha256": (None if chat_template is None else contract_sha256(chat_template)),
    }


def text_tokenizer_state(tokenizer: TextTokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, ContractStateProvider):
        return contract_state(tokenizer)
    backend = tokenizer_backend_state(tokenizer)
    if backend is not None:
        return {
            "grammar": "serialized-tokenizer-backend-v1",
            "backend": backend,
        }
    behavior = text_tokenizer_behavior(tokenizer)
    if isinstance(tokenizer, VocabularyProvider):
        vocab = tokenizer.get_vocab()
        if not isinstance(vocab, Mapping):
            raise TypeError("tokenizer get_vocab() must return a mapping.")
        return {
            "grammar": "get-vocab-behavior-v1",
            "vocab": vocab,
            "behavior": behavior,
        }
    return {
        "grammar": "text-tokenizer-interface-v2",
        "implementation": qualified_name(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
        "behavior": behavior,
    }


def tokenizer_backend_state(tokenizer: TextTokenizer) -> dict[str, Any] | None:
    if not isinstance(tokenizer, BackendTokenizerProvider):
        return None
    backend = tokenizer.backend_tokenizer
    if not isinstance(backend, TokenizerBackendSerializer):
        return None
    serialized = backend.to_str()
    if not isinstance(serialized, str):
        raise TypeError("tokenizer backend to_str() must return a string.")
    if not serialized:
        raise ValueError("tokenizer backend to_str() must not return an empty string.")
    try:
        state = json.loads(serialized)
    except json.JSONDecodeError:
        return {
            "format": "text-v1",
            "content_sha256": contract_sha256(serialized),
        }
    return {
        "format": "json-v1",
        "content_sha256": contract_sha256(state),
    }


def text_tokenizer_behavior(tokenizer: TextTokenizer) -> list[dict[str, object]]:
    return [
        {
            "text": text,
            "plain_ids": tokenizer_probe_ids(tokenizer.encode(text, add_special_tokens=False)),
            "special_ids": tokenizer_probe_ids(tokenizer.encode(text, add_special_tokens=True)),
        }
        for text in (
            "Hello, world!",
            "UPPER lower  spaces\tand\nlines",
            "中文标点，café e\u0301 — 🙂",
        )
    ]


def tokenizer_probe_ids(values: object) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("tokenizer encode() contract probe must return a sequence.")
    result: list[int] = []
    for value in values:
        if not isinstance(value, Integral):
            raise TypeError("tokenizer encode() contract probe must return integer token ids.")
        result.append(int(value))
    return result


def audio_tokenizer_contract(
    tokenizer: AudioTokenizer,
) -> dict[str, object]:
    state = contract_state(tokenizer)
    return {
        "implementation": qualified_name(tokenizer),
        "vocab_size": positive_int(
            tokenizer.vocab_size,
            "audio tokenizer vocabulary size",
        ),
        "state_grammar": state_grammar(state),
        "state_sha256": contract_sha256(state),
    }


def audio_private_grammar_contract(
    spec: AudioTokenSpec,
) -> dict[str, object]:
    return contract_state(spec.grammar)


def backbone_contract(model: ContractModel) -> dict[str, object]:
    encoder_state = canonical_value(model._encoder.contract_state())
    if not isinstance(encoder_state, dict):
        raise TypeError("backbone encoder contract_state() must return a mapping.")
    config = backbone_config_state(model.backbone.config)
    return {
        "implementation": qualified_name(model.backbone),
        "encoder": encoder_state,
        "hidden_size": positive_int(
            model.backbone.config.hidden_size,
            "backbone hidden size",
        ),
        "architecture_sha256": contract_sha256(config),
    }


def semantic_artifact_sha256(runtime: TokenModelRuntime) -> str | None:
    artifact = runtime.acoustic_generator_artifact
    if artifact is not None and (not isinstance(artifact, str) or not artifact):
        raise TypeError("acoustic generator artifact must be a non-empty string or None.")
    digest = optional_sha256(
        runtime.acoustic_generator_artifact_sha256,
        "acoustic generator artifact digest",
    )
    if (artifact is None) != (digest is None):
        raise ValueError(
            "acoustic generator artifact path and content digest must be configured together."
        )
    return digest


def flow_acoustic_contract(
    condition: _HiddenConditionAdapter,
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
    condition: _HiddenConditionAdapter,
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
    value: _HiddenConditionAdapter,
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


def build_model_contract(
    model: ContractModel,
    acoustic: Mapping[str, object],
) -> ModelCheckpointContract:
    """Build a contract from resolved runtime resources and realized modules."""
    if not isinstance(model, nn.Module):
        raise TypeError("checkpoint contract models must be torch modules.")
    return ModelCheckpointContract.from_components(
        {
            "runtime": runtime_contract(model),
            "interface": interface_contract(model),
            "acoustic": canonical_value(acoustic),
            "state_dict": state_dict_contract(model),
        }
    )


__all__ = [
    "MODEL_CONTRACT_GRAMMAR",
    "ModelCheckpointContract",
    "ModelCheckpointContractPayload",
    "build_model_contract",
    "canonical_value",
    "condition_adapter_contract",
    "contract_sha256",
    "flow_acoustic_contract",
    "rvq_acoustic_contract",
    "state_dict_contract",
    "validate_checkpoint_contract",
]
