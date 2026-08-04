from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from anydataset.types import Modality
from anytrain.codec import AcousticLayout

from ...runtime import AudioSequenceLayout
from ...runtime.protocol import TokenModelRuntime
from ...runtime.codec_contract import (
    acoustic_codec,
    codec_frame_rate,
    codec_sample_rate,
    fsq_level_values,
    fsq_levels,
    fsq_radix_order,
    semantic_feature_dim,
    structured_codec,
    supports_acoustic,
    supports_structured,
)
from ...runtime.tokenizer import TextTokenizer
from ._protocol import (
    BackendTokenizerProvider,
    ChatTemplateProvider,
    ContractModel,
    ContractStateProvider,
    TokenizerBackendSerializer,
    VocabularyProvider,
)
from ._value import (
    backbone_config_state,
    contract_state,
    non_negative_int,
    optional_sha256,
    positive_int,
    positive_sizes,
    qualified_name,
    state_grammar,
    token_block,
)
from .core import canonical_value, contract_sha256


def runtime_contract(model: ContractModel) -> dict[str, object]:
    return {
        "codec": codec_contract(model.runtime),
        "token_space": token_space_contract(model.runtime),
        "backbone": backbone_contract(model),
    }


def codec_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    codec = runtime.codec
    semantic_sizes = positive_sizes(
        runtime.semantic_codebook_sizes,
        "semantic codec codebooks",
    )
    acoustic: dict[str, object] | None = None
    if supports_acoustic(codec):
        resolved = acoustic_codec(codec)
        if supports_structured(codec):
            structured = structured_codec(codec)
            layout = structured.acoustic_layout
            unit_length = structured.acoustic_unit_length
        else:
            layout = AcousticLayout.FRAME_ALIGNED
            unit_length = None
        acoustic = {
            "feature_dim": resolved.acoustic_feature_dim,
            "codebook_sizes": list(resolved.acoustic_codebook_sizes),
            "layout": layout.value,
            "unit_length": unit_length,
        }
    name = runtime.codec_name
    if not isinstance(name, str) or not name:
        raise TypeError("runtime codec_name must be a non-empty string.")
    return {
        "name": name,
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
    }


def token_space_contract(runtime: TokenModelRuntime) -> dict[str, object]:
    blocks = runtime.layout.blocks
    if not isinstance(blocks, Mapping):
        raise TypeError("runtime token layout blocks must be a mapping.")
    token_blocks = {
        modality.value: list(token_block(blocks.get(modality.value), modality))
        for modality in (Modality.TEXT, Modality.AUDIO)
    }
    sequence_layout = runtime.audio_sequence_layout
    if not isinstance(sequence_layout, AudioSequenceLayout):
        raise TypeError("runtime audio sequence layout must be an AudioSequenceLayout.")
    return {
        "audio_sequence_layout": sequence_layout.value,
        "blocks": token_blocks,
        "special_ids": {
            "pad": non_negative_int(runtime.pad_token_id, "runtime pad_token_id"),
            "bos": non_negative_int(runtime.bos_token_id, "runtime bos_token_id"),
            "eos": non_negative_int(runtime.eos_token_id, "runtime eos_token_id"),
            "boa": non_negative_int(runtime.boa_token_id, "runtime boa_token_id"),
            "eoa": non_negative_int(runtime.eoa_token_id, "runtime eoa_token_id"),
            "mask": non_negative_int(
                runtime.mask_token_id,
                "runtime mask_token_id",
            ),
        },
        "text_tokenizer": text_tokenizer_contract(
            runtime.text_tokenizer,
            token_blocks["text"],
            configured_chat_template=runtime.backbone_chat_template,
        ),
        "audio_tokenizer": audio_tokenizer_contract(runtime),
    }


def text_tokenizer_contract(
    tokenizer: TextTokenizer,
    text_block: Sequence[int],
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
        "vocab_size": text_block[1] - text_block[0],
        "state_grammar": state_grammar(state),
        "state_sha256": contract_sha256(state),
        "special_tokens_sha256": contract_sha256(tokenizer.special_tokens_map),
        "chat_template_sha256": (
            None if chat_template is None else contract_sha256(chat_template)
        ),
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
            "plain_ids": tokenizer_probe_ids(
                tokenizer.encode(text, add_special_tokens=False)
            ),
            "special_ids": tokenizer_probe_ids(
                tokenizer.encode(text, add_special_tokens=True)
            ),
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
            raise TypeError(
                "tokenizer encode() contract probe must return integer token ids."
            )
        result.append(int(value))
    return result


def audio_tokenizer_contract(
    runtime: TokenModelRuntime,
) -> dict[str, object]:
    tokenizer = runtime.audio_tokenizer
    state = contract_state(tokenizer)
    result: dict[str, object] = {
        "implementation": qualified_name(tokenizer),
        "grammar": state_grammar(state),
        "vocab_size": positive_int(
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

__all__ = ["runtime_contract"]
