"""Canonical checkpoint-contract payloads and validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

import torch


MODEL_CONTRACT_GRAMMAR = "s2s-model-v3-contract-v2"
_MODEL_CONTRACT_FIELDS = frozenset({"grammar", "components", "sha256"})
_MISSING = "<missing>"
_DIFFERENCE_KEY_ORDER = {
    "components": ("runtime", "interface", "acoustic", "state_dict"),
    "components.runtime": ("token_space", "codec", "backbone"),
    "components.runtime.token_space": (
        "audio_sequence_layout",
        "blocks",
        "special_ids",
        "text_tokenizer",
        "audio_tokenizer",
    ),
    "components.interface": (
        "audio_embedding",
        "audio_projection",
        "audio_head",
        "source_audio_encoder",
        "ctc_decoders",
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
        raise ValueError(
            "checkpoint model contract fields do not match its grammar."
        )
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
        return {
            key: canonical_value(value[key])
            for key in sorted(value)
        }
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
    raise TypeError(
        f"unsupported model contract value: {type(value).__name__}."
    )


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
        order = {
            key: index
            for index, key in enumerate(_DIFFERENCE_KEY_ORDER.get(path, ()))
        }
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


__all__ = [
    "MODEL_CONTRACT_GRAMMAR",
    "ModelCheckpointContract",
    "ModelCheckpointContractPayload",
    "canonical_value",
    "contract_sha256",
    "validate_checkpoint_contract",
]
