from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Union, cast

from anydataset.types import AudioView, Modality
from anytrain.codec import AudioBackendIdentity, AudioCodeSpec
import torch

from .._compat import StrEnum, auto
from .backbone import BackboneInitialization, BackboneType, validate_backbone_readout
from .codec import (
    audio_backend_identity,
    audio_code_spec,
    has_audio_detokenizer_capability,
    has_audio_tokenizer_capability,
)


_FLOW_METHODS = frozenset(
    {
        "adaptive_heun",
        "bosh3",
        "dopri5",
        "dopri8",
        "euler",
        "explicit_adams",
        "fehlberg2",
        "fixed_adams",
        "heun2",
        "heun3",
        "implicit_adams",
        "midpoint",
        "rk4",
        "scipy_solver",
    }
)

_GENERATOR_ARTIFACT_FIELD = "acoustic_generator_artifact"
_LEGACY_GENERATOR_ARTIFACT_FIELD = "semantic_codec_artifact"
_AUDIO_INPUT_FIELDS = ("tokenizer", "bpe", "vocab_size", "frame_rate")


class _Unset:
    pass


_UNSET = _Unset()


def migrate_config_fields(fields: MutableMapping[str, Any]) -> None:
    """Migrate flat/input-first runtime aliases at the external config boundary."""

    output, output_present, output_side_migrated = _canonical_output(fields)
    input_, input_present, input_side_migrated = _canonical_input(fields)
    migrated: list[str] = []
    if output_side_migrated:
        migrated.append("audio_output.codec/tokenizer")
    if input_side_migrated:
        migrated.append("audio_input.codec/tokenizer")

    legacy_artifact = _pop_external(fields, _LEGACY_GENERATOR_ARTIFACT_FIELD)
    flat_artifact = _pop_external(fields, _GENERATOR_ARTIFACT_FIELD)
    if legacy_artifact is not _UNSET:
        if (
            flat_artifact is not _UNSET
            and flat_artifact != legacy_artifact
        ):
            raise ValueError(
                "runtime config contains conflicting acoustic_generator_artifact and "
                "legacy semantic_codec_artifact values."
            )
        flat_artifact = legacy_artifact
        migrated.append(_LEGACY_GENERATOR_ARTIFACT_FIELD)

    legacy_codec = _pop_external(fields, "codec")
    legacy_output = {
        "tokenizer": legacy_codec,
        "detokenizer": legacy_codec,
        "bpe": _pop_external(fields, "audio_tokenizer"),
        _GENERATOR_ARTIFACT_FIELD: flat_artifact,
    }
    for name, value in legacy_output.items():
        if value is _UNSET or (name in {"tokenizer", "detokenizer"} and value is None):
            continue
        _merge_leaf(output, name, value, side="audio_output")
        migrated.append(name if name == "tokenizer" else f"legacy {name}")

    legacy_input = _pop_external(fields, "input_audio")
    resolved_legacy_input, legacy_input_migrated = _optional_audio_input_mapping(
        legacy_input,
        name="runtime.input_audio",
    )
    if not isinstance(resolved_legacy_input, _Unset) and resolved_legacy_input is not None:
        if input_ is None:
            input_ = {}
        for name, value in resolved_legacy_input.items():
            _merge_leaf(input_, name, value, side="audio_input")
        input_present = True
        migrated.append("input_audio")
        if legacy_input_migrated:
            migrated.append("input_audio.codec/tokenizer")

    if output_present or output:
        fields["audio_output"] = output
    if input_present:
        fields["audio_input"] = _collapse_audio_input(input_)

    if migrated:
        warnings.warn(
            "legacy runtime audio fields are deprecated; use runtime.audio_input "
            "and runtime.audio_output.",
            FutureWarning,
            stacklevel=2,
        )


def _canonical_output(
    fields: MutableMapping[str, Any],
) -> tuple[dict[str, Any], bool, bool]:
    value = _external(fields, "audio_output")
    if value is _UNSET:
        return {}, False, False
    if isinstance(value, AudioOutputConfig):
        return {
            "tokenizer": value.tokenizer,
            "detokenizer": value.detokenizer,
            "bpe": value.bpe,
            _GENERATOR_ARTIFACT_FIELD: value.acoustic_generator_artifact,
        }, True, False
    result = _nested_mapping(value, "runtime.audio_output")
    return result, True, _migrate_side_fields(result, side="audio_output")


def _canonical_input(
    fields: MutableMapping[str, Any],
) -> tuple[dict[str, Any] | None, bool, bool]:
    value = _external(fields, "audio_input")
    if value is _UNSET:
        return None, False, False
    resolved, migrated = _optional_audio_input_mapping(
        value,
        name="runtime.audio_input",
    )
    if isinstance(resolved, _Unset):
        return None, False, migrated
    return _collapse_audio_input(resolved), True, migrated


def _optional_audio_input_mapping(
    value: object,
    *,
    name: str,
) -> tuple[dict[str, Any] | None | _Unset, bool]:
    if value is _UNSET:
        return _UNSET, False
    if value is None:
        return None, False
    if isinstance(value, AudioInputConfig):
        return _collapse_audio_input({
            "tokenizer": value.tokenizer,
            "bpe": value.bpe,
            "vocab_size": value.vocab_size,
            "frame_rate": value.frame_rate,
        }), False
    result = _nested_mapping(value, name)
    migrated = _migrate_side_fields(
        result,
        side="audio_input",
    )
    return _collapse_audio_input(result), migrated


def _migrate_side_fields(fields: dict[str, Any], *, side: str) -> bool:
    """Resolve the intermediate ``codec + tokenizer(BPE)`` side schema."""

    codec = fields.pop("codec", _UNSET)
    view = fields.pop("view", _UNSET)
    if codec is _UNSET:
        tokenizer = fields.get("tokenizer", _UNSET)
        _validate_migrated_view(tokenizer, view, side)
        return view is not _UNSET

    legacy_tokenizer = fields.pop("tokenizer", _UNSET)
    canonical_bpe = fields.pop("bpe", _UNSET)
    # ``codec`` was the backend name in the intermediate schema, while its
    # ``tokenizer`` leaf meant the BPE artifact.  Once the canonical schema
    # also uses ``tokenizer`` for the backend, an equal pair is unambiguously
    # the same backend expressed through both names, not a relative BPE path.
    legacy_bpe = (
        _UNSET
        if legacy_tokenizer is not _UNSET and legacy_tokenizer == codec
        else legacy_tokenizer
    )
    if (
        legacy_bpe is not _UNSET
        and canonical_bpe is not _UNSET
        and legacy_bpe != canonical_bpe
    ):
        raise ValueError(
            f"runtime config contains conflicting {side}.tokenizer and "
            f"{side}.bpe values under the legacy codec schema."
        )
    fields["tokenizer"] = codec
    if side == "audio_output":
        detokenizer = fields.get("detokenizer", _UNSET)
        if detokenizer is not _UNSET and detokenizer != codec:
            raise ValueError(
                "runtime config contains conflicting audio_output.detokenizer "
                "and legacy codec values."
            )
        fields["detokenizer"] = codec
    bpe = legacy_bpe if legacy_bpe is not _UNSET else canonical_bpe
    if bpe is not _UNSET:
        fields["bpe"] = bpe
    _validate_migrated_view(codec, view, side)
    return True


def _validate_migrated_view(tokenizer: object, view: object, side: str) -> None:
    if view is _UNSET or view is None:
        return
    if tokenizer is _UNSET or tokenizer is None:
        raise ValueError(f"runtime.{side}.view requires runtime.{side}.tokenizer.")
    if not isinstance(tokenizer, str):
        raise TypeError(f"runtime.{side}.tokenizer must be a string.")
    try:
        resolved_view = view if isinstance(view, AudioView) else AudioView(str(view))
    except ValueError as error:
        raise ValueError(f"unsupported runtime.{side}.view: {view!r}.") from error
    expected = tokenizer_audio_view(tokenizer)
    if resolved_view is not expected:
        raise ValueError(
            f"runtime.{side}.view must match tokenizer {tokenizer!r}: "
            f"{resolved_view.value!r} != {expected.value!r}."
        )


def _nested_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or null.")
    result: dict[str, Any] = {}
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
        if _is_missing(value, key):
            continue
        result[key] = value[key]
    return result


def _collapse_audio_input(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if set(value).issubset(_AUDIO_INPUT_FIELDS) and all(
        value.get(name) is None for name in _AUDIO_INPUT_FIELDS
    ):
        return None
    return value


def _merge_leaf(
    fields: dict[str, Any],
    name: str,
    value: object,
    *,
    side: str,
) -> None:
    if name in fields and fields[name] != value:
        raise ValueError(
            f"runtime config contains conflicting {side}.{name} and legacy values."
        )
    fields[name] = value


def _external(fields: Mapping[str, Any], name: str) -> object:
    if name not in fields.keys() or _is_missing(fields, name):
        return _UNSET
    return fields[name]


def _pop_external(fields: MutableMapping[str, Any], name: str) -> object:
    if name not in fields.keys():
        return _UNSET
    if _is_missing(fields, name):
        del fields[name]
        return _UNSET
    return fields.pop(name)


def _is_missing(fields: Mapping[str, Any], name: str) -> bool:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return False
    return bool(OmegaConf.is_config(fields) and OmegaConf.is_missing(fields, name))


def _audio_side_resources(
    *,
    tokenizer: Union[str, None, _Unset],
    bpe: Union[str, Path, None, _Unset],
    codec: Union[str, None, _Unset],
    default_tokenizer: Optional[str],
    side: str,
) -> tuple[Optional[str], Optional[Union[str, Path]]]:
    tokenizer_set = not isinstance(tokenizer, _Unset)
    bpe_set = not isinstance(bpe, _Unset)
    resolved_tokenizer = default_tokenizer if not tokenizer_set else tokenizer
    resolved_bpe = None if not bpe_set else bpe
    if isinstance(codec, _Unset):
        return (
            cast(Optional[str], resolved_tokenizer),
            cast(Optional[Union[str, Path]], resolved_bpe),
        )

    warnings.warn(
        f"{side}.codec is deprecated; use {side}.tokenizer for the backend "
        f"and {side}.bpe for the optional BPE artifact.",
        FutureWarning,
        stacklevel=3,
    )
    resolved_tokenizer = codec
    # ``codec`` may be combined with the canonical backend leaf during a
    # staged migration.  Equal names describe one backend.  A different
    # ``tokenizer`` value retains the intermediate meaning (the BPE artifact).
    canonical_tokenizer = tokenizer_set and tokenizer == codec
    if canonical_tokenizer:
        resolved_bpe = None if not bpe_set else bpe
    elif tokenizer_set and bpe_set:
        if tokenizer != bpe:
            raise ValueError(
                f"runtime config contains conflicting {side}.tokenizer and "
                f"{side}.bpe values under the legacy codec schema."
            )
        resolved_bpe = bpe
    elif tokenizer_set:
        resolved_bpe = tokenizer
    elif bpe_set:
        resolved_bpe = bpe
    else:
        resolved_bpe = None
    return (
        cast(Optional[str], resolved_tokenizer),
        cast(Optional[Union[str, Path]], resolved_bpe),
    )


def _artifact_identity(value: Optional[Union[str, Path]]) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


class AudioSequenceLayout(StrEnum):
    FLATTENED = auto()
    SEMANTIC = auto()


@dataclass(frozen=True)
class _AudioInputFields:
    tokenizer: Optional[str] = None
    bpe: Optional[Union[str, Path]] = None
    vocab_size: Optional[int] = None
    frame_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            if any(
                value is not None
                for value in (
                    self.bpe,
                    self.vocab_size,
                    self.frame_rate,
                )
            ):
                raise ValueError(
                    "runtime.audio_input tokenizer is required when configuring its "
                    "bpe, vocab_size, or frame_rate."
                )
            return
        if not isinstance(self.tokenizer, str):
            raise TypeError("runtime.audio_input tokenizer must be a string or None.")
        if not self.tokenizer:
            raise ValueError("runtime.audio_input tokenizer must not be empty.")
        spec = _audio_tokenizer_spec(
            self.tokenizer,
            "runtime.audio_input tokenizer",
        )
        _validate_bpe_path(self.bpe, "runtime.audio_input bpe")
        if self.vocab_size is not None or self.frame_rate is not None:
            warnings.warn(
                "runtime.audio_input vocab_size/frame_rate are deprecated; "
                "registered tokenizer presets own this metadata.",
                FutureWarning,
                stacklevel=3,
            )
        if self.vocab_size is not None:
            if isinstance(self.vocab_size, bool) or not isinstance(
                self.vocab_size,
                int,
            ):
                raise TypeError(
                    "runtime.audio_input vocab_size must be an integer or None."
                )
            if self.vocab_size <= 0:
                raise ValueError("runtime.audio_input vocab_size must be positive.")
            if self.bpe is None and self.vocab_size != spec.primary_codebook_sizes[0]:
                raise ValueError(
                    "runtime.audio_input vocab_size does not match the audio "
                    "tokenizer preset."
                )
        if self.frame_rate is not None:
            if isinstance(self.frame_rate, bool) or not isinstance(
                self.frame_rate,
                (int, float),
            ):
                raise TypeError(
                    "runtime.audio_input frame_rate must be numeric or None."
                )
            if not math.isfinite(float(self.frame_rate)) or self.frame_rate <= 0:
                raise ValueError(
                    "runtime.audio_input frame_rate must be finite and positive."
                )
            if not math.isclose(float(self.frame_rate), spec.frame_rate):
                raise ValueError(
                    "runtime.audio_input frame_rate does not match the audio "
                    "tokenizer preset."
                )

    @property
    def audio_view(self) -> AudioView:
        if self.tokenizer is None:
            raise RuntimeError("shared audio input does not have an independent view.")
        return tokenizer_audio_view(self.tokenizer)

    @property
    def token_space_identity(
        self,
    ) -> tuple[AudioBackendIdentity, AudioCodeSpec, str | None]:
        if self.tokenizer is None:
            raise RuntimeError("shared audio input has no independent token identity.")
        return (
            audio_backend_identity(self.tokenizer),
            audio_code_spec(self.tokenizer),
            _artifact_identity(self.bpe),
        )

    @property
    def codec(self) -> Optional[str]:
        """Deprecated backend-name alias retained for direct Python callers."""

        return self.tokenizer

    @property
    def view(self) -> Optional[AudioView]:
        """Deprecated derived-view alias retained for direct Python callers."""

        return None if self.tokenizer is None else self.audio_view


@dataclass(frozen=True, init=False)
class AudioInputConfig(_AudioInputFields):
    """Source audio backend plus its optional token-sequence BPE artifact."""

    def __init__(
        self,
        tokenizer: Union[str, None, _Unset] = _UNSET,
        view: Optional[AudioView] = None,
        bpe: Union[str, Path, None, _Unset] = _UNSET,
        vocab_size: Optional[int] = None,
        frame_rate: Optional[float] = None,
        *,
        codec: Union[str, None, _Unset] = _UNSET,
    ) -> None:
        resolved_tokenizer, resolved_bpe = _audio_side_resources(
            tokenizer=tokenizer,
            bpe=bpe,
            codec=codec,
            default_tokenizer=None,
            side="audio_input",
        )
        _validate_direct_view(resolved_tokenizer, view, "audio_input")
        _AudioInputFields.__init__(
            self,
            tokenizer=resolved_tokenizer,
            bpe=resolved_bpe,
            vocab_size=vocab_size,
            frame_rate=frame_rate,
        )


InputAudioConfig = AudioInputConfig


@dataclass(frozen=True)
class _AudioOutputFields:
    tokenizer: str = "longcat"
    detokenizer: Optional[str] = "longcat"
    bpe: Optional[Union[str, Path]] = None
    acoustic_generator_artifact: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, str):
            raise TypeError("runtime.audio_output tokenizer must be a string.")
        if not self.tokenizer:
            raise ValueError("runtime.audio_output tokenizer must not be empty.")
        tokenizer_spec = _audio_tokenizer_spec(
            self.tokenizer,
            "runtime.audio_output tokenizer",
        )
        _validate_optional_nonempty_string(
            self.detokenizer,
            "runtime.audio_output detokenizer",
        )
        if self.detokenizer is not None:
            detokenizer_spec = _audio_detokenizer_spec(
                self.detokenizer,
                "runtime.audio_output detokenizer",
            )
            if detokenizer_spec != tokenizer_spec:
                raise ValueError(
                    "runtime.audio_output tokenizer and detokenizer must use the "
                    "same audio code spec."
                )
        _validate_bpe_path(self.bpe, "runtime.audio_output bpe")
        _validate_optional_nonempty_string(
            self.acoustic_generator_artifact,
            "runtime.audio_output acoustic_generator_artifact",
        )

    @property
    def audio_view(self) -> AudioView:
        return tokenizer_audio_view(self.tokenizer)

    @property
    def token_space_identity(
        self,
    ) -> tuple[AudioBackendIdentity, AudioCodeSpec, str | None]:
        return (
            audio_backend_identity(self.tokenizer),
            audio_code_spec(self.tokenizer),
            _artifact_identity(self.bpe),
        )

    @property
    def codec(self) -> str:
        """Deprecated backend-name alias retained for direct Python callers."""

        return self.tokenizer

    @property
    def view(self) -> AudioView:
        """Deprecated derived-view alias retained for direct Python callers."""

        return self.audio_view


@dataclass(frozen=True, init=False)
class AudioOutputConfig(_AudioOutputFields):
    """Target/generation audio backend plus optional BPE and decoder artifact."""

    def __init__(
        self,
        tokenizer: Union[str, None, _Unset] = _UNSET,
        detokenizer: Union[str, None, _Unset] = _UNSET,
        view: Optional[AudioView] = None,
        bpe: Union[str, Path, None, _Unset] = _UNSET,
        acoustic_generator_artifact: Optional[str] = None,
        *,
        codec: Union[str, None, _Unset] = _UNSET,
    ) -> None:
        resolved_tokenizer, resolved_bpe = _audio_side_resources(
            tokenizer=tokenizer,
            bpe=bpe,
            codec=codec,
            default_tokenizer="longcat",
            side="audio_output",
        )
        if resolved_tokenizer is None:
            raise TypeError("runtime.audio_output tokenizer must be a string.")
        resolved_detokenizer = (
            resolved_tokenizer
            if isinstance(detokenizer, _Unset)
            else cast(Optional[str], detokenizer)
        )
        _validate_direct_view(resolved_tokenizer, view, "audio_output")
        _AudioOutputFields.__init__(
            self,
            tokenizer=resolved_tokenizer,
            detokenizer=resolved_detokenizer,
            bpe=resolved_bpe,
            acoustic_generator_artifact=acoustic_generator_artifact,
        )


@dataclass(frozen=True)
class _ConfigFields:
    audio_input: Optional[AudioInputConfig] = None
    audio_output: AudioOutputConfig = field(default_factory=AudioOutputConfig)
    backbone_type: BackboneType = BackboneType.HF_CAUSAL_LM
    backbone: str = "Qwen/Qwen3-0.6B"
    backbone_initialization: BackboneInitialization = BackboneInitialization.PRETRAINED
    backbone_trust_remote_code: bool = False
    backbone_chat_template: Optional[str] = None
    backbone_readout: str = "last_hidden_state"
    backbone_readouts: dict[str, str] = field(default_factory=dict)
    backbone_supports_cache_position: bool = True
    backbone_module: str = ""
    backbone_body: str = "base_model"
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    gradient_checkpointing: bool = False
    flow_method: str = "midpoint"
    flow_nfe: int = 20
    flow_num_steps: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.backbone_type, BackboneType):
            raise TypeError("backbone_type must be a BackboneType.")
        if not isinstance(self.backbone_initialization, BackboneInitialization):
            raise TypeError("backbone_initialization must be a BackboneInitialization.")
        if not isinstance(self.backbone_trust_remote_code, bool):
            raise TypeError("backbone_trust_remote_code must be a bool.")
        _validate_optional_nonempty_string(
            self.backbone_chat_template,
            "backbone_chat_template",
        )
        validate_backbone_readout(self.backbone_readout)
        _validate_backbone_readouts(self.backbone_readouts)
        if not isinstance(self.backbone_supports_cache_position, bool):
            raise TypeError("backbone_supports_cache_position must be a bool.")
        _validate_path(self.backbone_module, "backbone_module")
        _validate_path(self.backbone_body, "backbone_body", allow_empty=False)
        if self.audio_input is not None and not isinstance(
            self.audio_input,
            AudioInputConfig,
        ):
            raise TypeError("audio_input must be an AudioInputConfig or None.")
        if not isinstance(self.audio_output, AudioOutputConfig):
            raise TypeError("audio_output must be an AudioOutputConfig.")
        if self.audio_output.acoustic_generator_artifact is not None:
            if self.audio_view is AudioView.BICODEC:
                raise ValueError(
                    "BiCodec cannot use runtime.audio_output.acoustic_generator_artifact; "
                    "its global units belong to the token sequence."
                )
            if self.audio_view is not AudioView.LONGCAT:
                raise ValueError(
                    "acoustic generator artifacts currently require LongCat."
                )
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a bool.")
        if self.flow_method not in _FLOW_METHODS:
            raise ValueError(f"unsupported flow method: {self.flow_method}")
        if self.flow_nfe <= 0:
            raise ValueError(f"flow_nfe must be positive, got {self.flow_nfe}.")
        if self.flow_num_steps < 2:
            raise ValueError(
                "flow_num_steps must be at least 2, "
                f"got {self.flow_num_steps}."
            )

    @property
    def codec(self) -> str:
        return self.audio_output.tokenizer

    @property
    def input_audio(self) -> AudioInputConfig:
        return self.audio_input or AudioInputConfig()

    @property
    def audio_tokenizer(self) -> Optional[Union[str, Path]]:
        return self.audio_output.bpe

    @property
    def acoustic_generator_artifact(self) -> Optional[str]:
        return self.audio_output.acoustic_generator_artifact

    @property
    def audio_view(self) -> AudioView:
        return self.audio_output.audio_view


@dataclass(frozen=True, init=False)
class Config(_ConfigFields):
    """Canonical runtime config with direct-Python legacy constructor aliases."""

    def __init__(
        self,
        codec: Union[str, _Unset] = _UNSET,
        backbone_type: BackboneType = BackboneType.HF_CAUSAL_LM,
        backbone: str = "Qwen/Qwen3-0.6B",
        backbone_initialization: BackboneInitialization = BackboneInitialization.PRETRAINED,
        backbone_trust_remote_code: bool = False,
        backbone_chat_template: Optional[str] = None,
        backbone_readout: str = "last_hidden_state",
        backbone_readouts: Union[dict[str, str], _Unset] = _UNSET,
        backbone_supports_cache_position: bool = True,
        backbone_module: str = "",
        backbone_body: str = "base_model",
        input_audio: Union[AudioInputConfig, None, _Unset] = _UNSET,
        audio_tokenizer: Union[str, Path, None, _Unset] = _UNSET,
        acoustic_generator_artifact: Union[str, None, _Unset] = _UNSET,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
        flow_method: str = "midpoint",
        flow_nfe: int = 20,
        flow_num_steps: int = 10,
        *,
        audio_input: Union[AudioInputConfig, None, _Unset] = _UNSET,
        audio_output: Union[AudioOutputConfig, _Unset] = _UNSET,
    ) -> None:
        resolved_input = _audio_input_config(audio_input, input_audio)
        resolved_output = _audio_output_config(
            audio_output,
            codec=codec,
            bpe=audio_tokenizer,
            acoustic_generator_artifact=acoustic_generator_artifact,
        )
        if (
            resolved_input is not None
            and resolved_input.token_space_identity
            == resolved_output.token_space_identity
        ):
            resolved_input = None
        readouts = (
            {} if backbone_readouts is _UNSET else cast(dict[str, str], backbone_readouts)
        )
        _ConfigFields.__init__(
            self,
            audio_input=resolved_input,
            audio_output=resolved_output,
            backbone_type=backbone_type,
            backbone=backbone,
            backbone_initialization=backbone_initialization,
            backbone_trust_remote_code=backbone_trust_remote_code,
            backbone_chat_template=backbone_chat_template,
            backbone_readout=backbone_readout,
            backbone_readouts=readouts,
            backbone_supports_cache_position=backbone_supports_cache_position,
            backbone_module=backbone_module,
            backbone_body=backbone_body,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            flow_method=flow_method,
            flow_nfe=flow_nfe,
            flow_num_steps=flow_num_steps,
        )


def _audio_input_config(
    canonical: Union[AudioInputConfig, None, _Unset],
    legacy: Union[AudioInputConfig, None, _Unset],
) -> Optional[AudioInputConfig]:
    canonical_set = canonical is not _UNSET
    resolved = None if canonical is _UNSET else canonical
    if resolved is not None and not isinstance(resolved, AudioInputConfig):
        raise TypeError("audio_input must be an AudioInputConfig or None.")
    resolved = _configured_audio_input(cast(Optional[AudioInputConfig], resolved))
    if legacy is _UNSET:
        return cast(Optional[AudioInputConfig], resolved)
    if legacy is not None and not isinstance(legacy, AudioInputConfig):
        raise TypeError("input_audio must be an InputAudioConfig or None.")
    configured_legacy = _configured_audio_input(cast(Optional[AudioInputConfig], legacy))
    if canonical_set and resolved != configured_legacy:
        raise ValueError(
            "runtime config contains conflicting audio_input and legacy input_audio values."
        )
    return configured_legacy


def _configured_audio_input(
    value: Optional[AudioInputConfig],
) -> Optional[AudioInputConfig]:
    if value is None or value.tokenizer is None:
        return None
    return value


def _audio_output_config(
    canonical: Union[AudioOutputConfig, _Unset],
    *,
    codec: Union[str, _Unset],
    bpe: Union[str, Path, None, _Unset],
    acoustic_generator_artifact: Union[str, None, _Unset],
) -> AudioOutputConfig:
    canonical_set = canonical is not _UNSET
    if canonical_set:
        if not isinstance(canonical, AudioOutputConfig):
            raise TypeError("audio_output must be an AudioOutputConfig.")
        resolved = canonical
    else:
        resolved = AudioOutputConfig()
    values: dict[str, object] = {
        "tokenizer": resolved.tokenizer,
        "detokenizer": resolved.detokenizer,
        "bpe": resolved.bpe,
        _GENERATOR_ARTIFACT_FIELD: resolved.acoustic_generator_artifact,
    }
    aliases = {
        "tokenizer": codec,
        "detokenizer": codec,
        "bpe": bpe,
        _GENERATOR_ARTIFACT_FIELD: acoustic_generator_artifact,
    }
    changed = False
    for name, value in aliases.items():
        if value is _UNSET:
            continue
        if canonical_set and values[name] != value:
            raise ValueError(
                f"runtime config contains conflicting audio_output.{name} and "
                "legacy values."
            )
        values[name] = value
        changed = True
    if not changed:
        return resolved
    return AudioOutputConfig(
        tokenizer=cast(str, values["tokenizer"]),
        detokenizer=cast(Optional[str], values["detokenizer"]),
        bpe=cast(Optional[Union[str, Path]], values["bpe"]),
        acoustic_generator_artifact=cast(
            Optional[str],
            values[_GENERATOR_ARTIFACT_FIELD],
        ),
    )


def _audio_tokenizer_spec(name: str, field_name: str) -> AudioCodeSpec:
    spec = audio_code_spec(name)
    if not has_audio_tokenizer_capability(name):
        raise ValueError(f"{field_name} {name!r} has no tokenizer capability.")
    return spec


def _audio_detokenizer_spec(name: str, field_name: str) -> AudioCodeSpec:
    spec = audio_code_spec(name)
    if not has_audio_detokenizer_capability(name):
        raise ValueError(f"{field_name} {name!r} has no detokenizer capability.")
    return spec


def tokenizer_audio_view(tokenizer: str) -> AudioView:
    if not isinstance(tokenizer, str):
        raise TypeError("tokenizer must be a string.")
    if not tokenizer:
        raise ValueError("tokenizer must not be empty.")
    try:
        return AudioView(
            _audio_tokenizer_spec(tokenizer, "audio tokenizer").view
        )
    except ValueError as error:
        raise ValueError(f"unsupported audio tokenizer: {tokenizer}") from error


def codec_audio_view(codec: str) -> AudioView:
    """Deprecated alias for ``tokenizer_audio_view``."""

    return tokenizer_audio_view(codec)


def _validate_audio_view(
    tokenizer: str,
    view: Optional[AudioView],
    name: str,
) -> None:
    expected = tokenizer_audio_view(tokenizer)
    if view is None:
        return
    if not isinstance(view, AudioView):
        raise TypeError(f"{name} must be an AudioView or None.")
    if view is not expected:
        raise ValueError(
            f"{name} must match tokenizer {tokenizer!r}: "
            f"{view.value!r} != {expected.value!r}."
        )


def _validate_direct_view(
    tokenizer: Optional[str],
    view: Optional[AudioView],
    side: str,
) -> None:
    if view is None:
        return
    warnings.warn(
        f"runtime.{side}.view is deprecated; the view is derived from "
        f"runtime.{side}.tokenizer.",
        FutureWarning,
        stacklevel=3,
    )
    if tokenizer is None:
        raise ValueError(f"runtime.{side}.view requires runtime.{side}.tokenizer.")
    _validate_audio_view(tokenizer, view, f"runtime.{side} view")


def _validate_bpe_path(value: object, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a path or None.")
    if not str(value):
        raise ValueError(f"{name} must not be empty.")


def config_for_local_rank(config: Config) -> Config:
    """Bind an unindexed CUDA device to the current distributed local rank."""
    device = None if config.device is None else torch.device(config.device)
    if device is not None and device.type == "cuda" and device.index is None:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return replace(config, device=None if device is None else str(device))

def validate_sequence_layout_config(config: Config, layout: AudioSequenceLayout) -> None:
    if not isinstance(layout, AudioSequenceLayout):
        raise TypeError("audio_sequence_layout must be an AudioSequenceLayout.")
    if config.audio_view is AudioView.BICODEC:
        if layout is not AudioSequenceLayout.FLATTENED:
            raise ValueError(
                "BiCodec uses one self-describing structured sequence layout; "
                "set audio_sequence_layout=flattened."
            )
        if config.audio_output.acoustic_generator_artifact is not None:
            raise ValueError(
                "BiCodec global tokens are generated or reused in the token sequence "
                "and cannot use runtime.audio_output.acoustic_generator_artifact."
            )
        return
    if layout is AudioSequenceLayout.FLATTENED:
        if config.audio_output.bpe is not None:
            raise ValueError(
                "audio_sequence_layout=flattened cannot use audio_output.bpe "
                "for frame-code codecs."
            )
        if config.audio_output.acoustic_generator_artifact is not None:
            raise ValueError(
                "runtime.audio_output.acoustic_generator_artifact requires "
                "audio_sequence_layout=semantic."
            )




def _validate_path(value: object, name: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        if allow_empty:
            return
        raise ValueError(f"{name} must not be empty.")
    if value.startswith(".") or value.endswith(".") or ".." in value:
        raise ValueError(f"{name} must be a dotted attribute path.")
    for part in value.split("."):
        if not part.isidentifier():
            raise ValueError(f"{name} must contain identifier path components.")


def _validate_backbone_readouts(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("backbone_readouts must be a mapping.")
    for modality, readout in value.items():
        if not isinstance(modality, str):
            raise TypeError("backbone_readouts keys must be strings.")
        if modality not in {Modality.TEXT.value, Modality.AUDIO.value}:
            raise ValueError("backbone_readouts keys must be 'text' or 'audio'.")
        validate_backbone_readout(readout)


def _validate_optional_nonempty_string(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")

__all__ = [
    "AudioInputConfig",
    "AudioOutputConfig",
    "AudioSequenceLayout",
    "Config",
    "InputAudioConfig",
    "codec_audio_view",
    "config_for_local_rank",
    "migrate_config_fields",
    "tokenizer_audio_view",
    "validate_sequence_layout_config",
]
