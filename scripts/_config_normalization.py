from __future__ import annotations

from enum import Enum
from typing import Any, Type, TypeVar, cast

from omegaconf import DictConfig, ListConfig, OmegaConf
from peft import LoraConfig

from speech_to_speech.audio_route import (
    AudioStream,
    PromptSource,
    StreamSource,
)
from speech_to_speech.datamodule.dataset.speech import DatasetName
from speech_to_speech.datamodule.dataset.text import TextDatasetName
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.model import (
    AdapterType,
    AudioInputAdapterType,
    AudioOutputAdapterType,
)
from speech_to_speech.runtime import AudioRepresentation, BackboneInitialization
from speech_to_speech.stage import (
    ParameterGroup,
    ParameterPolicyName,
    StageName,
)

ConfigT = TypeVar("ConfigT")
EnumT = TypeVar("EnumT", bound=Enum)


def prepare(config: DictConfig) -> DictConfig:
    result = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))
    OmegaConf.resolve(result)
    _gradient_probe(result)
    semantic_adapter = result.model.get("semantic_audio_adapter")
    if semantic_adapter is not None:
        result.model.semantic_audio_adapter = _enum_name(
            AdapterType,
            semantic_adapter,
        )
    audio_input = result.model.get("audio_input_adapter")
    if isinstance(audio_input, DictConfig):
        value = audio_input.get("type")
        if value is not None:
            audio_input.type = _enum_name(AudioInputAdapterType, value)
    audio_output = result.model.get("audio_output_adapter")
    if isinstance(audio_output, DictConfig):
        value = audio_output.get("type")
        if value is not None:
            audio_output.type = _enum_name(AudioOutputAdapterType, value)
    _dataset(result.get("data"))
    _dataset(result.get("data", {}).get("dataset"))
    _data_shape(result.get("data"))
    _text_dataset(result.get("text_data", {}).get("dataset"))
    _audio_route(result.get("audio_route"))
    runtime = result.get("runtime")
    if runtime is not None:
        initialization = runtime.get("backbone_initialization")
        if initialization is not None:
            runtime.backbone_initialization = _enum_name(
                BackboneInitialization,
                initialization,
            )
        representation = runtime.get("audio_representation")
        if representation is not None:
            runtime.audio_representation = _enum_name(
                AudioRepresentation,
                representation,
            )
    stage = result.get("stage")
    if stage is not None:
        name = stage.get("name")
        if name is not None:
            stage.name = _enum_name(StageName, name)
    policy = result.get("parameter_policy")
    if policy is not None:
        name = policy.get("name")
        if name is not None:
            policy.name = _enum_name(ParameterPolicyName, name)
        for key in ("trainable_groups", "frozen_groups"):
            groups = policy.get(key)
            if groups is not None:
                policy[key] = [_enum_name(ParameterGroup, group) for group in groups]
    return result


def parse(config: DictConfig, schema: Type[ConfigT]) -> ConfigT:
    structured = OmegaConf.structured(schema)
    _writable(structured)
    merged = OmegaConf.merge(structured, config)
    OmegaConf.resolve(merged)
    return cast(ConfigT, OmegaConf.to_object(merged))


def peft_lora(config: DictConfig) -> LoraConfig | None:
    model = config.get("model")
    if not isinstance(model, DictConfig):
        raise TypeError("model config must be a mapping.")
    value = model.get("lora")
    model.lora = None
    if value is None:
        return None
    if not isinstance(value, DictConfig):
        raise TypeError("model.lora must be a mapping or null.")
    kwargs = OmegaConf.to_container(value, resolve=True)
    if not isinstance(kwargs, dict) or any(
        not isinstance(key, str) for key in kwargs
    ):
        raise TypeError("model.lora must contain string keys.")
    return LoraConfig(**cast(dict[str, Any], kwargs))


def _dataset(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    dataset = value.get("name")
    if dataset is not None:
        value.name = _enum_name(DatasetName, dataset)


def _data_shape(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    shape = value.get("shape")
    if shape is not None:
        value.shape = _enum_name(DataShape, shape)


def _text_dataset(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    dataset = value.get("name")
    if dataset is not None:
        value.name = _enum_name(TextDatasetName, dataset)


def _audio_route(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    prompt = value.get("prompt")
    output = value.get("output")
    decode = value.get("decode")
    if not all(isinstance(item, DictConfig) for item in (prompt, output, decode)):
        return
    prompt.source = _enum_name(PromptSource, prompt.source)
    prompt.streams = [_enum_name(AudioStream, stream) for stream in prompt.streams]
    output.streams = [_enum_name(AudioStream, stream) for stream in output.streams]
    decode.semantic = _enum_name(StreamSource, decode.semantic)
    decode.acoustic = _enum_name(StreamSource, decode.acoustic)


def _gradient_probe(config: DictConfig) -> None:
    callbacks = config.get("callbacks")
    if not isinstance(callbacks, DictConfig):
        return
    probe = callbacks.get("gradient_probe")
    if not isinstance(probe, DictConfig) or "loss_pairs" not in probe:
        return

    loss_pairs = probe.pop("loss_pairs")
    if probe.get("comparisons"):
        return

    probe.comparisons = [
        {
            "left": {"loss": str(left), "group": "batch"},
            "right": {"loss": str(right), "group": "batch"},
        }
        for left, right in loss_pairs
    ]


def _enum_name(enum: Type[EnumT], value: object) -> str:
    raw = str(value)
    return enum[raw].name if raw in enum.__members__ else enum(raw).name


def _writable(config: DictConfig | ListConfig) -> None:
    OmegaConf.set_readonly(config, False)
    nodes = (
        (config._get_node(key) for key in config.keys())
        if isinstance(config, DictConfig)
        else (config._get_node(index) for index in range(len(config)))
    )
    for node in nodes:
        if isinstance(node, (DictConfig, ListConfig)):
            _writable(node)
