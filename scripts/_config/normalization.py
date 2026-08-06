"""Hydra/OmegaConf normalization for strict entry schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Type, TypeVar, cast

from omegaconf import DictConfig, ListConfig, OmegaConf
from peft import LoraConfig

from speech_to_speech.task import Task
from speech_to_speech.datamodule.dataset.speech import DatasetName
from speech_to_speech.datamodule.dataset.text import TextDatasetName
from speech_to_speech.datamodule.sample import DataShape
from speech_to_speech.model import (
    AdapterType,
    AudioInputAdapterType,
    AudioOutputAdapterType,
    FsqFeature,
)
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.model.ctc import CTCDecoderType
from speech_to_speech.runtime import (
    AudioSequenceLayout,
    BackboneInitialization,
    BackboneType,
    migrate_config_fields,
)
from speech_to_speech.training.parameter_policy import (
    ParameterGroup,
    ParameterPolicyName,
)

ConfigT = TypeVar("ConfigT")
EnumT = TypeVar("EnumT", bound=Enum)


def prepare(config: DictConfig) -> DictConfig:
    result = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))
    OmegaConf.resolve(result)
    model = result.get("model")
    if isinstance(model, DictConfig):
        semantic_adapter = model.get("semantic_audio_adapter")
        if semantic_adapter is not None:
            model.semantic_audio_adapter = _enum_name(
                AdapterType,
                semantic_adapter,
            )
        audio_input = model.get("audio_input_adapter")
        if isinstance(audio_input, DictConfig):
            value = audio_input.get("type")
            if value is not None:
                audio_input.type = _enum_name(AudioInputAdapterType, value)
        audio_output = model.get("audio_output_adapter")
        if isinstance(audio_output, DictConfig):
            value = audio_output.get("type")
            if value is not None:
                audio_output.type = _enum_name(AudioOutputAdapterType, value)
        fsq_embedding = model.get("fsq_embedding")
        if isinstance(fsq_embedding, DictConfig):
            value = fsq_embedding.get("feature")
            if value is not None:
                fsq_embedding.feature = _enum_name(FsqFeature, value)
        _ctc(model.get("ctc"))
    acoustic = model.get("acoustic") if isinstance(model, DictConfig) else None
    if isinstance(acoustic, DictConfig):
        acoustic_type = acoustic.get("type")
        if acoustic_type is not None:
            acoustic.type = _enum_value(AcousticType, acoustic_type)
    _dataset(result.get("datamodule"))
    _dataset(result.get("datamodule", {}).get("dataset"))
    _data_shape(result.get("datamodule"))
    _data_tasks(result.get("datamodule"))
    _validation_loaders(result.get("validation"))
    _text_dataset(result.get("text_datamodule", {}).get("dataset"))
    _audio_sequence_layout(result)
    _reject_audio_representation(result)
    runtime = result.get("runtime")
    if runtime is not None:
        migrate_config_fields(runtime)
        backbone_type = runtime.get("backbone_type")
        if backbone_type is not None:
            runtime.backbone_type = _enum_name(
                BackboneType,
                backbone_type,
            )
        initialization = runtime.get("backbone_initialization")
        if initialization is not None:
            runtime.backbone_initialization = _enum_name(
                BackboneInitialization,
                initialization,
            )
    callbacks = result.get("callbacks")
    if isinstance(callbacks, DictConfig):
        policy = callbacks.get("parameter_policy")
    else:
        policy = None
    if isinstance(policy, DictConfig):
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
    if not isinstance(kwargs, dict) or any(not isinstance(key, str) for key in kwargs):
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


def _data_tasks(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    tasks = value.get("tasks")
    if isinstance(tasks, DictConfig):
        renamed: dict[str, Any] = {}
        for key in list(tasks.keys()):
            raw = str(key)
            task = Task[raw] if raw in Task.__members__ else Task(raw)
            # OmegaConf enum-key dictionaries accept member names at the schema boundary.
            renamed[task.name] = tasks[key]
        value.tasks = renamed


def _validation_loaders(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    loaders = value.get("loaders")
    if not isinstance(loaders, DictConfig):
        return
    for loader in loaders.values():
        if not isinstance(loader, DictConfig):
            continue
        _dataset(loader.get("dataset"))
        _data_shape(loader)


def _text_dataset(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    dataset = value.get("name")
    if dataset is not None:
        value.name = _enum_name(TextDatasetName, dataset)


def _ctc(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    for route_name in ("source", "target"):
        decoder = value.get(route_name)
        if not isinstance(decoder, DictConfig):
            continue
        decoder_type = decoder.get("type")
        if decoder_type is not None:
            decoder.type = _enum_name(CTCDecoderType, decoder_type)


def _audio_sequence_layout(config: DictConfig) -> None:
    layout = config.get("audio_sequence_layout")
    if layout is not None:
        config.audio_sequence_layout = _enum_name(AudioSequenceLayout, layout)


def _reject_audio_representation(config: DictConfig) -> None:
    runtime = config.get("runtime")
    if not isinstance(runtime, DictConfig):
        return
    if "audio_representation" not in runtime:
        return
    if OmegaConf.is_missing(runtime, "audio_representation"):
        return
    raise ValueError("runtime.audio_representation is internal; use audio_sequence_layout.")


def _enum_name(enum: Type[EnumT], value: object) -> str:
    raw = str(value)
    return enum[raw].name if raw in enum.__members__ else enum(raw).name


def _enum_value(enum: Type[EnumT], value: object) -> str:
    raw = str(value)
    member = enum[raw] if raw in enum.__members__ else enum(raw)
    return str(member.value)


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
