from __future__ import annotations

# ruff: noqa: F401

import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from anytrain.lightning import (
    GradientComparison,
    GradientProbe,
    GradientTarget,
    ModelCheckpoint,
    ParameterPolicyCallback,
)
from anytrain.lightning.schedule import UnitScheduleCallback
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig
from omegaconf.errors import (
    ConfigAttributeError,
    ConfigKeyError,
    InterpolationResolutionError,
)
from peft import LoraConfig

from scripts._config.overfit import (
    OverfitFlowConfig,
    OverfitRVQConfig,
    OverfitTokenConfig,
    overfit,
)
from scripts._config.train import (
    StagedTrainRVQConfig,
    StagedTrainTokenConfig,
    train as parse_train,
)
from scripts._entry import (
    performance as build_performance,
    runtime_config,
)
from scripts._logging import build as build_logger
from scripts import train as train_script
from scripts.overfit import (
    _gradient_logger,
    _prepare_generation_module,
)
from scripts.train import (
    build_datamodule as build_train_datamodule,
)
from speech_to_speech.datamodule import DataModule
from speech_to_speech.datamodule.module import LoaderKind
from speech_to_speech.datamodule.dataset.speech import DatasetName
from speech_to_speech.datamodule.types import DataShape, FusedBatch, ModelBatch
from speech_to_speech.callback import BatchUnits
from speech_to_speech.model import (
    AdapterType,
    AudioInputAdapterType,
    AudioOutputAdapterType,
    Config as ModelConfig,
    FsqFeature,
    ToyConfig,
)
from speech_to_speech.model.acoustic import AcousticType, DecoderConfig
from speech_to_speech.model.ctc import (
    CTCConfig,
    CTCDecoderConfig,
    CTCDecoderType,
    CTCRouteConfig,
)
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.pl_module.optim import Config as OptimConfig
from speech_to_speech.runtime import (
    AudioSequenceLayout,
    BackboneInitialization,
    BackboneType,
    Config as RuntimeConfig,
)
from speech_to_speech.datamodule.loader import LoaderConfig, LoaderStepMode
from speech_to_speech.training.parameter_policy import (
    ParameterGroup,
    ParameterPolicyName,
)
from speech_to_speech.task import PredictionModality
from speech_to_speech.task import Task


class _DeviceRestoreModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.moves: list[torch.device] = []

    def to(self, device: torch.device):  # type: ignore[override]
        self.moves.append(device)
        return self


class _OptimizerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    @property
    def lora_config(self) -> None:
        return None


_STAGED_JOINT_CASES = (
    (
        0,
        ParameterPolicyName.LORA,
        (("tts", "tts"), ("mt", "mt")),
        {"tts": 0.9, "mt": 0.1},
    ),
    (
        1,
        ParameterPolicyName.SPEECH_INTERFACE,
        (("asr", "asr"), ("tts", "tts"), ("mt", "mt")),
        {"asr": 0.45, "tts": 0.45, "mt": 0.1},
    ),
    (
        2,
        ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD,
        (
            ("asr", "asr"),
            ("s2tt", "s2tt"),
            ("tts", "tts"),
            ("t2st", "t2st"),
            ("mt", "mt"),
        ),
        {"asr": 0.225, "s2tt": 0.225, "tts": 0.225, "t2st": 0.225, "mt": 0.1},
    ),
    (
        3,
        ParameterPolicyName.FULL,
        (
            ("asr", "asr"),
            ("s2tt", "s2tt"),
            ("tts", "tts"),
            ("t2st", "t2st"),
            ("s2st", "s2st"),
            ("mt", "mt"),
        ),
        {"asr": 0.05, "s2tt": 0.05, "tts": 0.05, "t2st": 0.05, "s2st": 0.7, "mt": 0.1},
    ),
)


class ConfigTestCase(unittest.TestCase):
    def _assert_gradient_logger(
        self,
        grad_logger,
        config,
        acoustic_type,
        loss_name,
        probes,
    ):
        comparison = GradientComparison(
            GradientTarget("token"),
            GradientTarget(loss_name),
        )
        callback = _gradient_logger(config, acoustic_type, comparison)

        self.assertIs(callback, grad_logger.return_value)
        grad_logger.assert_called_once_with((comparison,), probes, every_n_steps=1)
        grad_logger.reset_mock()
        return comparison


def _default_gradient_probes() -> tuple[GradientProbe, ...]:
    return (
        GradientProbe(
            "backbone_l0_attn",
            (
                r"model\.backbone\.(?:layers|mimo_layers)\.0\.self_attn\.q_proj\.weight$",
                r"model\.backbone\.(?:layers|mimo_layers)\.0\.self_attn\.k_proj\.weight$",
                r"model\.backbone\.(?:layers|mimo_layers)\.0\.self_attn\.v_proj\.weight$",
                r"model\.backbone\.(?:layers|mimo_layers)\.0\.self_attn\.o_proj\.weight$",
            ),
            match="regex",
        ),
        GradientProbe(
            "backbone_l0_ffn",
            (
                r"model\.backbone\.(?:layers|mimo_layers)\.0\.mlp\.gate_proj\.weight$",
                r"model\.backbone\.(?:layers|mimo_layers)\.0\.mlp\.up_proj\.weight$",
                r"model\.backbone\.(?:layers|mimo_layers)\.0\.mlp\.down_proj\.weight$",
            ),
            match="regex",
        ),
    )


def _compose(config_name: str, *overrides: str) -> DictConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return compose(config_name=config_name, overrides=list(overrides))


def _overfit(*overrides: str):
    return overfit(_compose("overfit", *overrides))


def _performance_overfit(*overrides: str):
    return _overfit(
        "callbacks.performance.enabled=true",
        "callbacks.task_sample.enabled=false",
        *overrides,
    )


def _lora_overfit(*overrides: str):
    return _overfit(
        "+model/lora@model.lora=qwen",
        "callback/parameter_policy@callbacks.parameter_policy=lora",
        *overrides,
    )


def _train(*overrides: str):
    return parse_train(_compose("train", *overrides))


__all__ = [name for name in globals() if not name.startswith("__")]
