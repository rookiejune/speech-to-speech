from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, DictConfig

from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.runtime import AudioSequenceLayout, Config as RuntimeConfig
from speech_to_speech.task import Task

from speech_to_speech.training.config import non_negative_integer, positive_integer
from .normalization import parse, prepare


@dataclass
class GenerationSmokeConfig:
    task: str = Task.S2ST.value
    run_name: str = MISSING
    repo_output_root: str = MISSING
    output_subdir: str = MISSING
    output_dir: str = MISSING
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio_sequence_layout: AudioSequenceLayout = MISSING
    datamodule: SpeechConfig = MISSING
    sample_index: int = 0
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4])
    max_new_tokens: int = 2
    seed: int = 0


def generation_smoke(config: DictConfig) -> GenerationSmokeConfig:
    result = parse(prepare(config), GenerationSmokeConfig)
    _validate(result)
    return result


def _validate(config: GenerationSmokeConfig) -> None:
    Task(config.task)
    non_negative_integer(config.sample_index, "sample_index")
    non_negative_integer(config.seed, "seed")
    positive_integer(config.max_new_tokens, "max_new_tokens")
    if not config.batch_sizes:
        raise ValueError("batch_sizes must not be empty.")
    for index, value in enumerate(config.batch_sizes):
        positive_integer(value, f"batch_sizes[{index}]")
    _validate_output(config)


def _validate_output(config: GenerationSmokeConfig) -> None:
    subdir = Path(config.output_subdir)
    if subdir == Path(".") or subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "output_subdir must be a non-empty relative path without '..'."
        )
    expected = Path(config.repo_output_root).expanduser() / subdir
    if Path(config.output_dir).expanduser() != expected:
        raise ValueError("output_dir must equal repo_output_root/output_subdir.")


__all__ = ["GenerationSmokeConfig", "generation_smoke"]
