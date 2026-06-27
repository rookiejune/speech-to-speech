from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor

from ..config import BPEConfig, DatasetInput, TaskConfig
from ..runtime import longcat_codec, longcat_tokenizer
from ..types import LongCatSide, Task
from ..datamodule.pipeline import (
    SourceAutoregressionSample,
    SourceToTargetSample,
    TargetAutoregressionSample,
    TargetToSourceSample,
    TaskSample,
    TaskSampleStream,
)

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class TaskSampleLogger(Callback):
    datasets: Sequence[DatasetInput]
    cache_root: str | Path | None = None
    tasks: TaskConfig = field(default_factory=TaskConfig)
    bpe: BPEConfig = field(default_factory=BPEConfig)
    every_n_steps: int = 500
    samples_per_task: int = 1
    max_audio_samples: int | None = SAMPLE_RATE * 20

    def __post_init__(self) -> None:
        if self.every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive.")
        if self.samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive.")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_samples(trainer)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        self._log_samples(trainer)

    def _log_samples(self, trainer: Trainer) -> None:
        if not trainer.is_global_zero:
            return
        logger = _tensorboard_logger(trainer.loggers)
        step = trainer.global_step
        bpe = longcat_tokenizer(self.bpe)
        codec = longcat_codec()
        target_tasks = _enabled_task_names(self.tasks)
        counts: dict[str, int] = {}
        for sample in TaskSampleStream(
            self.datasets,
            cache_root=self.cache_root,
            tasks=self.tasks,
        ):
            task = _task_name(sample)
            count = counts.get(task, 0)
            if count >= self.samples_per_task:
                if _all_tasks_logged(counts, target_tasks, self.samples_per_task):
                    break
                continue
            _log_task_sample(
                logger,
                codec,
                bpe,
                sample,
                index=count,
                step=step,
                max_audio_samples=self.max_audio_samples,
            )
            counts[task] = count + 1
            if _all_tasks_logged(counts, target_tasks, self.samples_per_task):
                break


def _log_task_sample(
    logger: TensorBoardLogger,
    codec: object,
    bpe: object,
    sample: TaskSample,
    *,
    index: int,
    step: int,
    max_audio_samples: int | None,
) -> None:
    task = _task_name(sample)
    root = f"samples/{task}/{index}"
    logger.experiment.add_text(f"{root}/prompt", _prompt(sample), global_step=step)
    if isinstance(sample, SourceToTargetSample | TargetToSourceSample):
        logger.experiment.add_text(
            f"{root}/source/codes",
            _codes_text(_roundtrip_semantic_ids(bpe, sample.source.semantic_ids)),
            global_step=step,
        )
        logger.experiment.add_audio(
            f"{root}/source/audio",
            _decode_side(codec, bpe, sample.source, max_audio_samples=max_audio_samples),
            global_step=step,
            sample_rate=SAMPLE_RATE,
        )

    logger.experiment.add_text(
        f"{root}/label/codes",
        _codes_text(_roundtrip_semantic_ids(bpe, _target_side(sample).semantic_ids)),
        global_step=step,
    )
    logger.experiment.add_audio(
        f"{root}/label/audio",
        _decode_side(codec, bpe, _target_side(sample), max_audio_samples=max_audio_samples),
        global_step=step,
        sample_rate=SAMPLE_RATE,
    )


def _decode_side(
    codec: object,
    bpe: object,
    side: LongCatSide,
    *,
    max_audio_samples: int | None,
) -> Tensor:
    semantic_ids = _roundtrip_semantic_ids(bpe, side.semantic_ids)
    acoustic_ids = side.acoustic_ids.detach().to(dtype=torch.long)
    if acoustic_ids.dim() == 2:
        acoustic_length = acoustic_ids.size(-1)
    elif acoustic_ids.dim() == 3 and acoustic_ids.size(0) == 1:
        acoustic_ids = acoustic_ids.squeeze(0)
        acoustic_length = acoustic_ids.size(-1)
    else:
        raise ValueError("LongCat acoustic_codes must have shape [nq, time].")

    if semantic_ids.numel() != acoustic_length:
        raise ValueError(
            "BPE-expanded semantic length must match LongCat acoustic length: "
            f"semantic={semantic_ids.numel()} acoustic={acoustic_length}."
        )
    decode = getattr(codec, "decode", None)
    if not callable(decode):
        raise TypeError("LongCat codec must provide decode().")
    audio = decode(semantic_ids.unsqueeze(0), acoustic_ids.unsqueeze(0))
    if not isinstance(audio, Tensor):
        raise TypeError("LongCat codec decode() must return a Tensor.")
    audio = audio.detach().float().cpu()
    if audio.dim() == 3 and audio.size(0) == 1:
        audio = audio.squeeze(0)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError("decoded audio must have shape [channels, time].")
    if max_audio_samples is not None:
        if max_audio_samples <= 0:
            raise ValueError("max_audio_samples must be positive.")
        audio = audio[:, :max_audio_samples]
    return audio


def _roundtrip_semantic_ids(bpe: object, ids: Tensor) -> Tensor:
    encode_units = getattr(bpe, "encode_units", None)
    expand_ids = getattr(bpe, "expand_ids", None)
    if not callable(encode_units) or not callable(expand_ids):
        raise TypeError("LongCat BPE tokenizer must provide encode_units() and expand_ids().")
    units = [int(value) for value in ids.reshape(-1).detach().cpu().tolist()]
    encoded = encode_units(units)
    expanded = expand_ids(encoded)
    return torch.tensor([int(value) for value in expanded], dtype=torch.long)


def _codes_text(ids: Tensor, *, limit: int = 512) -> str:
    values = [int(value) for value in ids.reshape(-1).tolist()]
    suffix = "" if len(values) <= limit else f" ... ({len(values)} total)"
    return " ".join(str(value) for value in values[:limit]) + suffix


def _task_name(sample: TaskSample) -> str:
    if isinstance(sample, SourceAutoregressionSample):
        return "source_ar"
    if isinstance(sample, TargetAutoregressionSample):
        return "target_ar"
    if isinstance(sample, SourceToTargetSample):
        return "source_to_target"
    if isinstance(sample, TargetToSourceSample):
        return "target_to_source"
    raise TypeError("unknown task sample type.")


def _prompt(sample: TaskSample) -> str:
    if isinstance(sample, SourceAutoregressionSample | TargetAutoregressionSample):
        return "Continue the speech."
    if isinstance(sample, SourceToTargetSample | TargetToSourceSample):
        return "Translate the source speech."
    raise TypeError("unknown task sample type.")


def _target_side(sample: TaskSample) -> LongCatSide:
    if isinstance(sample, SourceAutoregressionSample | TargetAutoregressionSample):
        return sample.target
    if isinstance(sample, SourceToTargetSample | TargetToSourceSample):
        return sample.target
    raise TypeError("unknown task sample type.")


def _tensorboard_logger(loggers: Iterable[Any]) -> TensorBoardLogger:
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    names = ", ".join(type(logger).__name__ for logger in loggers)
    raise RuntimeError(
        "TaskSampleLogger requires TensorBoardLogger. "
        f"Configured loggers: {names or '<none>'}."
    )


def _enabled_task_names(tasks: TaskConfig) -> tuple[str, ...]:
    names: list[str] = []
    enabled = frozenset(Task(name) for name in tasks.enabled)
    if Task.AUTOREGRESSION in enabled:
        names.extend(("source_ar", "target_ar"))
    if Task.TRANSLATION in enabled:
        names.extend(("source_to_target", "target_to_source"))
    if not names:
        raise ValueError("tasks.enabled must contain at least one task.")
    return tuple(names)


def _all_tasks_logged(
    counts: Mapping[str, int],
    target_tasks: Sequence[str],
    samples_per_task: int,
) -> bool:
    return all(
        counts.get(task, 0) >= samples_per_task
        for task in target_tasks
    )
