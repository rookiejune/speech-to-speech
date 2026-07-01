from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor

from ..config import BPEConfig, DataModuleConfig, TaskConfig
from ..dataset import training_dataset
from ..datamodule.batch_builder import CausalLMBatchBuilder
from ..datamodule.example import longcat_pair_from_sample
from ..datamodule.types import (
    AutoregressionExample,
    CausalLMBatch,
    LongCatBatchSide,
    LongCatPair,
    LongCatSide,
    Task,
    TaskFamily,
    TranslationExample,
)
from ..runtime import longcat_codec, longcat_tokenizer
from ..model.acoustic import AcousticSampler, pooled_acoustic_condition_from_batch_side
from ..model.types import TeacherForcedWaveformGeneration
from .batch import batch_to_device
from ..datamodule.pipeline import (
    TaskSample,
    TaskSampleStream,
)

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class _GenerationSpec:
    name: str
    prompt: str
    batch: CausalLMBatch
    prefix: LongCatSide | None
    reference: LongCatSide


@dataclass
class TaskSampleLogger(Callback):
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    bpe: BPEConfig = field(default_factory=BPEConfig)
    every_n_steps: int = 500
    samples_per_task: int = 1
    max_audio_samples: int | None = SAMPLE_RATE * 20

    def __post_init__(self) -> None:
        if self.every_n_steps < 0:
            raise ValueError("every_n_steps must be non-negative.")
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
        if self.every_n_steps == 0:
            return
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
            self.datamodule.dataset_factory,
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


@dataclass
class TaskGenerationLogger(Callback):
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    bpe: BPEConfig = field(default_factory=BPEConfig)
    tokenizer: object | None = None
    every_n_steps: int = 500
    sample_index: int = 0
    flow_steps: int = 2
    chunk_size: int | None = None
    left_context_chunks: int | None = None
    guidance_scale: float = 1.0
    acoustic_sampler: str = AcousticSampler.SERIAL.value
    preview_tokens: int = 1024
    max_audio_samples: int | None = SAMPLE_RATE * 20
    _pair: LongCatPair | None = field(default=None, init=False)
    _last_logged_step: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive.")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative.")
        if self.flow_steps <= 0:
            raise ValueError("flow_steps must be positive.")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive when set.")
        if self.left_context_chunks is not None and self.left_context_chunks < 0:
            raise ValueError("left_context_chunks must be non-negative.")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative.")
        AcousticSampler(self.acoustic_sampler)
        if self.preview_tokens <= 0:
            raise ValueError("preview_tokens must be positive.")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        self._pair = _canary_pair(self.datamodule, self.sample_index)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if not _should_log_generation_step(step, self.every_n_steps, self._last_logged_step):
            return
        self._log_generations(trainer, pl_module)
        self._last_logged_step = step

    def _log_generations(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logger = _tensorboard_logger(trainer.loggers)
        step = int(trainer.global_step)
        pair = self._pair
        if pair is None:
            pair = _canary_pair(self.datamodule, self.sample_index)
            self._pair = pair

        model = getattr(pl_module, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TaskGenerationLogger requires pl_module.model to be a torch module.")
        teacher_forced_waveform = getattr(model, "teacher_forced_waveform", None)
        if not callable(teacher_forced_waveform):
            raise TypeError("TaskGenerationLogger requires model.teacher_forced_waveform().")

        if getattr(model, "dit", None) is None:
            _log_generation_skip(
                logger,
                step=step,
                reason="teacher-forced waveform logging requires model.dit.",
            )
            return
        bpe = longcat_tokenizer(self.bpe)
        codec = longcat_codec()
        builder = CausalLMBatchBuilder(model.idspace, tokenizer=self.tokenizer)
        device = _module_device(model)
        training_modes = _module_training_modes(model)
        codec = _move_runtime_to_device(codec, device)
        model.eval()
        try:
            for spec in _generation_specs(
                builder,
                bpe,
                pair,
            ):
                batch = _move_causal_lm_batch(spec.batch, device)
                acoustic_condition = _source_acoustic_condition(
                    model,
                    codec,
                    batch,
                )
                generation = teacher_forced_waveform(
                    batch,
                    bpe=bpe,
                    codec=codec,
                    acoustic_generator=model.acoustic_feature_generator(
                        num_steps=self.flow_steps,
                        chunk_size=self.chunk_size,
                        left_context_chunks=self.left_context_chunks,
                        guidance_scale=self.guidance_scale,
                        sampler=AcousticSampler(self.acoustic_sampler),
                        acoustic_condition=acoustic_condition,
                    ),
                )
                _log_generation_result(
                    logger,
                    codec,
                    bpe,
                    spec,
                    generation,
                    sample_index=self.sample_index,
                    flow_steps=self.flow_steps,
                    chunk_size=self.chunk_size,
                    left_context_chunks=self.left_context_chunks,
                    guidance_scale=self.guidance_scale,
                    acoustic_sampler=self.acoustic_sampler,
                    step=step,
                    preview_tokens=self.preview_tokens,
                    max_audio_samples=self.max_audio_samples,
                )
        finally:
            _restore_training_modes(training_modes)


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
    if sample.source is not None:
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


def _log_generation_result(
    logger: TensorBoardLogger,
    codec: object,
    bpe: object,
    spec: _GenerationSpec,
    generation: TeacherForcedWaveformGeneration,
    *,
    sample_index: int,
    flow_steps: int,
    chunk_size: int | None,
    left_context_chunks: int | None,
    guidance_scale: float,
    acoustic_sampler: str,
    step: int,
    preview_tokens: int,
    max_audio_samples: int | None,
) -> None:
    root = f"generations/{spec.name}"
    semantic_ids = generation.semantic_ids[generation.semantic_mask].detach().cpu()
    logger.experiment.add_text(
        f"{root}/summary",
        "\n".join(
            (
                f"sample_index: {sample_index}",
                f"prompt: {spec.prompt}",
                "mode: teacher_forced_waveform",
                f"flow_steps: {flow_steps}",
                f"chunk_size: {chunk_size}",
                f"left_context_chunks: {left_context_chunks}",
                f"guidance_scale: {guidance_scale}",
                f"acoustic_sampler: {acoustic_sampler}",
                f"semantic_frame_count: {semantic_ids.numel()}",
                f"acoustic_feature_shape: {tuple(generation.acoustic_features.shape)}",
                f"audio_shape: {tuple(generation.audio.shape)}",
            )
        ),
        global_step=step,
    )
    logger.experiment.add_text(
        f"{root}/teacher_forced/semantic_ids",
        _codes_text(semantic_ids, limit=preview_tokens),
        global_step=step,
    )
    logger.experiment.add_scalar(
        f"{root}/teacher_forced/semantic_frame_count",
        int(semantic_ids.numel()),
        global_step=step,
    )
    logger.experiment.add_audio(
        f"{root}/generated/audio",
        _first_audio(generation.audio, max_audio_samples=max_audio_samples),
        global_step=step,
        sample_rate=SAMPLE_RATE,
    )
    if spec.prefix is not None:
        _log_reference_side(
            logger,
            codec,
            bpe,
            spec.prefix,
            root=f"{root}/prefix",
            step=step,
            max_audio_samples=max_audio_samples,
        )
    _log_reference_side(
        logger,
        codec,
        bpe,
        spec.reference,
        root=f"{root}/reference",
        step=step,
        max_audio_samples=max_audio_samples,
    )


def _log_generation_skip(
    logger: TensorBoardLogger,
    *,
    step: int,
    reason: str,
) -> None:
    logger.experiment.add_text(
        "generations/skipped",
        reason,
        global_step=step,
    )


def _log_reference_side(
    logger: TensorBoardLogger,
    codec: object,
    bpe: object,
    side: LongCatSide,
    *,
    root: str,
    step: int,
    max_audio_samples: int | None,
) -> None:
    logger.experiment.add_text(
        f"{root}/codes",
        _codes_text(_roundtrip_semantic_ids(bpe, side.semantic_ids)),
        global_step=step,
    )
    logger.experiment.add_audio(
        f"{root}/audio",
        _decode_side(codec, bpe, side, max_audio_samples=max_audio_samples),
        global_step=step,
        sample_rate=SAMPLE_RATE,
    )


def _first_audio(audio: Tensor, *, max_audio_samples: int | None) -> Tensor:
    audio = audio.detach().float().cpu()
    if audio.dim() == 3:
        audio = audio[0]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError("audio must have shape [channels, time] or [batch, channels, time].")
    if max_audio_samples is not None:
        if max_audio_samples <= 0:
            raise ValueError("max_audio_samples must be positive.")
        audio = audio[:, :max_audio_samples]
    return audio


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
    encode_frames = getattr(bpe, "encode_frames", None)
    expand_ids = getattr(bpe, "expand_ids", None)
    if not callable(encode_frames) or not callable(expand_ids):
        raise TypeError("LongCat BPE tokenizer must provide encode_frames() and expand_ids().")
    frames = [[int(value)] for value in ids.reshape(-1).detach().cpu().tolist()]
    encoded = encode_frames(frames)
    expanded = expand_ids(encoded)
    return torch.tensor(_single_codebook_ids(expanded), dtype=torch.long)


def _encode_frames(bpe: object, ids: Tensor) -> Tensor:
    encode_frames = getattr(bpe, "encode_frames", None)
    if not callable(encode_frames):
        raise TypeError("LongCat BPE tokenizer must provide encode_frames().")
    frames = [[int(value)] for value in ids.reshape(-1).detach().cpu().tolist()]
    return torch.tensor([int(value) for value in encode_frames(frames)], dtype=torch.long)


def _single_codebook_ids(frames: object) -> list[int]:
    ids: list[int] = []
    if not isinstance(frames, Sequence):
        raise TypeError("LongCat BPE expand_ids() must return a sequence of frames.")
    for frame in frames:
        if isinstance(frame, int) or not isinstance(frame, Sequence):
            raise TypeError("LongCat BPE expand_ids() must return frame sequences.")
        if len(frame) != 1:
            raise ValueError("LongCat semantic BPE must use exactly one codebook.")
        ids.append(int(frame[0]))
    return ids


def _codes_text(ids: Tensor, *, limit: int = 512) -> str:
    values = [int(value) for value in ids.reshape(-1).tolist()]
    suffix = "" if len(values) <= limit else f" ... ({len(values)} total)"
    return " ".join(str(value) for value in values[:limit]) + suffix


def _canary_pair(config: DataModuleConfig, sample_index: int) -> LongCatPair:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    for index, sample in enumerate(training_dataset(config.dataset_factory)):
        if index == sample_index:
            return longcat_pair_from_sample(sample)
    raise IndexError(f"sample_index {sample_index} is outside the dataset.")


def _generation_specs(
    builder: CausalLMBatchBuilder,
    bpe: object,
    pair: LongCatPair,
) -> tuple[_GenerationSpec, ...]:
    source_ids = _encode_frames(bpe, pair.source.semantic_ids)
    target_ids = _encode_frames(bpe, pair.target.semantic_ids)
    return (
        _GenerationSpec(
            name="source_ar",
            prompt="Continue the speech.",
            batch=_task_batch(
                builder.autoregression(AutoregressionExample(audio_ids=source_ids)),
                source=None,
                target=pair.source,
            ),
            prefix=None,
            reference=pair.source,
        ),
        _GenerationSpec(
            name="target_ar",
            prompt="Continue the speech.",
            batch=_task_batch(
                builder.autoregression(AutoregressionExample(audio_ids=target_ids)),
                source=None,
                target=pair.target,
            ),
            prefix=None,
            reference=pair.target,
        ),
        _GenerationSpec(
            name="source_to_target",
            prompt="Translate the source speech.",
            batch=_task_batch(
                builder.translation(
                    TranslationExample(source_ids=source_ids, target_ids=target_ids)
                ),
                source=pair.source,
                target=pair.target,
            ),
            prefix=pair.source,
            reference=pair.target,
        ),
        _GenerationSpec(
            name="target_to_source",
            prompt="Translate the source speech.",
            batch=_task_batch(
                builder.translation(
                    TranslationExample(source_ids=target_ids, target_ids=source_ids)
                ),
                source=pair.target,
                target=pair.source,
            ),
            prefix=pair.target,
            reference=pair.source,
        ),
    )


def _task_batch(
    batch: CausalLMBatch,
    *,
    source: LongCatSide | None,
    target: LongCatSide,
) -> CausalLMBatch:
    return CausalLMBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        logits_to_keep=batch.logits_to_keep,
        loss_weights=batch.loss_weights,
        source_audio=_batch_side(source) if source is not None else None,
        target_audio=_batch_side(target),
    )


def _batch_side(side: LongCatSide) -> LongCatBatchSide:
    semantic_ids = side.semantic_ids.reshape(1, -1).detach().to(dtype=torch.long)
    acoustic_ids = side.acoustic_ids.detach().to(dtype=torch.long)
    if acoustic_ids.dim() == 2:
        acoustic_ids = acoustic_ids.unsqueeze(0)
    if acoustic_ids.dim() != 3 or acoustic_ids.size(0) != 1:
        raise ValueError("LongCat acoustic ids must have shape [nq, time].")
    if semantic_ids.size(1) != acoustic_ids.size(-1):
        raise ValueError("LongCat semantic and acoustic lengths must match.")
    mask = torch.ones((1, semantic_ids.size(1)), dtype=torch.bool)
    return LongCatBatchSide(
        semantic_ids=semantic_ids,
        semantic_mask=mask,
        acoustic_ids=acoustic_ids,
        acoustic_mask=mask,
    )


def _should_log_generation_step(
    step: int,
    every_n_steps: int,
    last_logged_step: int | None,
) -> bool:
    if step <= 0 or step == last_logged_step:
        return False
    return step == 1 or (step - 1) % every_n_steps == 0


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _module_training_modes(
    module: torch.nn.Module,
) -> tuple[tuple[torch.nn.Module, bool], ...]:
    return tuple((child, child.training) for child in module.modules())


def _restore_training_modes(modes: Iterable[tuple[torch.nn.Module, bool]]) -> None:
    for module, training in modes:
        module.train(training)


def _move_causal_lm_batch(batch: CausalLMBatch, device: torch.device) -> CausalLMBatch:
    return batch_to_device(batch, device)


def _source_acoustic_condition(
    model: torch.nn.Module,
    codec: object,
    batch: CausalLMBatch,
) -> Tensor | None:
    if batch.source_audio is None:
        return None
    dit = getattr(model, "dit", None)
    if not isinstance(dit, torch.nn.Module):
        return None
    return pooled_acoustic_condition_from_batch_side(
        batch.source_audio,
        feature_extractor=codec,
    )


def _move_runtime_to_device(value: object, device: torch.device) -> object:
    move = getattr(value, "to", None)
    if callable(move):
        moved = move(device)
        if moved is not None:
            value = moved
    if hasattr(value, "device"):
        setattr(value, "device", device)
    return value


def _task_name(sample: TaskSample) -> str:
    return sample.family.value


def _prompt(sample: TaskSample) -> str:
    match sample.family:
        case TaskFamily.SOURCE_AR | TaskFamily.TARGET_AR:
            return "Continue the speech."
        case TaskFamily.SOURCE_TO_TARGET | TaskFamily.TARGET_TO_SOURCE:
            return "Translate the source speech."
    raise ValueError(f"unsupported task family: {sample.family.value}")


def _target_side(sample: TaskSample) -> LongCatSide:
    return sample.target


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
