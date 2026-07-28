from __future__ import annotations

import math
from collections.abc import Mapping
from anydataset import types

from ..task import Task
from ._task import TaskWeights, allocate_tasks
from ._tokenization import token_ids
from .parser import parse_audio_codes, raw_speech, speech_from_codes
from .protocol import DataRuntime
from .sample import build_speech_sample, build_task_sample, chat_prompt
from .types import (
    ModelBatch,
    ModelSample,
    RawSpeech,
    RawSpeechBatch,
    Speech,
    AudioContextSample,
    SpeechTaskSample,
    Text,
)

_SINGLE_TASKS = frozenset({Task.ASR, Task.AUDIO_AR, Task.TEXT_AR, Task.TTS})


class SingleCollator:
    def __init__(
        self,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
        *,
        encode_missing_codes: bool = False,
    ) -> None:
        self.runtime = runtime
        self.encode_missing_codes = encode_missing_codes
        _validate_single_tasks(_positive_tasks(task_weights))
        self._task_weights = TaskWeights(task_weights)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        _validate_single_tasks(_positive_tasks(task_weights))
        self._task_weights.set(task_weights)

    @property
    def tasks(self) -> list[Task]:
        tasks, _ = self._task_weights.get()
        return tasks

    def _items(self, samples: list[types.Sample]) -> list[SpeechTaskSample]:
        available, weights = self._task_weights.get()
        tasks = allocate_tasks(available, weights, len(samples))
        return [
            _build_item(
                sample,
                task,
                self.runtime,
                encode_missing_codes=self.encode_missing_codes,
            )
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[types.Sample]) -> ModelBatch | RawSpeechBatch:
        items = self._items(samples)
        if any(item.needs_codec for item in items):
            return RawSpeechBatch(
                samples=tuple(items),
                pad_token_id=self.runtime.pad_token_id,
            )
        return ModelBatch.from_samples(
            [build_task_sample(item, self.runtime) for item in items],
            pad_token_id=self.runtime.pad_token_id,
        )


def parse_single_sample(sample: types.Sample, runtime: DataRuntime) -> Speech:
    audio_item, text_item = _single_items(sample)
    codes = _codec_codes(audio_item, runtime)
    text = text_item.views[types.TextView.TEXT]
    semantic_codes, _ = parse_audio_codes(codes, runtime)
    return speech_from_codes(
        codes,
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        language=_language(text_item),
        duration_seconds=_duration_seconds(
            audio_item,
            frames=semantic_codes.size(0),
            frame_rate=runtime.codec_frame_rate,
        ),
        runtime=runtime,
    )


def build_single_sample(
    utterance: Speech,
    task: Task,
    runtime: DataRuntime,
) -> ModelSample:
    _validate_single_tasks([task])
    prompt = chat_prompt(utterance.language, task, runtime)
    return build_speech_sample(utterance, utterance, task, runtime, prompt=prompt)


def _build_item(
    sample: types.Sample,
    task: Task,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool,
) -> SpeechTaskSample:
    _validate_single_tasks([task])
    audio_item, text_item = _single_items(sample)
    text = Text(
        text_token_ids=token_ids(
            text_item.views[types.TextView.TEXT],
            runtime.text_tokenizer,
        ),
        language=_language(text_item),
    )
    if (
        task.source_modality is not types.Modality.AUDIO
        and task.target_modality is not types.Modality.AUDIO
    ):
        return SpeechTaskSample(source=None, target=text, task=task)
    utterance = _utterance(
        sample,
        audio_item,
        runtime,
        encode_missing_codes=encode_missing_codes,
    )
    audio_context = None
    if isinstance(sample, AudioContextSample):
        context_audio, _ = _single_items(sample.audio_context)
        audio_context = _utterance(
            sample.audio_context,
            context_audio,
            runtime,
            encode_missing_codes=encode_missing_codes,
        )
    return _task_sample(utterance, text, task, audio_context=audio_context)


def _utterance(
    sample: types.Sample,
    audio_item: types.AudioItem,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool,
) -> Speech | RawSpeech:
    if runtime.audio_view in audio_item.views:
        return parse_single_sample(sample, runtime)
    if not encode_missing_codes:
        raise ValueError(
            f"single audio sample is missing {runtime.audio_view.value!r} codec "
            "codes; materialize codec views before training or enable explicit "
            "waveform fallback."
        )
    return parse_raw_single_sample(sample, runtime)


def parse_raw_single_sample(
    sample: types.Sample,
    runtime: DataRuntime,
) -> RawSpeech:
    audio_item, text_item = _single_items(sample)
    return raw_speech(audio_item, text_item, runtime)


def _task_sample(
    utterance: Speech | RawSpeech,
    text: Text,
    task: Task,
    *,
    audio_context: Speech | RawSpeech | None = None,
) -> SpeechTaskSample:
    source = None
    if task.source_modality is types.Modality.AUDIO:
        source = utterance
    elif task.source_modality is types.Modality.TEXT:
        source = text
    target = utterance if task.target_modality is types.Modality.AUDIO else text
    return SpeechTaskSample(
        source=source,
        target=target,
        task=task,
        audio_context=audio_context,
    )


def _single_items(sample: types.Sample) -> tuple[types.AudioItem, types.TextItem]:
    try:
        audio_item = sample[(types.Role.DEFAULT, types.Modality.AUDIO)]
        text_item = sample[(types.Role.DEFAULT, types.Modality.TEXT)]
    except KeyError as error:
        raise ValueError(
            "single data samples must use Role.DEFAULT text and audio items."
        ) from error
    if not isinstance(audio_item, types.AudioItem):
        raise TypeError("single audio item must be an AudioItem.")
    if not isinstance(text_item, types.TextItem):
        raise TypeError("single text item must be a TextItem.")
    return audio_item, text_item


def _codec_codes(audio_item: types.AudioItem, runtime: DataRuntime) -> object:
    try:
        codes = audio_item.views[runtime.audio_view]
    except KeyError as error:
        raise ValueError(
            f"single audio sample is missing {runtime.audio_view.value!r} codec codes."
        ) from error
    return codes


def _duration_seconds(
    audio_item: types.AudioItem,
    *,
    frames: int,
    frame_rate: float,
) -> float:
    value = audio_item.meta.get(types.AudioMeta.DURATION)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("AudioMeta.DURATION must be a number of seconds.")
        duration = float(value)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("AudioMeta.DURATION must be finite and non-negative.")
        return duration
    if frames < 0:
        raise ValueError("audio frame count must be non-negative.")
    rate = float(frame_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("codec frame_rate must be finite and positive.")
    return float(frames) / rate


def _language(text_item: types.TextItem):
    from .types import Language

    return Language(text_item.meta[types.TextMeta.LANG])


def _validate_single_tasks(tasks: list[Task]) -> None:
    for task in tasks:
        if task not in _SINGLE_TASKS:
            raise ValueError(f"{task.value} is not supported by the single data path.")


def _positive_tasks(values: Mapping[Task, float]) -> list[Task]:
    return [task for task, weight in values.items() if weight > 0]
