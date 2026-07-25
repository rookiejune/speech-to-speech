from __future__ import annotations

import math
from collections.abc import Mapping
from anydataset import types
from torch import Tensor

from ..task import Task
from ._tokenization import token_ids
from .collator import _TaskWeights, _allocate_tasks
from .parser import parse_audio_codes, speech_from_codes
from .protocol import DataRuntime
from .sample import build_speech_sample, chat_prompt
from .types import ModelBatch, ModelSample, RawSingleBatch, RawSingleSample, Speech

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
        self._task_weights = _TaskWeights(task_weights)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        _validate_single_tasks(_positive_tasks(task_weights))
        self._task_weights.set(task_weights)

    @property
    def tasks(self) -> list[Task]:
        tasks, _ = self._task_weights.get()
        return tasks

    def _items(self, samples: list[types.Sample]) -> list[ModelSample | RawSingleSample]:
        available, weights = self._task_weights.get()
        tasks = _allocate_tasks(available, weights, len(samples))
        return [
            _build_item(
                sample,
                task,
                self.runtime,
                encode_missing_codes=self.encode_missing_codes,
            )
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[types.Sample]) -> ModelBatch | RawSingleBatch:
        items = self._items(samples)
        model_samples = [item for item in items if isinstance(item, ModelSample)]
        if len(model_samples) == len(items):
            return ModelBatch.from_samples(
                model_samples,
                pad_token_id=self.runtime.pad_token_id,
            )
        raw_samples = [item for item in items if isinstance(item, RawSingleSample)]
        if len(raw_samples) == len(items):
            return RawSingleBatch(
                samples=tuple(raw_samples),
                pad_token_id=self.runtime.pad_token_id,
            )
        raise ValueError(
            "a single batch cannot mix precomputed codec samples and raw waveform "
            "fallback samples."
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


def build_single_sample_from_codes(
    sample: RawSingleSample,
    codes: Tensor,
    runtime: DataRuntime,
) -> ModelSample:
    speech = speech_from_codes(
        codes.cpu(),
        text_token_ids=sample.text_token_ids.cpu(),
        language=sample.language,
        duration_seconds=sample.duration_seconds,
        runtime=runtime,
    )
    return build_single_sample(speech, sample.task, runtime)


def _build_item(
    sample: types.Sample,
    task: Task,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool,
) -> ModelSample | RawSingleSample:
    _validate_single_tasks([task])
    audio_item, _ = _single_items(sample)
    if runtime.audio_view in audio_item.views:
        return build_single_sample(parse_single_sample(sample, runtime), task, runtime)
    if not encode_missing_codes:
        raise ValueError(
            f"single audio sample is missing {runtime.audio_view.value!r} codec "
            "codes; materialize codec views before training or enable explicit "
            "waveform fallback."
        )
    return parse_raw_single_sample(sample, runtime, task)


def parse_raw_single_sample(
    sample: types.Sample,
    runtime: DataRuntime,
    task: Task,
) -> RawSingleSample:
    audio_item, text_item = _single_items(sample)
    waveform, sample_rate = _waveform(audio_item)
    text = text_item.views[types.TextView.TEXT]
    return RawSingleSample(
        text_token_ids=token_ids(text, runtime.text_tokenizer),
        waveform=waveform,
        sample_rate=sample_rate,
        language=_language(text_item),
        task=task,
        duration_seconds=_raw_duration_seconds(audio_item, waveform, sample_rate),
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


def _codec_codes(audio_item: types.AudioItem, runtime: DataRuntime) -> Tensor:
    try:
        codes = audio_item.views[runtime.audio_view]
    except KeyError as error:
        raise ValueError(
            f"single audio sample is missing {runtime.audio_view.value!r} codec codes."
        ) from error
    if not isinstance(codes, Tensor):
        raise TypeError("single codec codes must be a Tensor.")
    return codes


def _waveform(audio_item: types.AudioItem) -> tuple[Tensor, int]:
    value = audio_item.views.get(types.AudioView.WAVEFORM)
    if value is None:
        raise ValueError(
            "waveform fallback requires AudioView.WAVEFORM when codec codes are missing."
        )
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
    waveform, sample_rate = value
    if not isinstance(waveform, Tensor):
        raise TypeError("AudioView.WAVEFORM waveform must be a Tensor.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("AudioView.WAVEFORM sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("AudioView.WAVEFORM sample_rate must be positive.")
    return waveform, sample_rate


def _raw_duration_seconds(
    audio_item: types.AudioItem,
    waveform: Tensor,
    sample_rate: int,
) -> float:
    value = audio_item.meta.get(types.AudioMeta.DURATION)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("AudioMeta.DURATION must be a number of seconds.")
        duration = float(value)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("AudioMeta.DURATION must be finite and non-negative.")
        return duration
    return float(waveform.size(-1)) / float(sample_rate)


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
