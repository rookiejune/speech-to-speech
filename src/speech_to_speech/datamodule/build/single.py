from __future__ import annotations

from collections.abc import Mapping
from anydataset import types

from ...prediction import PredictionModality
from ...source import SourceLayout
from ...task import Task
from ...task_spec import resolve_prediction
from ..config import TaskConfig
from .._helper.duration import from_frames
from .._helper.task import TaskWeights
from .._helper.tokenization import token_ids
from .ar import build_ar_sample, is_ar_task
from ..parse.parser import parse_audio_codes, raw_speech, speech_from_codes
from ..protocol import DataRuntime
from .sample import build_speech_sample, build_task_sample, chat_prompt
from ..types import (
    ModelBatch,
    ModelSample,
    RawSpeech,
    RawSpeechBatch,
    Speech,
    AudioContextSample,
    SpeechTaskSample,
    Text,
)

_SINGLE_TASKS = frozenset(
    {
        Task.ASR,
        Task.AUDIO_AR,
        Task.INTERLEAVED_AR,
        Task.MASKED_AR,
        Task.PARALLEL_AR,
        Task.TEXT_AR,
        Task.TTS,
    }
)


class SingleCollator:
    def __init__(
        self,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
        *,
        encode_missing_codes: bool = False,
        interleave_audio_frames: int = 25,
        mask_text_ratio: float = 0.5,
        mask_audio_ratio: float = 0.5,
        prediction: PredictionModality | None = None,
        tasks: Mapping[Task, TaskConfig] | None = None,
    ) -> None:
        self.runtime = runtime
        self.encode_missing_codes = encode_missing_codes
        self.interleave_audio_frames = interleave_audio_frames
        self.mask_text_ratio = mask_text_ratio
        self.mask_audio_ratio = mask_audio_ratio
        self.task_configs = tasks
        _validate_single_tasks(_positive_tasks(task_weights))
        self._task_weights = TaskWeights(task_weights, prediction=prediction)

    @property
    def tasks(self) -> list[Task]:
        return self._task_weights.tasks

    @property
    def prediction(self) -> PredictionModality | None:
        return self._task_weights.prediction

    def _items(self, samples: list[types.Sample]) -> list[SpeechTaskSample]:
        tasks = self._task_weights.allocate(len(samples))
        return [
            _build_item(
                sample,
                task,
                self.runtime,
                encode_missing_codes=self.encode_missing_codes,
                prediction=self.prediction,
            )
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[types.Sample]) -> ModelBatch | RawSpeechBatch:
        items = self._items(samples)
        if any(item.needs_codec for item in items):
            return RawSpeechBatch(
                samples=tuple(items),
                pad_token_id=self.runtime.pad_token_id,
                interleave_audio_frames=self.interleave_audio_frames,
                mask_text_ratio=self.mask_text_ratio,
                mask_audio_ratio=self.mask_audio_ratio,
            )
        return ModelBatch.from_samples(
            [
                build_task_sample(
                    item,
                    self.runtime,
                    interleave_audio_frames=self.interleave_audio_frames,
                    mask_text_ratio=self.mask_text_ratio,
                    mask_audio_ratio=self.mask_audio_ratio,
                    tasks=self.task_configs,
                )
                for item in items
            ],
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
        duration_seconds=from_frames(
            audio_item.meta.get(types.AudioMeta.DURATION),
            frames=semantic_codes.size(0),
            frame_rate=runtime.codec_frame_rate,
        ),
        runtime=runtime,
    )


def build_single_sample(
    utterance: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    interleave_audio_frames: int = 25,
    mask_text_ratio: float = 0.5,
    mask_audio_ratio: float = 0.5,
    prediction: PredictionModality | None = None,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> ModelSample:
    _validate_single_tasks([task])
    prediction = resolve_prediction(task, prediction)
    if task is Task.MASKED_AR:
        from .masked import build_masked_sample

        return build_masked_sample(
            utterance,
            task,
            runtime,
            prompt=chat_prompt(
                utterance.language,
                task,
                runtime,
                tasks=tasks,
            ),
            prediction=prediction,
            interleave_audio_frames=interleave_audio_frames,
            mask_text_ratio=mask_text_ratio,
            mask_audio_ratio=mask_audio_ratio,
        )
    if is_ar_task(task):
        target: Speech | Text
        if prediction.supervises_audio:
            target = utterance
        else:
            target = Text(
                text_token_ids=utterance.text_token_ids,
                language=utterance.language,
            )
        return build_ar_sample(
            target,
            task,
            runtime,
            prompt=chat_prompt(
                utterance.language,
                task,
                runtime,
                tasks=tasks,
            ),
            prediction=prediction,
            interleave_audio_frames=interleave_audio_frames,
        )
    prompt = chat_prompt(
        utterance.language,
        task,
        runtime,
        tasks=tasks,
    )
    return build_speech_sample(
        utterance,
        utterance,
        task,
        runtime,
        prompt=prompt,
        prediction=prediction,
    )


def _build_item(
    sample: types.Sample,
    task: Task,
    runtime: DataRuntime,
    *,
    encode_missing_codes: bool,
    prediction: PredictionModality | None,
) -> SpeechTaskSample:
    _validate_single_tasks([task])
    prediction = resolve_prediction(task, prediction)
    audio_item, text_item = _single_items(sample)
    text = Text(
        text_token_ids=token_ids(
            text_item.views[types.TextView.TEXT],
            runtime.text_tokenizer,
        ),
        language=_language(text_item),
    )
    if (
        task.source_layout is SourceLayout.NONE
        and not prediction.supervises_audio
    ):
        return SpeechTaskSample(
            source=None,
            target=text,
            task=task,
            prediction=prediction,
        )
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
    return _task_sample(
        utterance,
        text,
        task,
        prediction=prediction,
        audio_context=audio_context,
    )


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
    prediction: PredictionModality,
    audio_context: Speech | RawSpeech | None = None,
) -> SpeechTaskSample:
    source = None
    if task.source_layout is SourceLayout.TEXT_AUDIO:
        source = utterance
    elif task.source_modality is types.Modality.AUDIO:
        source = utterance
    elif task.source_modality is types.Modality.TEXT:
        source = text
    if prediction.supervises_audio:
        target: Speech | RawSpeech | Text = utterance
    else:
        target = text
    return SpeechTaskSample(
        source=source,
        target=target,
        task=task,
        prediction=prediction,
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


def _language(text_item: types.TextItem):
    from ..types import Language

    return Language(text_item.meta[types.TextMeta.LANG])


def _validate_single_tasks(tasks: list[Task]) -> None:
    for task in tasks:
        if task not in _SINGLE_TASKS:
            raise ValueError(f"{task.value} is not supported by the single data path.")


def _positive_tasks(values: Mapping[Task, float]) -> list[Task]:
    return [task for task, weight in values.items() if weight > 0]
