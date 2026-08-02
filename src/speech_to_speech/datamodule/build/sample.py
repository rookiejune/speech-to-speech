from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from anydataset.types import Modality
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from ...audio_stream import AudioStream
from ...prediction import PredictionModality
from ...runtime import AudioSequenceLayout
from ...runtime.audio_tokenizer import BiCodecAudioTokenizer
from ...task import Task
from ...task_spec import resolve_prediction
from ..config import TaskConfig, task_template_index
from .._helper.tokenization import token_ids
from .ar import build_ar_sample, is_ar_task
from ..protocol import DataRuntime, TextRuntime
from ..types import (
    AcousticTarget,
    Language,
    ModelSample,
    RawSpeech,
    Speech,
    SpeechPair,
    SpeechTaskSample,
    Text,
    TextPair,
)

_PLACEHOLDER = "$$$PLACEHOLDER$$$"


def build_sample(
    speech_pair: SpeechPair,
    task: Task,
    runtime: DataRuntime,
    *,
    prediction: PredictionModality | None = None,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> ModelSample:
    prompt = _prompt(speech_pair, task, runtime, tasks=tasks)
    source, target = _source_target(speech_pair, task)
    return build_speech_sample(
        source,
        target,
        task,
        runtime,
        prompt=prompt,
        prediction=prediction,
    )


def build_task_sample(
    sample: SpeechTaskSample,
    runtime: DataRuntime,
    *,
    interleave_audio_frames: int = 25,
    mask_text_ratio: float = 0.5,
    mask_audio_ratio: float = 0.5,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> ModelSample:
    if sample.needs_codec:
        raise ValueError("raw speech task samples must be materialized before building.")
    source = sample.source
    target = sample.target
    audio_context = sample.audio_context
    if (
        isinstance(source, RawSpeech)
        or isinstance(target, RawSpeech)
        or isinstance(audio_context, RawSpeech)
    ):
        raise AssertionError("materialized task samples must not contain RawSpeech.")
    if sample.task is Task.MASKED_AR:
        from .masked import build_masked_sample

        if not isinstance(target, Speech):
            raise TypeError("MASKED_AR target must be Speech.")
        return build_masked_sample(
            target,
            sample.task,
            runtime,
            prediction=sample.prediction,
            prompt=chat_prompt(
                target.language,
                sample.task,
                runtime,
                tasks=tasks,
            ),
            interleave_audio_frames=interleave_audio_frames,
            mask_text_ratio=mask_text_ratio,
            mask_audio_ratio=mask_audio_ratio,
        )
    if is_ar_task(sample.task):
        return build_ar_sample(
            target,
            sample.task,
            runtime,
            prompt=chat_prompt(
                target.language,
                sample.task,
                runtime,
                tasks=tasks,
            ),
            prediction=sample.prediction,
            interleave_audio_frames=interleave_audio_frames,
        )
    return _build_modal_sample(
        source,
        target,
        sample.task,
        runtime,
        prompt=chat_prompt(
            target.language,
            sample.task,
            runtime,
            tasks=tasks,
        ),
        audio_context=audio_context,
        prediction=sample.prediction,
    )


def build_speech_sample(
    source: Speech,
    target: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    prediction: PredictionModality | None = None,
) -> ModelSample:
    return _build_modal_sample(
        source,
        target,
        task,
        runtime,
        prompt=prompt,
        audio_context=None,
        prediction=prediction,
    )


def _build_modal_sample(
    source: Speech | Text | None,
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    audio_context: Speech | None,
    prediction: PredictionModality | None = None,
) -> ModelSample:
    prediction = resolve_prediction(task, prediction)
    source_modality = task.source_modality
    audio_input_positions: Tensor | None = None
    if source_modality is not None:
        if source is None:
            raise ValueError("tasks with a source modality require a source item.")
        prefix_text, suffix_text = _split(prompt, _PLACEHOLDER)
        tokenizer = runtime.text_tokenizer
        prefix = token_ids(prefix_text, tokenizer)
        suffix = token_ids(suffix_text, tokenizer)
        source_ids = _global_ids(source, source_modality, runtime)

        if source_modality is Modality.AUDIO:
            source_length = source_ids.numel()
            audio_input_positions = torch.arange(
                len(prefix) + 1,
                len(prefix) + 1 + source_length,
                dtype=torch.long,
                device=source_ids.device,
            )
            source_ids = _boa_eoa(source_ids, runtime)

        input_ids = torch.cat([prefix, source_ids, suffix])
    else:
        input_ids = token_ids(prompt, runtime.text_tokenizer)

    if prediction is PredictionModality.PARALLEL:
        return _parallel_response(
            input_ids,
            target,
            task,
            runtime,
            prediction=prediction,
            audio_input_positions=audio_input_positions,
            audio_context=audio_context,
            source=source,
        )

    target_modality = (
        Modality.TEXT
        if prediction is PredictionModality.TEXT
        else Modality.AUDIO
    )
    bicodec_tokenizer = _bicodec_tokenizer(
        target,
        target_modality,
        runtime,
    )
    reference_audio_codes = None
    if bicodec_tokenizer is not None:
        tokenizer = bicodec_tokenizer
        if runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
            response_streams = (AudioStream.ACOUSTIC, AudioStream.SEMANTIC)
        elif runtime.audio_sequence_layout is AudioSequenceLayout.SEMANTIC:
            if audio_context is None:
                raise ValueError(
                    "BiCodec semantic layout requires reference audio context."
                )
            prompt_ids = _global_bicodec_ids(
                audio_context,
                (AudioStream.ACOUSTIC,),
                tokenizer,
                runtime,
            )
            input_ids = torch.cat((input_ids, _boa_eoa(prompt_ids, runtime)))
            reference_audio_codes = _structured_codes(audio_context)
            response_streams = (AudioStream.SEMANTIC,)
        else:
            raise AssertionError(
                f"unsupported audio_sequence_layout: {runtime.audio_sequence_layout}"
            )

        response_local, response_groups = tokenizer.encode_streams_with_groups(
            _structured_codes(_speech(target, role="target")),
            response_streams,
        )
        response_ids = _boa_eoa(
            runtime.layout.to_global(Modality.AUDIO.value, response_local),
            runtime,
        )
        response_groups = torch.cat(
            (
                response_groups.new_tensor([tokenizer.forced_group]),
                response_groups,
                response_groups.new_tensor([tokenizer.forced_group]),
            )
        )
    else:
        response_ids = _global_ids(target, target_modality, runtime)
        response_groups = None
    target_acoustic_codes = None
    target_semantic_codes = None
    target_audio_token_positions = None
    audio_target: Speech | None = None

    if target_modality is Modality.AUDIO:
        if not isinstance(target, Speech):
            raise TypeError("audio target must be Speech.")
        audio_target = target
        if bicodec_tokenizer is None:
            response_ids = _boa_eoa(response_ids, runtime)
        if target.acoustic_codes is not None and runtime.semantic_codec_artifact is None and (
            bicodec_tokenizer is None
        ) and (
            runtime.audio_sequence_layout is not AudioSequenceLayout.FLATTENED
        ):
            target_semantic_codes = target.semantic_codes
            target_acoustic_codes = target.acoustic_codes
    else:
        response_ids = _append_eos(response_ids, runtime)

    full_ids = torch.cat([input_ids, response_ids])
    token_labels = torch.full_like(full_ids, -100)
    token_groups = None
    if target_modality is Modality.AUDIO:
        if response_groups is None:
            # BOA is a structural response prefix; supervise codec tokens and EOA.
            token_labels[len(input_ids) + 1 :] = response_ids[1:]
        else:
            token_groups = torch.full_like(full_ids, -1)
            group_slice = token_groups[len(input_ids) :]
            group_slice.copy_(response_groups)
            predicted = response_groups.ge(0)
            label_slice = token_labels[len(input_ids) :]
            label_slice[predicted] = response_ids[predicted]
    else:
        token_labels[len(input_ids) :] = response_ids

    if target_acoustic_codes is not None:
        if audio_target is None:
            raise AssertionError("acoustic target requires an audio target.")
        target_audio_token_positions = torch.repeat_interleave(
            torch.arange(
                len(input_ids) + 1,
                len(input_ids) + 1 + audio_target.audio_token_ids.numel(),
                dtype=torch.long,
            ),
            audio_target.audio_token_spans,
        )
        if target_audio_token_positions.numel() != target_acoustic_codes.size(0):
            raise ValueError("target acoustic frames and audio tokens must align.")

    acoustic_target = (
        None
        if target_acoustic_codes is None or target_audio_token_positions is None
        else AcousticTarget(
            semantic_codes=cast(Tensor, target_semantic_codes),
            codes=target_acoustic_codes,
            token_positions=target_audio_token_positions,
        )
    )
    prompt_length = (
        len(input_ids) + 1 if target_modality is Modality.AUDIO else len(input_ids)
    )
    return ModelSample.pack(
        prompt_ids=full_ids[:prompt_length],
        response_ids=full_ids[prompt_length:],
        token_labels=token_labels,
        token_groups=token_groups,
        acoustic_target=acoustic_target,
        task=task,
        prediction=prediction,
        audio_seconds=_audio_seconds(
            source,
            target,
            task,
            prediction,
            audio_context=audio_context if reference_audio_codes is not None else None,
        ),
        audio_input_positions=audio_input_positions,
        audio_context=reference_audio_codes,
    )


def _parallel_response(
    input_ids: Tensor,
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    prediction: PredictionModality,
    audio_input_positions: Tensor | None,
    audio_context: Speech | None,
    source: Speech | Text | None,
) -> ModelSample:
    if not isinstance(target, Speech):
        raise TypeError("PARALLEL prediction requires a Speech target.")
    if isinstance(runtime.audio_tokenizer, BiCodecAudioTokenizer):
        raise ValueError(
            "PARALLEL prediction is not supported with BiCodec sequence layouts."
        )
    text = _append_eos(
        runtime.layout.to_global(Modality.TEXT.value, target.text_token_ids),
        runtime,
    )
    audio = _boa_eoa(
        runtime.layout.to_global(Modality.AUDIO.value, target.audio_token_ids),
        runtime,
    )
    response = torch.cat([text, audio])
    full_ids = torch.cat([input_ids, response])
    labels = torch.full_like(full_ids, -100)
    labels[input_ids.numel() : input_ids.numel() + text.numel()] = text
    labels[input_ids.numel() + text.numel() + 1 :] = audio[1:]
    acoustic = None
    if target.acoustic_codes is not None and runtime.semantic_codec_artifact is None and (
        runtime.audio_sequence_layout is not AudioSequenceLayout.FLATTENED
    ):
        positions = torch.arange(
            input_ids.numel() + text.numel() + 1,
            input_ids.numel() + text.numel() + 1 + target.audio_token_ids.numel(),
            dtype=torch.long,
        )
        acoustic = AcousticTarget(
            semantic_codes=target.semantic_codes,
            codes=target.acoustic_codes,
            token_positions=torch.repeat_interleave(positions, target.audio_token_spans),
        )
    return ModelSample.pack(
        prompt_ids=input_ids,
        response_ids=response,
        token_labels=labels,
        token_groups=None,
        acoustic_target=acoustic,
        task=task,
        prediction=prediction,
        audio_seconds=_audio_seconds(
            source,
            target,
            task,
            prediction,
            audio_context=audio_context,
        ),
        audio_input_positions=audio_input_positions,
        audio_context=None,
    )


def build_text_sample(
    text_pair: TextPair,
    task: Task,
    runtime: TextRuntime,
    *,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> ModelSample:
    if (
        task.source_modality is Modality.AUDIO
        or task.prediction_modality is not PredictionModality.TEXT
    ):
        raise ValueError(f"{task.value} is not supported by the text-only data path.")

    prompt = _text_prompt(
        text_pair.target.language,
        task,
        runtime,
        tasks=tasks,
    )
    source, target = _text_source_target(text_pair, task)
    if task.source_modality is Modality.TEXT:
        prefix_text, suffix_text = _split(prompt, _PLACEHOLDER)
        tokenizer = runtime.text_tokenizer
        prefix = token_ids(prefix_text, tokenizer)
        suffix = token_ids(suffix_text, tokenizer)
        source_ids = _global_text_ids(source, runtime)
        input_ids = torch.cat([prefix, source_ids, suffix])
    else:
        input_ids = token_ids(prompt, runtime.text_tokenizer)

    response_ids = _append_eos(_global_text_ids(target, runtime), runtime)
    full_ids = torch.cat([input_ids, response_ids])
    token_labels = torch.full_like(full_ids, -100)
    token_labels[len(input_ids) :] = response_ids
    return ModelSample.pack(
        prompt_ids=input_ids,
        response_ids=response_ids,
        token_labels=token_labels,
        token_groups=None,
        acoustic_target=None,
        task=task,
        prediction=PredictionModality.TEXT,
    )


def _prompt(
    speech_pair: SpeechPair,
    task: Task,
    runtime: DataRuntime,
    *,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> str:
    return chat_prompt(
        speech_pair.target.language,
        task,
        runtime,
        tasks=tasks,
    )


def chat_prompt(
    language: Language,
    task: Task,
    runtime: TextRuntime,
    *,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> str:
    instruction = task.sample_template(task_template_index(tasks, task)).format(
        language=str(language),
        source=_PLACEHOLDER,
    )
    return cast(
        str,
        runtime.text_tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        ),
    )


def _text_prompt(
    language: Language,
    task: Task,
    runtime: TextRuntime,
    *,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> str:
    instruction = task.sample_template(task_template_index(tasks, task)).format(
        language=str(language),
        source=_PLACEHOLDER,
    )
    return cast(
        str,
        runtime.text_tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        ),
    )


def _source_target(speech_pair: SpeechPair, task: Task) -> tuple[Speech, Speech]:
    if task.uses_source_role:
        return speech_pair.source, speech_pair.target
    return speech_pair.target, speech_pair.target


def _audio_seconds(
    source: Speech | Text | None,
    target: Speech | Text,
    task: Task,
    prediction: PredictionModality,
    *,
    audio_context: Speech | None = None,
) -> float:
    seconds = 0.0
    if task.source_layout.includes_audio:
        seconds += _duration(_speech(source, role="source"), role="source")
    if prediction.supervises_audio:
        if source is not target:
            seconds += _duration(_speech(target, role="target"), role="target")
        elif not task.source_layout.includes_audio:
            seconds += _duration(_speech(target, role="target"), role="target")
    if audio_context is not None and audio_context is not source:
        seconds += _duration(audio_context, role="audio context")
    return seconds


def _duration(speech: Speech, *, role: str) -> float:
    if speech.duration_seconds is None:
        raise ValueError(
            f"{role} speech is missing duration_seconds; parse raw audio samples "
            "with a DataRuntime so duration can be read from metadata or inferred "
            "from codec frames."
        )
    return float(speech.duration_seconds)


def _speech(item: Speech | Text | None, *, role: str) -> Speech:
    if not isinstance(item, Speech):
        raise TypeError(f"audio {role} must be Speech.")
    return item


def _text_source_target(text_pair: TextPair, task: Task) -> tuple[Text, Text]:
    if task.uses_source_role:
        return text_pair.source, text_pair.target
    return text_pair.target, text_pair.target


def _split(sequence: str, delimiter: str) -> tuple[str, str]:
    parts = sequence.split(delimiter)
    if len(parts) != 2:
        raise ValueError("input placeholder must occur exactly once in chat template.")
    return parts[0], parts[1]


def _global_ids(
    item: Speech | Text,
    modality: Modality,
    runtime: DataRuntime,
) -> Tensor:
    if modality is Modality.TEXT:
        local_ids = item.text_token_ids
    elif modality is Modality.AUDIO:
        local_ids = _speech(item, role="item").audio_token_ids
    else:
        raise ValueError(f"unsupported modality: {modality.value}")
    return runtime.layout.to_global(modality.value, local_ids)


def _bicodec_tokenizer(
    target: Speech | Text,
    target_modality: Modality | None,
    runtime: DataRuntime,
) -> BiCodecAudioTokenizer | None:
    if target_modality is not Modality.AUDIO:
        return None
    if not isinstance(target, Speech):
        raise TypeError("audio target must be Speech.")
    tokenizer = runtime.audio_tokenizer
    if not isinstance(tokenizer, BiCodecAudioTokenizer):
        return None
    return tokenizer


def _global_bicodec_ids(
    speech: Speech,
    streams: tuple[AudioStream, ...],
    tokenizer: BiCodecAudioTokenizer,
    runtime: DataRuntime,
) -> Tensor:
    local_ids = tokenizer.encode_streams(_structured_codes(speech), streams)
    return runtime.layout.to_global(Modality.AUDIO.value, local_ids)


def _structured_codes(speech: Speech) -> SemanticAcousticCodes:
    if speech.acoustic_codes is None:
        raise ValueError("BiCodec sequence layouts require semantic and acoustic codes.")
    return SemanticAcousticCodes(
        semantic=speech.semantic_codes,
        acoustic=speech.acoustic_codes,
    )


def _global_text_ids(
    text: Text,
    runtime: TextRuntime,
) -> Tensor:
    return runtime.layout.to_global(Modality.TEXT.value, text.text_token_ids)


def _boa_eoa(ids: Tensor, runtime: DataRuntime) -> Tensor:
    return torch.cat(
        (
            ids.new_tensor([runtime.boa_token_id]),
            ids,
            ids.new_tensor([runtime.eoa_token_id]),
        )
    )


def _append_eos(ids: Tensor, runtime: TextRuntime) -> Tensor:
    return torch.cat([ids, ids.new_tensor([runtime.eos_token_id])])
