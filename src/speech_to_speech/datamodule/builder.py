from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from anydataset.types import Modality
from torch import Generator, Tensor

from ..audio import AudioCodes, AudioStream
from .loader.contract import ARFraming, validate_ar_framing
from ..runtime import AudioSequenceLayout
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.backbone.contract import TextTokenizer
from ..task import (
    FieldRole,
    PredictionModality,
    ResponseControl,
    ResponseSpec,
    Task,
    response_control_tokens,
    resolve_response,
    uses_source_ctc,
    uses_target_ctc,
)
from ..task.templates import format_response_instruction
from .config import TaskConfig, task_template_index
from .contract import DataRuntime, TextRuntime
from .batch import ModelSample
from .sample import (
    Language,
    RawSpeech,
    Speech,
    SpeechPair,
    SpeechTaskSample,
    Text,
    TextPair,
)
from .batch import (
    AcousticTarget,
    CTCTarget,
)

_PLACEHOLDER = "$$$PLACEHOLDER$$$"


def build_sample(
    speech_pair: SpeechPair,
    task: Task,
    runtime: DataRuntime,
    *,
    trace: str | None = None,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> ModelSample:
    response = resolve_response(task, trace=trace)
    prompt = _chat_prompt(
        speech_pair.target.language,
        task,
        runtime,
        response=response,
        tasks=tasks,
    )
    source, target = _source_target(speech_pair, task)
    return _build_modal_sample(
        source,
        target,
        task,
        runtime,
        prompt=prompt,
        audio_context=None,
        response=response,
    )


def build_task_sample(
    sample: SpeechTaskSample,
    runtime: DataRuntime,
    *,
    interleave_audio_frames: int = 25,
    mask_text_ratio: float = 0.5,
    mask_audio_ratio: float = 0.5,
    ar_framing: ARFraming = ARFraming.INSTRUCTION,
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
    validate_ar_framing(ar_framing, [sample.task])
    response = resolve_response(sample.task, trace=sample.trace)
    if ar_framing is ARFraming.PRETRAINING:
        return _build_pretraining_ar_sample(
            target,
            sample.task,
            runtime,
            spec=response,
        )
    prompt = _chat_prompt(
        target.language,
        sample.task,
        runtime,
        response=response,
        tasks=tasks,
    )
    if sample.task is Task.MASKED_AR:
        if not isinstance(target, Speech):
            raise TypeError("MASKED_AR target must be Speech.")
        return _build_masked_sample(
            target,
            sample.task,
            runtime,
            spec=response,
            prompt=prompt,
            interleave_audio_frames=interleave_audio_frames,
            mask_text_ratio=mask_text_ratio,
            mask_audio_ratio=mask_audio_ratio,
        )
    if is_ar_task(sample.task):
        return _build_ar_sample(
            target,
            sample.task,
            runtime,
            prompt=prompt,
            spec=response,
            interleave_audio_frames=interleave_audio_frames,
        )
    return _build_modal_sample(
        source,
        target,
        sample.task,
        runtime,
        prompt=prompt,
        audio_context=audio_context,
        response=response,
    )


def build_speech_sample(
    source: Speech,
    target: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    trace: str | None = None,
) -> ModelSample:
    response = resolve_response(task, trace=trace)
    return _build_modal_sample(
        source,
        target,
        task,
        runtime,
        prompt=prompt,
        audio_context=None,
        response=response,
    )


def _build_modal_sample(
    source: Speech | Text | None,
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    audio_context: Speech | None,
    response: ResponseSpec,
) -> ModelSample:
    prediction = response.prediction
    target_language = _target_language(response, target)
    input_ids, audio_input_positions = _modal_input_ids(
        source,
        task.source_modality,
        prompt,
        runtime,
    )
    source_ctc = _source_ctc(
        source,
        audio_input_positions,
        task,
        runtime,
    )
    if prediction is PredictionModality.PARALLEL:
        return _parallel_response(
            input_ids,
            target,
            task,
            runtime,
            audio_input_positions=audio_input_positions,
            audio_context=audio_context,
            source=source,
            source_ctc=source_ctc,
            response=response,
        )

    if prediction is PredictionModality.TEXT:
        return _text_response(
            input_ids,
            source,
            target,
            task,
            runtime,
            response=response,
            audio_input_positions=audio_input_positions,
            source_ctc=source_ctc,
        )

    target_modality = _target_modality(prediction)
    (
        input_ids,
        response_ids,
        reference_codes,
        uses_bicodec,
    ) = _target_response(
        input_ids,
        source if task.source_layout.includes_audio else None,
        target,
        target_modality,
        runtime,
        audio_context=audio_context,
    )

    full_ids = torch.cat([input_ids, response_ids])
    token_labels = _token_supervision(
        input_ids,
        response_ids,
    )
    target_ctc = _target_ctc(
        target,
        task,
        runtime,
        trace=response.name,
        input_length=input_ids.numel(),
        response_length=response_ids.numel(),
    )
    audio_target, target_semantic_codes, target_acoustic_codes = _acoustic_codes(
        target,
        target_modality,
        runtime,
        uses_bicodec=uses_bicodec,
    )
    acoustic_target = _acoustic_target(
        input_ids,
        audio_target,
        target_semantic_codes,
        target_acoustic_codes,
    )
    prompt_length = len(input_ids)
    return ModelSample.pack(
        prompt_ids=full_ids[:prompt_length],
        response_ids=full_ids[prompt_length:],
        token_labels=token_labels,
        acoustic_target=acoustic_target,
        source_ctc=source_ctc,
        target_ctc=target_ctc,
        task=task,
        trace=response.name,
        target_language=target_language,
        audio_seconds=_audio_seconds(
            source,
            target,
            task,
            prediction,
            audio_context=audio_context if reference_codes is not None else None,
        ),
        audio_input_positions=audio_input_positions,
    )


def _modal_input_ids(
    source: Speech | Text | None,
    source_modality: Modality | None,
    prompt: str,
    runtime: DataRuntime,
) -> tuple[Tensor, Tensor | None]:
    if source_modality is None:
        return token_ids(prompt, runtime.text_tokenizer), None
    if source is None:
        raise ValueError("tasks with a source modality require a source item.")
    prefix_text, suffix_text = _split(prompt, _PLACEHOLDER)
    prefix = token_ids(prefix_text, runtime.text_tokenizer)
    suffix = token_ids(suffix_text, runtime.text_tokenizer)
    source_ids = _global_ids(
        source,
        source_modality,
        runtime,
        input_audio=source_modality is Modality.AUDIO,
    )
    audio_input_positions = None
    if source_modality is Modality.AUDIO:
        audio_input_positions = torch.arange(
            len(prefix) + 2,
            len(prefix) + 2 + source_ids.numel(),
            dtype=torch.long,
            device=source_ids.device,
        )
        source_ids = _input_boa_eoa(source_ids, runtime)
    return torch.cat([prefix, source_ids, suffix]), audio_input_positions


def _target_language(
    response: ResponseSpec,
    target: Speech | Text,
) -> str | None:
    if not response.requires_target_language:
        return None
    return target.language.code


def _target_modality(prediction: PredictionModality) -> Modality:
    return Modality.TEXT if prediction is PredictionModality.TEXT else Modality.AUDIO


def _target_response(
    input_ids: Tensor,
    source: Speech | Text | None,
    target: Speech | Text,
    target_modality: Modality,
    runtime: DataRuntime,
    *,
    audio_context: Speech | None,
) -> tuple[Tensor, Tensor, AudioCodes | None, bool]:
    tokenizer = _bicodec_tokenizer(target, target_modality, runtime)
    if tokenizer is not None:
        input_ids, response_ids, reference = _bicodec_response(
            input_ids,
            source,
            target,
            runtime,
            tokenizer,
            audio_context=audio_context,
        )
        return input_ids, response_ids, reference, True
    response_ids = _global_ids(target, target_modality, runtime)
    if target_modality is Modality.AUDIO:
        response_ids = _boa_eoa(response_ids, runtime)
    else:
        response_ids = _append_eos(response_ids, runtime)
    return input_ids, response_ids, None, False


def _bicodec_response(
    input_ids: Tensor,
    source: Speech | Text | None,
    target: Speech | Text,
    runtime: DataRuntime,
    tokenizer: BiCodecAudioTokenizer,
    *,
    audio_context: Speech | None,
) -> tuple[Tensor, Tensor, AudioCodes | None]:
    reference_codes = None
    source_has_global = isinstance(source, Speech) and source.global_codes is not None
    if source_has_global and audio_context is not None and audio_context is not source:
        raise ValueError(
            "BiCodec tasks cannot select both source and reference global streams."
        )
    if audio_context is not None:
        prompt_ids = _global_bicodec_ids(
            audio_context,
            (AudioStream.GLOBAL,),
            tokenizer,
            runtime,
        )
        input_ids = torch.cat((input_ids, _boa_eoa(prompt_ids, runtime)))
        reference_codes = _structured_codes(audio_context)
    if source_has_global or audio_context is not None:
        response_streams = (AudioStream.SEMANTIC,)
    else:
        response_streams = (AudioStream.GLOBAL, AudioStream.SEMANTIC)
    response_local = tokenizer.encode_streams(
        _structured_codes(_speech(target, role="target")),
        response_streams,
    )
    response_ids = _boa_eoa(
        runtime.layout.to_global(Modality.AUDIO.value, response_local),
        runtime,
    )
    return input_ids, response_ids, reference_codes


def _token_supervision(
    input_ids: Tensor,
    response_ids: Tensor,
) -> Tensor:
    full_ids = torch.cat([input_ids, response_ids])
    labels = _ignored_labels(full_ids)
    _supervise_labels(labels, len(input_ids), response_ids)
    return labels


def _ignored_labels(ids: Tensor) -> Tensor:
    return torch.full_like(ids, -100)


def _supervise_labels(labels: Tensor, start: int, ids: Tensor) -> None:
    labels[start : start + ids.numel()] = ids


def _acoustic_codes(
    target: Speech | Text,
    target_modality: Modality,
    runtime: DataRuntime,
    *,
    uses_bicodec: bool,
) -> tuple[Speech | None, Tensor | None, Tensor | None]:
    if target_modality is not Modality.AUDIO:
        return None, None, None
    if not isinstance(target, Speech):
        raise TypeError("audio target must be Speech.")
    if (
        target.acoustic_codes is not None
        and runtime.acoustic_generator_artifact is None
        and not uses_bicodec
        and runtime.audio_sequence_layout is not AudioSequenceLayout.FLATTENED
    ):
        return target, target.semantic_codes, target.acoustic_codes
    return target, None, None


def _acoustic_target(
    input_ids: Tensor,
    audio_target: Speech | None,
    target_semantic_codes: Tensor | None,
    target_acoustic_codes: Tensor | None,
) -> AcousticTarget | None:
    if target_acoustic_codes is None:
        return None
    if audio_target is None:
        raise AssertionError("acoustic target requires an audio target.")
    positions = torch.repeat_interleave(
        torch.arange(
            len(input_ids) + 2,
            len(input_ids) + 2 + audio_target.audio_token_ids.numel(),
            dtype=torch.long,
        ),
        audio_target.audio_token_spans,
    )
    if positions.numel() != target_acoustic_codes.size(0):
        raise ValueError("target acoustic frames and audio tokens must align.")
    return AcousticTarget(
        semantic_codes=cast(Tensor, target_semantic_codes),
        codes=target_acoustic_codes,
        token_positions=positions,
    )


def _parallel_response(
    input_ids: Tensor,
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    audio_input_positions: Tensor | None,
    audio_context: Speech | None,
    source: Speech | Text | None,
    source_ctc: CTCTarget | None,
    response: ResponseSpec,
) -> ModelSample:
    if not isinstance(target, Speech):
        raise TypeError("PARALLEL prediction requires a Speech target.")
    if isinstance(runtime.audio_tokenizer, BiCodecAudioTokenizer):
        raise ValueError(
            "PARALLEL prediction is not supported with BiCodec sequence layouts."
        )
    text, text_labels = _text_response_ids(
        source,
        target,
        response,
        runtime,
        target_language=_target_language(response, target),
    )
    audio = _boa_eoa(
        runtime.layout.to_global(Modality.AUDIO.value, target.audio_token_ids),
        runtime,
    )
    response_ids = torch.cat([text, audio])
    full_ids = torch.cat([input_ids, response_ids])
    labels = _ignored_labels(full_ids)
    _supervise_labels(labels, input_ids.numel(), text_labels)
    _supervise_labels(labels, input_ids.numel() + text.numel(), audio)
    acoustic = None
    if target.acoustic_codes is not None and runtime.acoustic_generator_artifact is None and (
        runtime.audio_sequence_layout is not AudioSequenceLayout.FLATTENED
    ):
        positions = torch.arange(
            input_ids.numel() + text.numel() + 2,
            input_ids.numel() + text.numel() + 2 + target.audio_token_ids.numel(),
            dtype=torch.long,
        )
        acoustic = AcousticTarget(
            semantic_codes=target.semantic_codes,
            codes=target.acoustic_codes,
            token_positions=torch.repeat_interleave(positions, target.audio_token_spans),
        )
    return ModelSample.pack(
        prompt_ids=input_ids,
        response_ids=response_ids,
        token_labels=labels,
        acoustic_target=acoustic,
        source_ctc=source_ctc,
        target_ctc=None,
        task=task,
        trace=response.name,
        target_language=_target_language(response, target),
        audio_seconds=_audio_seconds(
            source,
            target,
            task,
            response.prediction,
            audio_context=audio_context,
        ),
        audio_input_positions=audio_input_positions,
    )


def _text_response(
    input_ids: Tensor,
    source: Speech | Text | None,
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    response: ResponseSpec,
    audio_input_positions: Tensor | None,
    source_ctc: CTCTarget | None,
) -> ModelSample:
    response_ids, response_labels = _text_response_ids(
        source,
        target,
        response,
        runtime,
        target_language=_target_language(response, target),
    )
    full_ids = torch.cat((input_ids, response_ids))
    labels = _ignored_labels(full_ids)
    _supervise_labels(labels, input_ids.numel(), response_labels)
    return ModelSample.pack(
        prompt_ids=input_ids,
        response_ids=response_ids,
        token_labels=labels,
        acoustic_target=None,
        source_ctc=source_ctc,
        target_ctc=None,
        task=task,
        trace=response.name,
        target_language=_target_language(response, target),
        audio_seconds=_audio_seconds(
            source,
            target,
            task,
            response.prediction,
        ),
        audio_input_positions=audio_input_positions,
    )


def _text_response_ids(
    source: Speech | Text | None,
    target: Speech | Text,
    response_spec: ResponseSpec,
    runtime: TextRuntime,
    *,
    target_language: str | None,
) -> tuple[Tensor, Tensor]:
    steps = [step for step in response_spec.steps if step.modality is Modality.TEXT]
    if not steps:
        raise ValueError("text response requires at least one text field.")
    values: list[Tensor] = []
    labels: list[Tensor] = []
    for step in steps:
        item = source if step.role is FieldRole.SOURCE else target
        if item is None:
            raise ValueError("source response text requires a source item.")
        payload = _global_text_ids(item, runtime)
        if step.control is ResponseControl.EOS:
            framed = _append_eos(payload, runtime)
            values.append(framed)
            labels.append(framed)
            continue
        controls = response_control_tokens(
            step.control,
            target_language=target_language,
        )
        if controls is None:
            raise ValueError("text response steps require EOS, ASR, or MT control.")
        prefix = payload.new_tensor(
            [runtime.control_token_id(token) for token in controls.prefix]
        )
        suffix = payload.new_tensor([runtime.control_token_id(controls.end)])
        framed = torch.cat((prefix, payload, suffix))
        values.append(framed)
        labels.append(framed)
    return torch.cat(values), torch.cat(labels)


def _source_ctc(
    source: Speech | Text | None,
    positions: Tensor | None,
    task: Task,
    runtime: DataRuntime,
) -> CTCTarget | None:
    if not uses_source_ctc(task):
        return None
    if not isinstance(source, Speech) or positions is None:
        raise TypeError("source CTC requires source speech and audio positions.")
    return ctc_target(positions, source, runtime)


def _target_ctc(
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    trace: str,
    input_length: int,
    response_length: int,
) -> CTCTarget | None:
    if not uses_target_ctc(task, trace=trace):
        return None
    if not isinstance(target, Speech):
        raise TypeError("target CTC requires target speech.")
    if response_length < 4:
        raise ValueError(
            "target audio response must contain BOA, schema, payload, and EOA."
        )
    positions = torch.arange(
        input_length + 2,
        input_length + response_length - 1,
        dtype=torch.long,
        device=target.audio_token_ids.device,
    )
    return ctc_target(positions, target, runtime)


def build_text_sample(
    text_pair: TextPair,
    task: Task,
    runtime: TextRuntime,
    *,
    ar_framing: ARFraming = ARFraming.INSTRUCTION,
    tasks: Mapping[Task, TaskConfig] | None = None,
    trace: str | None = None,
) -> ModelSample:
    response = resolve_response(task, trace=trace)
    if (
        task.source_modality is Modality.AUDIO
        or response.prediction is not PredictionModality.TEXT
    ):
        raise ValueError(f"{task.value} is not supported by the text-only data path.")
    validate_ar_framing(ar_framing, [task])
    if ar_framing is ARFraming.PRETRAINING:
        return _build_pretraining_ar_sample(
            text_pair.target,
            task,
            runtime,
            spec=response,
        )

    prompt = _chat_prompt(
        text_pair.target.language,
        task,
        runtime,
        response=response,
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

    response_ids, response_labels = _text_response_ids(
        source,
        target,
        response,
        runtime,
        target_language=_target_language(response, target),
    )
    full_ids = torch.cat([input_ids, response_ids])
    token_labels = _ignored_labels(full_ids)
    _supervise_labels(token_labels, len(input_ids), response_labels)
    return ModelSample.pack(
        prompt_ids=input_ids,
        response_ids=response_ids,
        token_labels=token_labels,
        acoustic_target=None,
        task=task,
        trace=response.name,
        target_language=_target_language(response, target),
    )


def chat_prompt(
    language: Language,
    task: Task,
    runtime: TextRuntime,
    *,
    trace: str | None = None,
    tasks: Mapping[Task, TaskConfig] | None = None,
) -> str:
    response = resolve_response(task, trace=trace)
    return _chat_prompt(
        language,
        task,
        runtime,
        response=response,
        tasks=tasks,
    )


def _chat_prompt(
    language: Language,
    task: Task,
    runtime: TextRuntime,
    *,
    response: ResponseSpec,
    tasks: Mapping[Task, TaskConfig] | None,
) -> str:
    instruction = task.sample_template(task_template_index(tasks, task)).format(
        language=str(language),
        source=_PLACEHOLDER,
    )
    instruction = format_response_instruction(
        instruction,
        response,
        language=str(language),
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
    *,
    input_audio: bool = False,
) -> Tensor:
    if modality is Modality.TEXT:
        local_ids = item.text_token_ids
    elif modality is Modality.AUDIO:
        local_ids = _speech(item, role="item").audio_token_ids
    else:
        raise ValueError(f"unsupported modality: {modality.value}")
    block = (
        runtime.input_audio_block_name
        if input_audio and modality is Modality.AUDIO
        else modality.value
    )
    return runtime.layout.to_global(block, local_ids)


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


def _structured_codes(speech: Speech) -> AudioCodes:
    if speech.global_codes is None:
        raise ValueError("BiCodec sequence layouts require semantic and global codes.")
    return AudioCodes(
        semantic_codes=speech.semantic_codes,
        global_codes=speech.global_codes,
    )


def _global_text_ids(
    text: Speech | Text,
    runtime: TextRuntime,
) -> Tensor:
    return runtime.layout.to_global(Modality.TEXT.value, text.text_token_ids)


def _boa_eoa(ids: Tensor, runtime: DataRuntime) -> Tensor:
    return torch.cat(
        (
            ids.new_tensor([runtime.boa_token_id]),
            ids.new_tensor([runtime.audio_schema_token_id]),
            ids,
            ids.new_tensor([runtime.eoa_token_id]),
        )
    )


def _input_boa_eoa(ids: Tensor, runtime: DataRuntime) -> Tensor:
    return torch.cat(
        (
            ids.new_tensor([runtime.input_boa_token_id]),
            ids.new_tensor([runtime.input_audio_schema_token_id]),
            ids,
            ids.new_tensor([runtime.input_eoa_token_id]),
        )
    )


def _append_eos(ids: Tensor, runtime: TextRuntime) -> Tensor:
    return torch.cat([ids, ids.new_tensor([runtime.eos_token_id])])


def is_ar_task(task: Task) -> bool:
    from ..task import TaskObjective

    return (
        not task.program.context
        and task.program.objective is TaskObjective.CAUSAL
    )


def build_pretraining_ar_sample(
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime | TextRuntime,
    *,
    trace: str | None = None,
) -> ModelSample:
    """Build an instruction-free causal-LM sample for a single modality.

    Both text and audio examples use the tokenizer BOS as a non-empty generation
    seed. The complete text/audio response framing remains in ``response_ids``
    and is supervised.
    """
    response = resolve_response(task, trace=trace)
    return _build_pretraining_ar_sample(
        target,
        task,
        runtime,
        spec=response,
    )


def _build_pretraining_ar_sample(
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime | TextRuntime,
    *,
    spec: ResponseSpec,
) -> ModelSample:
    if task not in {Task.AUDIO_AR, Task.TEXT_AR}:
        raise ValueError(
            "pretraining AR samples only support AUDIO_AR and TEXT_AR tasks."
        )
    prediction = spec.prediction
    if prediction is PredictionModality.TEXT:
        if not isinstance(target, Text):
            raise TypeError("TEXT_AR pretraining target must be Text.")
        bos = runtime.text_tokenizer.bos_token_id
        if bos is None:
            raise ValueError(
                "instruction-free text AR framing requires tokenizer bos_token_id."
            )
        if isinstance(bos, bool) or not isinstance(bos, int):
            raise TypeError("tokenizer bos_token_id must be an integer.")
        if bos < 0:
            raise ValueError("tokenizer bos_token_id must be non-negative.")
        prompt = target.text_token_ids.new_tensor([bos])
        response = _ar_append_eos(
            runtime.layout.to_global(Modality.TEXT.value, target.text_token_ids),
            runtime,
        )
        return _pack(
            prompt,
            response,
            task=task,
            spec=spec,
            target_language=_target_language(spec, target),
            supervise_from=0,
            acoustic_target=None,
            audio_seconds=0.0,
        )

    if not isinstance(target, Speech):
        raise TypeError("AUDIO_AR pretraining target must be Speech.")
    data_runtime = cast(DataRuntime, runtime)
    wrapped = _ar_boa_eoa(
        data_runtime.layout.to_global(Modality.AUDIO.value, target.audio_token_ids),
        data_runtime,
    )
    prompt = wrapped.new_tensor([data_runtime.bos_token_id])
    response = wrapped
    acoustic = _ar_acoustic_target(
        target,
        data_runtime,
        audio_token_start=prompt.numel() + 2,
    )
    return _pack(
        prompt,
        response,
        task=task,
        spec=spec,
        target_language=_target_language(spec, target),
        supervise_from=0,
        acoustic_target=acoustic,
        target_ctc=ctc_target(
            torch.arange(
                prompt.numel() + 2,
                prompt.numel() + 2 + target.audio_token_ids.numel(),
                dtype=torch.long,
                device=target.audio_token_ids.device,
            ),
            target,
            data_runtime,
        ),
        audio_seconds=_ar_duration(target),
    )


def build_ar_sample(
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    trace: str | None = None,
    interleave_audio_frames: int = 25,
) -> ModelSample:
    response = resolve_response(task, trace=trace)
    return _build_ar_sample(
        target,
        task,
        runtime,
        prompt=prompt,
        spec=response,
        interleave_audio_frames=interleave_audio_frames,
    )


def _build_ar_sample(
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    spec: ResponseSpec,
    interleave_audio_frames: int,
) -> ModelSample:
    if not is_ar_task(task):
        raise ValueError(f"{task.value} is not an autoregressive task.")
    if task.source_modality is not None:
        raise ValueError(f"{task.value} must not use a source modality.")
    prediction = spec.prediction
    marker = token_ids(prompt, runtime.text_tokenizer)

    if prediction is PredictionModality.TEXT:
        if not isinstance(target, Text):
            raise TypeError("TEXT_AR target must be Text.")
        response = _ar_append_eos(
            runtime.layout.to_global(Modality.TEXT.value, target.text_token_ids),
            runtime,
        )
        return _pack(
            marker,
            response,
            task=task,
            spec=spec,
            target_language=_target_language(spec, target),
            supervise_from=0,
            acoustic_target=None,
            audio_seconds=0.0,
        )

    if not isinstance(target, Speech):
        raise TypeError(f"{task.value} target must be Speech.")

    if prediction is PredictionModality.AUDIO:
        audio = _ar_boa_eoa(
            runtime.layout.to_global(Modality.AUDIO.value, target.audio_token_ids),
            runtime,
        )
        prompt_ids = marker
        response_ids = audio
        acoustic = _ar_acoustic_target(
            target,
            runtime,
            audio_token_start=prompt_ids.numel() + 2,
        )
        return _pack(
            prompt_ids,
            response_ids,
            task=task,
            spec=spec,
            target_language=_target_language(spec, target),
            supervise_from=0,
            acoustic_target=acoustic,
            target_ctc=ctc_target(
                torch.arange(
                    prompt_ids.numel() + 2,
                    prompt_ids.numel() + 2 + target.audio_token_ids.numel(),
                    dtype=torch.long,
                    device=target.audio_token_ids.device,
                ),
                target,
                runtime,
            ),
            audio_seconds=_ar_duration(target),
        )

    if prediction is PredictionModality.PARALLEL:
        return pack_parallel(
            marker,
            target,
            task,
            runtime,
            spec=spec,
        )

    if prediction is PredictionModality.INTERLEAVED:
        return pack_interleaved(
            marker,
            target,
            task,
            runtime,
            spec=spec,
            interleave_audio_frames=interleave_audio_frames,
        )

    raise ValueError(f"unsupported AR prediction modality: {prediction.value}")


def pack_parallel(
    marker: Tensor,
    speech: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    spec: ResponseSpec,
) -> ModelSample:
    text = _ar_append_eos(
        runtime.layout.to_global(Modality.TEXT.value, speech.text_token_ids),
        runtime,
    )
    audio = _ar_boa_eoa(
        runtime.layout.to_global(Modality.AUDIO.value, speech.audio_token_ids),
        runtime,
    )
    response = torch.cat([text, audio])
    labels = torch.full_like(torch.cat([marker, response]), -100)
    labels[marker.numel() : marker.numel() + text.numel()] = text
    labels[marker.numel() + text.numel() :] = audio
    acoustic = _ar_acoustic_target(
        speech,
        runtime,
        audio_token_start=marker.numel() + text.numel() + 2,
    )
    return ModelSample.pack(
        prompt_ids=marker,
        response_ids=response,
        token_labels=labels,
        acoustic_target=acoustic,
        task=task,
        trace=spec.name,
        target_language=_target_language(spec, speech),
        audio_seconds=_ar_duration(speech),
        audio_input_positions=None,
    )


def pack_interleaved(
    marker: Tensor,
    speech: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    spec: ResponseSpec,
    interleave_audio_frames: int,
) -> ModelSample:
    if (
        isinstance(interleave_audio_frames, bool)
        or not isinstance(interleave_audio_frames, int)
        or interleave_audio_frames < 1
    ):
        raise ValueError("interleave_audio_frames must be a positive integer.")
    audio_local = speech.audio_token_ids
    text_local = speech.text_token_ids
    spans = speech.audio_token_spans
    if audio_local.numel() == 0:
        raise ValueError("interleaved prediction requires non-empty audio tokens.")
    chunks: list[tuple[Tensor, Tensor]] = []
    index = 0
    while index < audio_local.numel():
        end = index
        frames = 0
        while end < audio_local.numel():
            next_frames = frames + int(spans[end].item())
            if end > index and next_frames > interleave_audio_frames:
                break
            frames = next_frames
            end += 1
        chunks.append((audio_local[index:end], spans[index:end]))
        index = end

    text_pieces = _split_proportional(text_local, len(chunks))
    pieces: list[Tensor] = []
    label_pieces: list[Tensor] = []
    audio_token_positions: list[Tensor] = []
    cursor = marker.numel()
    for text_chunk, (audio_chunk, span_chunk) in zip(text_pieces, chunks):
        text_ids = runtime.layout.to_global(Modality.TEXT.value, text_chunk)
        audio_ids = _ar_boa_eoa(
            runtime.layout.to_global(Modality.AUDIO.value, audio_chunk),
            runtime,
        )
        pieces.extend([text_ids, audio_ids])
        label_pieces.append(text_ids)
        label_pieces.append(audio_ids)
        audio_start = cursor + text_ids.numel() + 2
        audio_token_positions.append(
            torch.arange(
                audio_start,
                audio_start + audio_chunk.numel(),
                dtype=torch.long,
            )
        )
        cursor += text_ids.numel() + audio_ids.numel()

    response = torch.cat(pieces)
    response = _ar_append_eos(response, runtime)
    content_labels = torch.cat(label_pieces)
    labels = torch.full((marker.numel() + response.numel(),), -100, dtype=torch.long)
    labels[marker.numel() : marker.numel() + content_labels.numel()] = content_labels
    labels[-1] = runtime.eos_token_id
    positions = torch.cat(audio_token_positions)
    expanded = torch.repeat_interleave(positions, spans)
    acoustic = _acoustic_from_positions(speech, runtime, expanded)
    return ModelSample.pack(
        prompt_ids=marker,
        response_ids=response,
        token_labels=labels,
        acoustic_target=acoustic,
        task=task,
        trace=spec.name,
        target_language=_target_language(spec, speech),
        audio_seconds=_ar_duration(speech),
        audio_input_positions=None,
    )


def _split_proportional(values: Tensor, parts: int) -> list[Tensor]:
    if parts < 1:
        raise ValueError("interleaved layout requires at least one chunk.")
    total = int(values.numel())
    if total == 0:
        return [values.new_empty((0,), dtype=values.dtype) for _ in range(parts)]
    out: list[Tensor] = []
    for index in range(parts):
        start = (index * total) // parts
        end = ((index + 1) * total) // parts
        out.append(values[start:end])
    return out


def _pack(
    marker: Tensor,
    response: Tensor,
    *,
    task: Task,
    spec: ResponseSpec,
    target_language: str | None,
    supervise_from: int,
    acoustic_target: AcousticTarget | None,
    audio_seconds: float,
    target_ctc: CTCTarget | None = None,
) -> ModelSample:
    full = torch.cat([marker, response])
    labels = torch.full_like(full, -100)
    labels[marker.numel() + supervise_from :] = response[supervise_from:]
    return ModelSample.pack(
        prompt_ids=marker,
        response_ids=response,
        token_labels=labels,
        acoustic_target=acoustic_target,
        target_ctc=target_ctc,
        task=task,
        trace=spec.name,
        target_language=target_language,
        audio_seconds=audio_seconds,
        audio_input_positions=None,
    )


def _ar_acoustic_target(
    speech: Speech,
    runtime: DataRuntime,
    *,
    audio_token_start: int,
) -> AcousticTarget | None:
    positions = torch.arange(
        audio_token_start,
        audio_token_start + speech.audio_token_ids.numel(),
        dtype=torch.long,
    )
    expanded = torch.repeat_interleave(positions, speech.audio_token_spans)
    return _acoustic_from_positions(speech, runtime, expanded)


def _acoustic_from_positions(
    speech: Speech,
    runtime: DataRuntime,
    token_positions: Tensor,
) -> AcousticTarget | None:
    if speech.acoustic_codes is None or runtime.acoustic_generator_artifact is not None:
        return None
    if runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
        return None
    if token_positions.numel() != speech.acoustic_codes.size(0):
        raise ValueError("target acoustic frames and audio tokens must align.")
    return AcousticTarget(
        semantic_codes=speech.semantic_codes,
        codes=speech.acoustic_codes,
        token_positions=token_positions,
    )


def _ar_duration(speech: Speech) -> float:
    if speech.duration_seconds is None:
        raise ValueError(
            "speech is missing duration_seconds; parse raw audio samples with a "
            "DataRuntime so duration can be read from metadata or inferred from "
            "codec frames."
        )
    return float(speech.duration_seconds)


def _ar_boa_eoa(ids: Tensor, runtime: DataRuntime) -> Tensor:
    return torch.cat(
        (
            ids.new_tensor([runtime.boa_token_id]),
            ids.new_tensor([runtime.audio_schema_token_id]),
            ids,
            ids.new_tensor([runtime.eoa_token_id]),
        )
    )


def _ar_append_eos(ids: Tensor, runtime: DataRuntime | TextRuntime) -> Tensor:
    return torch.cat([ids, ids.new_tensor([runtime.eos_token_id])])


def build_masked_sample(
    speech: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    trace: str | None = None,
    interleave_audio_frames: int = 25,
    mask_text_ratio: float = 0.5,
    mask_audio_ratio: float = 0.5,
    generator: Generator | None = None,
) -> ModelSample:
    response = resolve_response(task, trace=trace)
    return _build_masked_sample(
        speech,
        task,
        runtime,
        prompt=prompt,
        spec=response,
        interleave_audio_frames=interleave_audio_frames,
        mask_text_ratio=mask_text_ratio,
        mask_audio_ratio=mask_audio_ratio,
        generator=generator,
    )


def _build_masked_sample(
    speech: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    spec: ResponseSpec,
    interleave_audio_frames: int,
    mask_text_ratio: float,
    mask_audio_ratio: float,
    generator: Generator | None = None,
) -> ModelSample:
    if task is not Task.MASKED_AR:
        raise ValueError(f"{task.value} is not a masked autoregressive task.")
    prediction = spec.prediction
    if not prediction.is_mixed:
        raise ValueError("MASKED_AR requires PARALLEL or INTERLEAVED prediction.")
    _validate_ratio(mask_text_ratio, name="mask_text_ratio")
    _validate_ratio(mask_audio_ratio, name="mask_audio_ratio")
    if not hasattr(runtime, "mask_token_id"):
        raise AttributeError("MASKED_AR requires runtime.mask_token_id.")

    marker = token_ids(prompt, runtime.text_tokenizer)
    masked_source = _masked_source(
        speech,
        runtime,
        mask_text_ratio=mask_text_ratio,
        mask_audio_ratio=mask_audio_ratio,
        generator=generator,
    )
    prefix = torch.cat([marker, masked_source])
    if prediction is PredictionModality.PARALLEL:
        return pack_parallel(
            prefix,
            speech,
            task,
            runtime,
            spec=spec,
        )
    return pack_interleaved(
        prefix,
        speech,
        task,
        runtime,
        spec=spec,
        interleave_audio_frames=interleave_audio_frames,
    )


def _masked_source(
    speech: Speech,
    runtime: DataRuntime,
    *,
    mask_text_ratio: float,
    mask_audio_ratio: float,
    generator: Generator | None,
) -> Tensor:
    mask_id = int(runtime.mask_token_id)
    text = runtime.layout.to_global(Modality.TEXT.value, speech.text_token_ids).clone()
    audio = runtime.layout.to_global(Modality.AUDIO.value, speech.audio_token_ids).clone()
    text[_mask_indices(text.numel(), mask_text_ratio, generator, device=text.device)] = (
        mask_id
    )
    audio[
        _mask_indices(audio.numel(), mask_audio_ratio, generator, device=audio.device)
    ] = mask_id
    return torch.cat(
        (
            text,
            audio.new_tensor([runtime.boa_token_id]),
            audio.new_tensor([runtime.audio_schema_token_id]),
            audio,
            audio.new_tensor([runtime.eoa_token_id]),
        )
    )


def _mask_indices(
    length: int,
    ratio: float,
    generator: Generator | None,
    *,
    device: torch.device,
) -> Tensor:
    if length == 0 or ratio <= 0.0:
        return torch.zeros(0, dtype=torch.long, device=device)
    count = min(length, max(1, int(round(length * ratio))))
    permutation = torch.randperm(length, generator=generator, device=device)
    return permutation[:count]


def _validate_ratio(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a float.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def ctc_target(
    positions: Tensor,
    speech: Speech,
    runtime: DataRuntime,
) -> CTCTarget:
    """Build local-text CTC supervision for one audio span."""
    labels = speech.text_token_ids
    if labels.numel() == 0:
        raise ValueError("CTC transcript must contain at least one text token.")
    text_start, _ = runtime.layout.blocks[Modality.TEXT.value]
    text_vocab_size = runtime.lexical_text_vocab_size
    if bool((labels < 0).any()) or bool((labels >= text_vocab_size).any()):
        raise ValueError(
            "CTC transcript contains an id outside the lexical text vocabulary."
        )
    blank = runtime.pad_token_id - text_start
    if not 0 <= blank < text_vocab_size:
        raise ValueError(
            "runtime pad token must belong to the lexical text vocabulary for CTC."
        )
    if bool(labels.eq(blank).any()):
        raise ValueError("CTC transcript must not contain the configured blank token.")
    return CTCTarget(
        token_positions=positions,
        text_token_ids=labels,
    )


def token_ids(text: str, tokenizer: TextTokenizer) -> Tensor:
    values = torch.as_tensor(
        tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
    )
    if values.dim() != 1:
        raise ValueError("text tokenizer must return a 1D token sequence.")
    return values


__all__ = [
    "build_ar_sample",
    "build_masked_sample",
    "build_pretraining_ar_sample",
    "build_sample",
    "build_speech_sample",
    "build_task_sample",
    "build_text_sample",
    "chat_prompt",
    "is_ar_task",
    "pack_interleaved",
    "pack_parallel",
]
