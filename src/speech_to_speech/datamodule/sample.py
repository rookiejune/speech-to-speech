from __future__ import annotations

from typing import cast

import torch
from anydataset.types import Modality
from torch import Tensor

from ..task import Task
from ._tokenization import token_ids
from .protocol import DataRuntime, TextRuntime
from .types import (
    AcousticPrompt,
    AcousticTarget,
    Language,
    ModelSample,
    Speech,
    SpeechPair,
    Text,
    TextPair,
)

_PLACEHOLDER = "$$$PLACEHOLDER$$$"


def build_sample(
    speech_pair: SpeechPair,
    task: Task,
    runtime: DataRuntime,
) -> ModelSample:
    prompt = _prompt(speech_pair, task, runtime)
    source, target = _source_target(speech_pair, task)

    source_acoustic_codes = None
    source_audio_token_positions = None
    source_modality = task.source_modality
    target_modality = task.target_modality
    if source_modality is not None:
        prefix_text, suffix_text = _split(prompt, _PLACEHOLDER)
        tokenizer = runtime.text_tokenizer
        prefix = token_ids(prefix_text, tokenizer)
        suffix = token_ids(suffix_text, tokenizer)
        source_ids = _global_ids(source, source_modality, runtime)

        if source_modality is Modality.AUDIO:
            source_ids = _boa_eoa(source_ids, runtime)
            source_acoustic_codes = source.acoustic_codes
            if source_acoustic_codes is not None:
                source_audio_token_positions = torch.repeat_interleave(
                    torch.arange(
                        len(prefix) + 1,
                        len(prefix) + 1 + source.audio_token_ids.numel(),
                        dtype=torch.long,
                    ),
                    source.audio_token_spans,
                )

        input_ids = torch.cat([prefix, source_ids, suffix])
    else:
        input_ids = token_ids(prompt, runtime.text_tokenizer)

    response_ids = _global_ids(target, target_modality, runtime)
    target_acoustic_codes = None
    target_semantic_codes = None
    target_audio_token_positions = None

    if target_modality is Modality.AUDIO:
        response_ids = _boa_eoa(response_ids, runtime)
        if target.acoustic_codes is not None:
            target_semantic_codes = target.semantic_codes
            target_acoustic_codes = target.acoustic_codes
    else:
        response_ids = _append_eos(response_ids, runtime)

    full_ids = torch.cat([input_ids, response_ids])
    token_labels = torch.full_like(full_ids, -100)
    if target_modality is Modality.AUDIO:
        # BOA is a structural response prefix; supervise semantic BPE tokens and EOA.
        token_labels[len(input_ids) + 1 :] = response_ids[1:]
    else:
        token_labels[len(input_ids) :] = response_ids

    if target_acoustic_codes is not None:
        target_audio_token_positions = torch.repeat_interleave(
            torch.arange(
                len(input_ids) + 1,
                len(input_ids) + 1 + target.audio_token_ids.numel(),
                dtype=torch.long,
            ),
            target.audio_token_spans,
        )
        if target_audio_token_positions.numel() != target_acoustic_codes.size(0):
            raise ValueError("target acoustic frames and audio tokens must align.")

    acoustic_prompt = (
        None
        if source_acoustic_codes is None or source_audio_token_positions is None
        else AcousticPrompt(
            codes=source_acoustic_codes,
            token_positions=source_audio_token_positions,
        )
    )
    acoustic_target = (
        None
        if target_acoustic_codes is None or target_audio_token_positions is None
        else AcousticTarget(
            semantic_codes=cast(Tensor, target_semantic_codes),
            codes=target_acoustic_codes,
            token_positions=target_audio_token_positions,
        )
    )
    return ModelSample(
        input_ids=full_ids,
        token_labels=token_labels,
        acoustic_prompt=acoustic_prompt,
        acoustic_target=acoustic_target,
        task=task,
    )


def build_text_sample(
    text_pair: TextPair,
    task: Task,
    runtime: TextRuntime,
) -> ModelSample:
    if (
        task.source_modality is Modality.AUDIO
        or task.target_modality is not Modality.TEXT
    ):
        raise ValueError(f"{task.value} is not supported by the text-only data path.")

    prompt = _text_prompt(text_pair.target.language, task, runtime)
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
    return ModelSample(
        input_ids=full_ids,
        token_labels=token_labels,
        acoustic_prompt=None,
        acoustic_target=None,
        task=task,
    )


def _prompt(
    speech_pair: SpeechPair,
    task: Task,
    runtime: DataRuntime,
) -> str:
    instruction = task.template.format(
        language=str(speech_pair.target.language),
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
) -> str:
    instruction = task.template.format(
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
    speech: Speech,
    modality: Modality,
    runtime: DataRuntime,
) -> Tensor:
    if modality is Modality.TEXT:
        local_ids = speech.text_token_ids
    elif modality is Modality.AUDIO:
        local_ids = speech.audio_token_ids
    else:
        raise ValueError(f"unsupported modality: {modality.value}")
    return runtime.layout.to_global(modality.value, local_ids)


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
