from __future__ import annotations

from typing import cast

import torch
from anydataset.types import Modality
from torch import Tensor

from ..runtime import runtime
from ._tokenization import text_ids
from .types import Sample, Speech, SpeechPair, Task

_PLACEHOLDER = "$$$PLACEHOLDER$$$"


def build_sample(speech_pair: SpeechPair, task: Task) -> Sample:
    prompt = _prompt(speech_pair, task)
    source, target = _source_target(speech_pair, task)

    source_acoustic_ids = None
    source_acoustic_positions = None
    source_modality = task.source_modality
    target_modality = task.target_modality
    if source_modality is not None:
        prefix_text, suffix_text = _split(prompt, _PLACEHOLDER)
        tokenizer = runtime().text_tokenizer
        prefix = text_ids(prefix_text, tokenizer)
        suffix = text_ids(suffix_text, tokenizer)
        source_ids = _global_ids(source, source_modality)

        if source_modality is Modality.AUDIO:
            source_ids = _boa_eoa(source_ids)
            source_acoustic_ids = source.acoustic_ids
            if source_acoustic_ids is not None:
                source_acoustic_positions = torch.repeat_interleave(
                    torch.arange(
                        len(prefix) + 1,
                        len(prefix) + 1 + source.bpe_ids.numel(),
                        dtype=torch.long,
                    ),
                    source.bpe_spans,
                )

        input_ids = torch.cat([prefix, source_ids, suffix])
    else:
        input_ids = text_ids(prompt, runtime().text_tokenizer)

    response_ids = _global_ids(target, target_modality)
    target_acoustic_labels = None
    target_semantic_frame_labels = None
    target_acoustic_positions = None

    if target_modality is Modality.AUDIO:
        response_ids = _boa_eoa(response_ids)
        if target.acoustic_ids is not None:
            target_semantic_frame_labels = target.semantic_ids
            target_acoustic_labels = target.acoustic_ids
    else:
        response_ids = _append_eos(response_ids)

    full_ids = torch.cat([input_ids, response_ids])
    labels = torch.full_like(full_ids, -100)
    if target_modality is Modality.AUDIO:
        # BOA is a structural response prefix; supervise semantic BPE tokens and EOA.
        labels[len(input_ids) + 1 :] = response_ids[1:]
    else:
        labels[len(input_ids) :] = response_ids

    if target_acoustic_labels is not None:
        target_acoustic_positions = torch.repeat_interleave(
            torch.arange(
                len(input_ids) + 1,
                len(input_ids) + 1 + target.bpe_ids.numel(),
                dtype=torch.long,
            ),
            target.bpe_spans,
        )
        if target_acoustic_positions.numel() != target_acoustic_labels.size(0):
            raise ValueError("target acoustic frames and BPE spans must align.")

    return Sample(
        input_ids=full_ids,
        labels=labels,
        acoustic_input_ids=source_acoustic_ids,
        acoustic_input_positions=source_acoustic_positions,
        semantic_frame_labels=target_semantic_frame_labels,
        acoustic_labels=target_acoustic_labels,
        acoustic_label_positions=target_acoustic_positions,
        task=task,
    )


def _prompt(speech_pair: SpeechPair, task: Task) -> str:
    instruction = task.template.format(
        language=str(speech_pair.target.language),
        source=_PLACEHOLDER,
    )
    return cast(
        str,
        runtime().text_tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        ),
    )


def _source_target(speech_pair: SpeechPair, task: Task) -> tuple[Speech, Speech]:
    if task.paired:
        return speech_pair.source, speech_pair.target
    return speech_pair.target, speech_pair.target


def _split(sequence: str, delimiter: str) -> tuple[str, str]:
    parts = sequence.split(delimiter)
    if len(parts) != 2:
        raise ValueError("input placeholder must occur exactly once in chat template.")
    return parts[0], parts[1]


def _global_ids(speech: Speech, modality: Modality) -> Tensor:
    if modality is Modality.TEXT:
        local_ids = speech.text_ids
    elif modality is Modality.AUDIO:
        local_ids = speech.bpe_ids
    else:
        raise ValueError(f"unsupported modality: {modality.value}")
    return runtime().layout.to_global(modality.value, local_ids)


def _boa_eoa(ids: Tensor) -> Tensor:
    return torch.cat(
        (
            ids.new_tensor([runtime().boa_token_id]),
            ids,
            ids.new_tensor([runtime().eoa_token_id]),
        )
    )


def _append_eos(ids: Tensor) -> Tensor:
    return torch.cat([ids, torch.tensor([runtime().eos_token_id])])
