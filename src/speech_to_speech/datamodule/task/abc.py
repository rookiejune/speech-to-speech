from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import ClassVar

import torch
from torch import Tensor

from ...runtime import runtime
from ..types import Sample, Speech, SpeechPair, Task


class TaskBase:
    """Task-specific view over a sampled speech pair.

    Semantic sample construction is cached because LBA length planning and the
    final DataLoader collate both need the same encoded causal row. Acoustic
    samples stay uncached because they only wrap existing speech tensors.
    """

    source: ClassVar[str | None]
    target: ClassVar[str]
    template: ClassVar[str]
    paired: ClassVar[bool]

    name: ClassVar[Task]
    _placeholder: ClassVar[str] = "$$$PLACEHOLDER$$$"

    def __init__(self) -> None:
        raise RuntimeError

    @classmethod
    def instruction(cls, sample: SpeechPair) -> str:
        fields = {"language": str(sample.target.language), "source": cls._placeholder}
        return cls.template.format(**fields)

    @classmethod
    def _prompt_ids(cls, sample: SpeechPair):
        return torch.tensor(
            runtime().text_tokenizer.apply_chat_template(
                [{"role": "user", "content": cls.instruction(sample)}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            ),
        )

    @classmethod
    def _parse_speech_pair(cls, sample: SpeechPair):
        if cls.paired:
            return sample.source, sample.target
        else:
            return sample.target, sample.target

    @classmethod
    def sample(cls, speech_pair: SpeechPair) -> Sample:
        _prompt_ids = cls._prompt_ids(speech_pair)

        source, target = cls._parse_speech_pair(speech_pair)

        source_acoustic_ids = None
        source_acoustic_positions = None
        if cls.source is not None:
            prefix, suffix = _split(_prompt_ids, cls.placeholder_ids())
            source_ids = _global_ids(source, cls.source)

            if cls.source == "audio":
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
            input_ids = _prompt_ids

        response_ids = _global_ids(target, cls.target)
        target_acoustic_labels = None
        target_acoustic_positions = None

        if cls.target == "audio":
            response_ids = _boa_eoa(response_ids)  # <boa> ... <eoa>
            target_acoustic_labels = target.acoustic_ids
        else:
            response_ids = _append_eos(response_ids)  # ... <eos>

        full_ids = torch.cat([input_ids, response_ids])
        labels = torch.full_like(full_ids, -100)
        if cls.target == "audio":
            # BOA is already present in input_ids and is a structural prefix,
            # so only semantic BPE tokens and EOA are supervised.
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
            acoustic_labels=target_acoustic_labels,
            acoustic_label_positions=target_acoustic_positions,
            task=cls.name,
        )

    @classmethod
    @cache
    def placeholder_ids(cls):
        return torch.tensor(runtime().text_tokenizer.encode(cls._placeholder))


class TaskFactory:
    _registry: ClassVar[dict[Task, type[TaskBase]]] = {}

    @classmethod
    def register(cls, task: Task) -> Callable[[type[TaskBase]], type[TaskBase]]:
        def decorator(task_cls: type[TaskBase]) -> type[TaskBase]:
            if task in cls._registry:
                raise ValueError(f"duplicate task registration: {task.value}")
            cls._registry[task] = task_cls
            task_cls.name = task
            return task_cls

        return decorator

    @classmethod
    def get(
        cls,
        task: Task,
    ):
        task_cls = cls._registry.get(task)
        if task_cls is None:
            registered = ", ".join(item.value for item in cls._registry)
            raise KeyError(
                f"unknown task: {task.value}. Registered tasks: {registered}."
            )
        return task_cls


def _split(
    sequence: Tensor,
    delimiter: Tensor,
) -> tuple[Tensor, Tensor]:
    limit = len(sequence) - len(delimiter) + 1
    for start in range(limit):
        if (sequence[start : start + len(delimiter)] == delimiter).all():
            return (
                sequence[:start],
                sequence[start + len(delimiter) :],
            )
    raise ValueError("input placeholder was not found in chat template ids.")


def _global_ids(speech: Speech, modality: str):
    if modality == "text":
        local_ids = speech.text_ids
    elif modality == "audio":
        local_ids = speech.bpe_ids
    else:
        raise ValueError
    return runtime().layout.to_global(modality, local_ids)


def _boa_eoa(ids: Tensor):
    special_ids = torch.tensor([runtime().eoa_token_id, runtime().boa_token_id])
    return torch.cat([ids, special_ids]).roll(1)


def _append_eos(ids: Tensor):
    special_ids = torch.tensor(
        [
            runtime().eos_token_id,
        ]
    )
    return torch.cat([ids, special_ids])
