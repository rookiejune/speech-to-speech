from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anydataset import AnyDataset
from anydataset import AudioItem, AudioView, Modality, Role
from anydataset.store import DatasetWriter
from anytrain.idspace import (
    IdSpace,
    IdSpaceEmbedding,
    Modality as IdModality,
    ModalityBlock,
)
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

TensorLike = torch.Tensor | Sequence[int]
ToyPair = tuple[TensorLike, TensorLike]


class MockTokenizer:
    def __init__(self) -> None:
        self.vocab_size = 16
        self.ids = {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "user": 3,
            "assistant": 4,
            "\n": 5,
            "<think>": 6,
            "</think>": 7,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids.get(token, -1)

    def __len__(self) -> int:
        return 16

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "<<<SPEECH_TO_SPEECH_SOURCE_AUDIO>>>":
            return [15]
        return [8, 9]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool = False,
    ) -> list[int]:
        del tokenize, enable_thinking, return_dict
        ids = [1, 10, 11]
        content = messages[0]["content"]
        if "<<<SPEECH_TO_SPEECH_SOURCE_AUDIO>>>" in content:
            ids.append(15)
        if add_generation_prompt:
            ids.extend([12, 13])
        return ids


class MockQwen(nn.Module):
    def __init__(self, *, vocab_size: int = 16, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        **kwargs: object,
    ):
        del attention_mask
        hidden = self.proj(inputs_embeds)
        output_hidden_states = kwargs.get("output_hidden_states")
        hidden_states = (inputs_embeds, hidden) if output_hidden_states else None
        return BaseModelOutputWithPast(last_hidden_state=hidden, hidden_states=hidden_states)


class MockFrameBPE:
    def __init__(
        self,
        *,
        expanded: Sequence[int] | None = None,
        vocab_size: int = 16,
    ) -> None:
        self.expanded = expanded
        self.vocab_size = vocab_size

    def encode_frames(self, frames: list[list[int]]) -> list[int]:
        return [int(frame[0]) for frame in frames]

    def expand_ids(self, ids: list[int]) -> list[tuple[int]]:
        values = self.expanded if self.expanded is not None else ids
        spans = []
        for value in values:
            if isinstance(value, Sequence) and not isinstance(value, int | str | bytes):
                spans.append(tuple(int(item) for item in value))
            else:
                spans.append((int(value),))
        return spans


@contextmanager
def isolated_anydataset_home(root: Path) -> Iterator[None]:
    with patch.dict(os.environ, {"ANYDATASET_HOME": str(root / "anydataset")}):
        yield


def toy_idspace(*, audio_vocab_size: int = 5) -> IdSpace:
    return IdSpace(
        {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "user": 3,
            "assistant": 4,
            "\n": 5,
            "<think>": 6,
            "</think>": 7,
            "boa": 16,
            "eoa": 17,
        },
        [
            ModalityBlock(IdModality.TEXT, 0, 16),
            ModalityBlock(IdModality.AUDIO, 18, audio_vocab_size),
        ],
    )


def toy_embedding(*, audio_vocab_size: int = 5, hidden_size: int = 4) -> IdSpaceEmbedding:
    space = toy_idspace(audio_vocab_size=audio_vocab_size)
    return IdSpaceEmbedding(space, hidden_size)


def toy_longcat_sample(
    source: TensorLike,
    target: TensorLike,
    *,
    include_acoustic: bool = True,
):
    source_view = toy_longcat_view(source, include_acoustic=include_acoustic)
    target_view = toy_longcat_view(target, include_acoustic=include_acoustic)
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: source_view}
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: target_view}
        ),
    }


def toy_longcat_view(
    semantic: TensorLike,
    *,
    include_acoustic: bool = True,
) -> dict[str, torch.Tensor]:
    semantic = _tensor(semantic)
    view = {"semantic_codes": semantic}
    if include_acoustic:
        view["acoustic_codes"] = torch.zeros((4, _time_length(semantic)), dtype=torch.long)
    return view


def write_toy_longcat_store(
    path: Path,
    *,
    pairs: Sequence[ToyPair] | None = None,
) -> Path:
    pairs = pairs or (
        (torch.tensor([0, 1, 2]), torch.tensor([2, 3])),
        (torch.tensor([3]), torch.tensor([4, 0, 1])),
    )
    DatasetWriter(path, dataset_id="toy-s2s", split="train").write(
        [toy_longcat_sample(source, target) for source, target in pairs]
    )
    return path


@contextmanager
def patched_wmt19_longcat(store: Path) -> Iterator[None]:
    zhuyin = types.ModuleType("zhuyin")
    datasets = types.ModuleType("zhuyin.datasets")
    wmt19_tts = types.ModuleType("zhuyin.datasets.wmt19_tts")
    wmt19_tts.wmt19_tts_longcat = lambda: AnyDataset(f"store://{store}:train")
    with patch.dict(
        sys.modules,
        {
            "zhuyin": zhuyin,
            "zhuyin.datasets": datasets,
            "zhuyin.datasets.wmt19_tts": wmt19_tts,
        },
    ):
        yield


def _tensor(value: TensorLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def _time_length(value: torch.Tensor) -> int:
    if value.dim() == 0:
        return 1
    return int(value.shape[-1])
