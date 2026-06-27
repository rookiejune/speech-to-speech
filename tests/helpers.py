from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast


class MockTokenizer:
    def __init__(self) -> None:
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
    ) -> list[int]:
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
        del kwargs
        return BaseModelOutputWithPast(last_hidden_state=self.proj(inputs_embeds))
