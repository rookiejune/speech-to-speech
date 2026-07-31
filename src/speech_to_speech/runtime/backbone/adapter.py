from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import torch
from torch import nn
from transformers.cache_utils import Cache

from ..types import Backbone, BackboneOutput, TextTokenizer, select_backbone_readout


BackboneExtra = Mapping[str, object]


@dataclass(frozen=True)
class BackboneOutputView:
    output: BackboneOutput
    last_hidden_state: torch.Tensor

    @property
    def past_key_values(self) -> Cache | None:
        return self.output.past_key_values

    @property
    def hidden_states(self) -> tuple[torch.Tensor, ...] | None:
        return self.output.hidden_states

    @property
    def attentions(self) -> tuple[torch.Tensor, ...] | None:
        return self.output.attentions


class BackboneAdapter(Protocol):
    @cached_property
    def model(self) -> Backbone: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def hidden_size(self) -> int: ...

    def input_embeddings(self) -> nn.Embedding: ...

    def encode(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView: ...


class BackboneEncoder(Protocol):
    def encode(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView: ...


@dataclass(frozen=True)
class BackboneBodyAdapter:
    body: Callable[..., BackboneOutput]
    readout: str
    supports_cache_position: bool

    def encode(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView:
        kwargs: dict[str, object] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden_states
            or self.readout != "last_hidden_state",
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
        }
        if self.supports_cache_position:
            kwargs["cache_position"] = cache_position
        if extra:
            kwargs.update(extra)
        output = self.body(**kwargs)
        return BackboneOutputView(
            output=output,
            last_hidden_state=select_backbone_readout(output, self.readout),
        )

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView:
        return self.encode(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            extra=extra,
        )


__all__ = [
    "BackboneAdapter",
    "BackboneBodyAdapter",
    "BackboneEncoder",
    "BackboneExtra",
    "BackboneOutputView",
]
