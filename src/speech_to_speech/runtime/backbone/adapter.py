from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

import torch
from anydataset.types import Modality
from torch import nn
from transformers.cache_utils import Cache

from ..tokenizer import TextTokenizer
from ..backbone.contract import (
    Backbone,
    BackboneOutput,
    BackboneReadout,
)


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

    def contract_state(self) -> Mapping[str, object]: ...

    @property
    def has_modality_readouts(self) -> bool: ...

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
        modality: Modality | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView: ...


class BackboneEncoder(Protocol):
    @property
    def has_modality_readouts(self) -> bool: ...

    def contract_state(self) -> Mapping[str, object]: ...

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
        modality: Modality | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView: ...


@dataclass(frozen=True)
class BackboneBodyAdapter:
    body: Callable[..., BackboneOutput]
    readout: BackboneReadout = field(default_factory=BackboneReadout)
    supports_cache_position: bool = True
    modality_readouts: Mapping[Modality, BackboneReadout] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.readout, BackboneReadout):
            raise TypeError("default backbone readout must be a BackboneReadout.")
        if not isinstance(self.supports_cache_position, bool):
            raise TypeError("supports_cache_position must be a bool.")
        for modality, readout in self.modality_readouts.items():
            if modality not in {Modality.TEXT, Modality.AUDIO}:
                raise ValueError(
                    f"unsupported modality-specific backbone readout: {modality.value}."
                )
            if not isinstance(readout, BackboneReadout):
                raise TypeError(
                    "modality-specific backbone readouts must be BackboneReadout values."
                )

    @property
    def has_modality_readouts(self) -> bool:
        return bool(self.modality_readouts)

    def readout_for(self, modality: Modality | None) -> BackboneReadout:
        if modality is None:
            return self.readout
        return self.modality_readouts.get(modality, self.readout)

    def contract_state(self) -> Mapping[str, object]:
        return {
            "grammar": "backbone-body-v1",
            "readout": self.readout.path,
            "readouts": {
                modality.value: readout.path
                for modality, readout in sorted(
                    self.modality_readouts.items(),
                    key=lambda item: item[0].value,
                )
            },
            "supports_cache_position": self.supports_cache_position,
        }

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
        modality: Modality | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView:
        readout = self.readout_for(modality)
        kwargs: dict[str, object] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden_states
            or readout.requires_hidden_states,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
        }
        if self.supports_cache_position:
            kwargs["cache_position"] = cache_position
        if extra:
            kwargs.update(extra)
        kwargs["return_dict"] = True
        output = self.body(**kwargs)
        return BackboneOutputView(
            output=output,
            last_hidden_state=readout.select(output),
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
        modality: Modality | None = None,
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
            modality=modality,
            extra=extra,
        )


__all__ = [
    "BackboneAdapter",
    "BackboneBodyAdapter",
    "BackboneEncoder",
    "BackboneExtra",
    "BackboneOutputView",
]
