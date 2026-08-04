from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from torch import Tensor, nn
from transformers.cache_utils import Cache


class TextTokenizer(Protocol):
    special_tokens_map: Mapping[str, str | Sequence[str]]
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None

    def __len__(self) -> int: ...

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        enable_thinking: bool = ...,
        return_dict: bool = ...,
    ) -> str | list[int]: ...


class BackboneConfig(Protocol):
    hidden_size: int


class BackboneOutput(Protocol):
    last_hidden_state: Tensor | Sequence[Tensor]
    past_key_values: Cache | None
    hidden_states: tuple[Tensor, ...] | None
    attentions: tuple[Tensor, ...] | None


@dataclass(frozen=True)
class BackboneReadout:
    """A validated output attribute with an optional sequence index.

    HuggingFace output objects normally expose a tensor as ``last_hidden_state``;
    multimodal backbones may expose a tuple under the same attribute.  Keeping
    the parsed path as a value object lets adapters decide whether a layer
    history is needed without treating a raw configuration string as runtime
    state.
    """

    path: str = "last_hidden_state"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError("backbone_readout must be a string.")
        _backbone_readout_path(self.path)

    @property
    def attribute(self) -> str:
        return _backbone_readout_path(self.path)[0]

    @property
    def index(self) -> int | None:
        return _backbone_readout_path(self.path)[1]

    @property
    def requires_hidden_states(self) -> bool:
        """Whether selecting this output requires the full layer history."""
        return self.attribute == "hidden_states"

    def select(self, output: BackboneOutput) -> Tensor:
        attribute = self.attribute
        index = self.index
        if not hasattr(output, attribute):
            raise ValueError(
                f"backbone output is missing readout attribute {attribute!r}."
            )
        value = getattr(output, attribute)
        if index is not None:
            if not isinstance(value, Sequence):
                raise TypeError(
                    f"backbone readout index [{index}] requires a sequence value."
                )
            if index >= len(value):
                raise ValueError(
                    f"backbone readout index [{index}] is out of range."
                )
            value = value[index]
        if not isinstance(value, Tensor):
            raise TypeError("backbone readout must resolve to a Tensor.")
        return value


def validate_backbone_readout(path: object) -> str:
    return BackboneReadout(cast(str, path)).path


def _backbone_readout_path(path: object) -> tuple[str, int | None]:
    if not isinstance(path, str):
        raise TypeError("backbone_readout must be a string.")
    if not path:
        raise ValueError("backbone_readout must not be empty.")
    if "[" not in path:
        if "]" in path:
            raise ValueError("backbone_readout index is missing opening '['.")
        attribute = path
        index = None
    else:
        start = path.find("[")
        if not path.endswith("]") or path.find("]", start + 1) != len(path) - 1:
            raise ValueError("backbone_readout index must end the path.")
        if path.find("[", start + 1) != -1:
            raise ValueError("backbone_readout accepts at most one index.")
        attribute = path[:start]
        raw = path[start + 1 : -1]
        if not raw.isdecimal():
            raise ValueError("backbone_readout indices must be non-negative integers.")
        index = int(raw)
    if not attribute.isidentifier():
        raise ValueError("backbone_readout must start with an identifier attribute.")
    return attribute, index


class BackboneBody(Protocol):
    def __call__(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None,
        use_cache: bool,
        position_ids: Tensor | None,
        cache_position: Tensor | None,
    ) -> BackboneOutput: ...


class Backbone(Protocol):
    @property
    def config(self) -> BackboneConfig: ...

    def get_input_embeddings(self) -> nn.Embedding: ...

    @property
    def base_model(self) -> BackboneBody: ...

__all__ = [
    "Backbone",
    "BackboneBody",
    "BackboneConfig",
    "BackboneOutput",
    "BackboneReadout",
    "TextTokenizer",
    "validate_backbone_readout",
]
