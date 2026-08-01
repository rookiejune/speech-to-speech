from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from typing import Protocol, cast

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.cache_utils import Cache

from ..types import Backbone, BackboneOutput, TextTokenizer
from .adapter import BackboneBodyAdapter, BackboneExtra, BackboneOutputView
from .config import AdapterConfig, BackboneInitialization, BackboneType


@dataclass(frozen=True)
class HuggingFaceBackboneAdapter:
    config: AdapterConfig

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        if self.config.type is BackboneType.QWEN2_5_OMNI_THINKER:
            processor = AutoProcessor.from_pretrained(
                self.config.path,
                trust_remote_code=self.config.trust_remote_code,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise TypeError("Qwen2.5-Omni processor must expose a text tokenizer.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.path,
                trust_remote_code=self.config.trust_remote_code,
            )
        bind_chat_bos(tokenizer)
        return cast(TextTokenizer, cast(object, tokenizer))

    @cached_property
    def root_model(self) -> nn.Module:
        kwargs = {}
        if self.config.dtype is not None:
            kwargs["dtype"] = dtype(self.config.dtype)
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation
        model = (
            self._pretrained(**kwargs)
            if self.config.initialization is BackboneInitialization.PRETRAINED
            else self._random(**kwargs)
        )
        if self.config.device is not None:
            model = cast(nn.Module, cast(object, model)).to(self.config.device)
        return cast(nn.Module, cast(object, model))

    @cached_property
    def model(self) -> Backbone:
        return cast(Backbone, cast(object, _path(self.root_model, self.config.module)))

    @cached_property
    def hidden_size(self) -> int:
        return _hidden_size(getattr(self.model, "config", None))

    @cached_property
    def body(self) -> BackboneBodyAdapter:
        target = _path(cast(object, self.model), self.config.body)
        if not callable(target):
            raise TypeError(f"backbone body path {self.config.body!r} is not callable.")
        return BackboneBodyAdapter(
            cast(Callable[..., BackboneOutput], target),
            readout=self.config.readout,
            supports_cache_position=self.config.supports_cache_position,
        )

    def input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

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
        return self.body.encode(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            extra=extra,
        )

    def _pretrained(self, **kwargs: object) -> object:
        if self.config.type is BackboneType.QWEN2_5_OMNI_THINKER:
            return _omni_model_factory().from_pretrained(
                self.config.path,
                trust_remote_code=self.config.trust_remote_code,
                **kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
            **kwargs,
        )

    def _random(self, **kwargs: object) -> object:
        config = AutoConfig.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.config.type is BackboneType.QWEN2_5_OMNI_THINKER:
            return _omni_model_factory()._from_config(config, **kwargs)
        return AutoModelForCausalLM.from_config(config, **kwargs)


def create(config: AdapterConfig) -> HuggingFaceBackboneAdapter:
    return HuggingFaceBackboneAdapter(config)


class _OmniModelFactory(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> object: ...

    def _from_config(self, config: object, **kwargs: object) -> object: ...


def _omni_model_factory() -> _OmniModelFactory:
    try:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Qwen2.5-Omni backbone requires a transformers build that exposes "
            "Qwen2_5OmniThinkerForConditionalGeneration."
        ) from error
    return cast(_OmniModelFactory, Qwen2_5OmniThinkerForConditionalGeneration)


def _path(root: object, path: str) -> object:
    current = root
    for item in (part for part in path.split(".") if part):
        if not hasattr(current, item):
            raise AttributeError(f"backbone object has no attribute {item!r}.")
        current = getattr(current, item)
    return current


def _hidden_size(config: object) -> int:
    for item in (
        config,
        getattr(config, "text_config", None),
        getattr(getattr(config, "thinker_config", None), "text_config", None),
    ):
        value = getattr(item, "hidden_size", None)
        if isinstance(value, bool) or not isinstance(value, Integral):
            continue
        size = int(value)
        if size <= 0:
            raise ValueError("backbone hidden size must be positive.")
        return size
    raise AttributeError("backbone config does not expose a text hidden size.")


# Qwen chat turn start; stock HF leaves bos_token unset while keeping this in vocab.
_CHAT_BOS_TOKEN = "<|im_start|>"


def bind_chat_bos(tokenizer: object) -> None:
    """Expose chat turn-start as bos when the tokenizer leaves bos unset."""
    if getattr(tokenizer, "bos_token_id", None) is not None:
        return
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return
    try:
        token_id = convert(_CHAT_BOS_TOKEN)
    except (AttributeError, NotImplementedError):
        return
    if isinstance(token_id, bool) or not isinstance(token_id, Integral):
        return
    token_id = int(token_id)
    if token_id < 0:
        return
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and token_id == int(unk):
        return
    setattr(tokenizer, "bos_token", _CHAT_BOS_TOKEN)


def dtype(value: str) -> torch.dtype:
    try:
        result = getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {value}") from error
    if not isinstance(result, torch.dtype):
        raise ValueError(f"unknown torch dtype: {value}")
    return result


__all__ = [
    "HuggingFaceBackboneAdapter",
    "bind_chat_bos",
    "create",
    "dtype",
]
