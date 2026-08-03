"""Small, explicit adapters for the remote Kimi-Audio implementation.

Kimi-Audio exposes a custom tokenizer and two causal-language-model output
branches.  This module keeps those remote objects intact while providing the
contracts used by the local runtime.  The implementation targets the
``moonshotai/Kimi-Audio-7B-Instruct`` remote-code snapshot and deliberately
does not manufacture an official chat template when the loaded tokenizer does
not provide one.  A caller may explicitly supply a project-specific Jinja
template for this repository's single-stream prompt protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from numbers import Integral
from typing import Literal, Protocol, TypeVar, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers.utils.chat_template_utils import render_jinja_template


class KimiRawTokenizer(Protocol):
    """The subset of ``TikTokenTokenizer`` consumed by the adapter."""

    @property
    def special_tokens(self) -> Mapping[str, int]: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    def encode(
        self,
        text: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | Collection[str] = ...,
        disallowed_special: Collection[str] = ...,
    ) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class KimiTokenizerAdapter:
    """Adapt Kimi's keyword-only tokenizer API to ``TextTokenizer``.

    The remote tokenizer is kept as a composed object.  In particular, this
    class does not monkey-patch its special-token attributes, which makes it
    safe to use alongside remote code and easy to replace in tests.  An
    explicitly configured chat template is treated as a project protocol, not
    as Kimi's official dual-stream prompt format.
    """

    def __init__(
        self,
        raw: KimiRawTokenizer,
        *,
        chat_template: str | None = None,
    ) -> None:
        if not callable(getattr(raw, "encode", None)):
            raise TypeError("Kimi tokenizer must expose encode().")
        if not callable(getattr(raw, "decode", None)):
            raise TypeError("Kimi tokenizer must expose decode().")

        self._raw = raw
        if chat_template is not None and not isinstance(chat_template, str):
            raise TypeError("Kimi chat_template must be a string or None.")
        if chat_template == "":
            raise ValueError("Kimi chat_template must not be empty.")
        self._configured_chat_template = chat_template
        self._vocab_size = _vocab_size(getattr(raw, "vocab_size", None))
        raw_special_tokens = _special_tokens(getattr(raw, "special_tokens", None))
        self._special_tokens = raw_special_tokens

        canonical_ids = {
            "bos_token": _token_id(getattr(raw, "bos_id", None), "bos_id"),
            "eos_token": _token_id(getattr(raw, "eos_id", None), "eos_id"),
            "pad_token": _token_id(getattr(raw, "pad_id", None), "pad_id"),
        }
        canonical_tokens = {
            name: _token_for_id(raw_special_tokens, token_id, name)
            for name, token_id in canonical_ids.items()
        }
        additional = tuple(
            token
            for token in raw_special_tokens
            if token not in canonical_tokens.values()
        )
        special_tokens_map: dict[str, str | Sequence[str]] = dict(canonical_tokens)
        if additional:
            special_tokens_map["additional_special_tokens"] = additional
        self._special_tokens_map = special_tokens_map

    @property
    def raw(self) -> KimiRawTokenizer:
        """Return the composed remote tokenizer without modifying it."""

        return self._raw

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __len__(self) -> int:
        return self._vocab_size

    @property
    def bos_token_id(self) -> int:
        return _token_id(self._raw.bos_id, "bos_token_id")

    @property
    def eos_token_id(self) -> int:
        return _token_id(self._raw.eos_id, "eos_token_id")

    @property
    def pad_token_id(self) -> int:
        return _token_id(self._raw.pad_id, "pad_token_id")

    @property
    def special_tokens_map(self) -> Mapping[str, str | Sequence[str]]:
        # Return a copy so callers cannot change the adapter's special-token
        # contract or accidentally mutate the remote tokenizer's mapping.
        return dict(self._special_tokens_map)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Kimi tokenizer text must be a string.")
        if not isinstance(add_special_tokens, bool):
            raise TypeError("add_special_tokens must be a boolean.")
        values = self._raw.encode(
            text,
            bos=add_special_tokens,
            eos=add_special_tokens,
            allowed_special="all",
            disallowed_special=(),
        )
        return _token_ids(values, self._vocab_size, "Kimi tokenizer encode() output")

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if not isinstance(skip_special_tokens, bool):
            raise TypeError("skip_special_tokens must be a boolean.")
        values = _token_ids(token_ids, self._vocab_size, "Kimi tokenizer decode() input")
        if skip_special_tokens:
            special_ids = set(self._special_tokens.values())
            values = [value for value in values if value not in special_ids]
        decoded = self._raw.decode(values)
        if not isinstance(decoded, str):
            raise TypeError("Kimi tokenizer decode() must return a string.")
        return decoded

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> int | list[int]:
        """Convert token strings while preserving the remote tokenizer's behavior."""
        if isinstance(tokens, str):
            if tokens in self._special_tokens:
                return self._special_tokens[tokens]
            converted = _convert_tokens(self._raw, tokens)
            return _token_id(converted, "converted token id", vocab_size=self._vocab_size)
        if not isinstance(tokens, Sequence):
            raise TypeError("tokens must be a string or a sequence of strings.")
        if any(not isinstance(token, str) for token in tokens):
            raise TypeError("Kimi tokenizer token sequences must contain strings.")
        if all(token in self._special_tokens for token in tokens):
            return [self._special_tokens[token] for token in tokens]
        converted = _convert_tokens(self._raw, tokens)
        return _token_ids(converted, self._vocab_size, "converted token ids")

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        return_dict: bool = False,
    ) -> str | list[int]:
        if self._configured_chat_template is not None:
            if return_dict:
                raise TypeError(
                    "configured Kimi chat templates do not return tokenizer mappings."
                )
            rendered = _render_chat_template(
                conversation,
                self._configured_chat_template,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
            if not tokenize:
                return rendered
            return self.encode(rendered, add_special_tokens=False)

        template = _chat_template(self._raw)
        if template is None:
            raise NotImplementedError(
                "Kimi remote tokenizer does not provide an official chat template; "
                "provide a tokenizer with apply_chat_template() for chat generation."
            )
        rendered = template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            return_dict=return_dict,
        )
        if isinstance(rendered, str):
            return rendered
        if isinstance(rendered, Mapping):
            raise TypeError("Kimi chat template return_dict=True is not a token sequence.")
        return _token_ids(
            rendered,
            self._vocab_size,
            "Kimi chat template token output",
        )


def kimi_body(model: object) -> Callable[..., object]:
    """Resolve ``MoonshotKimiaForCausalLM.model`` without wrapping it."""

    body = getattr(model, "model", None)
    if not callable(body):
        raise TypeError("Kimi model must expose a callable model body at .model.")
    return cast(Callable[..., object], body)


def remove_kimi_output_heads(model: nn.Module) -> tuple[str, ...]:
    """Unregister Kimi's text and MIMO heads while preserving body paths.

    The returned paths identify heads that were present and removed.  The
    operation is idempotent, and the ``model`` body is never replaced, so
    names such as ``model.layers.0`` remain unchanged for PEFT/state-dict
    matching.  This helper is intentionally explicit because calling the
    remote root after removing its heads is unsupported; callers should use
    :func:`kimi_body` for the training body.
    """

    body = kimi_body(model)
    body_input_embeddings = _input_embeddings(model)
    heads: list[tuple[str, nn.Module]] = []
    for name in ("lm_head", "mimo_output"):
        head = getattr(model, name, None)
        if head is None:
            continue
        if not isinstance(head, nn.Module):
            raise TypeError(f"Kimi output head {name!r} must be an nn.Module or None.")
        heads.append((name, head))

    for name, _ in heads:
        setattr(model, name, None)
    if getattr(model, "model", None) is not body:
        raise RuntimeError("removing Kimi output heads replaced the model body.")
    if body_input_embeddings is not None:
        after_input_embeddings = _input_embeddings(model)
        if after_input_embeddings is not body_input_embeddings:
            raise RuntimeError("removing Kimi output heads replaced input embeddings.")
    return tuple(name for name, _ in heads)


_OutputT = TypeVar("_OutputT")


def checkpoint_kimi_body(
    body: Callable[..., _OutputT],
    *,
    enabled: bool,
    kwargs: Mapping[str, object],
) -> _OutputT:
    """Call a Kimi body directly or through non-reentrant activation checkpointing."""

    call_kwargs = dict(kwargs)
    if not enabled:
        return body(**call_kwargs)

    use_cache = call_kwargs.get("use_cache")
    if use_cache is not None and not isinstance(use_cache, bool):
        raise TypeError("Kimi body use_cache must be a boolean when checkpointing.")
    if use_cache:
        raise ValueError("Kimi body activation checkpointing requires use_cache=False.")
    for cache_name in ("past_key_values", "cache_position"):
        if call_kwargs.get(cache_name) is not None:
            raise ValueError(
                "Kimi body activation checkpointing does not support a populated "
                f"{cache_name} cache."
            )
    if "use_reentrant" in call_kwargs:
        raise TypeError("use_reentrant is controlled by checkpoint_kimi_body.")

    call_kwargs["use_cache"] = False
    checkpoint_call = cast(Callable[..., _OutputT], checkpoint)
    return checkpoint_call(body, use_reentrant=False, **call_kwargs)


def call_kimi_body(
    body: Callable[..., _OutputT],
    *,
    checkpointed: bool,
    **kwargs: object,
) -> _OutputT:
    """Keyword-oriented alias for :func:`checkpoint_kimi_body`."""

    return checkpoint_kimi_body(body, enabled=checkpointed, kwargs=kwargs)


def should_checkpoint_kimi_body(model: nn.Module, enabled: bool) -> bool:
    """Return whether training-time Kimi body checkpointing should be active."""

    return bool(enabled and model.training and torch.is_grad_enabled())


def _vocab_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Kimi tokenizer vocab_size must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError("Kimi tokenizer vocab_size must be positive.")
    return result


def _special_tokens(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("Kimi tokenizer special_tokens must be a mapping.")
    result: dict[str, int] = {}
    for token, token_id in value.items():
        if not isinstance(token, str):
            raise TypeError("Kimi tokenizer special-token names must be strings.")
        result[token] = _token_id(token_id, f"special token {token!r}")
    return result


def _token_for_id(tokens: Mapping[str, int], token_id: int, name: str) -> str:
    for token, candidate_id in tokens.items():
        if candidate_id == token_id:
            return token
    raise ValueError(f"Kimi tokenizer special_tokens is missing {name} id {token_id}.")


def _token_id(value: object, name: str, *, vocab_size: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Kimi tokenizer {name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"Kimi tokenizer {name} must be non-negative.")
    if vocab_size is not None and result >= vocab_size:
        raise ValueError(f"Kimi tokenizer {name} must be smaller than vocab_size.")
    return result


def _token_ids(value: object, vocab_size: int, name: str) -> list[int]:
    if isinstance(value, Tensor):
        if value.dim() != 1:
            raise ValueError(f"{name} must be a one-dimensional token sequence.")
        value = value.detach().cpu().tolist()
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a token sequence, not text.")
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a one-dimensional token sequence.") from error
    return [
        _token_id(token_id, f"{name} item", vocab_size=vocab_size)
        for token_id in values
    ]


def _input_embeddings(model: nn.Module) -> nn.Module | None:
    getter = getattr(model, "get_input_embeddings", None)
    if not callable(getter):
        return None
    embeddings = getter()
    if embeddings is not None and not isinstance(embeddings, nn.Module):
        raise TypeError("Kimi get_input_embeddings() must return an nn.Module or None.")
    return embeddings


def _convert_tokens(raw: object, tokens: object) -> object:
    converter = getattr(raw, "convert_tokens_to_ids", None)
    if not callable(converter):
        raise KeyError(f"Kimi tokenizer has no token {tokens!r}.")
    try:
        return converter(tokens)
    except (AttributeError, NotImplementedError) as error:
        raise KeyError(f"Kimi tokenizer has no token {tokens!r}.") from error


def _chat_template(raw: object) -> Callable[..., object] | None:
    template = getattr(raw, "apply_chat_template", None)
    if not callable(template):
        return None
    declared = getattr(raw, "chat_template", None)
    if declared:
        return template
    if "apply_chat_template" in type(raw).__dict__:
        return template
    return None


def _render_chat_template(
    conversation: Sequence[Mapping[str, str]],
    template: str,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    messages: list[dict[str, str]] = []
    for message in conversation:
        if not isinstance(message, Mapping):
            raise TypeError("Kimi chat messages must be mappings.")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise TypeError("Kimi chat message roles must be non-empty strings.")
        if not isinstance(content, str):
            raise TypeError("Kimi chat message content must be a string.")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("Kimi chat conversations must not be empty.")

    rendered = cast(
        object,
        render_jinja_template(
            [messages],
            chat_template=template,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        ),
    )
    if not isinstance(rendered, tuple) or len(rendered) != 2:
        raise TypeError("Kimi chat template renderer returned an invalid result.")
    values = rendered[0]
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 1
        or not isinstance(values[0], str)
    ):
        raise TypeError("Kimi chat template must render exactly one string.")
    return values[0]


__all__ = [
    "KimiRawTokenizer",
    "KimiTokenizerAdapter",
    "call_kimi_body",
    "checkpoint_kimi_body",
    "kimi_body",
    "remove_kimi_output_heads",
    "should_checkpoint_kimi_body",
]
