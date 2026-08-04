from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import cast

from .audio_tokenizer import TorchCodecBPE
from .tokenizer import TextTokenizer


def audio_tokenizer(path: str | Path) -> TorchCodecBPE:
    from zhuyin.tokenizers.codec_bpe import codec_bpe

    tokenizer = codec_bpe(Path(path).expanduser())
    return TorchCodecBPE.wrap(tokenizer)


def text_tokenizer_vocab_size(tokenizer: object) -> int:
    try:
        return _positive_integral(len(cast(TextTokenizer, tokenizer)), "text tokenizer length")
    except (AttributeError, NotImplementedError, TypeError):
        pass
    for attribute in ("vocab_size", "total_vocab_size"):
        try:
            value = getattr(tokenizer, attribute)
        except AttributeError:
            continue
        if value is None:
            continue
        return _positive_integral(value, f"text tokenizer {attribute}")
    raise AttributeError("text tokenizer does not expose a positive vocabulary size.")


def _positive_integral(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def text_special_id(tokenizer: TextTokenizer, name: str) -> int:
    """Resolve a required text special-token id from HF tokenizer attributes."""
    if name not in {"pad_token_id", "bos_token_id", "eos_token_id"}:
        raise ValueError(f"unsupported text special token attribute: {name}.")
    token_id = getattr(tokenizer, name)
    if token_id is not None:
        return _token_id(token_id, name)
    map_key = name[: -len("_id")]
    token = tokenizer.special_tokens_map.get(map_key)
    if token is None:
        raise ValueError(f"text tokenizer is missing {name}.")
    if not isinstance(token, str):
        raise TypeError(f"text tokenizer {map_key} must be a string.")
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"text token {token!r} must map to one id.")
    return _token_id(ids[0], name)


def _token_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"text tokenizer {name} must be an integer.")
    token_id = int(value)
    if token_id < 0:
        raise ValueError(f"text tokenizer {name} must be non-negative.")
    return token_id

__all__ = ["audio_tokenizer", "text_special_id", "text_tokenizer_vocab_size"]
