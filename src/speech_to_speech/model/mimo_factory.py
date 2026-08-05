"""Runtime assembly for the standalone Kimi-style MIMO model.

The ordinary composition builder owns the serialized single-stream model.  A
MIMO run uses this small factory instead: it obtains the text embedding and
body from the runtime, keeps two local vocabularies, and calls the runtime's
already-adapted body exactly once per aligned sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Callable, Protocol, cast

import torch
from torch import Tensor, nn

from ..runtime.backbone import BackboneBodyAdapter
from ..runtime.backbone.mimo import DualStreamBodyAdapter
from ..runtime.backbone.contract import BackboneReadout
from ..mimo import MimoSpecialTokens
from .mimo import MimoModel, MimoModelConfig, TiedEmbeddingHead


@dataclass(frozen=True)
class MimoFactoryConfig:
    """Hydra-friendly knobs for MIMO model assembly."""

    text_readout: str | None = None
    audio_readout: str | None = None
    audio_vocab_size: int | None = None
    audio_embedding_dim: int | None = None
    audio_feature_dim: int | None = None
    audio_feature_scale: float = 2.0**0.5
    supports_cache_position: bool | None = None
    initialize_audio_from_runtime: bool = False
    toy: bool = False
    text_vocab_size: int | None = None
    text_blank_token_id: int | None = None
    audio_blank_token_id: int | None = None

    def __post_init__(self) -> None:
        for name in ("text_readout", "audio_readout"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None.")
        for name in (
            "audio_vocab_size",
            "audio_embedding_dim",
            "audio_feature_dim",
            "text_vocab_size",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None.")
        if (
            isinstance(self.audio_feature_scale, bool)
            or not isinstance(self.audio_feature_scale, (int, float))
            or not torch.isfinite(torch.tensor(float(self.audio_feature_scale)))
            or self.audio_feature_scale <= 0
        ):
            raise ValueError("audio_feature_scale must be finite and positive.")
        if self.supports_cache_position is not None and not isinstance(
            self.supports_cache_position, bool
        ):
            raise TypeError("supports_cache_position must be a boolean or None.")
        if not isinstance(self.initialize_audio_from_runtime, bool):
            raise TypeError("initialize_audio_from_runtime must be a boolean.")
        if not isinstance(self.toy, bool):
            raise TypeError("toy must be a boolean.")
        for name in ("text_blank_token_id", "audio_blank_token_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")


class MimoRuntime(Protocol):
    """Minimal runtime surface consumed by the factory."""

    @property
    def backbone_adapter(self) -> object: ...

    @property
    def backbone(self) -> object: ...

    @property
    def backbone_body(self) -> str: ...

    @property
    def layout(self) -> object: ...

    @property
    def audio_tokenizer(self) -> object: ...

    @property
    def backbone_readouts(self) -> Mapping[str, str]: ...

    @property
    def backbone_readout(self) -> str: ...

    @property
    def backbone_supports_cache_position(self) -> bool: ...

    @property
    def codec(self) -> object: ...


def build_mimo_model(
    runtime: MimoRuntime,
    config: MimoFactoryConfig | None = None,
    *,
    audio_embedding: nn.Embedding | None = None,
    audio_feature_projection: nn.Module | None = None,
) -> MimoModel:
    """Build a :class:`MimoModel` from a loaded runtime.

    ``runtime.backbone_adapter.body`` is deliberately used as the call path.
    For Kimi-Audio this callable adds ``return_dict=True`` and dispatches the
    remote-body activation-checkpoint wrapper; calling the raw module directly
    would lose both behaviours.  The underlying body module is still retained
    under ``model.body`` so Lightning sees all trainable parameters.
    """

    # Accept runtime-compatible test doubles and workspace adapters while
    # retaining the concrete Runtime annotation for production callers.
    required = (
        "backbone_adapter",
        "backbone",
        "backbone_body",
        "layout",
        "audio_tokenizer",
        "backbone_readouts",
        "backbone_readout",
        "backbone_supports_cache_position",
    )
    if any(not hasattr(runtime, name) for name in required):
        raise TypeError("build_mimo_model expects a runtime with backbone/layout capabilities.")
    layout_blocks = getattr(runtime.layout, "blocks", None)
    if isinstance(layout_blocks, Mapping) and "audio_input" in layout_blocks:
        raise ValueError(
            "MIMO currently requires one shared audio token space; use the token model "
            "for decoupled input/output audio tokenizers."
        )
    options = MimoFactoryConfig() if config is None else config
    if not isinstance(options, MimoFactoryConfig):
        raise TypeError("config must be a MimoFactoryConfig or None.")

    adapter = runtime.backbone_adapter
    backbone = runtime.backbone
    input_embeddings = getattr(adapter, "input_embeddings", None)
    if not callable(input_embeddings):
        raise TypeError("runtime backbone adapter must expose input_embeddings().")
    source_embedding = input_embeddings()
    if not isinstance(source_embedding, nn.Embedding):
        raise TypeError("runtime backbone input embeddings must be nn.Embedding.")
    text_vocab = (
        options.text_vocab_size if options.text_vocab_size is not None else _text_vocab(runtime)
    )
    _positive_int(text_vocab, "text vocabulary size")
    if source_embedding.num_embeddings < text_vocab:
        raise ValueError("runtime backbone input embeddings do not cover the text vocabulary.")

    hidden_size = _hidden_size(adapter, source_embedding)
    if options.audio_embedding_dim is not None and options.audio_embedding_dim != hidden_size:
        raise ValueError("audio_embedding_dim must match the runtime hidden size.")
    _, audio_vocab = _mimo_audio_vocab(runtime, options.audio_vocab_size)
    local_audio = audio_embedding or _audio_embedding(
        runtime,
        audio_vocab,
        hidden_size,
        reference=source_embedding.weight,
        initialize_from_runtime=options.initialize_audio_from_runtime,
    )
    if local_audio.num_embeddings != audio_vocab:
        raise ValueError("provided audio_embedding has the configured vocabulary size mismatch.")
    if local_audio.embedding_dim != hidden_size:
        raise ValueError("MIMO text/audio embeddings must share the runtime hidden size.")
    if local_audio.weight.device != source_embedding.weight.device:
        raise ValueError("MIMO text/audio embeddings must share a device.")

    feature_projection = audio_feature_projection
    if (
        feature_projection is None
        and options.audio_feature_dim is not None
        and options.audio_feature_dim != hidden_size
    ):
        feature_projection = nn.Linear(
            options.audio_feature_dim,
            hidden_size,
            bias=False,
            device=source_embedding.weight.device,
            dtype=source_embedding.weight.dtype,
        )

    _body_module(backbone, runtime.backbone_body)
    adapted_body = _adapted_body(adapter)
    text_head = (
        None
        if source_embedding.num_embeddings == text_vocab
        else TiedEmbeddingHead(source_embedding, text_vocab)
    )
    readouts = runtime.backbone_readouts
    text_readout = BackboneReadout(
        options.text_readout or readouts.get("text", runtime.backbone_readout)
    )
    audio_readout = BackboneReadout(
        options.audio_readout or readouts.get("audio", runtime.backbone_readout)
    )
    supports_cache_position = (
        runtime.backbone_supports_cache_position
        if options.supports_cache_position is None
        else options.supports_cache_position
    )
    model = RuntimeMimoModel(
        runtime,
        backbone=backbone,
        body_path=runtime.backbone_body,
        body_call=adapted_body,
        text_embedding_getter=cast(Callable[[], nn.Embedding], input_embeddings),
        text_embedding=source_embedding,
        audio_embedding=local_audio,
        text_head=text_head,
        text_readout=text_readout,
        audio_readout=audio_readout,
        audio_feature_projection=feature_projection,
        config=MimoModelConfig(
            audio_feature_scale=float(options.audio_feature_scale),
            supports_cache_position=bool(supports_cache_position),
        ),
    )
    return model


class RuntimeMimoModel(MimoModel):
    """MimoModel whose canonical body ownership is ``backbone.*``.

    The runtime adapter is a service rather than a Module owner.  It supplies
    the wrapped callable, while the exact same runtime backbone is registered
    once under ``backbone``.  The text embedding remains owned by that
    backbone and is retrieved through a non-registering getter.
    """

    def __init__(
        self,
        runtime: MimoRuntime,
        *,
        backbone: object,
        body_path: str,
        body_call: Callable[..., object],
        text_embedding_getter: Callable[[], nn.Embedding],
        text_embedding: nn.Embedding,
        audio_embedding: nn.Embedding,
        text_readout: BackboneReadout,
        audio_readout: BackboneReadout,
        text_head: nn.Module | None = None,
        audio_head: nn.Module | None = None,
        audio_feature_projection: nn.Module | None = None,
        config: MimoModelConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(backbone, nn.Module):
            raise TypeError("runtime backbone must be an nn.Module.")
        if not callable(body_call) or not callable(text_embedding_getter):
            raise TypeError("runtime MIMO body and embedding getter must be callable.")
        if text_embedding_getter() is not text_embedding:
            raise RuntimeError("runtime text embedding getter changed during assembly.")
        if not isinstance(audio_embedding, nn.Embedding):
            raise TypeError("audio_embedding must be nn.Embedding.")
        for name, value in (
            ("text_head", text_head),
            ("audio_head", audio_head),
            ("audio_feature_projection", audio_feature_projection),
        ):
            if value is not None and not isinstance(value, nn.Module):
                raise TypeError(f"{name} must be nn.Module or None.")
        self.runtime = runtime
        self.config = MimoModelConfig() if config is None else config
        self.backbone = backbone
        self._body_path = body_path
        self._text_embedding_getter = text_embedding_getter
        self.audio_embedding = audio_embedding
        self.text_head = text_head
        self.audio_head = audio_head
        self.audio_feature_projection = audio_feature_projection
        self.text_readout = text_readout
        self.audio_readout = audio_readout
        self._encoder = DualStreamBodyAdapter(
            body_call,
            text_readout=text_readout,
            audio_readout=audio_readout,
            supports_cache_position=self.config.supports_cache_position,
        )

    @property
    def text_embedding(self) -> nn.Embedding:
        value = self._text_embedding_getter()
        if not isinstance(value, nn.Embedding):
            raise TypeError("runtime text embedding getter must return nn.Embedding.")
        return value

    @property
    def body(self) -> nn.Module:
        return _body_module(self.backbone, self._body_path)


def _adapted_body(adapter: object) -> Callable[..., object]:
    body = getattr(adapter, "body", None)
    if not isinstance(body, BackboneBodyAdapter):
        # Test doubles and custom runtime adapters may expose a compatible
        # callable directly; retain that capability without requiring HF.
        if callable(body):
            return cast(Callable[..., object], body)
        raise TypeError("runtime backbone adapter must expose a callable body adapter.")
    return cast(Callable[..., object], body.body)


def _body_module(backbone: object, path: str) -> nn.Module:
    current = backbone
    for part in (part for part in path.split(".") if part):
        if not hasattr(current, part):
            raise AttributeError(f"runtime backbone has no body attribute {part!r}.")
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError("runtime backbone body path must resolve to nn.Module.")
    return current


def _text_vocab(runtime: MimoRuntime) -> int:
    try:
        start, end = getattr(cast(Any, runtime.layout), "blocks")["text"]
    except (AttributeError, KeyError, TypeError) as error:
        raise TypeError("runtime layout must expose a text vocabulary block.") from error
    if isinstance(start, bool) or isinstance(end, bool) or end <= start:
        raise ValueError("runtime text vocabulary block must be non-empty.")
    block_size = int(end - start)
    lexical_size = getattr(runtime, "lexical_text_vocab_size", block_size)
    if (
        isinstance(lexical_size, bool)
        or not isinstance(lexical_size, int)
        or lexical_size <= 0
        or lexical_size > block_size
    ):
        raise ValueError("runtime lexical text vocabulary must be positive and fit its text block.")
    return lexical_size


def _hidden_size(adapter: object, embedding: nn.Embedding) -> int:
    value = getattr(adapter, "hidden_size", None)
    if isinstance(value, bool) or not isinstance(value, int):
        value = embedding.embedding_dim
    if value <= 0:
        raise ValueError("runtime backbone hidden size must be positive.")
    if value != embedding.embedding_dim:
        raise ValueError("runtime hidden size must match input embedding width.")
    return value


def _audio_embedding(
    runtime: MimoRuntime,
    vocab_size: int,
    hidden_size: int,
    *,
    reference: Tensor,
    initialize_from_runtime: bool,
) -> nn.Embedding:
    output = nn.Embedding(
        vocab_size,
        hidden_size,
        device=reference.device,
        dtype=reference.dtype,
    )
    nn.init.normal_(output.weight, mean=0.0, std=hidden_size**-0.5)
    if not initialize_from_runtime:
        return output
    # Runtime codec initialization is optional because loading a codec can be
    # expensive.  Copy only the overlap; special and extra local rows remain
    # trainable random vectors when the codec vocabulary is smaller.
    try:
        codebook = getattr(runtime.codec, "semantic_codebook")
    except (AttributeError, RuntimeError, TypeError):
        return output
    if not isinstance(codebook, Tensor) or codebook.dim() not in {2, 3}:
        return output
    base = codebook.mean(dim=0) if codebook.dim() == 3 else codebook
    rows = min(vocab_size, base.size(0))
    width = min(hidden_size, base.size(1))
    with torch.no_grad():
        output.weight[:rows, :width].copy_(base[:rows, :width].to(output.weight))
    return output


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _mimo_audio_payload_vocab(runtime: MimoRuntime) -> int:
    """Return the local semantic-audio payload vocabulary for MIMO.

    Structured tokenizers such as BiCodec expose a complete serialization
    vocabulary through ``vocab_size``.  MIMO's semantic route consumes only
    the semantic payload, so prefer its explicit ``semantic_vocab_size`` when
    available and fall back to the tokenizer's ordinary vocabulary otherwise.
    """

    tokenizer = cast(Any, runtime.audio_tokenizer)
    semantic_size = getattr(tokenizer, "semantic_vocab_size", None)
    if semantic_size is not None:
        if (
            isinstance(semantic_size, bool)
            or not isinstance(semantic_size, int)
            or semantic_size <= 0
        ):
            raise ValueError("semantic audio payload vocabulary size must be a positive integer.")
        return semantic_size
    payload_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size <= 0:
        raise ValueError("audio tokenizer vocabulary size must be a positive integer.")
    return payload_size


def _mimo_audio_vocab(
    runtime: MimoRuntime,
    configured_size: int | None,
) -> tuple[int, int]:
    payload_size = _mimo_audio_payload_vocab(runtime)
    minimum_size = payload_size + 3
    audio_size = minimum_size if configured_size is None else configured_size
    _positive_int(audio_size, "audio vocabulary size")
    if audio_size < minimum_size:
        raise ValueError(
            "audio vocabulary size must cover the semantic audio payload and "
            "three MIMO special tokens."
        )
    return payload_size, audio_size


@dataclass(frozen=True)
class MimoVocab:
    """Local vocabulary and structural ids used by a MIMO data composer."""

    text_size: int
    audio_size: int
    text_blank: int
    text_bos: int
    text_eos: int
    audio_blank: int
    audio_bos: int
    audio_eos: int

    def special_tokens(self, *, audio_delay_tokens: int = 0) -> MimoSpecialTokens:
        return MimoSpecialTokens(
            text_bos=self.text_bos,
            text_eos=self.text_eos,
            text_blank=self.text_blank,
            audio_bos=self.audio_bos,
            audio_eos=self.audio_eos,
            audio_blank=self.audio_blank,
            audio_delay_tokens=audio_delay_tokens,
        )


def derive_mimo_vocab(
    runtime: MimoRuntime,
    config: MimoFactoryConfig | None = None,
) -> MimoVocab:
    """Derive local vocabularies and special ids without global id offsets."""

    options = MimoFactoryConfig() if config is None else config
    if not isinstance(options, MimoFactoryConfig):
        raise TypeError("config must be a MimoFactoryConfig or None.")
    text_size = options.text_vocab_size or _text_vocab(runtime)
    audio_payload, audio_size = _mimo_audio_vocab(
        runtime,
        options.audio_vocab_size,
    )
    _positive_int(text_size, "text vocabulary size")
    _positive_int(audio_size, "audio vocabulary size")
    text_blank = (
        options.text_blank_token_id
        if options.text_blank_token_id is not None
        else int(getattr(runtime, "pad_token_id", 0))
    )
    text_bos = int(getattr(runtime, "bos_token_id", text_blank))
    text_eos = int(getattr(runtime, "eos_token_id", text_blank))
    audio_blank = (
        options.audio_blank_token_id
        if options.audio_blank_token_id is not None
        else audio_payload + 2
    )
    audio_bos = audio_payload
    audio_eos = audio_payload + 1
    for name, value, size in (
        ("text_blank", text_blank, text_size),
        ("text_bos", text_bos, text_size),
        ("text_eos", text_eos, text_size),
        ("audio_blank", audio_blank, audio_size),
        ("audio_bos", audio_bos, audio_size),
        ("audio_eos", audio_eos, audio_size),
    ):
        if value < 0 or value >= size:
            raise ValueError(f"{name} must be inside its local vocabulary.")
    return MimoVocab(
        text_size=text_size,
        audio_size=audio_size,
        text_blank=text_blank,
        text_bos=text_bos,
        text_eos=text_eos,
        audio_blank=audio_blank,
        audio_bos=audio_bos,
        audio_eos=audio_eos,
    )


__all__ = [
    "MimoFactoryConfig",
    "MimoRuntime",
    "MimoVocab",
    "RuntimeMimoModel",
    "build_mimo_model",
    "derive_mimo_vocab",
]
