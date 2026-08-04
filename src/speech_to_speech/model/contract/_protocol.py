from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from semantic_acoustic_generator.model.dit import DiTDecoder

    from ..ctc import CTCDecoderRoutes

from ...runtime.backbone import BackboneEncoder
from ...runtime.protocol import TokenModelRuntime
from ...runtime.backbone.contract import Backbone
from ..audio_input import AudioInputTower
from ..token import TokenInterface


class ContractModel(Protocol):
    @property
    def runtime(self) -> TokenModelRuntime: ...

    @property
    def backbone(self) -> Backbone: ...

    @property
    def tokens(self) -> TokenInterface: ...

    @property
    def source_audio_encoder(self) -> AudioInputTower | None: ...

    @property
    def ctc_decoders(self) -> CTCDecoderRoutes: ...

    @property
    def _encoder(self) -> BackboneEncoder: ...


class Flow(Protocol):
    @property
    def decoder(self) -> DiTDecoder: ...


@runtime_checkable
class ContractStateProvider(Protocol):
    def contract_state(self) -> Mapping[str, object]: ...


@runtime_checkable
class VocabularyProvider(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


@runtime_checkable
class BackendTokenizerProvider(Protocol):
    @property
    def backend_tokenizer(self) -> object: ...


@runtime_checkable
class TokenizerBackendSerializer(Protocol):
    def to_str(self) -> str: ...


@runtime_checkable
class ChatTemplateProvider(Protocol):
    @property
    def chat_template(self) -> str | None: ...


@runtime_checkable
class ConfigStateProvider(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


@runtime_checkable
class ConfigOwner(Protocol):
    @property
    def config(self) -> object: ...

__all__ = [
    "BackendTokenizerProvider",
    "ChatTemplateProvider",
    "ConfigOwner",
    "ConfigStateProvider",
    "ContractModel",
    "ContractStateProvider",
    "Flow",
    "TokenizerBackendSerializer",
    "VocabularyProvider",
]
