from __future__ import annotations

from functools import cached_property
from typing import Protocol

from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor, nn

from ..runtime.types import AudioTokenizer, Backbone, Codec, TextTokenizer


class FlowSample(Protocol):
    final: Tensor


class FlowSamplingRuntime(Protocol):
    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        *,
        time_grid: Tensor | None = None,
        **model_extras: object,
    ) -> FlowSample: ...


class TokenModelRuntime(Protocol):
    @cached_property
    def layout(self) -> Layout: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer: ...

    @cached_property
    def codec(self) -> Codec: ...

    @cached_property
    def eos_token_id(self) -> int: ...

    @cached_property
    def bos_token_id(self) -> int: ...

    @property
    def eoa_token_id(self) -> int: ...

    @property
    def boa_token_id(self) -> int: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @cached_property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]: ...

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]: ...

    def is_codec_audio_id(self, token_id: int) -> bool: ...

    @cached_property
    def pad_token_id(self) -> int: ...

    @cached_property
    def backbone(self) -> Backbone: ...


class FlowModelRuntime(TokenModelRuntime, Protocol):
    @cached_property
    def flow_matching(self) -> FlowSamplingRuntime: ...
