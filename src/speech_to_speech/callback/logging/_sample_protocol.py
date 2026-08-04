from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, Protocol, TypedDict

from anydataset import types
from anytrain.module.idspace import Layout

from ...datamodule.batch import ModelBatch, TrainInput
from ...generation.result import Result
from ...runtime.codec_contract import CodecBackend
from ...runtime.tokenizer import TextTokenizer
from ...task import Request, Task
from ...datamodule.diagnostic import SampleSplit


class Module(Protocol):
    model: Any

    def materialize_batch(self, batch: TrainInput) -> ModelBatch: ...

    def generate(
        self,
        requests: Sequence[Request],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> list[Result]: ...


class GenerationKwargs(TypedDict):
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    use_cache: bool


class DataModule(Protocol):
    runtime: LoggingRuntime

    def diagnostic_samples(
        self,
        indices: Sequence[int],
        *,
        split: SampleSplit,
        loader_name: str,
    ) -> list[types.Sample]: ...

    def diagnostic_collator(
        self,
        task: Task,
        *,
        split: SampleSplit,
        loader_name: str,
    ) -> Callable[[list[types.Sample]], TrainInput]: ...


class LoggingRuntime(Protocol):
    @property
    def audio_view(self) -> types.AudioView: ...

    @property
    def codec(self) -> CodecBackend: ...

    @property
    def audio_tokenizer(self) -> object: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def layout(self) -> Layout: ...

__all__ = ["DataModule", "GenerationKwargs", "LoggingRuntime", "Module"]
