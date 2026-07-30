"""Audio stream ownership for autoregressive prompts, outputs, and decoding."""

from __future__ import annotations

from dataclasses import dataclass

from ._compat import StrEnum, auto


class AudioStream(StrEnum):
    GLOBAL = auto()
    ACOUSTIC = auto()
    SEMANTIC = auto()


class PromptSource(StrEnum):
    REFERENCE = auto()
    SOURCE = auto()


class StreamSource(StrEnum):
    PROMPT = auto()
    OUTPUT = auto()
    GENERATOR = auto()


_STREAM_ORDER = (AudioStream.GLOBAL, AudioStream.ACOUSTIC, AudioStream.SEMANTIC)


@dataclass
class Prompt:
    source: PromptSource
    streams: tuple[AudioStream, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, PromptSource):
            raise TypeError("audio route prompt source must be a PromptSource.")
        self.streams = _streams(self.streams, name="prompt")

    @property
    def canonical_streams(self) -> tuple[AudioStream, ...]:
        return _canonical_streams(self.streams)


@dataclass
class Output:
    streams: tuple[AudioStream, ...]

    def __post_init__(self) -> None:
        self.streams = _streams(self.streams, name="output")

    @property
    def canonical_streams(self) -> tuple[AudioStream, ...]:
        return _canonical_streams(self.streams)


@dataclass
class Decode:
    semantic: StreamSource
    acoustic: StreamSource

    def __post_init__(self) -> None:
        if not isinstance(self.semantic, StreamSource):
            raise TypeError("audio route semantic decode source must be a StreamSource.")
        if not isinstance(self.acoustic, StreamSource):
            raise TypeError("audio route acoustic decode source must be a StreamSource.")

    def source(self, stream: AudioStream) -> StreamSource:
        if stream is AudioStream.SEMANTIC:
            return self.semantic
        if stream in {AudioStream.GLOBAL, AudioStream.ACOUSTIC}:
            return self.acoustic
        raise TypeError("audio route stream must be an AudioStream.")


@dataclass
class Config:
    prompt: Prompt
    output: Output
    decode: Decode

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, Prompt):
            raise TypeError("audio route prompt must be a Prompt.")
        if not isinstance(self.output, Output):
            raise TypeError("audio route output must be an Output.")
        if not isinstance(self.decode, Decode):
            raise TypeError("audio route decode must be a Decode.")
        for stream in (AudioStream.GLOBAL, AudioStream.SEMANTIC):
            source = self.decode.source(stream)
            if source is StreamSource.PROMPT and not _provides_stream(
                self.prompt.streams,
                stream,
            ):
                raise ValueError(
                    f"audio route decode source prompt does not provide {_stream_label(stream)}."
                )
            if source is StreamSource.OUTPUT and not _provides_stream(
                self.output.streams,
                stream,
            ):
                raise ValueError(
                    f"audio route decode source output does not provide {_stream_label(stream)}."
                )


def _streams(
    streams: list[AudioStream] | tuple[AudioStream, ...],
    *,
    name: str,
) -> tuple[AudioStream, ...]:
    if not isinstance(streams, (list, tuple)):
        raise TypeError(f"audio route {name} streams must be a list or tuple.")
    if any(not isinstance(stream, AudioStream) for stream in streams):
        raise TypeError(f"audio route {name} streams must contain AudioStream values.")
    if len(streams) != len(set(streams)):
        raise ValueError(f"audio route {name} streams must not contain duplicates.")
    if AudioStream.GLOBAL in streams and AudioStream.ACOUSTIC in streams:
        raise ValueError(
            f"audio route {name} streams must not contain both global and "
            "acoustic streams."
        )
    return tuple(streams)


def _canonical_streams(streams: tuple[AudioStream, ...]) -> tuple[AudioStream, ...]:
    return tuple(stream for stream in _STREAM_ORDER if stream in streams)


def _stream_label(stream: AudioStream) -> str:
    return "global or acoustic" if stream is AudioStream.GLOBAL else stream.value


def _provides_stream(
    streams: tuple[AudioStream, ...],
    stream: AudioStream,
) -> bool:
    if stream is AudioStream.GLOBAL:
        return AudioStream.GLOBAL in streams or AudioStream.ACOUSTIC in streams
    return stream in streams


BICODEC_REUSE_PROMPT_GLOBAL = Config(
    prompt=Prompt(
        source=PromptSource.REFERENCE,
        streams=(AudioStream.GLOBAL,),
    ),
    output=Output(streams=(AudioStream.SEMANTIC,)),
    decode=Decode(
        semantic=StreamSource.OUTPUT,
        acoustic=StreamSource.PROMPT,
    ),
)

SEMANTIC_GENERATOR = Config(
    prompt=Prompt(
        source=PromptSource.SOURCE,
        streams=(),
    ),
    output=Output(streams=(AudioStream.SEMANTIC,)),
    decode=Decode(
        semantic=StreamSource.OUTPUT,
        acoustic=StreamSource.GENERATOR,
    ),
)

FULL_OUTPUT = Config(
    prompt=Prompt(
        source=PromptSource.SOURCE,
        streams=(),
    ),
    output=Output(streams=(AudioStream.ACOUSTIC, AudioStream.SEMANTIC)),
    decode=Decode(
        semantic=StreamSource.OUTPUT,
        acoustic=StreamSource.OUTPUT,
    ),
)

BICODEC_GENERATE_GLOBAL = Config(
    prompt=Prompt(
        source=PromptSource.SOURCE,
        streams=(),
    ),
    output=Output(streams=(AudioStream.GLOBAL, AudioStream.SEMANTIC)),
    decode=Decode(
        semantic=StreamSource.OUTPUT,
        acoustic=StreamSource.OUTPUT,
    ),
)


__all__ = [
    "BICODEC_GENERATE_GLOBAL",
    "BICODEC_REUSE_PROMPT_GLOBAL",
    "FULL_OUTPUT",
    "SEMANTIC_GENERATOR",
    "AudioStream",
    "Config",
    "Decode",
    "Output",
    "Prompt",
    "PromptSource",
    "StreamSource",
]
