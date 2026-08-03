from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Protocol, Union, cast, runtime_checkable

from anytrain.codec import AcousticLayout
from torch import Generator, Tensor, nn
from transformers.cache_utils import Cache


class SemanticCodec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    def decode(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: Generator | None = None,
    ) -> Tensor: ...


class SemanticCodebookCodec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def semantic_codebook(self) -> Tensor: ...


class Codec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...


class StructuredCodec(SemanticCodebookCodec, Protocol):
    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_layout(self) -> AcousticLayout: ...

    @property
    def acoustic_unit_length(self) -> int | None: ...

    @property
    def acoustic_feature_dim(self) -> int: ...

    def tokenize(self, audio: Tensor, sample_rate: int) -> object: ...

    def detokenize(self, codes: object) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


CodecBackend = Union[Codec, StructuredCodec]


class CodebookCodec(SemanticCodebookCodec, Protocol):
    pass


class AcousticCodec(CodebookCodec, Protocol):
    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


@runtime_checkable
class _CodebookCapability(Protocol):
    @property
    def semantic_codebook(self) -> Tensor: ...


@runtime_checkable
class _SemanticFeatureCapability(Protocol):
    @property
    def semantic_feature_dim(self) -> int: ...


@runtime_checkable
class _SampleRateCapability(Protocol):
    @property
    def sample_rate(self) -> int: ...


@runtime_checkable
class _FrameRateCapability(Protocol):
    @property
    def frame_rate(self) -> float: ...


@runtime_checkable
class _FrameCapability(_SampleRateCapability, _FrameRateCapability, Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...


@runtime_checkable
class _FrameCodebookCapability(Protocol):
    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...


@runtime_checkable
class _AcousticCapability(
    _SampleRateCapability,
    _FrameRateCapability,
    _CodebookCapability,
    Protocol,
):
    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


@runtime_checkable
class _StructuredCapability(
    _SampleRateCapability,
    _FrameRateCapability,
    _CodebookCapability,
    Protocol,
):
    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    @property
    def acoustic_layout(self) -> AcousticLayout: ...

    @property
    def acoustic_unit_length(self) -> int | None: ...

    @property
    def acoustic_feature_dim(self) -> int: ...

    def tokenize(self, audio: Tensor, sample_rate: int) -> object: ...

    def detokenize(self, codes: object) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


def codebook_codec(codec: object) -> CodebookCodec:
    if not isinstance(codec, _CodebookCapability):
        raise TypeError("codec-initialized audio embeddings require a semantic codebook.")
    _semantic_codebook(codec.semantic_codebook)
    return cast(CodebookCodec, codec)


def semantic_feature_dim(codec: object) -> int:
    if isinstance(codec, _CodebookCapability):
        return int(_semantic_codebook(codec.semantic_codebook).size(-1))
    if not isinstance(codec, _SemanticFeatureCapability):
        raise TypeError(
            "random audio embeddings require a semantic codebook or feature dimension."
        )
    value = codec.semantic_feature_dim
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("codec semantic feature dimension must be an integer.")
    if value <= 0:
        raise ValueError("codec semantic feature dimension must be positive.")
    return value


@runtime_checkable
class _FsqLevelsCapability(Protocol):
    @property
    def fsq_levels(self) -> tuple[tuple[int, ...], ...]: ...


def fsq_levels(codec: object) -> tuple[tuple[int, ...], ...] | None:
    """Return FSQ levels when the codec is a dim-1 FSQ source; otherwise None."""
    if not isinstance(codec, _SemanticFeatureCapability):
        return None
    if semantic_feature_dim(codec) != 1:
        return None
    if not isinstance(codec, _FsqLevelsCapability):
        return None
    levels = tuple(
        tuple(int(level) for level in stage) for stage in codec.fsq_levels
    )
    if not levels:
        raise ValueError("fsq_levels must be a non-empty tuple of stages.")
    for stage in levels:
        if not stage:
            raise ValueError("each FSQ stage must declare at least one level.")
        if any(level < 2 for level in stage):
            raise ValueError("FSQ levels must be at least 2.")
    if isinstance(codec, _FrameCodebookCapability):
        sizes = _codebook_sizes(codec.codebook_sizes, "FSQ codec")
        if len(sizes) != len(levels):
            raise ValueError("fsq_levels must align with codebook_sizes.")
        for size, stage in zip(sizes, levels):
            product = 1
            for level in stage:
                product *= level
            if product != size:
                raise ValueError(
                    f"FSQ levels {stage} must multiply to codebook size {size}."
                )
    return levels


def frame_codec(codec: object) -> Codec:
    if not isinstance(codec, _FrameCapability):
        raise TypeError("full frame-code encoding and decoding require a frame codec capability.")
    codec_sample_rate(codec)
    codec_frame_rate(codec)
    _codebook_sizes(codec.codebook_sizes, "frame codec")
    return cast(Codec, codec)


def frame_codebook_sizes(codec: object) -> tuple[int, ...]:
    if not isinstance(codec, _FrameCodebookCapability):
        raise TypeError("frame codec codebook metadata is required.")
    return _codebook_sizes(codec.codebook_sizes, "frame codec")


def acoustic_codec(codec: object) -> AcousticCodec:
    if not isinstance(codec, _AcousticCapability):
        raise TypeError("acoustic decoding requires an acoustic codec capability.")
    codec_sample_rate(codec)
    codec_frame_rate(codec)
    _codebook_sizes(codec.acoustic_codebook_sizes, "acoustic codec")
    _positive_int(codec.acoustic_feature_dim, "acoustic codec feature dimension")
    return cast(AcousticCodec, codec)


def supports_acoustic(codec: object) -> bool:
    if not isinstance(codec, _AcousticCapability):
        return False
    acoustic_codec(codec)
    return True


def structured_codec(codec: object) -> StructuredCodec:
    if not isinstance(codec, _StructuredCapability):
        raise TypeError("structured codec capability is required.")
    codec_sample_rate(codec)
    codec_frame_rate(codec)
    _codebook_sizes(codec.semantic_codebook_sizes, "structured semantic codec")
    _codebook_sizes(codec.acoustic_codebook_sizes, "structured acoustic codec")
    _positive_int(codec.acoustic_feature_dim, "structured codec acoustic feature dimension")
    layout = codec.acoustic_layout
    if not isinstance(layout, AcousticLayout):
        raise TypeError("structured codec acoustic layout must be an AcousticLayout.")
    unit_length = codec.acoustic_unit_length
    if layout is AcousticLayout.FIXED_LENGTH:
        _positive_int(unit_length, "fixed-length structured codec acoustic unit length")
    elif unit_length is not None:
        raise ValueError(
            "frame-aligned structured codec acoustic unit length must be None."
        )
    return cast(StructuredCodec, codec)


def supports_structured(codec: object) -> bool:
    if not isinstance(codec, _StructuredCapability):
        return False
    structured_codec(codec)
    return True


def codec_sample_rate(codec: object) -> int:
    if not isinstance(codec, _SampleRateCapability):
        raise TypeError("codec sample rate metadata is required.")
    return _positive_int(codec.sample_rate, "codec sample rate")


def codec_frame_rate(codec: object) -> float:
    if not isinstance(codec, _FrameRateCapability):
        raise TypeError("codec frame rate metadata is required.")
    value = codec.frame_rate
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("codec frame rate must be a number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("codec frame rate must be finite and positive.")
    return result


def _codebook_sizes(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} codebook sizes must be a tuple of integers.")
    if not value:
        raise ValueError(f"{name} codebook sizes must be non-empty.")
    for size in value:
        _positive_int(size, f"{name} codebook size")
    return cast(tuple[int, ...], value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _semantic_codebook(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("codec semantic codebook must be a Tensor.")
    if value.dim() not in {2, 3}:
        raise ValueError(
            "codec semantic codebook must have shape [vocab, dim] or "
            "[codebooks, vocab, dim]."
        )
    if any(size <= 0 for size in value.shape):
        raise ValueError("codec semantic codebook dimensions must be positive.")
    return value


class AudioTokenizer(Protocol):
    embedding_initialization: str

    @property
    def vocab_size(self) -> int: ...

    def encode(
        self, frames: Sequence[Sequence[int]] | Tensor
    ) -> list[int] | Tensor: ...

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor: ...

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor: ...


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

    @classmethod
    def from_path(cls, path: object) -> BackboneReadout:
        return cls(cast(str, path))

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
    return BackboneReadout.from_path(path).path


def select_backbone_readout(
    output: BackboneOutput,
    readout: BackboneReadout | str,
) -> Tensor:
    """Select a tensor from a backbone output.

    The string form remains accepted for callers that validate configuration
    at the boundary; adapters should retain the typed ``BackboneReadout``.
    """
    selected = (
        readout
        if isinstance(readout, BackboneReadout)
        else BackboneReadout.from_path(readout)
    )
    return selected.select(output)


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

    def get_output_embeddings(self) -> nn.Module | None: ...

    @property
    def base_model(self) -> BackboneBody: ...
