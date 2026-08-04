from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from ...audio_stream import AudioStream
from ._common import (
    codebook_size,
    token_tensor,
    validate_frame_ranges,
    validate_ids,
    validate_range,
)
from .bpe import TorchCodecBPE
from .native import NativeAudioTokenizer


@dataclass(frozen=True)
class BiCodecStreams:
    semantic: Tensor | None
    acoustic: Tensor | None


class BiCodecAudioTokenizer:
    """Serialize BiCodec semantic units and fixed-length acoustic speaker slots."""

    embedding_initialization = "random"

    def __init__(
        self,
        *,
        semantic_codebook_size: int,
        acoustic_codebook_sizes: Sequence[int],
        acoustic_unit_length: int | None,
        semantic_tokenizer: NativeAudioTokenizer | TorchCodecBPE | None = None,
    ) -> None:
        self._semantic_codebook_size = codebook_size(semantic_codebook_size)
        self._semantic_tokenizer = (
            NativeAudioTokenizer(vocab_size=self._semantic_codebook_size)
            if semantic_tokenizer is None
            else semantic_tokenizer
        )
        _validate_semantic_tokenizer(
            self._semantic_tokenizer,
            self._semantic_codebook_size,
        )
        self._semantic_vocab_size = codebook_size(self._semantic_tokenizer.vocab_size)
        self._acoustic_codebook_sizes = tuple(
            codebook_size(size) for size in acoustic_codebook_sizes
        )
        if not self._acoustic_codebook_sizes:
            raise ValueError("BiCodec tokenizer requires acoustic codebooks.")
        if acoustic_unit_length is None or acoustic_unit_length <= 0:
            raise ValueError("BiCodec tokenizer requires a positive acoustic unit length.")
        self._acoustic_unit_length = int(acoustic_unit_length)
        offsets = [self._semantic_vocab_size]
        for size in self._acoustic_codebook_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self._acoustic_offsets = tuple(offsets)
        marker_base = sum((self._semantic_vocab_size, *self._acoustic_codebook_sizes))
        self._semantic_token_id = marker_base
        self._acoustic_token_id = marker_base + 1
        self._end_token_id = marker_base + 2
        self._vocab_size = marker_base + 3

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def semantic_vocab_size(self) -> int:
        return self._semantic_vocab_size

    @property
    def semantic_codebook_size(self) -> int:
        return self._semantic_codebook_size

    @property
    def semantic_tokenizer(self) -> NativeAudioTokenizer | TorchCodecBPE:
        return self._semantic_tokenizer

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return self._acoustic_codebook_sizes

    @property
    def acoustic_offsets(self) -> tuple[int, ...]:
        return self._acoustic_offsets

    @property
    def acoustic_unit_length(self) -> int:
        return self._acoustic_unit_length

    @property
    def semantic_token_id(self) -> int:
        return self._semantic_token_id

    @property
    def acoustic_token_id(self) -> int:
        return self._acoustic_token_id

    @property
    def end_token_id(self) -> int:
        return self._end_token_id

    @property
    def semantic_token_range(self) -> tuple[int, int]:
        return 0, self._semantic_vocab_size

    @property
    def acoustic_token_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (offset, offset + size)
            for offset, size in zip(
                self._acoustic_offsets,
                self._acoustic_codebook_sizes,
            )
        )

    @property
    def prediction_token_ranges(self) -> tuple[tuple[int, int], ...]:
        return (self.semantic_token_range, *self.acoustic_token_ranges)

    def contract_state(self) -> dict[str, object]:
        """Return the effective structured token-ID grammar used by checkpoints."""
        return {
            "grammar": "bicodec-v1",
            "semantic_codebook_size": self.semantic_codebook_size,
            "semantic_vocab_size": self.semantic_vocab_size,
            "semantic_tokenizer": dict(self.semantic_tokenizer.contract_state()),
            "semantic_token_range": list(self.semantic_token_range),
            "acoustic_codebook_sizes": list(self.acoustic_codebook_sizes),
            "acoustic_offsets": list(self.acoustic_offsets),
            "acoustic_token_ranges": [
                list(bounds) for bounds in self.acoustic_token_ranges
            ],
            "acoustic_unit_length": self.acoustic_unit_length,
            "semantic_token_id": self.semantic_token_id,
            "acoustic_token_id": self.acoustic_token_id,
            "end_token_id": self.end_token_id,
            "vocab_size": self.vocab_size,
        }

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        """Encode semantic codes for the semantic-only route."""
        token_ids = self._semantic_tokenizer.encode(frames)
        if isinstance(token_ids, Tensor):
            return token_ids.to(dtype=torch.long)
        return torch.tensor(token_ids, dtype=torch.long)

    def encode_full(self, value: SemanticAcousticCodes) -> Tensor:
        return self.encode_streams(
            value,
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )

    def encode_acoustic(self, value: SemanticAcousticCodes) -> Tensor:
        """Encode only BiCodec's fixed-length acoustic speaker/style codes."""
        return self.encode_streams(value, (AudioStream.ACOUSTIC,))

    def encode_streams(
        self,
        value: SemanticAcousticCodes,
        streams: Sequence[AudioStream],
    ) -> Tensor:
        streams = _bicodec_streams(streams)
        semantic_value, acoustic_value = _bicodec_units(value, streams)
        semantic = (
            self.encode(semantic_value)
            if semantic_value is not None
            else None
        )
        acoustic = (
            _acoustic_tensor(acoustic_value, self._acoustic_codebook_sizes)
            if acoustic_value is not None
            else None
        )
        if acoustic is not None and acoustic.size(0) != self._acoustic_unit_length:
            raise ValueError(
                "BiCodec acoustic units must match the configured fixed length."
            )
        if (
            semantic is not None
            and acoustic is not None
            and semantic.device != acoustic.device
        ):
            raise ValueError("BiCodec stream tensors must share a device.")
        anchor = semantic if semantic is not None else acoustic
        if anchor is None:
            raise AssertionError("BiCodec serialization has no requested stream payload.")
        values: list[Tensor] = []
        if AudioStream.ACOUSTIC in streams:
            if acoustic is None:
                raise AssertionError("BiCodec acoustic serialization has no payload.")
            values.append(anchor.new_tensor([self._acoustic_token_id]))
            for slot in range(self._acoustic_unit_length):
                for index, offset in enumerate(self._acoustic_offsets):
                    values.append(
                        acoustic[slot, index].reshape(1).to(dtype=torch.long) + offset
                    )
        if AudioStream.SEMANTIC in streams:
            if semantic is None:
                raise AssertionError("BiCodec semantic serialization has no payload.")
            values.extend(
                (
                    anchor.new_tensor([self._semantic_token_id]),
                    semantic.to(dtype=torch.long),
                )
            )
        values.append(anchor.new_tensor([self._end_token_id]))
        return torch.cat(values).to(dtype=torch.long)

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        tensor = token_tensor(token_ids)
        if bool((tensor >= self._semantic_vocab_size).any()):
            raise ValueError("BiCodec full sequences must use decode_full().")
        return self._semantic_tokenizer.decode(token_ids)

    def decode_full(self, token_ids: Sequence[int] | Tensor) -> SemanticAcousticCodes:
        decoded = self.decode_streams(
            token_ids,
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )
        if decoded.semantic is None or decoded.acoustic is None:
            raise AssertionError("full BiCodec decode must produce both streams.")
        return SemanticAcousticCodes(
            semantic=decoded.semantic,
            acoustic=decoded.acoustic,
        )

    def decode_streams(
        self,
        token_ids: Sequence[int] | Tensor,
        streams: Sequence[AudioStream],
    ) -> BiCodecStreams:
        streams = _bicodec_streams(streams)
        tensor = token_tensor(token_ids)
        if tensor.numel() < 2:
            raise ValueError("BiCodec stream sequence is too short.")
        if int(tensor[-1]) != self._end_token_id:
            raise ValueError("BiCodec stream sequence must end with a sequence marker.")

        cursor = 0
        semantic: Tensor | None = None
        acoustic: Tensor | None = None
        if AudioStream.ACOUSTIC in streams:
            if int(tensor[cursor]) != self._acoustic_token_id:
                raise ValueError(
                    "BiCodec stream sequence is missing the acoustic marker."
                )
            cursor += 1
            payload_length = self._acoustic_unit_length * len(
                self._acoustic_codebook_sizes
            )
            payload = tensor[cursor : cursor + payload_length]
            if payload.numel() != payload_length:
                raise ValueError(
                    "BiCodec stream sequence has an invalid acoustic payload length."
                )
            values = []
            payload_cursor = 0
            for _ in range(self._acoustic_unit_length):
                slot = []
                for offset, size in zip(
                    self._acoustic_offsets,
                    self._acoustic_codebook_sizes,
                ):
                    value = payload[payload_cursor] - offset
                    if bool((value < 0) or (value >= size)):
                        raise ValueError("BiCodec acoustic tokens are out of range.")
                    slot.append(value)
                    payload_cursor += 1
                values.append(torch.stack(slot))
            acoustic = torch.stack(values, dim=0)
            cursor += payload_length

        if AudioStream.SEMANTIC in streams:
            if int(tensor[cursor]) != self._semantic_token_id:
                raise ValueError(
                    "BiCodec stream sequence is missing the semantic marker."
                )
            cursor += 1
            semantic = tensor[cursor:-1]
            if semantic.numel() < 1 or bool((semantic < 0).any()) or bool(
                (semantic >= self._semantic_vocab_size).any()
            ):
                raise ValueError("BiCodec semantic tokens are out of range.")
            decoded = self._semantic_tokenizer.decode(semantic)
            if not isinstance(decoded, Tensor):
                raise AssertionError("Tensor semantic tokens must decode to a Tensor.")
            semantic = _semantic_tensor(decoded, self._semantic_codebook_size)
            cursor = tensor.numel() - 1

        if cursor != tensor.numel() - 1:
            raise ValueError("BiCodec stream sequence contains unexpected tokens.")
        return BiCodecStreams(semantic=semantic, acoustic=acoustic)

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor:
        tensor = token_tensor(token_ids)
        semantic = (tensor >= 0) & (tensor < self._semantic_vocab_size)
        spans = torch.zeros_like(tensor, dtype=torch.long)
        if bool(semantic.any()):
            semantic_spans = self._semantic_tokenizer.frame_spans(tensor[semantic])
            if not isinstance(semantic_spans, Tensor):
                raise AssertionError("Tensor semantic tokens must produce Tensor spans.")
            spans[semantic] = semantic_spans
        if isinstance(token_ids, Tensor):
            return spans.to(device=token_ids.device)
        return [int(value) for value in spans.tolist()]


def _validate_semantic_tokenizer(
    tokenizer: NativeAudioTokenizer | TorchCodecBPE,
    codebook_size: int,
) -> None:
    if isinstance(tokenizer, NativeAudioTokenizer):
        if tokenizer.vocab_size != codebook_size:
            raise ValueError(
                "BiCodec native semantic tokenizer vocabulary must match the "
                "semantic codebook size."
            )
        return
    if isinstance(tokenizer, TorchCodecBPE):
        if tuple(tokenizer.codebook_sizes) != (codebook_size,):
            raise ValueError(
                "BiCodec semantic BPE codebook sizes must match the raw semantic "
                "codebook."
            )
        return
    raise TypeError(
        "BiCodec semantic tokenizer must be NativeAudioTokenizer or TorchCodecBPE."
    )


def _bicodec_streams(
    streams: Sequence[AudioStream],
) -> tuple[AudioStream, ...]:
    values = tuple(streams)
    if not values:
        raise ValueError("BiCodec serialization requires at least one audio stream.")
    if any(not isinstance(stream, AudioStream) for stream in values):
        raise TypeError("BiCodec streams must contain AudioStream values.")
    unknown = set(values) - {AudioStream.ACOUSTIC, AudioStream.SEMANTIC}
    if unknown:
        labels = ", ".join(sorted(stream.value for stream in unknown))
        raise ValueError(f"BiCodec streams do not support: {labels}.")
    return tuple(
        stream
        for stream in (AudioStream.ACOUSTIC, AudioStream.SEMANTIC)
        if stream in values
    )


def _bicodec_units(
    value: SemanticAcousticCodes,
    streams: Sequence[AudioStream],
) -> tuple[Tensor | None, Tensor | None]:
    if not isinstance(value, SemanticAcousticCodes):
        raise TypeError("BiCodec stream input must be SemanticAcousticCodes.")
    semantic = value.semantic if AudioStream.SEMANTIC in streams else None
    acoustic = value.acoustic if AudioStream.ACOUSTIC in streams else None
    return semantic, acoustic


def _semantic_tensor(value: Sequence[Sequence[int]] | Tensor, vocab_size: int) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value
    else:
        tensor = torch.tensor(value, dtype=torch.long)
    validate_ids(tensor, "semantic codes")
    if tensor.dim() == 1:
        tensor = tensor[:, None]
    if tensor.dim() != 2 or tensor.size(1) != 1:
        raise ValueError("BiCodec semantic codes must have shape [time, 1].")
    validate_range(tensor, "semantic codes", vocab_size)
    return tensor.to(dtype=torch.long)


def _acoustic_tensor(value: Tensor, codebook_sizes: Sequence[int]) -> Tensor:
    validate_ids(value, "acoustic codes")
    if value.dim() != 2 or value.size(1) != len(codebook_sizes):
        raise ValueError("BiCodec acoustic codes must have shape [slots, codebooks].")
    tensor = value.to(dtype=torch.long)
    validate_frame_ranges(tensor, codebook_sizes)
    return tensor
