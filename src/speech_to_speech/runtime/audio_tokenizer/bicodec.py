from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import torch
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from ...audio_route import AudioStream
from ._common import (
    codebook_size,
    token_tensor,
    validate_frame_ranges,
    validate_ids,
    validate_range,
)


@dataclass(frozen=True)
class BiCodecStreams:
    semantic: Tensor | None
    acoustic: Tensor | None


class BiCodecAudioTokenizer:
    """Serialize BiCodec semantic units and fixed speaker/global slots."""

    embedding_initialization = "random"
    forced_group = -1
    semantic_group = 0
    semantic_or_end_group = 1
    acoustic_group_start = 2

    def __init__(
        self,
        *,
        semantic_vocab_size: int,
        acoustic_codebook_sizes: Sequence[int],
        acoustic_unit_length: int | None,
    ) -> None:
        self._semantic_vocab_size = codebook_size(semantic_vocab_size)
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
    def acoustic_codebook_sizes(self) -> tuple[int, ...]:
        return self._acoustic_codebook_sizes

    @property
    def global_codebook_sizes(self) -> tuple[int, ...]:
        """Return the fixed-length BiCodec global codebook sizes."""
        return self._acoustic_codebook_sizes

    @property
    def acoustic_offsets(self) -> tuple[int, ...]:
        return self._acoustic_offsets

    @property
    def acoustic_unit_length(self) -> int:
        return self._acoustic_unit_length

    @property
    def global_unit_length(self) -> int:
        return self._acoustic_unit_length

    @property
    def semantic_token_id(self) -> int:
        return self._semantic_token_id

    @property
    def acoustic_token_id(self) -> int:
        return self._acoustic_token_id

    @property
    def global_token_id(self) -> int:
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
    def global_token_ranges(self) -> tuple[tuple[int, int], ...]:
        return self.acoustic_token_ranges

    @property
    def prediction_token_ranges(self) -> tuple[tuple[int, int], ...]:
        return (self.semantic_token_range, *self.acoustic_token_ranges)

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        """Encode semantic codes for the semantic-only route."""
        tensor = _semantic_tensor(frames, self._semantic_vocab_size)
        return tensor[:, 0].to(dtype=torch.long)

    def encode_full(self, value: SemanticAcousticCodes) -> Tensor:
        return self.encode_streams(
            value,
            (AudioStream.GLOBAL, AudioStream.SEMANTIC),
        )

    def encode_global(self, value: SemanticAcousticCodes) -> Tensor:
        """Encode only BiCodec's fixed-length global speaker/style codes."""
        return self.encode_streams(value, (AudioStream.GLOBAL,))

    def encode_streams(
        self,
        value: SemanticAcousticCodes,
        streams: Sequence[AudioStream],
    ) -> Tensor:
        streams = _bicodec_streams(streams)
        semantic_value, acoustic_value = _bicodec_units(value, streams)
        semantic = (
            _semantic_tensor(semantic_value, self._semantic_vocab_size)
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
                "BiCodec global units must match the configured fixed length."
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
        if AudioStream.GLOBAL in streams:
            if acoustic is None:
                raise AssertionError("BiCodec global serialization has no payload.")
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
                    semantic[:, 0].to(dtype=torch.long),
                )
            )
        values.append(anchor.new_tensor([self._end_token_id]))
        return torch.cat(values).to(dtype=torch.long)

    def encode_streams_with_groups(
        self,
        value: SemanticAcousticCodes,
        streams: Sequence[AudioStream],
    ) -> tuple[Tensor, Tensor]:
        streams = _bicodec_streams(streams)
        token_ids = self.encode_streams(value, streams)
        groups = token_ids.new_full(token_ids.shape, self.forced_group)
        cursor = 0
        if AudioStream.GLOBAL in streams:
            cursor += 1
            for _ in range(self._acoustic_unit_length):
                for codebook in range(len(self._acoustic_codebook_sizes)):
                    groups[cursor] = self.acoustic_group_start + codebook
                    cursor += 1
        if AudioStream.SEMANTIC in streams:
            cursor += 1
            semantic_count = token_ids.numel() - cursor - 1
            if semantic_count < 1:
                raise AssertionError("BiCodec semantic serialization produced no payload.")
            groups[cursor] = self.semantic_group
            if semantic_count > 1:
                groups[cursor + 1 : cursor + semantic_count] = (
                    self.semantic_or_end_group
                )
            groups[-1] = self.semantic_or_end_group
        return token_ids, groups

    def prediction_ids(self, group: int, *, device: torch.device) -> Tensor:
        if isinstance(group, bool) or not isinstance(group, int):
            raise TypeError("BiCodec prediction group must be an integer.")
        if group == self.semantic_group:
            start, end = self.semantic_token_range
            return torch.arange(start, end, dtype=torch.long, device=device)
        if group == self.semantic_or_end_group:
            start, end = self.semantic_token_range
            semantic = torch.arange(start, end, dtype=torch.long, device=device)
            return torch.cat((semantic, semantic.new_tensor([self._end_token_id])))
        codebook = group - self.acoustic_group_start
        if codebook < 0 or codebook >= len(self.acoustic_token_ranges):
            raise ValueError(f"unknown BiCodec prediction group: {group}.")
        start, end = self.acoustic_token_ranges[codebook]
        return torch.arange(start, end, dtype=torch.long, device=device)

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        tensor = token_tensor(token_ids)
        if bool((tensor >= self._semantic_vocab_size).any()):
            raise ValueError("BiCodec full sequences must use decode_full().")
        return (
            tensor[:, None]
            if isinstance(token_ids, Tensor)
            else [(int(value),) for value in tensor]
        )

    def decode_full(self, token_ids: Sequence[int] | Tensor) -> SemanticAcousticCodes:
        decoded = self.decode_streams(
            token_ids,
            (AudioStream.GLOBAL, AudioStream.SEMANTIC),
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
        if AudioStream.GLOBAL in streams:
            if int(tensor[cursor]) != self._acoustic_token_id:
                raise ValueError("BiCodec stream sequence is missing the global marker.")
            cursor += 1
            payload_length = self._acoustic_unit_length * len(
                self._acoustic_codebook_sizes
            )
            payload = tensor[cursor : cursor + payload_length]
            if payload.numel() != payload_length:
                raise ValueError(
                    "BiCodec stream sequence has an invalid global payload length."
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
                        raise ValueError("BiCodec global tokens are out of range.")
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
            semantic = semantic[:, None]
            cursor = tensor.numel() - 1

        if cursor != tensor.numel() - 1:
            raise ValueError("BiCodec stream sequence contains unexpected tokens.")
        return BiCodecStreams(semantic=semantic, acoustic=acoustic)

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor:
        tensor = token_tensor(token_ids)
        spans = ((tensor >= 0) & (tensor < self._semantic_vocab_size)).to(dtype=torch.long)
        if isinstance(token_ids, Tensor):
            return spans.to(device=token_ids.device)
        return [int(value) for value in spans.tolist()]


def _bicodec_streams(
    streams: Sequence[AudioStream],
) -> tuple[AudioStream, ...]:
    values = tuple(streams)
    if not values:
        raise ValueError("BiCodec serialization requires at least one audio stream.")
    if any(not isinstance(stream, AudioStream) for stream in values):
        raise TypeError("BiCodec streams must contain AudioStream values.")
    if AudioStream.ACOUSTIC in values:
        raise ValueError("BiCodec streams must use global instead of acoustic.")
    return tuple(
        stream
        for stream in (AudioStream.GLOBAL, AudioStream.SEMANTIC)
        if stream in values
    )


def _bicodec_units(
    value: SemanticAcousticCodes,
    streams: Sequence[AudioStream],
) -> tuple[Tensor | None, Tensor | None]:
    if not isinstance(value, SemanticAcousticCodes):
        raise TypeError("BiCodec stream input must be SemanticAcousticCodes.")
    semantic = value.semantic if AudioStream.SEMANTIC in streams else None
    acoustic = value.acoustic if AudioStream.GLOBAL in streams else None
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
