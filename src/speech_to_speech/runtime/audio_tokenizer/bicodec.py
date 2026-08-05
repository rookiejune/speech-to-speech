from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ...audio import AudioCodes, AudioStream
from ..audio_schema import AudioGrammarVariant, AudioTokenBlock, AudioTokenGrammar
from ._common import (
    codebook_size,
    token_tensor,
    validate_frame_ranges,
    validate_ids,
    validate_range,
)
from .bpe import TorchCodecBPE
from .native import NativeAudioTokenizer


class BiCodecAudioTokenizer:
    """Serialize BiCodec semantic units and fixed-length global speaker slots."""

    embedding_initialization = "random"

    def __init__(
        self,
        *,
        semantic_codebook_size: int,
        global_codebook_sizes: Sequence[int],
        global_unit_length: int,
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
        self._global_codebook_sizes = tuple(
            codebook_size(size) for size in global_codebook_sizes
        )
        if not self._global_codebook_sizes:
            raise ValueError("BiCodec tokenizer requires global codebooks.")
        if isinstance(global_unit_length, bool) or not isinstance(global_unit_length, int):
            raise TypeError("BiCodec global unit length must be an integer.")
        if global_unit_length <= 0:
            raise ValueError("BiCodec tokenizer requires a positive global unit length.")
        self._global_unit_length = global_unit_length
        offsets = [self._semantic_vocab_size]
        for size in self._global_codebook_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self._global_offsets = tuple(offsets)
        marker_base = sum((self._semantic_vocab_size, *self._global_codebook_sizes))
        self._semantic_token_id = marker_base
        self._global_token_id = marker_base + 1
        self._vocab_size = marker_base + 2
        global_block = AudioTokenBlock(
            name="global",
            marker_id=self._global_token_id,
            token_ranges=self.global_token_ranges,
            min_repeats=self._global_unit_length,
            max_repeats=self._global_unit_length,
        )
        semantic_block = AudioTokenBlock(
            name="semantic",
            marker_id=self._semantic_token_id,
            token_ranges=(self.semantic_token_range,),
        )
        self._grammar = AudioTokenGrammar(
            name="bicodec-streams-v1",
            variants=(
                AudioGrammarVariant(
                    name="global_semantic",
                    blocks=(global_block, semantic_block),
                ),
                AudioGrammarVariant(name="global", blocks=(global_block,)),
                AudioGrammarVariant(name="semantic", blocks=(semantic_block,)),
            ),
            default_variant="global_semantic",
            generation_variants=("global_semantic", "semantic"),
            prompt_continuations=(
                ("global", "semantic"),
                ("global_semantic", "semantic"),
            ),
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def grammar(self) -> AudioTokenGrammar:
        return self._grammar

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
    def global_codebook_sizes(self) -> tuple[int, ...]:
        return self._global_codebook_sizes

    @property
    def global_offsets(self) -> tuple[int, ...]:
        return self._global_offsets

    @property
    def global_unit_length(self) -> int:
        return self._global_unit_length

    @property
    def semantic_token_id(self) -> int:
        return self._semantic_token_id

    @property
    def global_token_id(self) -> int:
        return self._global_token_id

    @property
    def semantic_token_range(self) -> tuple[int, int]:
        return 0, self._semantic_vocab_size

    @property
    def global_token_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (offset, offset + size)
            for offset, size in zip(
                self._global_offsets,
                self._global_codebook_sizes,
            )
        )

    @property
    def prediction_token_ranges(self) -> tuple[tuple[int, int], ...]:
        return (self.semantic_token_range, *self.global_token_ranges)

    def contract_state(self) -> dict[str, object]:
        """Return the effective structured token-ID grammar used by checkpoints."""
        return {
            "grammar": "bicodec-v3",
            "semantic_codebook_size": self.semantic_codebook_size,
            "semantic_vocab_size": self.semantic_vocab_size,
            "semantic_tokenizer": dict(self.semantic_tokenizer.contract_state()),
            "semantic_token_range": list(self.semantic_token_range),
            "global_codebook_sizes": list(self.global_codebook_sizes),
            "global_offsets": list(self.global_offsets),
            "global_token_ranges": [
                list(bounds) for bounds in self.global_token_ranges
            ],
            "global_unit_length": self.global_unit_length,
            "semantic_token_id": self.semantic_token_id,
            "global_token_id": self.global_token_id,
            "vocab_size": self.vocab_size,
        }

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        """Encode semantic payload codes without structured stream markers."""
        token_ids = self._semantic_tokenizer.encode(frames)
        if isinstance(token_ids, Tensor):
            return token_ids.to(dtype=torch.long)
        return torch.tensor(token_ids, dtype=torch.long)

    def encode_full(self, value: AudioCodes) -> Tensor:
        return self.encode_streams(
            value,
            (AudioStream.GLOBAL, AudioStream.SEMANTIC),
        )

    def encode_global(self, value: AudioCodes) -> Tensor:
        """Encode only BiCodec's fixed-length global speaker/style codes."""
        return self.encode_streams(value, (AudioStream.GLOBAL,))

    def encode_streams(
        self,
        value: AudioCodes,
        streams: Sequence[AudioStream],
    ) -> Tensor:
        streams = _bicodec_streams(streams)
        semantic_value, global_value = _bicodec_units(value, streams)
        semantic = (
            self.encode(semantic_value)
            if semantic_value is not None
            else None
        )
        global_codes = (
            _global_tensor(global_value, self._global_codebook_sizes)
            if global_value is not None
            else None
        )
        if global_codes is not None and global_codes.size(0) != self._global_unit_length:
            raise ValueError(
                "BiCodec global units must match the configured fixed length."
            )
        if (
            semantic is not None
            and global_codes is not None
            and semantic.device != global_codes.device
        ):
            raise ValueError("BiCodec stream tensors must share a device.")
        anchor = semantic if semantic is not None else global_codes
        if anchor is None:
            raise AssertionError("BiCodec serialization has no requested stream payload.")
        values: list[Tensor] = []
        if AudioStream.GLOBAL in streams:
            if global_codes is None:
                raise AssertionError("BiCodec global serialization has no payload.")
            values.append(anchor.new_tensor([self._global_token_id]))
            for slot in range(self._global_unit_length):
                for index, offset in enumerate(self._global_offsets):
                    values.append(
                        global_codes[slot, index].reshape(1).to(dtype=torch.long) + offset
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
        return torch.cat(values).to(dtype=torch.long)

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        tensor = token_tensor(token_ids)
        if bool((tensor >= self._semantic_vocab_size).any()):
            raise ValueError("BiCodec full sequences must use decode_full().")
        return self._semantic_tokenizer.decode(token_ids)

    def decode_full(self, token_ids: Sequence[int] | Tensor) -> AudioCodes:
        decoded = self.decode_streams(token_ids)
        if decoded.semantic_codes is None or decoded.global_codes is None:
            raise AssertionError("full BiCodec decode must produce both streams.")
        return AudioCodes(
            semantic_codes=decoded.semantic_codes,
            global_codes=decoded.global_codes,
        )

    def decode_streams(
        self,
        token_ids: Sequence[int] | Tensor,
        streams: Sequence[AudioStream] | None = None,
    ) -> AudioCodes:
        tensor = token_tensor(token_ids)
        if tensor.numel() < 2:
            raise ValueError("BiCodec stream sequence is too short.")

        cursor = 0
        semantic: Tensor | None = None
        global_codes: Tensor | None = None
        if int(tensor[cursor]) == self._global_token_id:
            cursor += 1
            payload_length = self._global_unit_length * len(
                self._global_codebook_sizes
            )
            payload = tensor[cursor : cursor + payload_length]
            if payload.numel() != payload_length:
                raise ValueError(
                    "BiCodec stream sequence has an invalid global payload length."
                )
            values = []
            payload_cursor = 0
            for _ in range(self._global_unit_length):
                slot = []
                for offset, size in zip(
                    self._global_offsets,
                    self._global_codebook_sizes,
                ):
                    value = payload[payload_cursor] - offset
                    if bool((value < 0) or (value >= size)):
                        raise ValueError("BiCodec global tokens are out of range.")
                    slot.append(value)
                    payload_cursor += 1
                values.append(torch.stack(slot))
            global_codes = torch.stack(values, dim=0)
            cursor += payload_length

        if cursor < tensor.numel() and int(tensor[cursor]) == self._semantic_token_id:
            cursor += 1
            semantic = tensor[cursor:]
            if semantic.numel() < 1 or bool((semantic < 0).any()) or bool(
                (semantic >= self._semantic_vocab_size).any()
            ):
                raise ValueError("BiCodec semantic tokens are out of range.")
            decoded = self._semantic_tokenizer.decode(semantic)
            if not isinstance(decoded, Tensor):
                raise AssertionError("Tensor semantic tokens must decode to a Tensor.")
            semantic = _semantic_tensor(decoded, self._semantic_codebook_size)
            cursor = tensor.numel()

        if global_codes is None and semantic is None:
            raise ValueError(
                "BiCodec stream sequence must begin with a global or semantic marker."
            )
        if cursor != tensor.numel():
            raise ValueError("BiCodec stream sequence contains unexpected tokens.")
        decoded_streams = AudioCodes(
            semantic_codes=semantic,
            global_codes=global_codes,
        )
        if streams is not None:
            expected = frozenset(_bicodec_streams(streams))
            actual = frozenset(
                stream
                for stream, value in (
                    (AudioStream.GLOBAL, decoded_streams.global_codes),
                    (AudioStream.SEMANTIC, decoded_streams.semantic_codes),
                )
                if value is not None
            )
            if actual != expected:
                raise ValueError(
                    "BiCodec stream sequence markers do not match the expected streams."
                )
        return decoded_streams

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
    unknown = set(values) - {AudioStream.GLOBAL, AudioStream.SEMANTIC}
    if unknown:
        labels = ", ".join(sorted(stream.value for stream in unknown))
        raise ValueError(f"BiCodec streams do not support: {labels}.")
    return tuple(
        stream
        for stream in (AudioStream.GLOBAL, AudioStream.SEMANTIC)
        if stream in values
    )


def _bicodec_units(
    value: AudioCodes,
    streams: Sequence[AudioStream],
) -> tuple[Tensor | None, Tensor | None]:
    if not isinstance(value, AudioCodes):
        raise TypeError("BiCodec stream input must be AudioCodes.")
    semantic = value.semantic_codes if AudioStream.SEMANTIC in streams else None
    global_codes = value.global_codes if AudioStream.GLOBAL in streams else None
    return semantic, global_codes


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


def _global_tensor(value: Tensor, codebook_sizes: Sequence[int]) -> Tensor:
    validate_ids(value, "global codes")
    if value.dim() != 2 or value.size(1) != len(codebook_sizes):
        raise ValueError("BiCodec global codes must have shape [slots, codebooks].")
    tensor = value.to(dtype=torch.long)
    validate_frame_ranges(tensor, codebook_sizes)
    return tensor
