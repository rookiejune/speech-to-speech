from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ..audio_schema import AudioGrammarVariant, AudioTokenBlock, AudioTokenGrammar
from ._common import (
    codebook_size,
    frame_tensor,
    token_tensor,
    validate_frame_ranges,
    validate_range,
)


class FlattenedAudioTokenizer:
    """Flatten fixed-width codec codebooks into one audio token sequence."""

    embedding_initialization = "random"

    def __init__(self, *, codebook_sizes: Sequence[int], codec_name: str) -> None:
        if not codebook_sizes:
            raise ValueError("flattened audio tokenizer requires codebook sizes.")
        if not codec_name:
            raise ValueError("flattened audio tokenizer requires a codec name.")
        sizes = [codebook_size(size) for size in codebook_sizes]
        self._codec_name = codec_name
        self._codebook_sizes = tuple(sizes)
        offsets = [0]
        for size in sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self._offsets = tuple(offsets)
        self._code_vocab_size = sum(sizes)
        self._codebook_token_ids = tuple(
            self._code_vocab_size + index for index in range(len(sizes))
        )
        self._vocab_size = self._code_vocab_size + len(sizes)
        block_names = tuple(
            f"codebook_{index}" for index in range(len(self._codebook_sizes))
        )
        self._grammar = AudioTokenGrammar(
            name="flattened-codebook-blocks-v1",
            variants=(
                AudioGrammarVariant(
                    name="full",
                    blocks=tuple(
                        AudioTokenBlock(
                            name=name,
                            marker_id=marker,
                            token_ranges=(bounds,),
                        )
                        for name, marker, bounds in zip(
                            block_names,
                            self._codebook_token_ids,
                            self.codebook_ranges,
                        )
                    ),
                    equal_repeat_groups=(block_names,) if len(block_names) > 1 else (),
                ),
            ),
            default_variant="full",
            generation_variants=("full",),
        )

    @property
    def codec_name(self) -> str:
        return self._codec_name

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return self._codebook_sizes

    @property
    def codebook_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (offset, offset + size)
            for offset, size in zip(self._offsets, self.codebook_sizes)
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def grammar(self) -> AudioTokenGrammar:
        return self._grammar

    @property
    def codebook_token_ids(self) -> tuple[int, ...]:
        return self._codebook_token_ids

    @property
    def special_tokens(self) -> dict[str, int]:
        return {
            f"codec:{self.codec_name}:codebook:{index}": token_id
            for index, token_id in enumerate(self.codebook_token_ids)
        }

    def contract_state(self) -> dict[str, object]:
        """Return the effective flattened token-ID grammar used by checkpoints."""
        return {
            "grammar": "flattened-v1",
            "codec_name": self.codec_name,
            "codebook_sizes": list(self.codebook_sizes),
            "codebook_ranges": [list(bounds) for bounds in self.codebook_ranges],
            "codebook_token_ids": list(self.codebook_token_ids),
            "vocab_size": self.vocab_size,
        }

    def encode(self, frames: Sequence[Sequence[int]] | Tensor) -> Tensor:
        tensor = frame_tensor(frames, self.codebook_sizes)
        validate_frame_ranges(tensor, self.codebook_sizes)
        values = []
        for index, offset in enumerate(self._offsets):
            values.append(tensor.new_tensor([self.codebook_token_ids[index]]))
            values.append(tensor[:, index] + offset)
        return torch.cat(values).to(dtype=torch.long)

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        tensor = token_tensor(token_ids)
        payload = flattened_payload(tensor, self)
        if payload.numel() % len(self.codebook_sizes) != 0:
            raise ValueError(
                "flattened token sequence length must be divisible by codebook count."
            )
        frames = payload.reshape(len(self.codebook_sizes), -1).transpose(0, 1).clone()
        offsets = frames.new_tensor(self._offsets)
        frames -= offsets
        validate_frame_ranges(frames, self.codebook_sizes)
        if isinstance(token_ids, Tensor):
            return frames.to(device=token_ids.device, dtype=torch.long)
        return [tuple(int(value) for value in row) for row in frames.tolist()]

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor:
        tensor = token_tensor(token_ids)
        validate_flattened_sequence(tensor, self)
        spans = flattened_frame_spans(tensor, self)
        if isinstance(token_ids, Tensor):
            return spans.to(device=token_ids.device, dtype=torch.long)
        return [int(value) for value in spans.tolist()]


def flattened_payload(token_ids: Tensor, tokenizer: FlattenedAudioTokenizer) -> Tensor:
    if token_ids.numel() < 2 * len(tokenizer.codebook_sizes):
        raise ValueError("flattened token sequence is missing codec codebook markers.")
    payloads = []
    index = 0
    expected_frames: int | None = None
    for codebook, marker in enumerate(tokenizer.codebook_token_ids):
        if index >= token_ids.numel() or int(token_ids[index].item()) != marker:
            raise ValueError(
                f"flattened token sequence is missing codebook {codebook} marker."
            )
        index += 1
        next_markers = set(tokenizer.codebook_token_ids[codebook + 1 :])
        end = index
        while end < token_ids.numel() and int(token_ids[end].item()) not in next_markers:
            end += 1
        values = token_ids[index:end]
        if values.numel() == 0:
            raise ValueError("flattened codebook blocks must not be empty.")
        if expected_frames is None:
            expected_frames = values.numel()
        elif values.numel() != expected_frames:
            raise ValueError("flattened codebook blocks must have equal lengths.")
        validate_range(
            values - tokenizer._offsets[codebook],
            f"codebook {codebook} token ids",
            tokenizer.codebook_sizes[codebook],
        )
        payloads.append(values)
        index = end
    if index != token_ids.numel():
        raise ValueError("flattened token sequence has trailing unknown markers.")
    return torch.cat(payloads)


def validate_flattened_sequence(
    token_ids: Tensor,
    tokenizer: FlattenedAudioTokenizer,
) -> None:
    if _is_flattened_vocab_range(token_ids, tokenizer):
        return
    flattened_payload(token_ids, tokenizer)


def flattened_frame_spans(
    token_ids: Tensor,
    tokenizer: FlattenedAudioTokenizer,
) -> Tensor:
    spans = torch.zeros_like(token_ids, dtype=torch.long)
    first_start = tokenizer._offsets[0]
    first_end = first_start + tokenizer.codebook_sizes[0]
    spans[(token_ids >= first_start) & (token_ids < first_end)] = 1
    return spans


def _is_flattened_vocab_range(
    token_ids: Tensor,
    tokenizer: FlattenedAudioTokenizer,
) -> bool:
    return token_ids.numel() == tokenizer.vocab_size and torch.equal(
        token_ids.cpu(),
        torch.arange(tokenizer.vocab_size, dtype=token_ids.dtype),
    )
