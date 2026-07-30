from __future__ import annotations

from collections.abc import Sequence
from typing import cast, overload

import torch
from anytrain.tokenizer import CodecBPE
from torch import Tensor

from ._common import validate_ids


class TorchCodecBPE(CodecBPE):
    """CodecBPE with tensor conveniences for model/runtime integration."""

    embedding_initialization = "codec"

    @classmethod
    def wrap(cls, tokenizer: CodecBPE) -> TorchCodecBPE:
        if isinstance(tokenizer, cls):
            return tokenizer
        return cls(tokenizer._core, tokenizer._codec)

    @overload
    def encode(self, frames: Sequence[Sequence[int]]) -> list[int]: ...

    @overload
    def encode(self, frames: Tensor) -> Tensor: ...

    def encode(
        self,
        frames: Sequence[Sequence[int]] | Tensor,
    ) -> list[int] | Tensor:
        if not isinstance(frames, Tensor):
            return super().encode(frames)
        token_ids = super().encode(_frames(frames, self.codebook_sizes))
        return torch.tensor(token_ids, dtype=torch.long, device=frames.device)

    @overload
    def decode(self, token_ids: Sequence[int]) -> list[tuple[int, ...]]: ...

    @overload
    def decode(self, token_ids: Tensor) -> Tensor: ...

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor:
        if not isinstance(token_ids, Tensor):
            return super().decode(token_ids)
        frames = super().decode(_ids(token_ids))
        return torch.tensor(frames, dtype=torch.long, device=token_ids.device)

    @overload
    def frame_spans(self, token_ids: Sequence[int]) -> list[int]: ...

    @overload
    def frame_spans(self, token_ids: Tensor) -> Tensor: ...

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor:
        values = _ids(token_ids) if isinstance(token_ids, Tensor) else token_ids
        spans = [len(self._core.tokens[int(token_id)]) for token_id in values]
        if isinstance(token_ids, Tensor):
            return token_ids.new_tensor(spans, dtype=torch.long)
        return spans


def _frames(frames: Tensor, codebook_sizes: Sequence[int]) -> list[list[int]]:
    validate_ids(frames, "frames")
    if frames.dim() == 1:
        if len(codebook_sizes) != 1:
            raise ValueError(
                "1D frame tensors are only valid for single-codebook tokenizers."
            )
        return [[value] for value in frames.detach().cpu().tolist()]
    if frames.dim() != 2:
        raise ValueError("frame tensor must have shape [frames, codebooks].")
    if frames.size(-1) != len(codebook_sizes):
        raise ValueError("frame tensor must match tokenizer codebook count.")
    return cast(list[list[int]], frames.detach().cpu().tolist())


def _ids(ids: Tensor) -> list[int]:
    validate_ids(ids, "token ids")
    if ids.dim() != 1:
        raise ValueError("token id tensor must have shape [tokens].")
    return ids.detach().cpu().tolist()
