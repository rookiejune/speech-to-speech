from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from anydataset.dataset.collate import Batch
from anydataset.provider.abc import AudioProvider
from anydataset.types import (
    AudioView,
    Modality,
    Role,
    SemanticGlobalView,
)
from anytrain.codec import (
    SemanticGlobalCodec,
    SemanticGlobalCodes,
)
from torch import Tensor, nn


class BiCodecProvider(nn.Module, AudioProvider):
    """Materialize semantic/global BiCodec units in anydataset's store schema."""

    output = AudioView.BICODEC

    def __init__(self, codec: SemanticGlobalCodec) -> None:
        super().__init__()
        if codec.global_unit_length <= 0:
            raise ValueError("BiCodec materialization requires global units.")
        self.codec = codec
        if isinstance(codec, nn.Module):
            nn.Module.eval(codec)

    @torch.inference_mode()
    def forward(self, views: Mapping[AudioView, Any]) -> SemanticGlobalView:
        waveform, sample_rate = self._audio_batch(views)
        codes = _codes(self.codec.tokenize(waveform, sample_rate), self.codec)
        if codes.semantic.shape[0] != 1:
            raise ValueError("single-sample BiCodec provider expects one codec result.")
        return self._view(codes, 0)

    @torch.inference_mode()
    def call_batch(
        self,
        batch: Batch,
    ) -> (
        Sequence[SemanticGlobalView]
        | Mapping[tuple[Role, Modality], Sequence[SemanticGlobalView]]
    ):
        refs = _audio_refs(batch)
        outputs = {ref: self._encode_ref_batch(batch, ref) for ref in refs}
        if len(refs) == 1:
            return outputs[refs[0]]
        return outputs

    def _encode_ref_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> Sequence[SemanticGlobalView]:
        waveform, sample_rates, lengths = self._waveform_batch(batch, ref)
        if waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        sample_rate = _sample_rate(sample_rates)
        outputs: dict[int, SemanticGlobalView] = {}
        for length, indexes in self._length_groups(lengths):
            clipped = waveform[list(indexes), ..., :length].contiguous()
            codes = _codes(self.codec.tokenize(clipped, sample_rate), self.codec)
            if codes.semantic.shape[0] != len(indexes):
                raise ValueError(
                    "BiCodec tokenize must return one output per input waveform."
                )
            outputs.update(
                (sample_index, self._view(codes, batch_index))
                for batch_index, sample_index in enumerate(indexes)
            )
        return [outputs[index] for index in range(len(lengths))]

    def _audio_batch(
        self,
        views: Mapping[AudioView, Any],
    ) -> tuple[Tensor, int]:
        waveform, sample_rate = self._waveform(views)
        waveform = (
            waveform
            if isinstance(waveform, Tensor)
            else torch.as_tensor(waveform)
        )
        if waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError("BiCodec waveform input must have shape [channel, time].")
        return waveform.unsqueeze(0), sample_rate

    def _view(self, codes: SemanticGlobalCodes, index: int) -> SemanticGlobalView:
        return {
            "semantic": self._tensor(codes.semantic[index]),
            "global": self._tensor(codes.global_codes[index]),
        }


def _audio_refs(batch: Batch) -> tuple[tuple[Role, Modality], ...]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.AUDIO
        and (
            AudioView.WAVEFORM in batch.sample[ref].views
            or AudioView.FILE in batch.sample[ref].views
        )
    )
    if not refs:
        raise ValueError("BiCodecProvider expects at least one waveform audio input.")
    return refs


def _sample_rate(values: Tensor) -> int:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("BiCodec sample rates must be a non-empty vector.")
    first = values[0].item()
    if not torch.equal(values, values.new_full(values.shape, first)):
        raise ValueError("BiCodec materialization requires one sample rate per batch.")
    return int(first)


def _codes(
    value: object,
    codec: SemanticGlobalCodec,
) -> SemanticGlobalCodes:
    if not isinstance(value, SemanticGlobalCodes):
        raise TypeError("BiCodec tokenize() must return SemanticGlobalCodes.")
    semantic = _units(
        value.semantic,
        name="semantic",
        codebook_sizes=codec.semantic_codebook_sizes,
    )
    global_codes = _units(
        value.global_codes,
        name="global",
        codebook_sizes=codec.global_codebook_sizes,
    )
    if semantic.shape[0] != global_codes.shape[0]:
        raise ValueError("BiCodec semantic and global units must share the batch axis.")
    unit_length = codec.global_unit_length
    if global_codes.shape[1] != unit_length:
        raise ValueError(
            "BiCodec global units must use the configured fixed unit length "
            f"{unit_length}, got {global_codes.shape[1]}."
        )
    return SemanticGlobalCodes(semantic=semantic, global_codes=global_codes)


def _units(
    value: object,
    *,
    name: str,
    codebook_sizes: tuple[int, ...],
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"BiCodec {name} units must be a Tensor.")
    if value.ndim != 3 or min(value.shape) < 1:
        raise ValueError(
            f"BiCodec {name} units must have non-empty [batch, unit, codebook] shape."
        )
    if value.shape[-1] != len(codebook_sizes):
        raise ValueError(
            f"BiCodec {name} units must contain all configured "
            f"{len(codebook_sizes)} codebooks."
        )
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TypeError(f"BiCodec {name} units must contain integer ids.")
    minimum = value.amin(dim=(0, 1))
    maximum = value.amax(dim=(0, 1))
    limits = torch.as_tensor(codebook_sizes, dtype=torch.int64, device=value.device)
    invalid = (minimum < 0) | (maximum >= limits)
    if invalid.any().item():
        observed = torch.stack((minimum, maximum), dim=1).cpu().tolist()
        details = "; ".join(
            f"codebook {index} observed [{low}, {high}], expected [0, {size})"
            for index, ((low, high), size) in enumerate(
                zip(observed, codebook_sizes)
            )
            if low < 0 or high >= size
        )
        raise ValueError(f"BiCodec {name} ids are outside configured ranges: {details}.")
    return value


__all__ = ["BiCodecProvider"]
