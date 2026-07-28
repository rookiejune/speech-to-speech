from __future__ import annotations

import torch
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from ..datamodule.protocol import DatasetRuntime
from ..datamodule.single import build_single_sample_from_codes
from ..datamodule.types import (
    AcousticTarget,
    ConcreteTrainInput,
    ModelBatch,
    RawSingleBatch,
    TrainBatch,
    TrainInputBatch,
)
from ..runtime.types import frame_codec, structured_codec, supports_structured


class OnDeviceCodecMaterializer:
    """Materialize explicit raw waveform fallback batches before loss computation."""

    def __init__(self, runtime: DatasetRuntime) -> None:
        self.runtime = runtime

    @torch.no_grad()
    def __call__(
        self,
        batch: TrainInputBatch,
        *,
        device: torch.device | None = None,
    ) -> TrainBatch:
        if isinstance(batch, tuple):
            return tuple(self._concrete(item, device=device) for item in batch)
        return self._concrete(batch, device=device)

    def _concrete(
        self,
        batch: ConcreteTrainInput,
        *,
        device: torch.device | None,
    ) -> ModelBatch:
        if isinstance(batch, ModelBatch):
            return _move_model_batch(batch, device)
        if isinstance(batch, RawSingleBatch):
            return self._raw_single(batch, device=device)
        raise TypeError(f"unsupported train batch: {type(batch).__name__}")

    def _raw_single(
        self,
        batch: RawSingleBatch,
        *,
        device: torch.device | None,
    ) -> ModelBatch:
        samples = []
        for sample in batch.samples:
            waveform = sample.waveform
            if device is not None:
                waveform = waveform.to(device=device)
            batched_waveform = _batched_waveform(waveform)
            if supports_structured(self.runtime.codec):
                encoded = structured_codec(self.runtime.codec).tokenize(
                    batched_waveform,
                    sample.sample_rate,
                )
                if not isinstance(encoded, SemanticAcousticCodes):
                    raise TypeError("structured codec tokenize must return SemanticAcousticCodes.")
                if encoded.semantic.size(0) != 1 or encoded.acoustic.size(0) != 1:
                    raise ValueError("per-sample codec fallback expects one encoded item.")
                codes: object = SemanticAcousticCodes(
                    semantic=encoded.semantic[0].detach().cpu(),
                    acoustic=encoded.acoustic[0].detach().cpu(),
                )
            else:
                codes = _encoded_codes(
                    frame_codec(self.runtime.codec).encode(
                        batched_waveform,
                        sample.sample_rate,
                    )
                )
            samples.append(build_single_sample_from_codes(sample, codes, self.runtime))
        return _move_model_batch(
            ModelBatch.from_samples(samples, pad_token_id=batch.pad_token_id),
            device,
        )


def _batched_waveform(waveform: Tensor) -> Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError("raw fallback waveform must have shape [time] or [channel, time].")
    return waveform.unsqueeze(0)


def _encoded_codes(codes: Tensor) -> Tensor:
    if not isinstance(codes, Tensor):
        raise TypeError("codec encode must return a Tensor.")
    if codes.dim() == 3:
        if codes.size(0) != 1:
            raise ValueError("per-sample codec fallback expects one encoded item.")
        return codes[0].detach().cpu()
    if codes.dim() == 2:
        return codes.detach().cpu()
    raise ValueError("codec encode must return [frames, codebooks] or [1, frames, codebooks].")


def _move_model_batch(batch: ModelBatch, device: torch.device | None) -> ModelBatch:
    if device is None:
        return batch
    return ModelBatch(
        input_ids=batch.input_ids.to(device=device),
        token_labels=batch.token_labels.to(device=device),
        acoustic_target=_move_target(batch.acoustic_target, device),
        tasks=list(batch.tasks),
        pad_token_id=batch.pad_token_id,
        audio_seconds=(
            None if batch.audio_seconds is None else batch.audio_seconds.to(device=device)
        ),
    )


def _move_target(
    target: AcousticTarget | None,
    device: torch.device,
) -> AcousticTarget | None:
    if target is None:
        return None
    return AcousticTarget(
        semantic_codes=target["semantic_codes"].to(device=device),
        codes=target["codes"].to(device=device),
        token_positions=target["token_positions"].to(device=device),
    )
