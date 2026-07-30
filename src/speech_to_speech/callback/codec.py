from __future__ import annotations

import torch
from anytrain.codec import SemanticAcousticCodes
from anydataset.types import AudioView
from torch import Tensor

from ..datamodule.protocol import DatasetRuntime
from ..datamodule.parser import speech_from_codes
from ..datamodule.sample import build_task_sample
from ..datamodule.types import (
    ModelBatch,
    RawSpeech,
    RawSpeechBatch,
    Speech,
    SpeechTaskSample,
    Text,
    TrainInput,
)
from ..runtime.types import frame_codec, structured_codec, supports_structured


class OnDeviceCodecMaterializer:
    """Materialize explicit raw waveform fallback batches before loss computation."""

    def __init__(self, runtime: DatasetRuntime) -> None:
        self.runtime = runtime

    @torch.no_grad()
    def __call__(
        self,
        batch: TrainInput,
        *,
        device: torch.device | None = None,
    ) -> ModelBatch:
        return self._concrete(batch, device=device)

    def _concrete(
        self,
        batch: TrainInput,
        *,
        device: torch.device | None,
    ) -> ModelBatch:
        if isinstance(batch, ModelBatch):
            return _move_model_batch(batch, device)
        if isinstance(batch, RawSpeechBatch):
            return self._raw_speech(batch, device=device)
        raise TypeError(f"unsupported train batch: {type(batch).__name__}")

    def _raw_speech(
        self,
        batch: RawSpeechBatch,
        *,
        device: torch.device | None,
    ) -> ModelBatch:
        samples = [
            build_task_sample(self._task_sample(sample, device=device), self.runtime)
            for sample in batch.samples
        ]
        return _move_model_batch(
            ModelBatch.from_samples(samples, pad_token_id=batch.pad_token_id),
            device,
        )

    def _task_sample(
        self,
        sample: SpeechTaskSample,
        *,
        device: torch.device | None,
    ) -> SpeechTaskSample:
        source = self._item(sample.source, device=device)
        target = self._item(sample.target, device=device)
        audio_context = self._item(sample.audio_context, device=device)
        if target is None:
            raise AssertionError("speech task target must not be None.")
        if isinstance(audio_context, Text):
            raise AssertionError("audio context materialization returned text.")
        return SpeechTaskSample(
            source=source,
            target=target,
            task=sample.task,
            prediction=sample.prediction,
            audio_context=audio_context,
        )

    def _item(
        self,
        item: Speech | Text | RawSpeech | None,
        *,
        device: torch.device | None,
    ) -> Speech | Text | None:
        if not isinstance(item, RawSpeech):
            return item
        codes = self._encode(item, device=device)
        return speech_from_codes(
            codes,
            text_token_ids=item.text_token_ids.cpu(),
            language=item.language,
            duration_seconds=item.duration_seconds,
            runtime=self.runtime,
        )

    def _encode(
        self,
        sample: RawSpeech,
        *,
        device: torch.device | None,
    ) -> object:
        # Codec backends are materialization dependencies, not part of the model's
        # mixed-precision graph. Keep their input and internal kernels in FP32.
        waveform = sample.waveform.to(dtype=torch.float32)
        if device is not None:
            waveform = waveform.to(device=device)
        batched_waveform = _batched_waveform(waveform)
        with torch.autocast(
            device_type=batched_waveform.device.type,
            enabled=False,
        ):
            if self.runtime.audio_view is AudioView.BICODEC:
                if not supports_structured(self.runtime.codec):
                    raise TypeError(
                        "BiCodec waveform fallback requires a structured codec."
                    )
                encoded = structured_codec(self.runtime.codec).tokenize(
                    batched_waveform,
                    sample.sample_rate,
                )
                if not isinstance(encoded, SemanticAcousticCodes):
                    raise TypeError(
                        "structured codec tokenize must return SemanticAcousticCodes."
                    )
                if encoded.semantic.size(0) != 1 or encoded.acoustic.size(0) != 1:
                    raise ValueError("per-sample codec fallback expects one encoded item.")
                return SemanticAcousticCodes(
                    semantic=encoded.semantic[0].detach().cpu(),
                    acoustic=encoded.acoustic[0].detach().cpu(),
                )
            return _encoded_codes(
                frame_codec(self.runtime.codec).encode(
                    batched_waveform,
                    sample.sample_rate,
                )
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
    return batch if device is None else batch.to(device)
