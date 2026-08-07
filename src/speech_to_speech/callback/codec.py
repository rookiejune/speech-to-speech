from __future__ import annotations

import torch
from anytrain.codec import SemanticGlobalCodes
from anydataset.types import AudioView
from torch import Tensor

from ..audio import AudioCodes, AudioStream
from ..datamodule.contract import DatasetRuntime
from ..datamodule.parse import speech_from_codes
from ..datamodule.builder import build_task_sample
from ..datamodule.batch import (
    ModelBatch,
    TrainInput,
)
from ..datamodule.sample import (
    RawSpeech,
    RawSpeechBatch,
    Speech,
    SpeechTaskSample,
    Text,
)
from ..runtime.codec_contract import (
    frame_tokenizer,
    global_codec,
    supports_global,
)


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
            build_task_sample(
                self._task_sample(sample, device=device),
                self.runtime,
                interleave_audio_frames=batch.interleave_audio_frames,
                mask_text_ratio=batch.mask_text_ratio,
                mask_audio_ratio=batch.mask_audio_ratio,
                ar_framing=batch.ar_framing,
            )
            for sample in batch.samples
        ]
        return _move_model_batch(
            ModelBatch.from_samples(
                samples,
                pad_token_id=batch.pad_token_id,
                layout=self.runtime.layout,
            ),
            device,
        )

    def _task_sample(
        self,
        sample: SpeechTaskSample,
        *,
        device: torch.device | None,
    ) -> SpeechTaskSample:
        source = self._item(sample.source, device=device, input_audio=True)
        target = self._item(sample.target, device=device, input_audio=False)
        audio_context = self._item(
            sample.audio_context,
            device=device,
            input_audio=False,
        )
        if target is None:
            raise AssertionError("speech task target must not be None.")
        if isinstance(audio_context, Text):
            raise AssertionError("audio context materialization returned text.")
        return SpeechTaskSample(
            source=source,
            target=target,
            task=sample.task,
            trace=sample.trace,
            audio_context=audio_context,
        )

    def _item(
        self,
        item: Speech | Text | RawSpeech | None,
        *,
        device: torch.device | None,
        input_audio: bool,
    ) -> Speech | Text | None:
        if not isinstance(item, RawSpeech):
            return item
        codes = self._encode(item, device=device, input_audio=input_audio)
        return speech_from_codes(
            codes,
            text_token_ids=item.text_token_ids.cpu(),
            language=item.language,
            duration_seconds=item.duration_seconds,
            runtime=self.runtime,
            input_audio=input_audio,
        )

    def _encode(
        self,
        sample: RawSpeech,
        *,
        device: torch.device | None,
        input_audio: bool,
    ) -> object:
        # Codec backends are materialization dependencies, not part of the model's
        # mixed-precision graph. Keep their input and internal kernels in FP32.
        streams = tuple(getattr(self.runtime, "input_audio_stream_views", ()))
        waveform = sample.waveform.to(dtype=torch.float32)
        if device is not None:
            waveform = waveform.to(device=device)
        batched_waveform = _batched_waveform(waveform)
        with torch.autocast(
            device_type=batched_waveform.device.type,
            enabled=False,
        ):
            if input_audio and len(streams) > 1:
                return self._composed_input_codes(
                    batched_waveform,
                    sample.sample_rate,
                    streams=streams,
                )
            view = (
                self.runtime.input_audio_view
                if input_audio
                else self.runtime.audio_view
            )
            backend = (
                self.runtime.input_codec
                if input_audio and self.runtime.input_audio_decoupled
                else self.runtime.codec
            )
            if view is AudioView.BICODEC:
                if not supports_global(backend):
                    raise TypeError(
                        "BiCodec waveform fallback requires a semantic-global codec."
                    )
                codec = global_codec(backend)
                semantic, global_codes = _semantic_global_codes(
                    codec.tokenize(
                        batched_waveform,
                        sample.sample_rate,
                    )
                )
                return AudioCodes(
                    semantic_codes=semantic,
                    global_codes=global_codes,
                )
            return _encoded_codes(
                frame_tokenizer(backend).encode(
                    batched_waveform,
                    sample.sample_rate,
                )
            )

    def _composed_input_codes(
        self,
        waveform: Tensor,
        sample_rate: int,
        *,
        streams: tuple[tuple[AudioStream, AudioView], ...],
    ) -> AudioCodes:
        if (
            len(streams) != 2
            or {stream for stream, _ in streams}
            != {AudioStream.SEMANTIC, AudioStream.GLOBAL}
        ):
            raise ValueError(
                "composed input audio requires exactly semantic and global streams."
            )
        semantic = _encoded_codes(
            frame_tokenizer(self.runtime.input_codec).encode(
                waveform,
                sample_rate,
            )
        )
        _, global_codes = _semantic_global_codes(
            global_codec(self.runtime.codec).tokenize(
                waveform,
                sample_rate,
            )
        )
        return AudioCodes(
            semantic_codes=semantic,
            global_codes=global_codes,
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


def _semantic_global_codes(codes: object) -> tuple[Tensor, Tensor]:
    if not isinstance(codes, SemanticGlobalCodes):
        raise TypeError("BiCodec tokenize must return SemanticGlobalCodes.")
    if codes.semantic.size(0) != 1 or codes.global_codes.size(0) != 1:
        raise ValueError("per-sample codec fallback expects one encoded item.")
    return (
        codes.semantic[0].detach().cpu(),
        codes.global_codes[0].detach().cpu(),
    )


def _move_model_batch(batch: ModelBatch, device: torch.device | None) -> ModelBatch:
    return batch if device is None else batch.to(device)
