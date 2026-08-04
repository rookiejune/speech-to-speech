from __future__ import annotations

from typing import Any

from anydataset import types
from torch import Tensor

from ...datamodule.diagnostic import source_item, target_item
from ...generation.decode import decode_reference_codes
from ...generation.eval.acoustic import reference_audio
from ...runtime.codec_contract import (
    acoustic_codec,
    codec_sample_rate,
)
from ...task import Task
from ._sample_protocol import DataModule, Module


def log_target_reference_audio(
    audio_writer: Any | None,
    datamodule: DataModule,
    module: Module,
    batch: Any,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> tuple[Tensor, int] | None:
    if not task.prediction_modality.supervises_audio:
        return None
    if batch.acoustic_target is None:
        target, sample_rate = sample_audio(datamodule, sample, task, source=False)
        if audio_writer is not None:
            audio_writer.add_audio(
                f"{tag}/target", target, step, sample_rate=sample_rate
            )
        return target, sample_rate
    codec = acoustic_codec(datamodule.runtime.codec)
    target, reference = reference_audio(module.model, batch, codec, seed=0)
    sample_rate = codec_sample_rate(codec)
    target = target.detach().cpu()
    if audio_writer is not None:
        audio_writer.add_audio(
            f"{tag}/target", target, step, sample_rate=sample_rate
        )
        audio_writer.add_audio(
            f"{tag}/reference_generation",
            reference.detach().cpu(),
            step,
            sample_rate=sample_rate,
        )
    return target, sample_rate


def log_source_audio(
    audio_writer: Any,
    datamodule: DataModule,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> None:
    if task.source_modality is not types.Modality.AUDIO:
        return
    waveform, sample_rate = sample_audio(datamodule, sample, task, source=True)
    audio_writer.add_audio(
        f"{tag}/source", waveform, step, sample_rate=sample_rate
    )


def log_target_audio(
    audio_writer: Any,
    datamodule: DataModule,
    sample: types.Sample,
    task: Task,
    tag: str,
    step: int,
) -> None:
    if not task.prediction_modality.supervises_audio:
        return
    waveform, sample_rate = sample_audio(datamodule, sample, task, source=False)
    audio_writer.add_audio(
        f"{tag}/target",
        waveform,
        step,
        sample_rate=sample_rate,
    )


def sample_audio(
    datamodule: DataModule,
    sample: types.Sample,
    task: Task,
    *,
    source: bool,
) -> tuple[Tensor, int]:
    ref = source_item(sample, task) if source else target_item(sample, task)
    if ref is None:
        raise ValueError("sample audio reference is missing.")
    _, item = ref
    if not isinstance(item, types.AudioItem):
        raise TypeError("sample audio reference must contain an AudioItem.")
    raw = item.views.get(types.AudioView.WAVEFORM)
    if raw is not None:
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
        waveform, sample_rate = raw
        if not isinstance(waveform, Tensor):
            raise TypeError("AudioView.WAVEFORM waveform must be a Tensor.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("AudioView.WAVEFORM sample_rate must be an integer.")
        if sample_rate <= 0:
            raise ValueError("AudioView.WAVEFORM sample_rate must be positive.")
        return waveform.detach().cpu(), sample_rate
    runtime = datamodule.runtime
    codes = item.views[runtime.audio_view]
    waveform = decode_reference_codes(codes, codec=runtime.codec)
    if waveform.dim() < 2:
        raise ValueError("codec sample decode must return a batched waveform.")
    return waveform[0].detach().cpu(), codec_sample_rate(runtime.codec)

__all__ = [
    "log_source_audio",
    "log_target_audio",
    "log_target_reference_audio",
    "sample_audio",
]
