from __future__ import annotations

from collections.abc import Sequence
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ..datamodule.types import ModelBatch
from ..runtime.types import AcousticCodec, codec_sample_rate
from .batch import requests_from_batch
from .reporting import audio_output

if TYPE_CHECKING:
    from ..model.acoustic import FlowModel, RVQModel
    from ..pl_module import SpeechToSpeechModule


@torch.no_grad()
def evaluate_autoregressive(
    module: SpeechToSpeechModule[Any],
    batch: ModelBatch,
    *,
    sample_rate: int,
) -> dict[str, object]:
    requests = requests_from_batch(batch)
    if len(requests) != 1:
        raise ValueError("autoregressive evaluation requires exactly one sample.")
    device = next(module.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = module.generate(
        requests,
        max_new_tokens=64,
        do_sample=False,
    )[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    audio = audio_output(result, "autoregressive evaluation")
    features = audio["features"]
    waveform = audio["waveform"]
    if features is None:
        raise RuntimeError("autoregressive evaluation did not return acoustic features.")
    if waveform.numel() == 0:
        raise RuntimeError("autoregressive evaluation returned an empty waveform.")
    if not bool(torch.isfinite(features).all() and torch.isfinite(waveform).all()):
        raise RuntimeError("autoregressive evaluation returned non-finite output.")
    duration = waveform.numel() / sample_rate
    return {
        "token_ids": result["response_ids"].detach().cpu().tolist(),
        "feature_shape": list(features.shape),
        "waveform_shape": list(waveform.shape),
        "duration_seconds": duration,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / duration,
        "finite": True,
    }


@torch.no_grad()
def evaluate(
    model: FlowModel | RVQModel,
    batch: ModelBatch,
    codec: AcousticCodec,
    *,
    seeds: Sequence[int],
) -> dict[str, float]:
    batch = device_batch(batch, next(model.parameters()).device)
    target_data = batch.acoustic_target
    if target_data is None or batch.acoustic_target_mask is None:
        raise RuntimeError("acoustic evaluation requires complete target fields")
    if batch.input_ids.size(0) != 1:
        raise ValueError("acoustic evaluation currently requires batch size 1")

    was_training = model.training
    sample_rate = codec_sample_rate(codec)
    model.eval()
    try:
        hidden_states = model.token_hidden_states(
            batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        condition = model.target_frame_condition(
            hidden_states, target_data["token_positions"]
        )
        safe_codes = target_data["codes"].clamp_min(0)
        target = codec.acoustic_codes_to_features(safe_codes)
        mask = batch.acoustic_target_mask
        valid = mask[0]
        semantic = target_data["semantic_codes"][0, valid].unsqueeze(0)
        target = target[0, valid].unsqueeze(0)
        reference = mono(codec.decode_features(semantic, target))

        values: dict[str, list[float]] = {}
        for seed in seeds:
            generator = torch.Generator(device=condition.device).manual_seed(seed)
            if condition.is_cuda:
                torch.cuda.synchronize(condition.device)
            started = time.perf_counter()
            sampled = model.sample_acoustic_features(
                condition,
                mask=mask,
                generator=generator,
            )
            if condition.is_cuda:
                torch.cuda.synchronize(condition.device)
            elapsed = time.perf_counter() - started
            sampled = sampled[0, valid].unsqueeze(0)
            waveform = mono(codec.decode_features(semantic, sampled))
            append(values, "feature_mse", torch.mean((sampled.float() - target.float()) ** 2))
            for name, value in stft_distance(waveform, reference).items():
                append(values, name, value)
            append(values, "waveform_rms", waveform.square().mean().sqrt())
            append(values, "waveform_peak", waveform.abs().max())
            values.setdefault("duration_seconds", []).append(
                waveform.numel() / sample_rate
            )
            values.setdefault("sampling_seconds", []).append(elapsed)
            values.setdefault("sampling_rtf", []).append(
                elapsed / (waveform.numel() / sample_rate)
            )
        return {name: sum(items) / len(items) for name, items in values.items()}
    finally:
        model.train(was_training)


@torch.no_grad()
def reference_audio(
    model: "FlowModel | RVQModel",
    batch: ModelBatch,
    codec: AcousticCodec,
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Decode target audio and a target-semantic teacher-forced sample."""
    batch = device_batch(batch, next(model.parameters()).device)
    target_data = batch.acoustic_target
    if target_data is None or batch.acoustic_target_mask is None:
        raise RuntimeError("reference audio requires complete target fields")
    if batch.input_ids.size(0) != 1:
        raise ValueError("reference audio currently requires batch size 1")

    was_training = model.training
    model.eval()
    try:
        hidden_states = model.token_hidden_states(
            batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        condition = model.target_frame_condition(
            hidden_states, target_data["token_positions"]
        )
        target_features = codec.acoustic_codes_to_features(
            target_data["codes"].clamp_min(0)
        )
        valid = batch.acoustic_target_mask[0]
        semantic = target_data["semantic_codes"][0, valid].unsqueeze(0)
        target_features = target_features[0, valid].unsqueeze(0)
        target_waveform = mono(codec.decode_features(semantic, target_features))
        sampled = model.sample_acoustic_features(
            condition,
            mask=batch.acoustic_target_mask,
            generator=torch.Generator(device=condition.device).manual_seed(seed),
        )
        sampled = sampled[0, valid].unsqueeze(0)
        reference_waveform = mono(codec.decode_features(semantic, sampled))
        return target_waveform, reference_waveform
    finally:
        model.train(was_training)


def device_batch(batch: ModelBatch, device: torch.device) -> ModelBatch:
    return batch.to(device)


def stft_distance(sample: Tensor, reference: Tensor) -> dict[str, Tensor]:
    spectral_convergence = sample.new_zeros(())
    log_magnitude = sample.new_zeros(())
    for n_fft in (256, 512, 1024):
        window = torch.hann_window(n_fft, device=sample.device, dtype=sample.dtype)
        sample_magnitude = torch.stft(
            sample,
            n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        ).abs()
        reference_magnitude = torch.stft(
            reference,
            n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        ).abs()
        spectral_convergence += torch.linalg.vector_norm(
            sample_magnitude - reference_magnitude
        ) / torch.linalg.vector_norm(reference_magnitude).clamp_min(1e-7)
        log_magnitude += torch.mean(
            torch.abs(
                torch.log(sample_magnitude.clamp_min(1e-7))
                - torch.log(reference_magnitude.clamp_min(1e-7))
            )
        )
    return {
        "stft_spectral_convergence": spectral_convergence / 3,
        "stft_log_magnitude": log_magnitude / 3,
    }


def mono(waveform: Tensor) -> Tensor:
    value = waveform.float()
    while value.dim() > 1 and value.size(0) == 1:
        value = value.squeeze(0)
    if value.dim() == 2:
        value = value.mean(dim=0)
    if value.dim() != 1:
        raise ValueError("codec decode must produce a mono waveform")
    return value


def append(values: dict[str, list[float]], name: str, value: Tensor) -> None:
    values.setdefault(name, []).append(float(value))


def cast_tensor(value: Tensor | None) -> Tensor:
    if value is None:
        raise RuntimeError("required batch tensor is unavailable")
    return value
