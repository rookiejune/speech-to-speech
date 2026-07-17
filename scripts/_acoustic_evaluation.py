from __future__ import annotations

from collections.abc import Sequence
import time

import torch
from torch import Tensor

from speech_to_speech.datamodule import ModelBatch
from speech_to_speech.model import SpeechToSpeechFlowModel, SpeechToSpeechRVQModel
from speech_to_speech.runtime.types import Codec


@torch.no_grad()
def evaluate(
    model: SpeechToSpeechFlowModel | SpeechToSpeechRVQModel,
    batch: ModelBatch,
    codec: Codec,
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
    model.eval()
    try:
        prompt = batch.acoustic_prompt
        hidden_states = model.token_hidden_states(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            acoustic_prompt_codes=None if prompt is None else prompt["codes"],
            acoustic_prompt_positions=None if prompt is None else prompt["token_positions"],
            acoustic_prompt_mask=batch.acoustic_prompt_mask,
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
                waveform.numel() / codec.sample_rate
            )
            values.setdefault("sampling_seconds", []).append(elapsed)
            values.setdefault("sampling_rtf", []).append(
                elapsed / (waveform.numel() / codec.sample_rate)
            )
        return {name: sum(items) / len(items) for name, items in values.items()}
    finally:
        model.train(was_training)


def device_batch(batch: ModelBatch, device: torch.device) -> ModelBatch:
    def move(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.to(device)

    prompt = batch.acoustic_prompt
    target = batch.acoustic_target
    return ModelBatch(
        input_ids=cast_tensor(move(batch.input_ids)),
        token_labels=cast_tensor(move(batch.token_labels)),
        acoustic_prompt=(
            None
            if prompt is None
            else {
                "codes": cast_tensor(move(prompt["codes"])),
                "token_positions": cast_tensor(move(prompt["token_positions"])),
            }
        ),
        acoustic_target=(
            None
            if target is None
            else {
                "semantic_codes": cast_tensor(move(target["semantic_codes"])),
                "codes": cast_tensor(move(target["codes"])),
                "token_positions": cast_tensor(move(target["token_positions"])),
            }
        ),
        tasks=batch.tasks,
        pad_token_id=batch.pad_token_id,
    )


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
