from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from speech_to_speech.datamodule import ModelBatch
from speech_to_speech.model import SpeechToSpeechFlowModel
from speech_to_speech.runtime.types import Codec


@torch.no_grad()
def evaluate(
    model: SpeechToSpeechFlowModel,
    batch: ModelBatch,
    codec: Codec,
    *,
    seeds: Sequence[int],
) -> dict[str, float]:
    batch = device_batch(batch, next(model.parameters()).device)
    if (
        batch.acoustic_labels is None
        or batch.acoustic_label_positions is None
        or batch.semantic_frame_labels is None
        or batch.acoustic_target_mask is None
    ):
        raise RuntimeError("acoustic evaluation requires complete target fields")
    if batch.input_ids.size(0) != 1:
        raise ValueError("acoustic evaluation currently requires batch size 1")

    was_training = model.training
    model.eval()
    try:
        output = model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            acoustic_input_ids=batch.acoustic_input_ids,
            acoustic_input_positions=batch.acoustic_input_positions,
            acoustic_input_mask=batch.acoustic_input_mask,
            output_hidden_states=True,
        )
        if output.hidden_states is None:
            raise RuntimeError("model did not return acoustic condition states")
        condition = model.target_frame_condition(
            output.hidden_states[-1], batch.acoustic_label_positions
        )
        target = model.acoustic_target_latent(batch.acoustic_labels)
        mask = batch.acoustic_target_mask
        valid = mask[0]
        semantic = batch.semantic_frame_labels[0, valid].unsqueeze(0)
        target = target[0, valid].unsqueeze(0)
        reference = mono(codec.decode_features(semantic, target))

        values: dict[str, list[float]] = {}
        for seed in seeds:
            generator = torch.Generator(device=condition.device).manual_seed(seed)
            sampled = model.acoustic_flow.sample(condition, generator=generator)
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
        return {name: sum(items) / len(items) for name, items in values.items()}
    finally:
        model.train(was_training)


def device_batch(batch: ModelBatch, device: torch.device) -> ModelBatch:
    def move(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.to(device)

    return ModelBatch(
        input_ids=cast_tensor(move(batch.input_ids)),
        labels=cast_tensor(move(batch.labels)),
        acoustic_input_ids=move(batch.acoustic_input_ids),
        acoustic_input_positions=move(batch.acoustic_input_positions),
        semantic_frame_labels=move(batch.semantic_frame_labels),
        acoustic_labels=move(batch.acoustic_labels),
        acoustic_label_positions=move(batch.acoustic_label_positions),
        tasks=batch.tasks,
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
