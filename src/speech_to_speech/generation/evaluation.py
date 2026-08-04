from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import TYPE_CHECKING, Any, TypedDict

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from torch import Tensor

from .._oom import annotate, tensor_report
from ..datamodule.batch import ModelBatch
from ..runtime.codec_contract import (
    AcousticCodec,
    codec_sample_rate,
)
from ..task import PredictionModality, Request, Task
from .contract import (
    AudioOutput,
    GenerationRuntime,
    Result,
    TextEvaluationModel,
)
from .service import generate_responses, requests_from_batch
from .text import decode_text_ids as _decode_text_ids

if TYPE_CHECKING:
    from ..model.acoustic.flow import FlowModel
    from ..model.acoustic.rvq import RVQModel
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
            prediction=batch.prediction_modality,
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
            prediction=batch.prediction_modality,
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


class TextProbe(TypedDict):
    instruction: str
    reference: str


class TextProbeResult(TypedDict):
    generated: str
    nll: float


@torch.no_grad()
def evaluate_text(
    probes: Mapping[str, TextProbe],
    model: TextEvaluationModel,
    *,
    max_new_tokens: int,
) -> dict[str, TextProbeResult]:
    runtime = model.runtime
    prompts = {
        name: _prompt_ids(runtime, probe["instruction"])
        for name, probe in probes.items()
    }
    requests = [
        Request(
            prompt_ids=prompts[name],
            task=Task.T2TT,
            audio_input_positions=None,
        )
        for name in probes
    ]
    try:
        generations = generate_responses(
            requests,
            model,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    except torch.OutOfMemoryError as error:
        annotate(
            error,
            phase="text_evaluation_generation",
            inputs={
                "type": "TextGenerationRequests",
                "prompt_ids": [tensor_report(value) for value in prompts.values()],
                "padded_prompt_shape": [
                    len(prompts),
                    max((value.numel() for value in prompts.values()), default=0),
                ],
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            },
        )
        raise

    results: dict[str, TextProbeResult] = {}
    for (name, probe), generation in zip(probes.items(), generations):
        results[name] = TextProbeResult(
            generated=_decode_text_ids(runtime, generation["response_ids"]),
            nll=_reference_nll(model, prompts[name], probe["reference"]),
        )
    return results


def _prompt_ids(runtime: GenerationRuntime, instruction: str) -> Tensor:
    ids = runtime.text_tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    local_ids = torch.as_tensor(ids, dtype=torch.long)
    return runtime.layout.to_global(Modality.TEXT.value, local_ids)


def _reference_nll(
    model: TextEvaluationModel,
    prompt_ids: Tensor,
    reference: str,
) -> float:
    runtime = model.runtime
    text_start, _ = runtime.layout.blocks[Modality.TEXT.value]
    local_reference = torch.tensor(
        runtime.text_tokenizer.encode(reference, add_special_tokens=False),
        dtype=torch.long,
    )
    reference_ids = runtime.layout.to_global(Modality.TEXT.value, local_reference)
    response_ids = torch.cat(
        (reference_ids, torch.tensor([runtime.eos_token_id], dtype=torch.long))
    )
    input_shape = [1, prompt_ids.numel() + response_ids.numel()]
    try:
        device = model.backbone.get_input_embeddings().weight.device
        input_ids = torch.cat((prompt_ids, response_ids)).to(device=device)[None]
        hidden_states = model.token_hidden_states(
            input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            prediction=PredictionModality.TEXT,
        )
        predictors = hidden_states[0, prompt_ids.numel() - 1 : -1]
        prediction = model.token_logits(predictors, Modality.TEXT).float()
        target = input_ids[0, prompt_ids.numel() :] - text_start
        return float(F.cross_entropy(prediction, target).detach().cpu())
    except torch.OutOfMemoryError as error:
        annotate(
            error,
            phase="text_evaluation_reference_nll",
            inputs={
                "type": "TextReferenceNLL",
                "input_ids_shape": input_shape,
                "prompt_tokens": prompt_ids.numel(),
                "reference_tokens_with_eos": response_ids.numel(),
            },
        )
        raise


def summary(run_output: dict[str, Any]) -> dict[str, Any]:
    result = run_output["result"]
    audio = audio_output(result, "generation result")
    features = audio["features"]
    waveform = audio["waveform"]
    if features is None:
        raise RuntimeError("generation smoke requires acoustic features.")
    return {
        "token_ids": result["response_ids"].detach().cpu().tolist(),
        "acoustic_shape": list(features.shape),
        "waveform_shape": list(waveform.shape),
        "finite": bool(torch.isfinite(features).all() and torch.isfinite(waveform).all()),
        "calls": run_output["calls"],
        "top_logits": [
            top_logits(values, run_output["allowed_ids"])
            for values in run_output["allowed_logits"]
        ],
        "elapsed_seconds": run_output["elapsed_seconds"],
        "peak_cuda_bytes": run_output["peak_cuda_bytes"],
    }


def compare(cached_run: dict[str, Any], full_run: dict[str, Any]) -> dict[str, Any]:
    cached = cached_run["result"]
    full = full_run["result"]
    cached_audio = audio_output(cached, "cached result")
    full_audio = audio_output(full, "full-recompute result")
    cached_features = cached_audio["features"]
    full_features = full_audio["features"]
    cached_waveform = cached_audio["waveform"]
    full_waveform = full_audio["waveform"]
    cached_tokens = cached["response_ids"]
    full_tokens = full["response_ids"]
    if cached_features is None or full_features is None:
        raise RuntimeError("generation smoke requires acoustic features.")
    return {
        "tokens_equal": bool(torch.equal(cached_tokens, full_tokens)),
        "first_token_difference": first_difference(cached_tokens, full_tokens),
        "logit_steps": compare_logits(
            cached_run["allowed_logits"], full_run["allowed_logits"]
        ),
        "acoustic_shapes_equal": cached_features.shape == full_features.shape,
        "waveform_shapes_equal": cached_waveform.shape == full_waveform.shape,
        "acoustic_max_abs": optional_max_abs(cached_features, full_features),
        "waveform_max_abs": optional_max_abs(cached_waveform, full_waveform),
        "cached_finite": bool(
            torch.isfinite(cached_features).all()
            and torch.isfinite(cached_waveform).all()
        ),
        "full_finite": bool(
            torch.isfinite(full_features).all() and torch.isfinite(full_waveform).all()
        ),
    }


def optional_max_abs(left: Tensor, right: Tensor) -> float | None:
    if left.shape != right.shape:
        return None
    return float((left.float() - right.float()).abs().max())


def compare_logits(
    cached: list[Tensor], full: list[Tensor]
) -> list[dict[str, float | int]]:
    if len(cached) != len(full):
        raise ValueError("cached and full generation must contain the same logit steps.")
    return [
        {
            "step": step,
            "max_abs": float((cached_values - full_values).abs().max()),
        }
        for step, (cached_values, full_values) in enumerate(zip(cached, full))
    ]


def allowed_values(logits: Tensor, allowed_ids: Sequence[int]) -> Tensor:
    ids = torch.as_tensor(allowed_ids, device=logits.device, dtype=torch.long)
    return logits.index_select(0, ids).detach().float().cpu()


def selected_id(logits: Tensor, allowed_ids: Sequence[int]) -> int:
    values = allowed_values(logits, allowed_ids)
    return allowed_ids[int(values.argmax())]


def tensor_max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def hidden_last(output: Any, name: str) -> Tensor:
    if output.hidden_states is None:
        raise RuntimeError(f"generation did not return {name} hidden states.")
    return output.hidden_states[-1]


def hidden_layer_max_abs(output: Any, reference: Any) -> list[float]:
    if output.hidden_states is None or reference.hidden_states is None:
        raise RuntimeError("probe did not return layer hidden states.")
    return [
        tensor_max_abs(left[0, -1], right[0, -1])
        for left, right in zip(output.hidden_states, reference.hidden_states)
    ]


def top_logits(
    values: Tensor, allowed_ids: Sequence[int], count: int = 5
) -> dict[str, Any]:
    top_values, local_ids = values.topk(min(count, values.numel()))
    margin = float(top_values[0] - top_values[1]) if top_values.numel() > 1 else None
    return {
        "token_ids": [allowed_ids[index] for index in local_ids.tolist()],
        "values": top_values.tolist(),
        "top1_margin": margin,
    }


def first_difference(left: Tensor, right: Tensor) -> int | None:
    shared = min(left.numel(), right.numel())
    difference = (left[:shared] != right[:shared]).nonzero()
    if difference.numel():
        return int(difference[0].item())
    if left.numel() != right.numel():
        return shared
    return None


def audio_output(result: Result, name: str) -> AudioOutput:
    audio = result["audio"]
    if audio is None:
        raise RuntimeError(f"{name} did not return audio output.")
    return audio


__all__ = [
    "TextProbe",
    "TextProbeResult",
    "allowed_values",
    "audio_output",
    "compare",
    "evaluate",
    "evaluate_autoregressive",
    "evaluate_text",
    "first_difference",
    "hidden_last",
    "hidden_layer_max_abs",
    "mono",
    "reference_audio",
    "selected_id",
    "stft_distance",
    "summary",
    "tensor_max_abs",
    "top_logits",
]
