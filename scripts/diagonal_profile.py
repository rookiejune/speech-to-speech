from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import partial

import torch
from lightning.pytorch import seed_everything
from torch import Tensor, nn

from speech_to_speech.config import DatasetFactoryConfig
from speech_to_speech.dataset import dataset_metadata, training_dataset
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.datamodule.example import longcat_pair_from_sample, speech_pair_from_sample
from speech_to_speech.model.DiT.model import DiT
from speech_to_speech.model.acoustic import null_acoustic_condition
from speech_to_speech.model.diagonal import (
    diagonal_flow_sample,
    serial_flow_sample,
    serial_forward_count,
)
from speech_to_speech.model.orchestrator import Orchestrator
from speech_to_speech.model.qwen3 import Qwen3Config
from speech_to_speech.runtime import prepare_longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import load_config
from speech_to_speech.types import (
    AcousticCondition,
    CausalLMBatch,
    LongCatBatchSide,
    LongCatPair,
    LongCatSide,
    SpeechPair,
    TranslationExample,
)


@dataclass(frozen=True)
class TimedSample:
    seconds: float
    peak_memory_mb: float | None


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = time.perf_counter()
    config = load_config(args.config_name, overrides=args.overrides, config_dir=args.config_dir)
    seed_everything(config.train.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    _log_stage(args, "load tokenizer", started_at)
    tokenizer = qwen3_tokenizer(config.model)
    _log_stage(args, "prepare bpe", started_at)
    bpe = prepare_longcat_tokenizer(
        partial(_speech_pairs, config.datamodule.dataset_factory),
        datasets=dataset_metadata(config.datamodule.dataset_factory),
        config=config.bpe,
    )
    _log_stage(args, "load sample", started_at)
    pair = _longcat_pair_at(config.datamodule.dataset_factory, args.sample_index)

    device = torch.device(args.device)
    flow_dtype = _flow_dtype(args.flow_dtype, precision=config.train.precision, device=device)
    dit_config = _dit_config(layers=args.dit_layers, heads=args.dit_heads)
    model_config = replace(config.model, train_dit=True)
    _log_stage(args, "load qwen and dit", started_at)
    model = Orchestrator(
        dit=DiT(dit_config),
        model_config=model_config,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    ).eval()
    model = model.to(device)
    if model.dit is None:
        raise RuntimeError("diagonal profile requires a DiT decoder.")
    model.dit.to(dtype=flow_dtype)
    model.acoustic_condition_proj.to(dtype=flow_dtype)

    _log_stage(args, "build acoustic condition", started_at)
    batch = _move_batch(_target_translation_batch(model, tokenizer, bpe, pair), device)
    condition = model.acoustic_condition(batch, bpe)
    hidden, mask = _flow_condition(model, condition, dtype=flow_dtype)
    if args.frames is not None:
        hidden, mask = _tile_time(hidden, mask, frames=args.frames)
    initial = hidden.new_zeros(hidden.shape)
    acoustic_condition = null_acoustic_condition(model.dit, initial)
    sampler_kwargs = {
        "last_hidden_state": hidden,
        "acoustic_condition": acoustic_condition,
        "mask": mask,
        "num_steps": args.flow_steps,
        "chunk_size": args.chunk_size,
        "guidance_scale": args.guidance_scale,
    }

    serial_fn = lambda: serial_flow_sample(model.dit, initial, **sampler_kwargs)
    diagonal_fn = lambda: diagonal_flow_sample(
        model.dit,
        initial,
        wave_stride=args.wave_stride,
        **sampler_kwargs,
    )
    _log_stage(args, "warmup samplers", started_at)
    for _ in range(args.warmup):
        serial_fn()
        diagonal_fn()

    _log_stage(args, "time serial sampler", started_at)
    serial_timing = _time_repeats(args.repeats, serial_fn, device=device)
    serial_sample = serial_fn()
    _log_stage(args, "time diagonal sampler", started_at)
    diagonal_timing = _time_repeats(args.repeats, diagonal_fn, device=device)
    diagonal_sample = diagonal_fn()
    _log_stage(args, "done", started_at)

    diff = (serial_sample.final - diagonal_sample.final).abs().detach().float()
    active = mask.unsqueeze(-1).expand_as(diff)
    active_diff = diff[active]
    cfg_multiplier = 1 if args.guidance_scale == 1.0 else 2
    max_wave_width = max(len(batch.cells) for batch in diagonal_sample.schedule)
    speedup = (
        float("inf")
        if diagonal_timing.seconds == 0
        else serial_timing.seconds / diagonal_timing.seconds
    )

    return {
        "device": str(device),
        "flow_dtype": str(flow_dtype).removeprefix("torch."),
        "sample_index": args.sample_index,
        "source_frame_count": int(pair.source.semantic_ids.numel()),
        "target_frame_count": int(pair.target.semantic_ids.numel()),
        "condition_frame_count": int(condition.mask.sum().detach().cpu()),
        "profile_frame_count": int(hidden.size(1)),
        "chunk_size": args.chunk_size,
        "flow_steps": args.flow_steps,
        "wave_stride": args.wave_stride,
        "guidance_scale": args.guidance_scale,
        "dit_layers": args.dit_layers,
        "dit_heads": args.dit_heads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "serial_seconds": serial_timing.seconds,
        "diagonal_seconds": diagonal_timing.seconds,
        "speedup": speedup,
        "serial_velocity_count": serial_sample.forward_count,
        "diagonal_velocity_count": diagonal_sample.forward_count,
        "serial_dit_forward_count": serial_sample.forward_count * cfg_multiplier,
        "diagonal_dit_forward_count": diagonal_sample.forward_count * cfg_multiplier,
        "expected_serial_velocity_count": serial_forward_count(
            frame_count=hidden.size(1),
            chunk_size=args.chunk_size,
            num_steps=args.flow_steps,
        ),
        "diagonal_packed_row_count": diagonal_sample.packed_row_count,
        "diagonal_schedule_length": len(diagonal_sample.schedule),
        "diagonal_max_wave_width": max_wave_width,
        "serial_peak_memory_mb": serial_timing.peak_memory_mb,
        "diagonal_peak_memory_mb": diagonal_timing.peak_memory_mb,
        "max_abs_diff": float(diff.max().detach().cpu()),
        "active_max_abs_diff": float(active_diff.max().detach().cpu())
        if active_diff.numel() > 0
        else 0.0,
        "mean_abs_diff": float(diff.mean().detach().cpu()),
    }


def _flow_condition(
    model: Orchestrator,
    condition: AcousticCondition,
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    hidden = condition.hidden_states
    projection = model.acoustic_condition_proj
    projection_dtype = _module_dtype(projection, hidden.dtype)
    hidden = projection(hidden.to(dtype=projection_dtype)).to(dtype=dtype)
    mask = condition.mask.to(device=hidden.device, dtype=torch.bool)
    return hidden, mask


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return fallback


def _tile_time(hidden: Tensor, mask: Tensor, *, frames: int) -> tuple[Tensor, Tensor]:
    if frames <= 0:
        raise ValueError("frames must be positive.")
    if hidden.size(1) == 0:
        raise ValueError("cannot tile an empty acoustic condition.")
    repeats = (frames + hidden.size(1) - 1) // hidden.size(1)
    hidden = hidden.repeat(1, repeats, 1)[:, :frames].contiguous()
    mask = mask.repeat(1, repeats)[:, :frames].contiguous()
    return hidden, mask


def _speech_pairs(config: DatasetFactoryConfig) -> Iterable[SpeechPair]:
    for sample in training_dataset(config):
        yield speech_pair_from_sample(sample)


def _longcat_pair_at(config: DatasetFactoryConfig, index: int) -> LongCatPair:
    if index < 0:
        raise ValueError("sample_index must be non-negative.")
    for sample_index, sample in enumerate(training_dataset(config)):
        if sample_index == index:
            return longcat_pair_from_sample(sample)
    raise IndexError(f"sample_index {index} is outside the dataset.")


def _target_translation_batch(
    model: Orchestrator,
    tokenizer: object,
    bpe: object,
    pair: LongCatPair,
) -> CausalLMBatch:
    builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
    source_ids = _encode_frames(bpe, pair.source.semantic_ids)
    target_ids = _encode_frames(bpe, pair.target.semantic_ids)
    batch = builder.translation(TranslationExample(source_ids=source_ids, target_ids=target_ids))
    return CausalLMBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        logits_to_keep=batch.logits_to_keep,
        source_audio=_batch_side(pair.source),
        target_audio=_batch_side(pair.target),
    )


def _batch_side(side: LongCatSide) -> LongCatBatchSide:
    semantic_ids = side.semantic_ids.reshape(1, -1).detach().to(dtype=torch.long)
    acoustic_ids = side.acoustic_ids.detach().to(dtype=torch.long)
    if acoustic_ids.dim() == 2:
        acoustic_ids = acoustic_ids.unsqueeze(0)
    if acoustic_ids.dim() != 3:
        raise ValueError("LongCat acoustic ids must have shape [nq, time] or [batch, nq, time].")
    if acoustic_ids.size(0) != 1:
        raise ValueError("diagonal profile expects a single LongCat sample.")
    length = semantic_ids.size(1)
    if acoustic_ids.size(-1) != length:
        raise ValueError("LongCat semantic and acoustic lengths must match.")
    mask = torch.ones((1, length), dtype=torch.bool)
    return LongCatBatchSide(
        semantic_ids=semantic_ids,
        semantic_mask=mask,
        acoustic_ids=acoustic_ids,
        acoustic_mask=mask,
    )


def _encode_frames(bpe: object, ids: torch.Tensor) -> torch.Tensor:
    encode_frames = bpe.encode_frames
    frames = [[int(value)] for value in ids.reshape(-1).detach().cpu().tolist()]
    return torch.tensor(encode_frames(frames), dtype=torch.long)


def _dit_config(*, layers: int, heads: int) -> Qwen3Config:
    if layers <= 0:
        raise ValueError("dit_layers must be positive.")
    if heads <= 0:
        raise ValueError("dit_heads must be positive.")
    config = Qwen3Config()
    config.hidden_size = 1024
    config.num_hidden_layers = layers
    config.num_attention_heads = heads
    config.num_key_value_heads = heads
    config.intermediate_size = 3072
    return config


def _move_batch(batch: CausalLMBatch, device: torch.device) -> CausalLMBatch:
    logits_to_keep = batch.logits_to_keep
    if isinstance(logits_to_keep, torch.Tensor):
        logits_to_keep = logits_to_keep.to(device=device)
    return CausalLMBatch(
        input_ids=batch.input_ids.to(device=device),
        attention_mask=batch.attention_mask.to(device=device),
        labels=batch.labels.to(device=device),
        logits_to_keep=logits_to_keep,
        source_audio=_move_side(batch.source_audio, device),
        target_audio=_move_side(batch.target_audio, device),
    )


def _move_side(side: LongCatBatchSide | None, device: torch.device) -> LongCatBatchSide | None:
    if side is None:
        return None
    return LongCatBatchSide(
        semantic_ids=side.semantic_ids.to(device=device),
        semantic_mask=side.semantic_mask.to(device=device),
        acoustic_ids=side.acoustic_ids.to(device=device),
        acoustic_mask=side.acoustic_mask.to(device=device),
    )


def _flow_dtype(value: str, *, precision: str, device: torch.device) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value != "auto":
        raise ValueError(f"unsupported flow dtype: {value}")
    if device.type == "cuda" and "bf16" in precision:
        return torch.bfloat16
    return torch.float32


def _time_repeats(
    count: int,
    fn: Callable[[], object],
    *,
    device: torch.device,
) -> TimedSample:
    if count <= 0:
        raise ValueError("repeats must be positive.")
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(count):
        fn()
    _sync(device)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return TimedSample(
        seconds=(time.perf_counter() - start) / count,
        peak_memory_mb=peak_memory_mb,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _log_stage(args: argparse.Namespace, stage: str, started_at: float) -> None:
    if not args.verbose:
        return
    elapsed = time.perf_counter() - started_at
    print(f"[{elapsed:8.2f}s] {stage}", file=sys.stderr, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile real DiT serial vs diagonal flow sampling."
    )
    parser.add_argument("config_name", nargs="?", default="config")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--wave-stride", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--dit-layers", type=int, default=1)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument(
        "--flow-dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
