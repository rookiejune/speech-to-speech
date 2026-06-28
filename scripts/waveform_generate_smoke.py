from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import torch
from anydataset import AnyDataset, MultipleAnyDataset, WeightedRandomStrategy
from lightning.pytorch import seed_everything

from speech_to_speech.config import DataConfig
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.datamodule.example import speech_pair_from_sample
from speech_to_speech.model.DiT.model import DiT
from speech_to_speech.model.orchestrator import AcousticSampler, Orchestrator
from speech_to_speech.model.qwen3 import Qwen3Config
from speech_to_speech.runtime import longcat_codec, prepare_longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import load_config
from speech_to_speech.types import GenerationBatch, SpeechPair


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = time.perf_counter()
    config = load_config(args.config)
    seed_everything(config.train.seed, workers=True)

    tokenizer = qwen3_tokenizer(config.model)
    bpe = prepare_longcat_tokenizer(
        _speech_pairs(config.data),
        datasets=config.data.datasets,
        config=config.bpe,
    )
    model_config = replace(config.model, train_dit=True)
    model = Orchestrator(
        dit=DiT(_dit_config(layers=args.dit_layers, heads=args.dit_heads)),
        model_config=model_config,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    ).eval()
    device = torch.device(args.device)
    model = model.to(device)

    pair = _speech_pair_at(config.data, args.sample_index)
    source_bpe_ids = _encode_units(bpe, pair.source_ids)
    builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
    batch = _move_generation_batch(builder.translation_generation(source_bpe_ids), device)
    _sync(device)
    generation_started_at = time.perf_counter()
    generation = model.generate_waveform(
        batch,
        bpe=bpe,
        codec=longcat_codec(),
        acoustic_generator=model.acoustic_feature_generator(
            num_steps=args.flow_steps,
            chunk_size=args.chunk_size,
            guidance_scale=args.guidance_scale,
            sampler=AcousticSampler(args.acoustic_sampler),
        ),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    _sync(device)
    generation_seconds = time.perf_counter() - generation_started_at
    return {
        "sample_index": args.sample_index,
        "acoustic_sampler": args.acoustic_sampler,
        "max_new_tokens": args.max_new_tokens,
        "flow_steps": args.flow_steps,
        "chunk_size": args.chunk_size,
        "guidance_scale": args.guidance_scale,
        "source_frame_count": int(pair.source_ids.numel()),
        "target_frame_count": int(pair.target_ids.numel()),
        "generated_token_shape": list(generation.token_ids.shape),
        "generated_semantic_shape": list(generation.semantic_ids.shape),
        "generated_frame_count": int(generation.semantic_mask.sum().item()),
        "acoustic_feature_shape": list(generation.acoustic_features.shape),
        "audio_shape": list(generation.audio.shape),
        "audio_abs_mean": float(generation.audio.abs().mean().detach().cpu()),
        "generation_seconds": generation_seconds,
        "total_seconds": time.perf_counter() - started_at,
    }


def _speech_pairs(data: DataConfig) -> Iterable[SpeechPair]:
    for sample in _dataset(data):
        yield speech_pair_from_sample(sample)


def _speech_pair_at(data: DataConfig, index: int) -> SpeechPair:
    if index < 0:
        raise ValueError("sample_index must be non-negative.")
    for sample_index, sample in enumerate(_dataset(data)):
        if sample_index == index:
            return speech_pair_from_sample(sample)
    raise IndexError(f"sample_index {index} is outside the dataset.")


def _dataset(data: DataConfig) -> AnyDataset | MultipleAnyDataset:
    datasets = tuple(AnyDataset(dataset, cache_root=data.cache_root) for dataset in data.datasets)
    if len(datasets) == 1:
        return datasets[0]
    return MultipleAnyDataset(datasets, strategy=WeightedRandomStrategy())


def _encode_units(bpe: object, ids: torch.Tensor) -> torch.Tensor:
    encode_units = bpe.encode_units
    units = [int(value) for value in ids.reshape(-1).detach().cpu().tolist()]
    return torch.tensor(encode_units(units), dtype=torch.long)


def _move_generation_batch(batch: GenerationBatch, device: torch.device) -> GenerationBatch:
    return GenerationBatch(
        input_ids=batch.input_ids.to(device=device),
        attention_mask=batch.attention_mask.to(device=device),
    )


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


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full waveform generation sanity check.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--acoustic-sampler",
        choices=tuple(sampler.value for sampler in AcousticSampler),
        default=AcousticSampler.DIAGONAL.value,
    )
    parser.add_argument("--dit-layers", type=int, default=1)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument("--device", default=_default_device())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
