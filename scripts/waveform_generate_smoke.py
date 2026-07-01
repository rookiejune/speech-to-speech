from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from functools import partial

import torch
from lightning.pytorch import seed_everything

from speech_to_speech.config import DatasetFactoryConfig, with_acoustic_decoder
from speech_to_speech.dataset import dataset_metadata, training_dataset
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.datamodule.example import speech_pair_from_sample
from speech_to_speech.model.acoustic import AcousticSampler
from speech_to_speech.model.orchestrator import Orchestrator
from speech_to_speech.runtime import longcat_codec, prepare_longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import load_config
from speech_to_speech.types import GenerationBatch, SpeechPair


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = time.perf_counter()
    config = load_config(args.config_name, overrides=args.overrides, config_dir=args.config_dir)
    seed_everything(config.train.seed, workers=True)

    tokenizer = qwen3_tokenizer(config.model)
    bpe = prepare_longcat_tokenizer(
        partial(_speech_pairs, config.datamodule.dataset_factory),
        datasets=dataset_metadata(config.datamodule.dataset_factory),
        config=config.bpe,
    )
    model_config = with_acoustic_decoder(
        config.model,
        enabled=True,
        train=True,
        dit=replace(
            config.model.acoustic.dit,
            num_hidden_layers=args.dit_layers,
            num_attention_heads=args.dit_heads,
            num_key_value_heads=args.dit_heads,
        ),
    )
    model = Orchestrator(
        model_config=model_config,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    ).eval()
    device = torch.device(args.device)
    model = model.to(device)

    pair = _speech_pair_at(config.datamodule.dataset_factory, args.sample_index)
    source_bpe_ids = _encode_frames(bpe, pair.source_ids)
    builder = CausalLMBatchBuilder(model.idspace, tokenizer=tokenizer)
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
            left_context_chunks=args.left_context_chunks,
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
        "left_context_chunks": args.left_context_chunks,
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


def _speech_pairs(config: DatasetFactoryConfig) -> Iterable[SpeechPair]:
    for sample in training_dataset(config):
        yield speech_pair_from_sample(sample)


def _speech_pair_at(config: DatasetFactoryConfig, index: int) -> SpeechPair:
    if index < 0:
        raise ValueError("sample_index must be non-negative.")
    for sample_index, sample in enumerate(training_dataset(config)):
        if sample_index == index:
            return speech_pair_from_sample(sample)
    raise IndexError(f"sample_index {index} is outside the dataset.")


def _encode_frames(bpe: object, ids: torch.Tensor) -> torch.Tensor:
    encode_frames = bpe.encode_frames
    frames = [[int(value)] for value in ids.reshape(-1).detach().cpu().tolist()]
    return torch.tensor(encode_frames(frames), dtype=torch.long)


def _move_generation_batch(batch: GenerationBatch, device: torch.device) -> GenerationBatch:
    return GenerationBatch(
        input_ids=batch.input_ids.to(device=device),
        attention_mask=batch.attention_mask.to(device=device),
    )


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full waveform generation sanity check.")
    parser.add_argument("config_name", nargs="?", default="config")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--left-context-chunks", type=int)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--acoustic-sampler",
        choices=tuple(sampler.value for sampler in AcousticSampler),
        default=AcousticSampler.SERIAL.value,
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
