from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from functools import partial

import torch
from lightning.pytorch import seed_everything

from speech_to_speech.config import DatasetFactoryConfig
from speech_to_speech.dataset import dataset_metadata, training_dataset
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.datamodule.example import speech_pair_from_sample
from speech_to_speech.model.orchestrator import Orchestrator
from speech_to_speech.runtime import prepare_longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import load_config
from speech_to_speech.types import GenerationBatch, SpeechPair


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config_name, overrides=args.overrides, config_dir=args.config_dir)
    seed_everything(config.train.seed, workers=True)

    tokenizer = qwen3_tokenizer(config.model)
    bpe = prepare_longcat_tokenizer(
        partial(_speech_pairs, config.datamodule.dataset_factory),
        datasets=dataset_metadata(config.datamodule.dataset_factory),
        config=config.bpe,
    )
    model = Orchestrator(
        model_config=config.model,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    ).eval()

    pair = _speech_pair_at(config.datamodule.dataset_factory, args.sample_index)
    source_bpe_ids = _encode_frames(bpe, pair.source_ids)
    builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
    batch = builder.translation_generation(source_bpe_ids)
    batch = _move_generation_batch(batch, _module_device(model))

    generation = model.generate_semantic(
        batch,
        bpe=bpe,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    semantic_ids = generation.semantic_ids[generation.semantic_mask].detach().cpu().tolist()
    token_ids = generation.token_ids.detach().cpu().reshape(-1).tolist()
    return {
        "sample_index": args.sample_index,
        "source_frame_count": int(pair.source_ids.numel()),
        "target_frame_count": int(pair.target_ids.numel()),
        "source_bpe_token_count": int(source_bpe_ids.numel()),
        "prompt_token_count": int(batch.input_ids.size(1)),
        "generated_token_count": len(token_ids),
        "generated_semantic_frame_count": len(semantic_ids),
        "generated_token_ids": token_ids[: args.preview_tokens],
        "generated_semantic_ids": semantic_ids[: args.preview_tokens],
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


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _move_generation_batch(batch: GenerationBatch, device: torch.device) -> GenerationBatch:
    return GenerationBatch(
        input_ids=batch.input_ids.to(device=device),
        attention_mask=batch.attention_mask.to(device=device),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic generation sanity check.")
    parser.add_argument("config_name", nargs="?", default="config")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--preview-tokens", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
