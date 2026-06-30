from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from functools import partial

import torch
from lightning.pytorch import seed_everything

from speech_to_speech.config import DatasetFactoryConfig, with_acoustic_decoder
from speech_to_speech.dataset import dataset_metadata, training_dataset
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.datamodule.example import longcat_pair_from_sample, speech_pair_from_sample
from speech_to_speech.model.orchestrator import Orchestrator
from speech_to_speech.runtime import (
    longcat_acoustic_features,
    longcat_codec,
    prepare_longcat_tokenizer,
    qwen3_tokenizer,
)
from speech_to_speech.smoke import load_config
from speech_to_speech.types import (
    CausalLMBatch,
    LongCatBatchSide,
    LongCatPair,
    LongCatSide,
    SpeechPair,
    TranslationExample,
)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = time.perf_counter()
    config = load_config(args.config_name, overrides=args.overrides, config_dir=args.config_dir)
    seed_everything(config.train.seed, workers=True)

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
    _log_stage(args, "longcat acoustic features", started_at)
    target_features = longcat_acoustic_features(pair.target.acoustic_ids)
    target_features = _normalize_features(target_features)
    if args.max_frames is not None:
        target_features = target_features[:, : args.max_frames]
    codec = longcat_codec()

    model_config = with_acoustic_decoder(
        config.model,
        enabled=True,
        train=True,
        dit=replace(
            config.model.acoustic.dit,
            hidden_size=target_features.size(-1),
            intermediate_size=max(target_features.size(-1) * 3, 1),
            num_hidden_layers=args.dit_layers,
            num_attention_heads=args.dit_heads,
            num_key_value_heads=args.dit_heads,
        ),
    )
    _log_stage(args, "load qwen and dit", started_at)
    model = Orchestrator(
        model_config=model_config,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    ).eval()
    device = torch.device(args.device)
    model = model.to(device)

    batch = _move_batch(_target_translation_batch(model, tokenizer, bpe, pair), device)
    target_features = target_features.to(device=device, dtype=torch.float32)
    noise = torch.zeros_like(target_features)
    timesteps = torch.full((target_features.size(0),), 0.5, device=device)

    _log_stage(args, "acoustic flow loss", started_at)
    loss = model.acoustic_flow_loss(
        batch,
        bpe,
        target_features,
        noise=noise,
        timesteps=timesteps,
        source_feature_extractor=codec,
    )
    _log_stage(args, "done", started_at)
    condition = model.acoustic_condition(batch, bpe)
    return {
        "sample_index": args.sample_index,
        "target_frame_count": int(pair.target.semantic_ids.numel()),
        "target_feature_shape": list(target_features.shape),
        "condition_shape": list(condition.hidden_states.shape),
        "condition_frame_count": int(condition.mask.sum().item()),
        "source_frame_count": int(batch.source_audio.acoustic_mask.sum().item())
        if batch.source_audio is not None
        else 0,
        "loss": float(loss.detach().cpu()),
    }


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
        raise ValueError("acoustic loss smoke expects a single LongCat sample.")
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


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    if features.dim() == 2:
        return features.unsqueeze(0)
    if features.dim() == 3:
        return features
    raise ValueError("LongCat acoustic features must have shape [time, dim] or [batch, time, dim].")


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


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _log_stage(args: argparse.Namespace, stage: str, started_at: float) -> None:
    if not args.verbose:
        return
    elapsed = time.perf_counter() - started_at
    print(f"[{elapsed:8.2f}s] {stage}", file=sys.stderr, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data acoustic flow loss sanity check.")
    parser.add_argument("config_name", nargs="?", default="config")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dit-layers", type=int, default=1)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
