from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import seed_everything
from torch import Tensor

from speech_to_speech.config import DatasetFactoryConfig, SpeechToSpeechConfig
from speech_to_speech.dataset import training_dataset
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.datamodule.example import longcat_pair_from_sample
from speech_to_speech.model.DiT.model import DiT
from speech_to_speech.model.orchestrator import AcousticSampler, Orchestrator, dit_config
from speech_to_speech.runtime import longcat_codec, longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import load_config
from speech_to_speech.types import (
    AudioBoundary,
    GenerationBatch,
    LongCatPair,
    LongCatSide,
)

SAMPLE_RATE = 16_000


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    config = load_config(args.config_name, overrides=args.overrides, config_dir=args.config_dir)
    seed_everything(config.train.seed, workers=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = qwen3_tokenizer(config.model)
    bpe = longcat_tokenizer(config.bpe)
    model = _load_model(config, bpe_vocab_size=bpe.vocab_size, ckpt_path=args.ckpt_path)
    device = torch.device(args.device)
    model = model.to(device).eval()
    codec = _move_runtime_to_device(longcat_codec(), device)
    builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)

    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        for sample_index in args.sample_indices:
            pair = _longcat_pair_at(config.datamodule.dataset_factory, sample_index)
            for direction, source, reference in _directions(pair):
                result_dir = output_dir / f"sample_{sample_index:04d}" / direction
                result_dir.mkdir(parents=True, exist_ok=True)
                source_audio = _decode_reference(codec, bpe, source)
                reference_audio = _decode_reference(codec, bpe, reference)
                _save_audio(result_dir / "source.wav", source_audio)
                _save_audio(result_dir / "reference.wav", reference_audio)

                source_bpe_ids = _encode_frames(bpe, source.semantic_ids)
                batch = _move_generation_batch(
                    builder.translation_generation(source_bpe_ids),
                    device,
                )
                generated_at = time.perf_counter()
                _seed_generation(config.train.seed, sample_index=sample_index, direction=direction)
                semantic = model.generate_semantic(
                    batch,
                    bpe=bpe,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                generated_frame_count = int(semantic.semantic_mask.sum().item())
                generated_wav: str | None = None
                generation_error: str | None = None
                if generated_frame_count > 0:
                    _seed_generation(
                        config.train.seed,
                        sample_index=sample_index,
                        direction=direction,
                    )
                    generation = model.generate_waveform(
                        batch,
                        bpe=bpe,
                        codec=codec,
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
                    generated_audio = _first_audio(generation.audio)
                    generated_wav = str(result_dir / "generated.wav")
                    _save_audio(result_dir / "generated.wav", generated_audio)
                    generated_token_count = int(generation.token_ids.numel())
                    generated_frame_count = int(generation.semantic_mask.sum().item())
                    eoa_hit = _eoa_hit(model, generation.token_ids)
                    token_preview = _preview_ids(generation.token_ids)
                    semantic_preview = _preview_ids(
                        generation.semantic_ids[generation.semantic_mask]
                    )
                else:
                    generated_token_count = int(semantic.token_ids.numel())
                    eoa_hit = _eoa_hit(model, semantic.token_ids)
                    generation_error = "no generated semantic frames"
                    token_preview = _preview_ids(semantic.token_ids)
                    semantic_preview = []
                row = {
                    "sample_index": sample_index,
                    "direction": direction,
                    "checkpoint": str(args.ckpt_path),
                    "source_wav": str(result_dir / "source.wav"),
                    "reference_wav": str(result_dir / "reference.wav"),
                    "generated_wav": generated_wav,
                    "generation_error": generation_error,
                    "source_frame_count": int(source.semantic_ids.numel()),
                    "reference_frame_count": int(reference.semantic_ids.numel()),
                    "generated_token_count": generated_token_count,
                    "generated_semantic_frame_count": generated_frame_count,
                    "generated_token_preview": token_preview,
                    "generated_semantic_preview": semantic_preview,
                    "eoa_hit": eoa_hit,
                    "manual_note": "",
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "flow_steps": args.flow_steps,
                    "chunk_size": args.chunk_size,
                    "guidance_scale": args.guidance_scale,
                    "acoustic_sampler": args.acoustic_sampler,
                    "generation_seconds": time.perf_counter() - generated_at,
                    "total_seconds": time.perf_counter() - started_at,
                }
                summary_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                summary_file.flush()
    print(summary_path)


def _load_model(
    config: SpeechToSpeechConfig,
    *,
    bpe_vocab_size: int,
    ckpt_path: Path,
) -> Orchestrator:
    model_config = replace(config.model, train_dit=True)
    model = Orchestrator(
        dit=DiT(dit_config()),
        model_config=model_config,
        bpe_config=config.bpe,
        tokenizer=qwen3_tokenizer(model_config),
        bpe_vocab_size=bpe_vocab_size,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a mapping.")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint state_dict must be a mapping.")
    model.load_state_dict(_model_state_dict(state))
    return model


def _model_state_dict(state: dict[str, Any]) -> dict[str, Tensor]:
    prefix = "model."
    selected: dict[str, Tensor] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        if not isinstance(value, Tensor):
            raise TypeError(f"checkpoint tensor expected for {key}.")
        selected[key.removeprefix(prefix)] = value
    if not selected:
        raise RuntimeError("checkpoint does not contain model.* state_dict entries.")
    return selected


def _seed_generation(seed: int, *, sample_index: int, direction: str) -> None:
    direction_offset = 0 if direction == "source_to_target" else 1
    torch.manual_seed(seed + sample_index * 2 + direction_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + sample_index * 2 + direction_offset)


def _longcat_pair_at(config: DatasetFactoryConfig, index: int) -> LongCatPair:
    if index < 0:
        raise ValueError("sample index must be non-negative.")
    for sample_index, sample in enumerate(training_dataset(config)):
        if sample_index == index:
            return longcat_pair_from_sample(sample)
    raise IndexError(f"sample index {index} is outside the dataset.")


def _directions(pair: LongCatPair) -> tuple[tuple[str, LongCatSide, LongCatSide], ...]:
    return (
        ("source_to_target", pair.source, pair.target),
        ("target_to_source", pair.target, pair.source),
    )


def _encode_frames(bpe: object, ids: Tensor) -> Tensor:
    encode_frames = getattr(bpe, "encode_frames", None)
    if not callable(encode_frames):
        raise TypeError("LongCat BPE tokenizer must provide encode_frames().")
    frames = [[int(value)] for value in ids.reshape(-1).detach().cpu().tolist()]
    return torch.tensor(encode_frames(frames), dtype=torch.long)


def _decode_reference(codec: object, bpe: object, side: LongCatSide) -> Tensor:
    semantic_ids = _roundtrip_semantic_ids(bpe, side.semantic_ids)
    acoustic_ids = side.acoustic_ids.detach().to(dtype=torch.long)
    if acoustic_ids.dim() == 3 and acoustic_ids.size(0) == 1:
        acoustic_ids = acoustic_ids.squeeze(0)
    if acoustic_ids.dim() != 2:
        raise ValueError("LongCat acoustic ids must have shape [nq, time].")
    if semantic_ids.numel() != acoustic_ids.size(-1):
        raise ValueError("semantic and acoustic reference lengths must match.")
    decode = getattr(codec, "decode", None)
    if not callable(decode):
        raise TypeError("LongCat codec must provide decode().")
    audio = decode(semantic_ids.unsqueeze(0), acoustic_ids.unsqueeze(0))
    if not isinstance(audio, Tensor):
        raise TypeError("LongCat codec decode() must return a Tensor.")
    return _first_audio(audio)


def _roundtrip_semantic_ids(bpe: object, ids: Tensor) -> Tensor:
    encode_frames = getattr(bpe, "encode_frames", None)
    expand_ids = getattr(bpe, "expand_ids", None)
    if not callable(encode_frames) or not callable(expand_ids):
        raise TypeError("LongCat BPE tokenizer must provide encode_frames() and expand_ids().")
    frames = [[int(value)] for value in ids.reshape(-1).detach().cpu().tolist()]
    return torch.tensor(_single_codebook_ids(expand_ids(encode_frames(frames))), dtype=torch.long)


def _single_codebook_ids(frames: object) -> list[int]:
    if not isinstance(frames, Sequence) or isinstance(frames, str | bytes):
        raise TypeError("LongCat BPE expand_ids() must return a sequence of frames.")
    ids: list[int] = []
    for frame in frames:
        if isinstance(frame, int) or not isinstance(frame, Sequence):
            raise TypeError("LongCat BPE expand_ids() must return frame sequences.")
        if len(frame) != 1:
            raise ValueError("LongCat semantic BPE must use exactly one codebook.")
        ids.append(int(frame[0]))
    return ids


def _move_generation_batch(batch: GenerationBatch, device: torch.device) -> GenerationBatch:
    return GenerationBatch(
        input_ids=batch.input_ids.to(device=device),
        attention_mask=batch.attention_mask.to(device=device),
    )


def _move_runtime_to_device(value: object, device: torch.device) -> object:
    move = getattr(value, "to", None)
    if callable(move):
        moved = move(device)
        if moved is not None:
            value = moved
    if hasattr(value, "device"):
        setattr(value, "device", device)
    return value


def _first_audio(audio: Tensor) -> Tensor:
    audio = audio.detach().float().cpu()
    if audio.dim() == 3:
        audio = audio[0]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError("audio must have shape [channels, time].")
    return audio.contiguous()


def _preview_ids(ids: Tensor, *, limit: int = 64) -> list[int]:
    return [int(value) for value in ids.reshape(-1).detach().cpu().tolist()[:limit]]


def _save_audio(path: Path, audio: Tensor) -> None:
    import torchaudio

    torchaudio.save(str(path), audio, SAMPLE_RATE)


def _eoa_hit(model: Orchestrator, token_ids: Tensor) -> bool:
    eoa_id = model.embed_tokens.space.special_token_id(AudioBoundary.EOA)
    return bool(token_ids.detach().cpu().eq(eoa_id).any())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint free-running waveform.")
    parser.add_argument("config_name", nargs="?", default="config")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides matching the training run.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--acoustic-sampler",
        choices=tuple(sampler.value for sampler in AcousticSampler),
        default=AcousticSampler.DIAGONAL.value,
    )
    parser.add_argument("--device", default=_default_device())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
