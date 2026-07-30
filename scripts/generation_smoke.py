from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from speech_to_speech.datamodule.collate.collator import Collator
from speech_to_speech.datamodule.dataset.speech import DatasetConfig, load_dataset
from speech_to_speech.datamodule.types import ModelBatch
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.generation.eval.reporting import compare, summary
from speech_to_speech.model.acoustic import FlowModel
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import Runtime
from speech_to_speech.task import Task

if __package__:
    from ._generation_benchmark import benchmark_batch
    from ._generation_probe import run, second_step
else:
    from _generation_benchmark import benchmark_batch
    from _generation_probe import run, second_step


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed(args.seed, torch.device(args.device))
    runtime = Runtime(
        RuntimeConfig(
            codec=args.codec,
            backbone=args.backbone,
            audio_tokenizer=args.audio_tokenizer,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
    )
    dataset = load_dataset(DatasetConfig(split=args.split), runtime)
    batch = _prepared_batch(
        Collator(runtime, {Task.S2ST: 1.0})([dataset[args.sample_index]])
    )
    request = requests_from_batch(batch)[0]

    model = FlowModel(runtime=runtime).eval()

    probe = second_step(model, request)
    cached = run(
        model,
        request,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    full = run(
        model,
        request,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        use_cache=False,
    )
    comparison = compare(cached, full)
    batch_sizes = args.batch_sizes
    batch_requests = [
        requests_from_batch(
            _prepared_batch(Collator(runtime, {Task.S2ST: 1.0})([dataset[index]]))
        )[0]
        for index in range(args.sample_index, args.sample_index + max(batch_sizes))
    ]
    for prefix_length, batch_request in enumerate(batch_requests):
        if prefix_length == 0:
            continue
        prefix = batch_request["prompt_ids"].new_full(
            (prefix_length,), runtime.bos_token_id
        )
        batch_request["prompt_ids"] = torch.cat((prefix, batch_request["prompt_ids"]))
    batch_benchmark = [
        benchmark_batch(
            model,
            batch_requests[:batch_size],
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        for batch_size in batch_sizes
    ]

    result = {
        "task": Task.S2ST.value,
        "sample_index": args.sample_index,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt_tokens": int(request["prompt_ids"].numel()),
        "second_step_probe": probe,
        "cached": summary(cached),
        "full_recompute": summary(full),
        "comparison": comparison,
        "batch_benchmark": batch_benchmark,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    _validate(comparison, batch_benchmark)


def _validate(
    comparison: Mapping[str, Any],
    batch_benchmark: Sequence[Mapping[str, Any]],
) -> None:
    if not comparison["tokens_equal"]:
        raise RuntimeError("cached and full-recompute greedy tokens differ.")
    if not comparison["cached_finite"] or not comparison["full_finite"]:
        raise RuntimeError("generation produced non-finite acoustic output.")
    if not all(item["tokens_equal"] for item in batch_benchmark):
        raise RuntimeError("batch and per-request greedy tokens differ.")
    if not all(item["finite"] for item in batch_benchmark):
        raise RuntimeError(
            "batch or per-request generation produced non-finite acoustic output."
        )


def _batch_sizes(value: object) -> list[int]:
    if not isinstance(value, str):
        raise TypeError("batch sizes must be a comma-separated string.")
    items = value.split(",")
    if not items or any(not item.strip() for item in items):
        raise ValueError("batch sizes must be positive integers.")
    try:
        return [_positive_int(item) for item in items]
    except (TypeError, ValueError) as error:
        raise ValueError("batch sizes must be positive integers.") from error


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("value must be a positive integer.")
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError("value must be a positive integer.") from error
    if result < 1:
        raise ValueError("value must be a positive integer.")
    return result


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("value must be a non-negative integer.")
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError("value must be a non-negative integer.") from error
    if result < 0:
        raise ValueError("value must be a non-negative integer.")
    return result


def _seed(seed: int, device: torch.device) -> None:
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _prepared_batch(batch: ModelBatch | object) -> ModelBatch:
    if not isinstance(batch, ModelBatch):
        raise TypeError("generation smoke requires prepared codec data.")
    return batch


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-tokenizer", required=True)
    parser.add_argument("--sample-index", type=_non_negative_int, default=0)
    parser.add_argument("--batch-sizes", type=_batch_sizes, default="1,2,4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser


if __name__ == "__main__":
    main()
