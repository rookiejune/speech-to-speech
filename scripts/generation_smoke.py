from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from speech_to_speech.datamodule import Collator
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.model import SpeechToSpeechFlowModel
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import init_runtime
from speech_to_speech.task import Task
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

if __package__:
    from ._generation_benchmark import benchmark_batch
    from ._generation_probe import run, second_step
    from ._generation_reporting import compare, summary
else:
    from _generation_benchmark import benchmark_batch
    from _generation_probe import run, second_step
    from _generation_reporting import compare, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    runtime = init_runtime(
        RuntimeConfig(
            codec=args.codec,
            backbone=args.backbone,
            audio_tokenizer=args.audio_tokenizer,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
    )
    dataset = wmt19_tts_codec(codec=args.codec, split=args.split)
    batch = Collator(runtime, {Task.S2ST: 1.0})([dataset[args.sample_index]])
    request = requests_from_batch(batch)[0]

    model = SpeechToSpeechFlowModel(runtime=runtime).eval()
    with torch.no_grad():
        model.acoustic_prompt_gate.fill_(args.acoustic_prompt_gate)

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
    batch_sizes = _batch_sizes(args.batch_sizes)
    batch_requests = [
        requests_from_batch(
            Collator(runtime, {Task.S2ST: 1.0})([dataset[index]])
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
        acoustic_prompt = batch_request["acoustic_prompt"]
        if acoustic_prompt is not None:
            acoustic_prompt["token_positions"] += prefix_length
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
        "source_acoustic_frames": (
            0
            if request["acoustic_prompt"] is None
            else int(request["acoustic_prompt"]["codes"].size(0))
        ),
        "acoustic_prompt_gate": args.acoustic_prompt_gate,
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
    if not comparison["tokens_equal"]:
        raise RuntimeError("cached and full-recompute greedy tokens differ.")
    if not comparison["cached_finite"] or not comparison["full_finite"]:
        raise RuntimeError("generation produced non-finite acoustic output.")
    if not all(item["tokens_equal"] for item in batch_benchmark):
        raise RuntimeError("batch and per-request greedy tokens differ.")


def _batch_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split(",")]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("batch sizes must be positive integers.")
    return sizes


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-tokenizer", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--acoustic-prompt-gate", type=float, default=1.0)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser


if __name__ == "__main__":
    main()
