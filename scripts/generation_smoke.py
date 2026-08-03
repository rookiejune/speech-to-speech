from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig

from speech_to_speech.callback import OnDeviceCodecMaterializer
from speech_to_speech.datamodule.collate.collator import Collator
from speech_to_speech.datamodule.dataset.speech import load_dataset
from speech_to_speech.datamodule.types import ModelBatch, TrainInput
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.generation.eval.reporting import compare, summary
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.runtime import Runtime, runtime_for_sequence_layout
from speech_to_speech.task import Task

if __package__:
    from ._generation_benchmark import benchmark_batch
    from ._generation_smoke_config import (
        GenerationSmokeConfig,
        generation_smoke as parse_config,
    )
    from ._entry import runtime_config
    from ._generation_probe import run as probe_run, second_step
else:
    from _generation_benchmark import benchmark_batch
    from _generation_smoke_config import (
        GenerationSmokeConfig,
        generation_smoke as parse_config,
    )
    from _entry import runtime_config
    from _generation_probe import run as probe_run, second_step


@hydra.main(version_base=None, config_path="../configs", config_name="generation_smoke")
def main(config: DictConfig) -> None:
    run(parse_config(config))


def run(config: GenerationSmokeConfig) -> None:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rt_config = runtime_config(config.runtime)
    device = None if rt_config.device is None else torch.device(rt_config.device)
    _seed(config.seed, device or torch.device("cpu"))
    runtime = runtime_for_sequence_layout(rt_config, config.audio_sequence_layout)
    task = Task(config.task)
    dataset = load_dataset(config.datamodule.dataset, runtime)
    collator = Collator(
        runtime,
        {task: 1.0},
        encode_missing_codes=config.datamodule.encode_missing_codes,
        interleave_audio_frames=config.datamodule.interleave_audio_frames,
        mask_text_ratio=config.datamodule.mask_text_ratio,
        mask_audio_ratio=config.datamodule.mask_audio_ratio,
    )
    batch = _prepared_batch(
        collator([dataset[config.sample_index]]),
        runtime,
        config,
        device=device,
    )
    request = requests_from_batch(batch)[0]

    model = FlowModel(runtime=runtime).eval()

    probe = second_step(model, request)
    cached = probe_run(
        model,
        request,
        seed=config.seed,
        max_new_tokens=config.max_new_tokens,
        use_cache=True,
    )
    full = probe_run(
        model,
        request,
        seed=config.seed,
        max_new_tokens=config.max_new_tokens,
        use_cache=False,
    )
    comparison = compare(cached, full)
    batch_sizes = config.batch_sizes
    batch_requests = [
        requests_from_batch(
            _prepared_batch(
                collator([dataset[index]]),
                runtime,
                config,
                device=device,
            )
        )[0]
        for index in range(config.sample_index, config.sample_index + max(batch_sizes))
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
            seed=config.seed,
            max_new_tokens=config.max_new_tokens,
        )
        for batch_size in batch_sizes
    ]

    result = {
        "task": task.value,
        "dataset": {
            "split": config.datamodule.dataset.split,
            "data_root": config.datamodule.dataset.root,
            "filter": config.datamodule.dataset.filter,
            "split_manifest": config.datamodule.dataset.split_manifest,
            "split_label": config.datamodule.dataset.split_label,
            "split_manifest_sha256": (
                _sha256(config.datamodule.dataset.split_manifest)
                if config.datamodule.dataset.split_manifest
                else None
            ),
        },
        "sample_index": config.sample_index,
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
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


def _seed(seed: int, device: torch.device) -> None:
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def _prepared_batch(
    batch: TrainInput | object,
    runtime: Runtime,
    config: GenerationSmokeConfig,
    *,
    device: torch.device | None,
) -> ModelBatch:
    if not isinstance(batch, ModelBatch):
        if config.datamodule.encode_missing_codes:
            return OnDeviceCodecMaterializer(runtime)(
                cast(TrainInput, batch),
                device=device,
            )
        raise TypeError(
            "generation smoke requires prepared codec data; set "
            "datamodule.encode_missing_codes=true to online encode raw waveform samples."
        )
    return batch


if __name__ == "__main__":
    main()
