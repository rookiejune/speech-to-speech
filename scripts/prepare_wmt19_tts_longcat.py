from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

import torch
from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    DatasetWriter,
    FilterRule,
    Modality,
    ModalityMaterializer,
    Preset,
    Role,
    Sample,
    Source,
    Spec,
    TextItem,
    TextView,
    ViewMaterializer,
)
from anydataset.provider.longcat import LongCatProvider
from anydataset.provider.moss_tts import MossTTSProvider
from anydataset.quality.speech import Predicate as SpeechQuality
from anydataset.quality.speech import Profile as SpeechQualityProfile
from anydataset.store.reader import read_store_dataset
from anytrain.evaluator.speech import SpeechEvaluator, UTMOSEvaluator, WhisperASREvaluator
from anytrain.tts import TTSOptions


DEFAULT_ROOT = Path("storage/wmt19-zh-en-tts-longcat-1000")
DEFAULT_HF_HOME = Path("/mnt/pami202/zhuyin/huggingface")
DEFAULT_LONGCAT_CACHE = DEFAULT_HF_HOME / "longcat-audio-codec"
DEFAULT_TORCH_HOME = Path("/mnt/pami202/zhuyin/torch")
DEFAULT_WHISPER_ROOT = Path("/mnt/pami202/zhuyin/whisper")


@dataclass(frozen=True)
class Paths:
    root: Path
    text: Path
    tts_delta: Path
    tts: Path
    longcat_delta: Path
    full: Path
    filter_cache: Path
    reports: Path


@dataclass(frozen=True)
class Stage:
    name: str
    path: str | None
    reused: bool
    seconds: float | None
    sample_count: int | None = None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_env(args)
    paths = resolve_paths(args.root)
    paths.reports.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    stages = [
        write_text_store(args, paths),
        write_tts_store(args, paths),
        write_longcat_store(args, paths),
    ]
    filter_summary = apply_speech_filter(args, paths)
    summary = {
        "config": run_config(args, paths),
        "stages": [asdict(stage) for stage in stages],
        "filter": filter_summary,
        "seconds": time.perf_counter() - started_at,
    }
    write_json(paths.reports / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def write_text_store(args: argparse.Namespace, paths: Paths) -> Stage:
    if is_ready_store(paths.text):
        return ready_stage("text", paths.text)

    start = time.perf_counter()
    samples = limited_wmt19_samples(
        split=args.split,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        limit=args.limit,
    )
    DatasetWriter(
        paths.text,
        dataset_id=args.dataset_id,
        split=args.split,
        max_shard_samples=args.max_shard_samples,
    ).write(samples)
    return Stage(
        name="text",
        path=str(paths.text),
        reused=False,
        seconds=time.perf_counter() - start,
        sample_count=len(read_store_dataset(paths.text)),
    )


def write_tts_store(args: argparse.Namespace, paths: Paths) -> Stage:
    if is_ready_store(paths.tts):
        return ready_stage("tts", paths.tts)

    start = time.perf_counter()
    if not is_ready_store(paths.tts_delta):
        ModalityMaterializer(
            paths.tts_delta,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
            batch_size=args.tts_batch_size,
        ).write(
            dataset_factory=TextStoreFactory(paths.text, args.split),
            provider_factory=MossTTSFactory(args),
            devices=args.devices,
        )

    copy_store(paths.text, paths.tts)
    merge_store(paths.tts, paths.tts_delta, args.split)
    return Stage(
        name="tts",
        path=str(paths.tts),
        reused=False,
        seconds=time.perf_counter() - start,
        sample_count=len(read_store_dataset(paths.tts)),
    )


def write_longcat_store(args: argparse.Namespace, paths: Paths) -> Stage:
    if is_ready_store(paths.full):
        return ready_stage("longcat", paths.full)

    start = time.perf_counter()
    if not is_ready_store(paths.longcat_delta):
        ViewMaterializer(
            paths.longcat_delta,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
            batch_size=args.longcat_batch_size,
        ).write(
            dataset_factory=TTSStoreFactory(paths.tts, args.split),
            provider_factory=LongCatFactory(args),
            devices=args.devices,
        )

    copy_store(paths.tts, paths.full)
    merge_store(paths.full, paths.longcat_delta, args.split)
    return Stage(
        name="longcat",
        path=str(paths.full),
        reused=False,
        seconds=time.perf_counter() - start,
        sample_count=len(read_store_dataset(paths.full)),
    )


def apply_speech_filter(args: argparse.Namespace, paths: Paths) -> dict[str, Any]:
    dataset = AnyDataset(
        Spec(source=Source.STORE, path=str(paths.full), split=args.split),
        cache_root=paths.root / "dataset-cache",
    )
    evaluator = SpeechEvaluator(
        asr=WhisperASREvaluator(
            model_name=args.whisper_model,
            device=args.quality_device,
            download_root=args.whisper_root,
            decode_options={"temperature": 0.0},
        ),
        utmos=UTMOSEvaluator(
            device=args.quality_device,
            backend_load_options={"trust_repo": True},
        ),
    )
    predicate = SpeechQuality(
        profile=SpeechQualityProfile(
            min_utmos=args.min_utmos,
            max_wer=args.max_wer,
            min_chrf=args.min_chrf,
            min_bleu=args.min_bleu,
        ),
        evaluator=evaluator,
        decode_options={},
    )
    rule = FilterRule(args.filter_rule_name, predicate)
    start = time.perf_counter()
    result = rule.apply(
        dataset,
        metrics=True,
        num_workers=1,
        commit_samples=args.filter_commit_samples,
        max_shard_samples=args.max_shard_samples,
        cache_root=paths.filter_cache,
    )
    metrics_report = paths.reports / "speech_quality_metrics.jsonl"
    write_metrics_jsonl(metrics_report, result.iter_metrics())
    return {
        "seconds": time.perf_counter() - start,
        "counts": dict(result.counts),
        "labels": list(result.labels),
        "accepted": result.counts.get("accept", 0),
        "rejected": result.counts.get("reject", 0),
        "cache_path": str(result.cache_path),
        "metrics_path": None if result.metrics_path is None else str(result.metrics_path),
        "metrics_jsonl": str(metrics_report),
        "preview": preview_metrics(metrics_report, limit=args.preview_metrics),
    }


@dataclass(frozen=True)
class TextStoreFactory:
    path: Path
    split: str

    def __call__(self) -> AnyDataset:
        return store_dataset(self.path, self.split)


@dataclass(frozen=True)
class TTSStoreFactory:
    path: Path
    split: str

    def __call__(self) -> AnyDataset:
        return store_dataset(self.path, self.split)


@dataclass(frozen=True)
class MossTTSFactory:
    args: argparse.Namespace

    def __call__(self, device: str) -> MossTTSProvider:
        return MossTTSProvider(
            resolve_pretrained_source(self.args.moss_model, self.args),
            options=tts_options(self.args),
            cache_dir=self.args.hf_home,
            codec_model=resolve_pretrained_source(self.args.moss_codec_model, self.args),
            device=device if device != "cpu" else None,
            local_files_only=self.args.local_files_only,
            trust_remote_code=self.args.trust_remote_code,
            dtype=self.args.tts_dtype,
            attn_implementation=self.args.tts_attn_implementation,
            runtime_kwargs=tts_runtime_kwargs(self.args),
        )


@dataclass(frozen=True)
class LongCatFactory:
    args: argparse.Namespace

    def __call__(self, device: str) -> LongCatProvider:
        return LongCatProvider(
            cache_dir=self.args.longcat_cache_dir,
            decoders=(self.args.longcat_decoder,),
            device=device if device != "cpu" else None,
            local_files_only=self.args.local_files_only,
        )


def limited_wmt19_samples(
    *,
    split: str,
    source_lang: str,
    target_lang: str,
    limit: int,
) -> Iterable[Sample]:
    dataset = Preset.WMT19.create(
        split=split,
        source_lang=source_lang,
        target_lang=target_lang,
        streaming=True,
    )
    yield from islice(dataset, limit)


def copy_store(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def merge_store(base: Path, delta: Path, split: str) -> None:
    store_dataset(base, split).merge(store_dataset(delta, split))


def store_dataset(path: Path, split: str) -> AnyDataset:
    return AnyDataset(
        Spec(source=Source.STORE, path=str(path), split=split),
        cache_root=path.parent / "dataset-cache",
    )


def is_ready_store(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        read_store_dataset(path)
    except Exception:
        return False
    return True


def ready_stage(name: str, path: Path) -> Stage:
    store = read_store_dataset(path)
    return Stage(
        name=name,
        path=str(path),
        reused=True,
        seconds=None,
        sample_count=len(store),
    )


def tts_options(args: argparse.Namespace) -> TTSOptions:
    kwargs: dict[str, Any] = {
        "sample_rate": args.tts_sample_rate,
        "max_new_tokens": args.tts_max_new_tokens,
        "temperature": args.tts_temperature,
        "top_p": args.tts_top_p,
        "seed": args.tts_seed,
    }
    return TTSOptions(**{key: value for key, value in kwargs.items() if value is not None})


def tts_runtime_kwargs(args: argparse.Namespace) -> dict[str, object]:
    output: dict[str, object] = {"do_sample": args.tts_do_sample}
    return {key: value for key, value in output.items() if value is not None}


def resolve_pretrained_source(source: str, args: argparse.Namespace) -> str:
    path = Path(source).expanduser()
    if path.exists() or not args.local_files_only:
        return str(path) if path.exists() else source

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return source

    return snapshot_download(
        source,
        cache_dir=str(args.hf_home / "hub"),
        local_files_only=True,
    )


def write_metrics_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def preview_metrics(path: Path, *, limit: int) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in islice(handle, limit):
            output.append(cast(Mapping[str, Any], json.loads(line)))
    return output


def inspect_sample(path: Path, split: str) -> dict[str, Any]:
    dataset = store_dataset(path, split)
    sample = dataset[0]
    output: dict[str, Any] = {}
    for role in (Role.SOURCE, Role.TARGET):
        text = cast(TextItem, sample[role, Modality.TEXT])
        audio = cast(AudioItem, sample[role, Modality.AUDIO])
        waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        longcat = audio.views[AudioView.LONGCAT]
        output[role.value] = {
            "text": text.views[TextView.TEXT],
            "sample_rate": int(sample_rate),
            "waveform_shape": list(torch.as_tensor(waveform).shape),
            "longcat_shapes": {
                key: list(value.shape) if hasattr(value, "shape") else None
                for key, value in longcat.items()
            },
        }
    return output


def configure_env(args: argparse.Namespace) -> None:
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_CACHE"] = str(args.hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(args.hf_home / "datasets")
    os.environ["ANYTRAIN_HOME"] = str(args.anytrain_home)
    os.environ["TORCH_HOME"] = str(args.torch_home)
    os.environ["ANYTRAIN_WHISPER_ROOT"] = str(args.whisper_root)


def resolve_paths(root: Path) -> Paths:
    root = root.expanduser().resolve()
    return Paths(
        root=root,
        text=root / "text-store",
        tts_delta=root / "tts-delta",
        tts=root / "tts-store",
        longcat_delta=root / "longcat-delta",
        full=root / "full-store",
        filter_cache=root / "filter-cache",
        reports=root / "reports",
    )


def run_config(args: argparse.Namespace, paths: Paths) -> dict[str, Any]:
    return {
        "root": str(paths.root),
        "split": args.split,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "limit": args.limit,
        "devices": args.devices,
        "tts_batch_size": args.tts_batch_size,
        "longcat_batch_size": args.longcat_batch_size,
        "moss_model": args.moss_model,
        "moss_codec_model": args.moss_codec_model,
        "longcat_decoder": args.longcat_decoder,
        "quality_device": args.quality_device,
        "whisper_model": args.whisper_model,
        "filter_rule_name": args.filter_rule_name,
        "thresholds": {
            "min_utmos": args.min_utmos,
            "max_wer": args.max_wer,
            "min_chrf": args.min_chrf,
            "min_bleu": args.min_bleu,
        },
        "first_sample": inspect_sample(paths.full, args.split)
        if is_ready_store(paths.full)
        else None,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WMT19 zh-en text with TTS waveform, LongCat views, and speech quality filters."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--limit", type=positive_int, default=1000)
    parser.add_argument("--dataset-id", default="wmt19-zh-en-tts-longcat-1000")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-shard-samples", type=positive_int, default=100_000)
    parser.add_argument("--tts-batch-size", type=positive_int, default=1)
    parser.add_argument("--longcat-batch-size", type=positive_int, default=1)
    parser.add_argument("--moss-model", default="OpenMOSS-Team/MOSS-TTS-v1.5")
    parser.add_argument("--moss-codec-model", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--tts-sample-rate", type=positive_int, default=None)
    parser.add_argument("--tts-max-new-tokens", type=positive_int, default=None)
    parser.add_argument("--tts-temperature", type=float, default=None)
    parser.add_argument("--tts-top-p", type=float, default=None)
    parser.add_argument("--tts-seed", type=int, default=None)
    parser.add_argument("--tts-do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tts-dtype", default="auto")
    parser.add_argument("--tts-attn-implementation", default="sdpa")
    parser.add_argument("--longcat-cache-dir", type=Path, default=DEFAULT_LONGCAT_CACHE)
    parser.add_argument(
        "--longcat-decoder",
        choices=("16k_4codebooks", "24k_2codebooks", "24k_4codebooks"),
        default="16k_4codebooks",
    )
    parser.add_argument("--quality-device", default="cuda:0")
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument("--whisper-root", type=Path, default=DEFAULT_WHISPER_ROOT)
    parser.add_argument("--min-utmos", type=float, default=3.0)
    parser.add_argument("--max-wer", type=float, default=0.4)
    parser.add_argument("--min-chrf", type=float, default=50.0)
    parser.add_argument("--min-bleu", type=float, default=None)
    parser.add_argument(
        "--filter-rule-name",
        default="wmt19_zh_en_tts_speech_quality_v1_utmos3_wer04_chrf50",
    )
    parser.add_argument("--filter-commit-samples", type=positive_int, default=16)
    parser.add_argument("--preview-metrics", type=positive_int, default=5)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--anytrain-home", type=Path, default=Path("/mnt/pami202/zhuyin"))
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


if __name__ == "__main__":
    main()
