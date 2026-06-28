from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anydataset import AnyDataset, DatasetWriter, FilterRule, Source, Spec
from anydataset.filter import FilteredDataset
from anydataset.quality.translation import Predicate as TranslationQuality
from anydataset.store.reader import read_store_dataset
from anydataset.types import Preset, Sample


DEFAULT_ROOT = Path("storage/wmt19-zh-en-tts-longcat-1000")


@dataclass(frozen=True)
class Paths:
    root: Path
    full: Path
    filter_cache: Path
    reports: Path
    train: Path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args.root)
    paths.reports.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    dataset = AnyDataset(
        Spec(source=Source.STORE, path=str(paths.full), split=args.split),
        cache_root=paths.root / "dataset-cache",
    )
    speech = FilteredDataset(
        dataset,
        FilterRule(args.speech_rule_name, missing_speech_cache),
        labels=args.speech_labels,
        cache_root=paths.filter_cache,
    )
    text_rule = FilterRule(
        args.text_rule_name,
        TranslationQuality.from_preset(
            Preset.WMT19,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        ),
    )
    text_result = text_rule.apply(
        speech,
        metrics=True,
        num_workers=args.num_workers,
        commit_samples=args.commit_samples,
        max_shard_samples=args.max_shard_samples,
        cache_root=paths.filter_cache,
    )
    selected = text_result.select(*args.text_labels)

    if args.write_store:
        DatasetWriter(
            paths.train,
            dataset_id=args.dataset_id,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
        ).write(selected)

    metrics_report = paths.reports / "translation_quality_metrics.jsonl"
    write_metrics_jsonl(metrics_report, text_result.iter_metrics())
    summary = {
        "config": {
            "root": str(paths.root),
            "source_store": str(paths.full),
            "train_store": str(paths.train),
            "split": args.split,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "speech_rule_name": args.speech_rule_name,
            "speech_labels": list(args.speech_labels),
            "text_rule_name": args.text_rule_name,
            "text_labels": list(args.text_labels),
        },
        "counts": {
            "base": len(dataset),
            "speech_selected": len(speech),
            "text": dict(text_result.counts),
            "selected": len(selected),
        },
        "cache": {
            "speech": str(speech.cache_path),
            "text": str(text_result.cache_path),
            "text_metrics": None
            if text_result.metrics_path is None
            else str(text_result.metrics_path),
        },
        "metrics_jsonl": str(metrics_report),
        "preview": preview_metrics(metrics_report, limit=args.preview_metrics),
        "flag_counts": flag_counts(metrics_report),
        "seconds": time.perf_counter() - started_at,
    }
    if paths.train.exists():
        summary["train_store"] = {
            "path": str(paths.train),
            "sample_count": len(read_store_dataset(paths.train)),
        }
    write_json(paths.reports / "quality_filter_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def resolve_paths(root: Path) -> Paths:
    root = root.expanduser().resolve()
    return Paths(
        root=root,
        full=root / "full-store",
        filter_cache=root / "filter-cache",
        reports=root / "reports",
        train=root / "train-store",
    )


def missing_speech_cache(sample: Sample) -> str:
    del sample
    raise RuntimeError(
        "speech quality cache is missing; run prepare_wmt19_tts_longcat.py first."
    )


def write_metrics_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def preview_metrics(path: Path, *, limit: int) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            output.append(json.loads(line))
    return output


def flag_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metrics = row.get("metrics", {})
            if isinstance(metrics, Mapping):
                flags = metrics.get("flags", [])
                if isinstance(flags, list):
                    counts.update(str(flag) for flag in flags)
    return dict(counts)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter prepared WMT19 TTS/LongCat data by speech and text quality."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument(
        "--speech-rule-name",
        default="wmt19_zh_en_tts_speech_quality_v1_utmos3_wer04_chrf50",
    )
    parser.add_argument("--speech-labels", nargs="+", default=("accept",))
    parser.add_argument(
        "--text-rule-name",
        default="wmt19_zh_en_text_quality_rules_v1_clean_usable",
    )
    parser.add_argument("--text-labels", nargs="+", default=("clean", "usable"))
    parser.add_argument("--dataset-id", default="wmt19-zh-en-tts-longcat-quality-train")
    parser.add_argument("--num-workers", type=positive_int, default=1)
    parser.add_argument("--commit-samples", type=positive_int, default=16)
    parser.add_argument("--max-shard-samples", type=positive_int, default=100_000)
    parser.add_argument("--preview-metrics", type=positive_int, default=5)
    parser.add_argument("--write-store", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


if __name__ == "__main__":
    main()
