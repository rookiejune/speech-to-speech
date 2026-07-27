from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def build_manifest(
    candidate_path: Path,
    audit_path: Path,
    data_root: Path,
    *,
    artifact_role: str = "stage1_pilot",
    split_method: str = "candidate_artifact",
) -> dict[str, Any]:
    candidate = _load_object(candidate_path, "split candidate")
    audit = _load_object(audit_path, "root audit")
    dataset = _string(candidate, "dataset", "split candidate")
    codec = _string(candidate, "codec", "split candidate")
    split = _optional_string(audit, "split", "root audit", default="train")
    audit_dataset = _optional_string(
        audit,
        "dataset",
        "root audit",
        default=dataset,
    )
    audit_codec = _optional_string(
        audit,
        "codec",
        "root audit",
        default=codec,
    )
    if audit_dataset != dataset:
        raise ValueError("split candidate and root audit datasets do not match.")
    if audit_codec != codec:
        raise ValueError("split candidate and root audit codecs do not match.")

    dataset_length = _dataset_length(audit)
    split_metadata = audit.get("split_candidate")
    if split_metadata is not None:
        audit_method = _string(
            _object(split_metadata, "root audit split_candidate"),
            "method",
            "root audit split_candidate",
        )
        if audit_method != split_method:
            raise ValueError(
                "requested split method does not match the root audit: "
                f"{split_method!r} != {audit_method!r}."
            )
    root_fingerprint = _root_fingerprint(audit.get("files"))
    splits: dict[str, list[int]] = {}
    seen: set[int] = set()
    for label in ("train", "dev", "test"):
        splits[label] = _indices(
            candidate.get(label),
            label=label,
            count=dataset_length,
            seen=seen,
        )
    if len(seen) != dataset_length:
        raise ValueError(
            "split candidate must cover every audited dataset index exactly once."
        )

    return {
        "version": 1,
        "artifact_role": artifact_role,
        "dataset": dataset,
        "codec": codec,
        "split": split,
        "dataset_root": str(data_root.expanduser()),
        "dataset_length": dataset_length,
        "split_method": split_method,
        "source_artifacts": {
            "candidate": str(candidate_path.expanduser()),
            "candidate_sha256": _sha256(candidate_path),
            "audit": str(audit_path.expanduser()),
            "audit_sha256": _sha256(audit_path),
        },
        "root_fingerprint": root_fingerprint,
        "splits": splits,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    candidate = Path(args.candidate).expanduser()
    audit = Path(args.audit).expanduser()
    output = Path(args.output).expanduser()
    manifest = build_manifest(
        candidate,
        audit,
        Path(args.data_root).expanduser(),
        artifact_role=args.artifact_role,
        split_method=args.split_method,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "splits": manifest["splits"]}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-role", default="stage1_pilot")
    parser.add_argument("--split-method", default="candidate_artifact")
    return parser


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} does not exist: {path}") from error
    return _object(payload, label)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return cast(dict[str, Any], value)


def _string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} field {key!r} must be a non-empty string.")
    return value


def _optional_string(
    payload: dict[str, Any],
    key: str,
    label: str,
    *,
    default: str,
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} field {key!r} must be a non-empty string.")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{label} must be a positive integer.")
    return value


def _dataset_length(audit: dict[str, Any]) -> int:
    value = audit.get("dataset_len")
    if value is not None:
        return _positive_int(value, "dataset_len")
    files = audit.get("files")
    if not isinstance(files, list):
        raise TypeError("root audit files must be a list.")
    for offset, item in enumerate(files):
        entry = _object(item, f"root audit files[{offset}]")
        if entry.get("relative_path") != "samples.parquet":
            continue
        parquet = _object(entry.get("parquet"), "root audit samples.parquet parquet")
        return _positive_int(parquet.get("num_rows"), "samples.parquet num_rows")
    raise ValueError("root audit does not contain samples.parquet row count.")


def _indices(
    value: object,
    *,
    label: str,
    count: int,
    seen: set[int],
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"split candidate split {label!r} must be a non-empty list.")
    result: list[int] = []
    for offset, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"split candidate split {label!r} index {offset} must be an integer."
            )
        if item < 0 or item >= count:
            raise IndexError(
                f"split candidate split {label!r} index {offset} is outside "
                f"dataset length {count}: {item}."
            )
        if item in seen:
            raise ValueError(
                f"split candidate repeats dataset index {item} in split {label!r}."
            )
        seen.add(item)
        result.append(item)
    return result


def _root_fingerprint(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        raise TypeError("root audit files must be a list.")
    result: dict[str, str] = {}
    for offset, item in enumerate(value):
        entry = _object(item, f"root audit files[{offset}]")
        relative_path = _string(entry, "relative_path", f"root audit files[{offset}]")
        sha256 = _string(entry, "sha256", f"root audit files[{offset}]")
        if relative_path in result:
            raise ValueError(f"root audit repeats file {relative_path!r}.")
        result[relative_path] = sha256
    if not result:
        raise ValueError("root audit files must not be empty.")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
