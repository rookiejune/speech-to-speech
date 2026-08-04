from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024
_DIRECTORY_GRAMMAR = b"speech-to-speech-artifact-directory-v1\0"


def content_sha256(path: Path) -> str:
    """Hash a file or a directory without including its machine-local root path."""
    if path.is_file():
        return _file_sha256(path)
    if path.is_dir():
        return _directory_sha256(path)
    if not path.exists():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    raise ValueError(f"artifact must be a regular file or directory: {path}")


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256(_DIRECTORY_GRAMMAR)
    entries = sorted(
        root.rglob("*"),
        key=lambda entry: entry.relative_to(root).as_posix(),
    )
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink() and entry.is_dir():
            raise ValueError(
                "artifact directory must not contain directory symlinks: "
                f"{relative}"
            )
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError(
                "artifact directory entries must be regular files: "
                f"{relative}"
            )
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(bytes.fromhex(_file_sha256(entry)))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
