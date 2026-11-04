from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def append_record(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    return path


def array(path, name, shape, dtype):
    return np.lib.format.open_memmap(Path(path) / name, mode="w+", shape=shape, dtype=dtype)


def checked_manifest(path):
    manifest = read_json(Path(path) / "manifest.json")
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete preparation: {path}")
    if digest(Path(path) / "requests.jsonl") != manifest["requests_sha256"]:
        raise ValueError("Prepared requests changed after their manifest was written.")
    for name, checksum in manifest.get("array_sha256", {}).items():
        if digest(Path(path) / name) != checksum:
            raise ValueError(f"Prepared array changed: {name}")
    return manifest
