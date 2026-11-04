from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data import (
    DatasetConfig,
    build_targets,
    deterministic_split_indices,
    resolve_dataset_path,
    serialize_http_request,
)
from .common import append_record, array, digest, new_directory, write_json


def prepare(dataset_config, output, max_rows=0, seed=14):
    """Preserve original split membership; optionally sample within each split."""
    start = time.perf_counter()
    config = DatasetConfig.from_yaml(dataset_config)
    source = resolve_dataset_path(config)
    if config.format != "csv":
        raise ValueError("This independent streaming workflow currently requires CSV input.")
    if max_rows < 0 or max_rows and max_rows < 6:
        raise ValueError("--max-rows must be zero or at least 6.")
    out = new_directory(output)
    # Splitting needs the labels, not 900k serialized HTTP strings in memory.
    columns = set(config.label_columns.values())
    for name in (
        config.split.time_column if config.split.strategy == "time" else None,
        config.split.group_column if config.split.strategy == "group" else None,
    ):
        if name:
            columns.add(name)
    frame = pd.read_csv(source, usecols=sorted(columns), low_memory=False)
    splits = deterministic_split_indices(frame, config)
    original_n = len(frame)
    membership = np.empty(original_n, dtype=np.int8)
    selected = np.zeros(original_n, dtype=bool)
    rng = np.random.default_rng(seed)
    for number, indices in enumerate(splits):
        membership[indices] = number
        count = (
            len(indices)
            if not max_rows
            else min(len(indices), max(2, round(max_rows * len(indices) / original_n)))
        )
        chosen = indices if count == len(indices) else rng.choice(indices, count, replace=False)
        selected[chosen] = True
    del frame
    n = int(selected.sum())
    split_array = array(out, "splits.npy", (n,), np.int8)
    labels = array(out, "labels.npy", (n, len(config.label_columns)), np.int8)
    hashes = array(out, "input_hashes.npy", (n,), "S64")
    # IDs are strings; fail rather than truncate unusually long custom identifiers.
    id_width = 160 if config.id_column else 24
    ids = array(out, "row_ids.npy", (n,), f"U{id_width}")
    used = set(config.text_columns.values()) | set(config.label_columns.values())
    if config.id_column:
        used.add(config.id_column)
    offset = position = 0
    with (out / "requests.jsonl").open("w", encoding="utf-8") as target:
        for chunk in tqdm(
            pd.read_csv(
                source,
                usecols=sorted(used),
                chunksize=2000,
                dtype={c: str for c in config.text_columns.values()},
            ),
            total=(original_n + 1999) // 2000,
            desc="Prepare requests",
        ):
            chunk_labels = build_targets(chunk, config.label_columns)
            # Keep the model's existing serialization and missing-value behavior.
            for j in np.flatnonzero(selected[offset : offset + len(chunk)]):
                row = chunk.iloc[int(j)].to_dict()
                original_position = offset + int(j)
                row_id = (
                    str(row[config.id_column]) if config.id_column else f"row-{original_position}"
                )
                if len(row_id) > id_width:
                    raise ValueError(f"Request ID exceeds {id_width} characters.")
                fields = {
                    k: "" if pd.isna(row.get(v)) else str(row[v])
                    for k, v in config.text_columns.items()
                }
                fingerprint = hashlib.sha256(
                    serialize_http_request(row, config.text_columns).encode("utf-8")
                ).hexdigest()
                split_array[position] = membership[original_position]
                labels[position] = chunk_labels[int(j)]
                hashes[position] = fingerprint
                ids[position] = row_id
                append_record(
                    target,
                    {
                        "position": position,
                        "row_id": row_id,
                        "fields": fields,
                        "input_hash": fingerprint,
                    },
                )
                position += 1
            offset += len(chunk)
    for values in (split_array, labels, hashes, ids):
        values.flush()
    if position != n or len(np.unique(ids)) != n:
        raise ValueError("Missing rows or duplicate request IDs in the prepared dataset.")
    elapsed = time.perf_counter() - start
    manifest = {
        "complete": True,
        "rows": n,
        "original_rows": original_n,
        "sampled": n < original_n,
        "dataset_sha256": digest(source),
        "requests_sha256": digest(out / "requests.jsonl"),
        "label_names": list(config.label_columns),
        "text_columns": config.text_columns,
        "split_config": asdict(config.split),
        "sampling_seed": seed,
        "split_counts": dict(Counter(("train", "validation", "test")[int(s)] for s in split_array)),
        "seconds": elapsed,
        "rows_per_second": n / max(elapsed, 0.001),
        "array_sha256": {
            name: digest(out / name)
            for name in ("splits.npy", "labels.npy", "input_hashes.npy", "row_ids.npy")
        },
        "limitations": [
            "CSV reconstruction cannot recover omitted raw-wire fields.",
            "Row stratification does not guarantee content disjointness.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    return manifest
