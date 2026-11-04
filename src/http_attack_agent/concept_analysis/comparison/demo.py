"""Synthetic independent-detector fixture. Not a real CRS/SecBERT experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import append_record, digest, new_directory, write_json


def fixture(output):
    root = new_directory(output)
    prepared, waf, encoded, checkpoint = [
        root / name for name in ("prepared", "waf", "encoded", "checkpoint")
    ]
    for directory in (prepared, waf, encoded, checkpoint):
        directory.mkdir()
    rng = np.random.default_rng(20)
    n = 600
    categories = np.tile([0, 1, 2], n // 3)
    labels = np.column_stack([categories == 1, categories == 2, np.zeros(n, bool)]).astype(np.int8)
    label_names = ["sql_injection", "path_traversal", "fake_source_data"]
    ids = np.asarray([f"demo-{i}" for i in range(n)], dtype=str)
    splits = np.asarray([0] * 360 + [1] * 120 + [2] * 120, dtype=np.int8)
    hashes = np.asarray([f"{i:064x}" for i in range(n)], dtype="S64")
    hashes[480:482] = hashes[:2]  # known historical duplicate inputs
    model_attack = categories != 0
    crs_attack = model_attack.copy()
    model_attack[rng.choice(n, 60, replace=False)] ^= True
    crs_attack[rng.choice(n, 100, replace=False)] ^= True
    logits = np.full((n, 3), -3.0, dtype=np.float32)
    for i in range(n):
        if model_attack[i]:
            logits[i, 1 if categories[i] == 2 else 0] = 3
    logits += rng.normal(0, 0.8, logits.shape).astype(np.float32)
    scores = np.column_stack([np.where(crs_attack, 7.0, 1.0), np.full(n, 5.0)])
    scores[490:495] = np.nan
    columns = {"method": "method", "request_target": "request_target"}
    with (prepared / "requests.jsonl").open("w", encoding="utf-8") as target:
        for i in range(n):
            append_record(
                target,
                {
                    "position": i,
                    "row_id": str(ids[i]),
                    "fields": {"method": "GET", "request_target": f"/?id={i}"},
                    "input_hash": hashes[i].decode(),
                },
            )
    for name, data in [
        ("labels", labels),
        ("splits", splits),
        ("row_ids", ids),
        ("input_hashes", hashes),
    ]:
        np.save(prepared / f"{name}.npy", data)
    checksum = digest(prepared / "requests.jsonl")
    write_json(
        prepared / "manifest.json",
        {
            "complete": True,
            "rows": n,
            "original_rows": n,
            "sampled": True,
            "synthetic": True,
            "requests_sha256": checksum,
            "label_names": label_names,
            "text_columns": columns,
        },
    )
    np.save(waf / "decision_scores.npy", scores)
    catalog = [
        {"rule_id": "942100", "tags": ["attack-sqli", "capec/1000/152/248/66"]},
        {"rule_id": "930100", "tags": ["attack-lfi", "capec/1000/255/153/126"]},
    ]
    write_json(waf / "rule_catalog.json", catalog)
    with (waf / "events.jsonl").open("w", encoding="utf-8") as target:
        for i in range(n):
            if crs_attack[i] and np.isfinite(scores[i]).all():
                rule = catalog[1 if categories[i] == 2 else 0]
                append_record(target, {**rule, "position": i, "row_id": str(ids[i])})
    write_json(
        waf / "manifest.json",
        {
            "complete": True,
            "rows": n,
            "prepared_sha256": checksum,
            "synthetic": True,
            "decision_scores_sha256": digest(waf / "decision_scores.npy"),
            "rule_catalog_sha256": digest(waf / "rule_catalog.json"),
            "events_sha256": digest(waf / "events.jsonl"),
            "engine": "synthetic fixture",
            "crs_tree_sha256": "synthetic",
            "paranoia_level": 1,
            "seconds": 0.0,
            "rows_per_second": None,
            "workers": 1,
        },
    )
    metadata = {
        "label_names": label_names,
        "text_columns": columns,
        "max_length": 256,
        "test_metrics": {"thresholds": {name: 0.5 for name in label_names}},
    }
    write_json(checkpoint / "metadata.json", metadata)
    np.save(encoded / "logits.npy", logits)
    write_json(
        encoded / "manifest.json",
        {
            "complete": True,
            "rows": n,
            "prepared_sha256": checksum,
            "checkpoint_sha256": "synthetic",
            "checkpoint_metadata_sha256": digest(checkpoint / "metadata.json"),
            "array_sha256": {"logits.npy": digest(encoded / "logits.npy")},
            "device": "synthetic",
            "batch_size": 32,
            "seconds": 0.0,
            "rows_per_second": None,
        },
    )
    return {"prepared": prepared, "annotations": waf, "encoded": encoded, "checkpoint": checkpoint}


def demo(output):
    from .evaluate import compare

    args = fixture(output)
    result = compare(**args, output=Path(output) / "comparison", bootstrap=50)
    result["warning"] = (
        "Synthetic fixture only. This demonstrates reporting, not model superiority."
    )
    return result
