from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..data import serialize_http_request
from ..models.checkpoint import load_local_checkpoint
from .common import array, checked_manifest, digest, new_directory, records, write_json
from .replay import _batches


def encode(prepared, checkpoint, output, batch_size=32, device=None):
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    manifest = checked_manifest(prepared)
    metadata_path = Path(checkpoint) / "metadata.json"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, metadata = load_local_checkpoint(checkpoint, device=device)
    model.requires_grad_(False)
    if metadata["label_names"] != manifest["label_names"]:
        raise ValueError("Checkpoint and prepared label names/order differ.")
    if list(metadata["text_columns"].items()) != list(manifest["text_columns"].items()):
        raise ValueError("Checkpoint and prepared input fields/order differ.")
    if metadata.get("dataset_rows") != manifest["original_rows"]:
        raise ValueError("Checkpoint dataset size differs from the supplied original CSV.")
    # Check the old held-out IDs when the training artifact contains them.
    legacy = Path(checkpoint) / "test_embeddings.npz"
    ids = np.load(Path(prepared) / "row_ids.npy", mmap_mode="r")
    splits = np.load(Path(prepared) / "splits.npy", mmap_mode="r")
    split_check = "unverified: historical test_embeddings.npz missing"
    if legacy.is_file():
        with np.load(legacy, allow_pickle=False) as saved:
            expected = set(saved["row_ids"].astype(str))
        actual = set(ids[splits == 2].astype(str))
        if not actual <= expected or not manifest["sampled"] and actual != expected:
            raise ValueError("Reconstructed test split differs from the historical checkpoint.")
        split_check = "test IDs match historical export (subset for a sampled run)"
    out = new_directory(output)
    n, dimensions = manifest["rows"], metadata["hidden_size"]
    embeddings = array(out, "embeddings.npy", (n, dimensions), np.float32)
    logits = array(out, "logits.npy", (n, len(manifest["label_names"])), np.float32)
    columns = list(manifest["text_columns"])
    coverage = array(out, "field_coverage.npy", (n, len(columns)), np.int8)
    coverage[:] = -1
    start = time.perf_counter()
    count = 0
    for batch in tqdm(
        _batches(records(Path(prepared) / "requests.jsonl"), batch_size),
        total=(n + batch_size - 1) // batch_size,
        desc="SecBERT embeddings",
    ):
        texts = [serialize_http_request(row["fields"], {k: k for k in columns}) for row in batch]
        token_options = {
            "padding": True,
            "truncation": True,
            "max_length": metadata["max_length"],
            "return_tensors": "pt",
        }
        offsets_enabled = bool(tokenizer.is_fast and tokenizer.truncation_side == "right")
        encoded = tokenizer(texts, return_offsets_mapping=offsets_enabled, **token_options)
        offsets = encoded.pop("offset_mapping", None)
        with torch.inference_mode():
            result = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        positions = [row["position"] for row in batch]
        embeddings[positions] = result.embeddings.float().cpu().numpy()
        logits[positions] = result.logits.float().cpu().numpy()
        if offsets is not None:
            for i, row in enumerate(batch):
                last = int(offsets[i, :, 1].max())
                cursor = 0
                for j, key in enumerate(columns):
                    marker = f"[{key.upper().replace(' ', '_')}]\n"
                    value = row["fields"][key].replace("\x00", "<NUL>")
                    begin = cursor + len(marker)
                    end = begin + len(value.rstrip())
                    coverage[positions[i], j] = (
                        0 if not value or last <= begin else 2 if last >= end else 1
                    )
                    cursor = begin + len(value) + 1
        count += len(batch)
    for values in (embeddings, logits, coverage):
        values.flush()
    seconds = time.perf_counter() - start
    report = {
        "complete": count == n,
        "rows": n,
        "dimensions": dimensions,
        "prepared_sha256": manifest["requests_sha256"],
        "checkpoint_sha256": digest(Path(checkpoint) / "model.pt"),
        "checkpoint_metadata_sha256": digest(metadata_path),
        "head_sha256": digest(Path(checkpoint) / "embedding_head.pt"),
        "freeze_backbone_during_training": metadata.get("freeze_backbone"),
        "max_length": metadata["max_length"],
        "field_names": columns,
        "coverage_codes": {
            "-1": "unknown",
            "0": "absent_or_truncated",
            "1": "partially_visible",
            "2": "field_fully_visible",
        },
        "coverage_limit": "Field visibility only; no exact attribution span is asserted.",
        "historical_split_check": split_check,
        "device": str(device),
        "batch_size": batch_size,
        "seconds": seconds,
        "rows_per_second": n / max(seconds, 0.001),
        "array_sha256": {
            name: digest(out / name)
            for name in ("embeddings.npy", "logits.npy", "field_coverage.npy")
        },
    }
    write_json(out / "manifest.json", report)
    return report
