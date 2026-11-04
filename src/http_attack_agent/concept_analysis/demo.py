"""Synthetic analysis integration demo, explicitly NOT an actual SecBERT/CRS run."""

from __future__ import annotations

import numpy as np

from .common import append_record, digest, new_directory, write_json


def demo(output):
    import torch

    from ..models.hf_classifier import build_embedding_head
    from .probes import fit
    from .report import report
    from .sensitivity import analyze, explain_request

    root = new_directory(output)
    prepared, waf, encoded, checkpoint = [
        root / name for name in ("prepared", "waf", "encoded", "checkpoint")
    ]
    for directory in (prepared, waf, encoded, checkpoint):
        directory.mkdir()
    rng = np.random.default_rng(14)
    n, dimensions = 600, 8
    torch.manual_seed(14)
    labels = rng.integers(0, 2, (n, 2), dtype=np.int8)
    embeddings = rng.normal(size=(n, dimensions)).astype(np.float32)
    embeddings[:, :2] += labels * 5
    ids = np.asarray([f"demo-{i}" for i in range(n)], dtype=str)
    splits = np.asarray([0] * 360 + [1] * 120 + [2] * 120, dtype=np.int8)
    # Intentionally leave one inspection unknown; tests ensure it is not treated as negative.
    concepts = labels.copy()
    concepts[0] = -1
    with (prepared / "requests.jsonl").open("w", encoding="utf-8") as target:
        for i in range(n):
            append_record(
                target,
                {
                    "position": i,
                    "row_id": str(ids[i]),
                    "fields": {
                        "method": "GET",
                        "request_target": f"/?demo={i}",
                        "protocol": "HTTP/1.1",
                    },
                    "input_hash": f"{i:064x}",
                },
            )
    for name, values in [
        ("labels", labels),
        ("splits", splits),
        ("row_ids", ids),
        ("input_hashes", np.asarray([f"{i:064x}" for i in range(n)], dtype="S64")),
    ]:
        np.save(prepared / f"{name}.npy", values)
    checksum = digest(prepared / "requests.jsonl")
    columns = {"method": "method", "request_target": "request_target", "protocol": "protocol"}
    names = ["sql_injection", "path_traversal"]
    write_json(
        prepared / "manifest.json",
        {
            "complete": True,
            "rows": n,
            "original_rows": n,
            "sampled": True,
            "requests_sha256": checksum,
            "label_names": names,
            "text_columns": columns,
            "synthetic": True,
            "dataset_sha256": "synthetic",
            "seconds": 0.0,
            "rows_per_second": 0.0,
        },
    )
    np.save(waf / "concepts.npy", concepts)
    concept_names = ["synthetic_sqli", "synthetic_lfi"]
    write_json(
        waf / "manifest.json",
        {
            "complete": True,
            "prepared_sha256": checksum,
            "rows": n,
            "concept_names": concept_names,
            "concepts_sha256": digest(waf / "concepts.npy"),
            "mapping": {"concepts": {name: {} for name in concept_names}},
            "synthetic": True,
            "seconds": 0.0,
            "rows_per_second": 0.0,
        },
    )
    (waf / "events.jsonl").write_text("", encoding="utf-8")
    with (waf / "status.jsonl").open("w", encoding="utf-8") as target:
        for i in range(n):
            append_record(
                target,
                {
                    "row_id": str(ids[i]),
                    "position": i,
                    "status": "unknown" if i == 0 else "complete",
                },
            )
    head = build_embedding_head(dimensions, 2, dropout=0)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)
    features = torch.tensor(embeddings[:360])
    targets = torch.tensor(labels[:360], dtype=torch.float32)
    head.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(features), targets)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(embeddings)).numpy()
    torch.save(head.state_dict(), checkpoint / "embedding_head.pt")
    metadata = {
        "hidden_size": dimensions,
        "label_names": names,
        "dropout": 0,
        "text_columns": columns,
        "dataset_rows": n,
        "test_metrics": {"thresholds": {name: 0.5 for name in names}},
    }
    write_json(checkpoint / "metadata.json", metadata)
    np.save(encoded / "embeddings.npy", embeddings)
    np.save(encoded / "logits.npy", logits)
    np.save(encoded / "field_coverage.npy", np.full((n, 3), -1, dtype=np.int8))
    write_json(
        encoded / "manifest.json",
        {
            "complete": True,
            "prepared_sha256": checksum,
            "checkpoint_sha256": "synthetic",
            "head_sha256": digest(checkpoint / "embedding_head.pt"),
            "field_names": list(columns),
            "coverage_codes": {"-1": "unknown"},
            "synthetic": True,
            "seconds": 0.0,
            "rows_per_second": 0.0,
        },
    )
    fit(prepared, waf, encoded, root / "bank", cap=300, permutations=2, repeats=2, conditional=True)
    analyze(
        prepared,
        encoded,
        root / "bank",
        checkpoint,
        max_samples=60,
        random_directions=2,
        device="cpu",
    )
    explain_request(
        prepared,
        waf,
        encoded,
        root / "bank",
        checkpoint,
        "demo-500",
        root / "example_explanation.json",
        device="cpu",
    )
    result = report(root / "bank")
    result["warning"] = "Synthetic smoke demo only: no SecBERT or real CRS inspection occurred."
    write_json(root / "DEMO.json", result)
    return result
