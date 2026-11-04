from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .common import (
    append_record,
    array,
    checked_manifest,
    digest,
    new_directory,
    records,
    write_json,
)
from .comparison.mapping import catalog_rules
from .events import annotate_events, load_concepts
from .modsecurity import ModSecurity, check_engine

_ENGINE = None


def _initialize(crs, paranoia, library):
    global _ENGINE
    _ENGINE = ModSecurity(crs, paranoia, library)


def _inspect_batch(batch):
    return [(row, _ENGINE.inspect(row["fields"])) for row in batch]


def _batches(source, size=64):
    batch = []
    for row in source:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _results(source, crs, paranoia, library, workers):
    # Bound pending work on Python 3.10 too; Executor.map would eagerly consume 900k rows.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize, initargs=(crs, paranoia, library)
    ) as executor:
        pending = deque()
        for batch in _batches(source):
            pending.append(executor.submit(_inspect_batch, batch))
            if len(pending) >= 2 * workers:
                yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()


def replay(prepared, output, crs, concept_config, workers=4, paranoia=1, library=None):
    if workers < 1:
        raise ValueError("workers must be positive.")
    manifest = checked_manifest(prepared)
    config, names = load_concepts(concept_config)
    # Fail on engine/rules errors before launching a large replay.
    engine = ModSecurity(crs, paranoia, library)
    try:
        check_engine(engine)
        version, crs_hash, configuration = engine.version, engine.crs_hash, engine.configuration
    finally:
        engine.close()
    out = new_directory(output)
    (out / "effective-modsecurity.conf").write_text(configuration, encoding="utf-8")
    targets = array(out, "concepts.npy", (manifest["rows"], len(names)), np.int8)
    targets[:] = -1
    decisions = array(out, "decision_scores.npy", (manifest["rows"], 2), np.float64)
    decisions[:] = np.nan
    write_json(out / "rule_catalog.json", catalog_rules(crs, paranoia))
    counts = Counter()
    start = time.perf_counter()
    iterator = _results(
        records(Path(prepared) / "requests.jsonl"),
        str(Path(crs).resolve()),
        paranoia,
        library,
        workers,
    )
    with (
        (out / "events.jsonl").open("w", encoding="utf-8") as events_file,
        (out / "status.jsonl").open("w", encoding="utf-8") as status_file,
    ):
        for row, result in tqdm(iterator, total=manifest["rows"], desc="CRS replay"):
            active, events = annotate_events(result["events"], row["fields"], config)
            if result["status"] == "complete":
                targets[row["position"]] = [int(name in active) for name in names]
                if result.get("decision"):
                    decisions[row["position"]] = [
                        result["decision"]["anomaly_score"],
                        result["decision"]["threshold"],
                    ]
            counts[result["status"]] += 1
            for event in events:
                append_record(
                    events_file, {**event, "row_id": row["row_id"], "position": row["position"]}
                )
            append_record(
                status_file,
                {
                    "row_id": row["row_id"],
                    "position": row["position"],
                    **{k: v for k, v in result.items() if k != "events"},
                },
            )
    targets.flush()
    decisions.flush()
    seconds = time.perf_counter() - start
    result = {
        "complete": sum(counts.values()) == manifest["rows"],
        "prepared_sha256": manifest["requests_sha256"],
        "rows": manifest["rows"],
        "concept_names": names,
        "mapping": config,
        "mapping_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "engine": version,
        "crs_tree_sha256": crs_hash,
        "paranoia_level": paranoia,
        "statuses": dict(counts),
        "workers": workers,
        "seconds": seconds,
        "rows_per_second": manifest["rows"] / max(seconds, 0.001),
        "events_sha256": digest(out / "events.jsonl"),
        "concepts_sha256": digest(out / "concepts.npy"),
        "decision_scores_sha256": digest(out / "decision_scores.npy"),
        "rule_catalog_sha256": digest(out / "rule_catalog.json"),
        "status_sha256": digest(out / "status.jsonl"),
        "comparison_complete_rows": int(np.isfinite(decisions).all(axis=1).sum()),
        "scope": "request phases 1/2; CSV reconstruction; default CRS request rules",
        "wire_limitations": "Original header ordering/duplicates/framing unavailable. "
        "Content-Length is synthesized for nonempty bodies.",
    }
    write_json(out / "manifest.json", result)
    return result
