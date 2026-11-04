from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..common import checked_manifest, digest, new_directory, read_json, records, write_json
from .mapping import load_mapping, matches
from .metrics import (
    binary_metrics,
    label_metrics,
    paired_intervals,
    threshold_for_f1,
    threshold_for_fpr,
)


def operating_points(y_validation, crs_scores, model_scores, fpr=0.01):
    result = {}
    for name, scores in [("CRS", crs_scores), ("SecBERT", model_scores)]:
        result[name] = {}
        if len(y_validation) >= 4 and len(np.unique(y_validation)) == 2:
            for mode, threshold in [
                ("validation_f1", threshold_for_f1(y_validation, scores)),
                ("validation_fpr", threshold_for_fpr(y_validation, scores, fpr)),
            ]:
                result[name][mode] = {
                    "threshold": threshold,
                    "validation": binary_metrics(y_validation, scores >= threshold, scores),
                }
    return result


def compare(
    prepared,
    annotations,
    encoded,
    checkpoint,
    output,
    mapping=None,
    target_fpr=0.01,
    bootstrap=500,
    seed=14,
    examples=50,
):
    if bootstrap < 0 or examples < 0:
        raise ValueError("bootstrap and examples must be nonnegative.")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1).")
    manifest = checked_manifest(prepared)
    waf = read_json(Path(annotations) / "manifest.json")
    enc = read_json(Path(encoded) / "manifest.json")
    metadata_path = Path(checkpoint) / "metadata.json"
    metadata = read_json(metadata_path)
    for artifact in (waf, enc):
        if (
            not artifact.get("complete")
            or artifact["prepared_sha256"] != manifest["requests_sha256"]
        ):
            raise ValueError("Prepared, WAF, and encoded artifacts must be complete and aligned.")
    synthetic = bool(manifest.get("synthetic"))
    if enc.get("checkpoint_metadata_sha256") != digest(metadata_path):
        raise ValueError(
            "Checkpoint metadata differ from encode provenance. Re-export with this checkpoint."
        )
    labels = manifest["label_names"]
    if labels != metadata["label_names"] or list(metadata["text_columns"].items()) != list(
        manifest["text_columns"].items()
    ):
        raise ValueError("Checkpoint label/input ordering differs from prepared data.")
    thresholds_by_label = metadata.get("test_metrics", {}).get("thresholds")
    if not isinstance(thresholds_by_label, dict) or set(thresholds_by_label) != set(labels):
        raise ValueError("Checkpoint must contain saved validation-selected per-label thresholds.")
    thresholds = np.asarray([thresholds_by_label[name] for name in labels], dtype=float)
    if not np.isfinite(thresholds).all() or not ((thresholds > 0) & (thresholds < 1)).all():
        raise ValueError("Saved probability thresholds must lie strictly between 0 and 1.")
    required = {
        "decision_scores.npy": "decision_scores_sha256",
        "rule_catalog.json": "rule_catalog_sha256",
        "events.jsonl": "events_sha256",
    }
    for name, key in required.items():
        path = Path(annotations) / name
        if not path.is_file() or key not in waf:
            raise ValueError(
                "WAF replay lacks comparison telemetry/catalog. Rerun replay with the updated code."
            )
        if digest(path) != waf[key]:
            raise ValueError(f"WAF artifact changed: {name}")
    expected_logit_hash = enc.get("array_sha256", {}).get("logits.npy")
    if not expected_logit_hash or digest(Path(encoded) / "logits.npy") != expected_logit_hash:
        raise ValueError(
            "Encoded logits are missing integrity provenance or have changed. Rerun encode."
        )
    n = manifest["rows"]
    logits = np.asarray(np.load(Path(encoded) / "logits.npy", mmap_mode="r"), dtype=np.float64)
    scores = np.load(Path(annotations) / "decision_scores.npy", mmap_mode="r")
    y = np.load(Path(prepared) / "labels.npy", mmap_mode="r")
    splits = np.load(Path(prepared) / "splits.npy", mmap_mode="r")
    ids = np.load(Path(prepared) / "row_ids.npy", mmap_mode="r")
    hashes = np.load(Path(prepared) / "input_hashes.npy", mmap_mode="r")
    if logits.shape != (n, len(labels)) or scores.shape != (n, 2) or y.shape != logits.shape:
        raise ValueError("Artifact row/label shapes differ.")
    if not np.isfinite(logits).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid model logits or ground-truth labels.")
    # Both outputs are scored against ORIGINAL attack targets, never against CRS annotations.
    y_binary = np.any(y > 0, axis=1)
    probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    model_labels = probability >= thresholds
    model_default = np.any(model_labels, axis=1)
    # Continuous margin agrees with the original per-label threshold decision at zero.
    model_score = np.max(logits - np.log(thresholds / (1 - thresholds)), axis=1)
    valid = np.isfinite(scores).all(axis=1) & (scores[:, 0] >= 0) & (scores[:, 1] > 0)
    crs_default = np.zeros(n, dtype=bool)
    crs_default[valid] = scores[valid, 0] >= scores[valid, 1]
    all_test = np.flatnonzero(splits == 2)
    test = all_test[valid[all_test]]
    if not len(test):
        raise ValueError(
            "No test request has a completed CRS score. Inspect status.jsonl/check-waf."
        )
    validation = np.flatnonzero((splits == 1) & valid)
    validation = validation[~np.isin(hashes[validation], hashes[splits == 0])]
    _, unique = np.unique(hashes[validation], return_index=True)
    validation = validation[unique]
    calibration = operating_points(
        y_binary[validation], scores[validation, 0], model_score[validation], target_fpr
    )
    clean = test[~np.isin(hashes[test], hashes[splits != 2])]
    _, unique = np.unique(hashes[clean], return_index=True)
    clean = clean[unique]
    subsets = {"common_test": test, "clean_common_test": clean}
    definitions = load_mapping(labels, mapping)
    catalog = read_json(Path(annotations) / "rule_catalog.json")
    coverage = {
        name: [rule["rule_id"] for rule in catalog if matches(rule, definitions[name])]
        for name in labels
    }
    label_hits = np.zeros((n, len(labels)), dtype=bool)
    case_groups = {
        "crs_miss_model_detect": test[y_binary[test] & ~crs_default[test] & model_default[test]],
        "model_miss_crs_detect": test[y_binary[test] & crs_default[test] & ~model_default[test]],
        "crs_false_alarm_model_correct": test[
            ~y_binary[test] & crs_default[test] & ~model_default[test]
        ],
        "model_false_alarm_crs_correct": test[
            ~y_binary[test] & ~crs_default[test] & model_default[test]
        ],
    }
    rng = np.random.default_rng(seed)
    chosen_cases = {}
    for category, positions in case_groups.items():
        selected = rng.choice(positions, min(examples, len(positions)), replace=False)
        chosen_cases.update({int(i): category for i in selected})
    case_rules = {position: set() for position in chosen_cases}
    mapping_cache = {}
    for event in tqdm(records(Path(annotations) / "events.jsonl"), desc="Map CRS rule events"):
        i = int(event["position"])
        if not 0 <= i < n or event["row_id"] != str(ids[i]):
            raise ValueError("WAF event request ID/position does not match prepared input.")
        # Cache by metadata too: malformed/mixed rule versions must not silently share mappings.
        key = (str(event["rule_id"]), tuple(sorted(event.get("tags", []))))
        if key not in mapping_cache:
            mapping_cache[key] = [
                j for j, name in enumerate(labels) if matches(event, definitions[name])
            ]
            if any(not coverage[labels[j]] for j in mapping_cache[key]):
                raise ValueError(
                    "A mapped event is missing from declared rule coverage. "
                    "Check the CRS rule catalog/mapping before comparison."
                )
        label_hits[i, mapping_cache[key]] = True
        if i in case_rules:
            case_rules[i].add(str(event["rule_id"]))
    crs_labels = label_hits & crs_default[:, None]
    results = {}
    intervals = {}
    classification = {}
    shared = [j for j, name in enumerate(labels) if coverage[name]]
    for subset_name, indices in subsets.items():
        results[subset_name] = {}
        intervals[subset_name] = {}
        defaults = {"CRS": crs_default, "SecBERT": model_default}
        continuous = {"CRS": scores[:, 0], "SecBERT": model_score}
        for mode in ("saved_policy", "validation_f1", "validation_fpr"):
            results[subset_name][mode] = {}
            predictions = {}
            for detector in ("CRS", "SecBERT"):
                if mode == "saved_policy":
                    predicted = defaults[detector][indices]
                    threshold = (
                        "recorded per-request CRS threshold"
                        if detector == "CRS"
                        else "saved per-label model thresholds"
                    )
                elif mode in calibration[detector]:
                    threshold = calibration[detector][mode]["threshold"]
                    predicted = continuous[detector][indices] >= threshold
                else:
                    continue
                predictions[detector] = predicted
                results[subset_name][mode][detector] = {
                    **binary_metrics(y_binary[indices], predicted, continuous[detector][indices]),
                    "threshold": threshold,
                }
            if len(predictions) == 2:
                intervals[subset_name][mode] = paired_intervals(
                    y_binary[indices],
                    predictions["CRS"],
                    predictions["SecBERT"],
                    hashes[indices],
                    bootstrap,
                    seed,
                )
        classification[subset_name] = {
            "all_labels": {
                "CRS": label_metrics(y[indices], crs_labels[indices], labels),
                "SecBERT": label_metrics(y[indices], model_labels[indices], labels),
            },
            "mapped_labels_only": {
                "CRS": label_metrics(
                    y[indices][:, shared],
                    crs_labels[indices][:, shared],
                    [labels[j] for j in shared],
                ),
                "SecBERT": label_metrics(
                    y[indices][:, shared],
                    model_labels[indices][:, shared],
                    [labels[j] for j in shared],
                ),
            },
        }
    excluded = all_test[~valid[all_test]]
    failure_scenarios = {}
    for fill, label in [(False, "all_failed_as_normal"), (True, "all_failed_as_attack")]:
        predictions = crs_default[all_test].copy()
        predictions[~valid[all_test]] = fill
        failure_scenarios[label] = binary_metrics(y_binary[all_test], predictions)
    out = new_directory(output)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "row_id",
                "position",
                "clean_common_test",
                "true_attacks",
                "crs_status",
                "crs_anomaly_score",
                "crs_threshold",
                "crs_attack",
                "secbert_attack",
                "secbert_margin",
                "crs_labels",
                "secbert_labels",
            ],
        )
        writer.writeheader()
        clean_set = set(map(int, clean))
        for i in all_test:
            writer.writerow(
                {
                    "row_id": str(ids[i]),
                    "position": int(i),
                    "clean_common_test": int(i) in clean_set,
                    "true_attacks": json.dumps([name for j, name in enumerate(labels) if y[i, j]]),
                    "crs_status": "complete" if valid[i] else "unknown",
                    "crs_anomaly_score": float(scores[i, 0]) if valid[i] else "",
                    "crs_threshold": float(scores[i, 1]) if valid[i] else "",
                    "crs_attack": bool(crs_default[i]) if valid[i] else "",
                    "secbert_attack": bool(model_default[i]),
                    "secbert_margin": float(model_score[i]),
                    "crs_labels": json.dumps(
                        [name for j, name in enumerate(labels) if crs_labels[i, j]]
                    )
                    if valid[i]
                    else "",
                    "secbert_labels": json.dumps(
                        [name for j, name in enumerate(labels) if model_labels[i, j]]
                    ),
                }
            )
    cases = [
        {
            "row_id": str(ids[i]),
            "position": i,
            "category": category,
            "actual_attack_labels": [name for j, name in enumerate(labels) if y[i, j]],
            "rule_ids": sorted(case_rules[i]),
            "crs_anomaly_score": float(scores[i, 0]),
            "secbert_margin": float(model_score[i]),
        }
        for i, category in sorted(chosen_cases.items())
    ]
    write_json(out / "disagreement_examples.json", cases)
    report = {
        "synthetic": synthetic,
        "sampled": manifest["sampled"],
        "rows": n,
        "label_names": labels,
        "prepared_sha256": manifest["requests_sha256"],
        "checkpoint_sha256": enc["checkpoint_sha256"],
        "engine": waf.get("engine"),
        "crs_tree_sha256": waf.get("crs_tree_sha256"),
        "paranoia_level": waf.get("paranoia_level"),
        "binary_results": results,
        "paired_intervals": intervals,
        "classification": classification,
        "calibration": calibration,
        "calibration_samples": len(validation),
        "target_validation_fpr": target_fpr,
        "coverage": {
            "test_rows": len(all_test),
            "common_test_rows": len(test),
            "clean_common_test_rows": len(clean),
            "excluded_unknown_crs_rows": len(excluded),
            "excluded_attacks": int(y_binary[excluded].sum()),
            "excluded_normals": int((~y_binary[excluded]).sum()),
            "crs_evaluation_fraction": len(test) / len(all_test),
        },
        "unknown_crs_scenarios": failure_scenarios,
        "secbert_all_test": binary_metrics(
            y_binary[all_test], model_default[all_test], model_score[all_test]
        ),
        "label_mapping": definitions,
        "declared_mapped_rules": coverage,
        "unmapped_labels": [name for name in labels if not coverage[name]],
        "classification_policy": "CRS label predicted when a mapped rule matches AND recorded "
        "anomaly threshold is reached. Mapped-label metrics describe a "
        "taxonomy projection, not native CRS attack classification.",
        "disagreement_counts": {name: len(positions) for name, positions in case_groups.items()},
        "score_definitions": {
            "CRS": "Recorded blocking_inbound_anomaly_score.",
            "SecBERT": "max(class_logit - logit(saved_class_threshold)).",
        },
        "runtime": {
            "CRS": {key: waf.get(key) for key in ("rows", "seconds", "rows_per_second", "workers")},
            "SecBERT": {
                key: enc.get(key)
                for key in ("rows", "seconds", "rows_per_second", "device", "batch_size")
            },
            "note": "Observed full replay/export stage throughput. CPU workers and batched GPU "
            "inference are different workloads; not online per-request latency.",
        },
        "limitations": [
            (
                "CSV reconstruction lacks original wire/header framing; CRS sees reconstructed "
                "available fields while SecBERT is token-truncated to its saved max_length."
            ),
            (
                "Unknown CRS requests are excluded from BOTH detectors in paired tables; "
                "coverage and failed-request scenarios are reported separately."
            ),
            (
                "Full-task class scores include zero predictions for unmapped CRS labels; "
                "they alone cannot establish superiority on common attack categories."
            ),
            (
                "Clean test removes exact prepared train/validation inputs and repeated test "
                "inputs, not near duplicates or historical training exposure."
            ),
            (
                "Sampled preparation audits only selected rows. Bootstrap intervals are "
                "conditional on this model, dataset, and input groups; no broad deployment "
                "or multiple-comparison significance claim follows."
            ),
        ],
        "max_length": metadata.get("max_length"),
        "integrity": {
            "waf_scores": waf["decision_scores_sha256"],
            "events": waf["events_sha256"],
            "catalog": waf["rule_catalog_sha256"],
            "logits": expected_logit_hash,
            "metadata": enc["checkpoint_metadata_sha256"],
        },
    }
    write_json(out / "comparison.json", report)
    from .report import render_report

    render_report(out)
    return {
        "report": str(out / "README.md"),
        "common_test": len(test),
        "clean_test": len(clean),
        "unmapped_labels": report["unmapped_labels"],
        "synthetic": synthetic,
    }
