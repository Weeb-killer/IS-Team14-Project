from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from .common import checked_manifest, digest, new_directory, read_json, write_json


def _metrics(y, score, threshold):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "status": "insufficient_examples",
            "positives": int(y.sum()),
            "negatives": int(len(y) - y.sum()),
        }
    predicted = score >= threshold
    return {
        "status": "ok",
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def _threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def _fit(x, y, train, validation, cap, seed, override=None):
    rng = np.random.default_rng(seed)
    # Balanced sampling affects probe training only. Test prevalence is retained.
    sampled = []
    for value in (0, 1):
        candidates = train[y[train] == value]
        sampled.extend(
            rng.choice(candidates, min(len(candidates), max(1, cap // 2)), replace=False).tolist()
        )
    sampled = np.asarray(sampled, dtype=int)
    scaler = StandardScaler().fit(np.asarray(x[sampled], dtype=np.float32))
    xt = scaler.transform(np.asarray(x[sampled], dtype=np.float32))
    xv = scaler.transform(np.asarray(x[validation], dtype=np.float32))
    yt, yv = y[sampled], y[validation]
    if override is not None:
        yt, yv = rng.permutation(yt), rng.permutation(yv)
    best = None
    for c in (0.01, 0.1, 1.0):
        estimator = LinearSVC(
            C=c, class_weight="balanced", dual=False, max_iter=3000, tol=1e-3, random_state=seed
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(xt, yt)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        score = estimator.decision_function(xv)
        ap = float(average_precision_score(yv, score))
        if best is None or ap > best[0]:
            best = ap, estimator, score, c, converged
    _, estimator, score, c, converged = best
    # Convert the complete affine separator back to ORIGINAL embedding coordinates.
    w = estimator.coef_[0] / scaler.scale_
    b = float(estimator.intercept_[0] - w @ scaler.mean_)
    negative_reference = np.asarray(x[sampled[y[sampled] == 0]], dtype=np.float32).mean(axis=0)
    return (
        w,
        b,
        _threshold(yv, score),
        negative_reference,
        {
            "C": c,
            "training_rows": len(sampled),
            "converged": converged,
            "preprocessing": "train-only StandardScaler; saved separator is in raw coordinates",
        },
    )


def _one(x, y, splits, hashes, seed, cap, permutations, repeats, min_support):
    train = np.flatnonzero((splits == 0) & (y >= 0))
    validation = np.flatnonzero((splits == 1) & (y >= 0))
    test = np.flatnonzero((splits == 2) & (y >= 0))
    supports = {
        name: {"positives": int((y[indices] == 1).sum()), "negatives": int((y[indices] == 0).sum())}
        for name, indices in [("train", train), ("validation", validation), ("test", test)]
    }
    if min(supports["train"].values()) < min_support or min(supports["validation"].values()) < 2:
        return {"status": "insufficient_training_or_validation", "support": supports}, None
    # Avoid repeating identical inputs inside a probe's training sample.
    _, unique = np.unique(hashes[train], return_index=True)
    train = train[unique]
    # Validation selection must not see content already used for fitting.
    validation = validation[~np.isin(hashes[validation], hashes[train])]
    if min(int((y[validation] == a).sum()) for a in (0, 1)) < 2:
        return {"status": "insufficient_unique_validation", "support": supports}, None
    if min(int((y[train] == a).sum()) for a in (0, 1)) < min_support:
        return {"status": "insufficient_unique_training", "support": supports}, None
    w, b, threshold, reference, fit_info = _fit(x, y, train, validation, cap, seed)
    scores = np.asarray(x[test]) @ w + b
    metrics = _metrics(y[test], scores, threshold)
    # A second evaluation excludes inputs seen in any original training/validation row.
    seen = hashes[splits != 2]
    clean = test[~np.isin(hashes[test], seen)]
    _, unique = np.unique(hashes[clean], return_index=True)
    clean = clean[unique]
    clean_x = np.asarray(x[clean])
    clean_metrics = _metrics(y[clean], clean_x @ w + b, threshold)
    rng = np.random.default_rng(seed + 10000)
    null = []
    for number in range(permutations):
        nw, nb, _, _, _ = _fit(x, y, train, validation, cap, seed + number + 1000, override=True)
        if len(np.unique(y[clean])) == 2:
            null.append(float(roc_auc_score(rng.permutation(y[clean]), clean_x @ nw + nb)))
    vectors = [w / max(np.linalg.norm(w), 1e-12)]
    repeated_auc = []
    for number in range(1, repeats):
        rw, rb, _, _, _ = _fit(x, y, train, validation, cap, seed + number)
        vectors.append(rw / max(np.linalg.norm(rw), 1e-12))
        if len(np.unique(y[clean])) == 2:
            repeated_auc.append(float(roc_auc_score(y[clean], clean_x @ rw + rb)))
    report = {
        "status": "ok",
        "support": supports,
        "fit": fit_info,
        "threshold": threshold,
        "test": metrics,
        "clean_test": clean_metrics,
        "test_reliable_support": min(supports["test"].values()) >= min_support,
        "clean_test_reliable_support": min(int((y[clean] == a).sum()) for a in (0, 1))
        >= min_support,
        "clean_test_excluded": len(test) - len(clean),
        "permutation_runs": len(null),
        "permutation_auc_mean": float(np.mean(null)) if null else None,
        "permutation_tail_fraction": (
            (1 + sum(s >= clean_metrics["roc_auc"] for s in null)) / (1 + len(null))
        )
        if null and clean_metrics["status"] == "ok"
        else None,
        "repeat_clean_auc": repeated_auc,
        "direction_cosine_to_first": [float(vectors[0] @ v) for v in vectors],
        "example_positions": {
            "high_score_positives": test[(y[test] == 1)][
                np.argsort(scores[y[test] == 1])[::-1][:5]
            ].tolist(),
            "high_score_false_positives": test[(y[test] == 0) & (scores >= threshold)][
                np.argsort(scores[(y[test] == 0) & (scores >= threshold)])[::-1][:5]
            ].tolist(),
        },
    }
    return report, (w, b, threshold, reference, np.asarray(vectors))


def fit(
    prepared,
    annotations,
    encoded,
    output,
    seed=14,
    cap=20000,
    permutations=20,
    repeats=3,
    min_support=20,
    conditional=False,
):
    if cap < 4 or repeats < 1 or permutations < 0 or min_support < 2:
        raise ValueError("cap >= 4, repeats >= 1, min_support >= 2 and permutations >= 0 required.")
    manifest = checked_manifest(prepared)
    waf = read_json(Path(annotations) / "manifest.json")
    enc = read_json(Path(encoded) / "manifest.json")
    for artifact in (waf, enc):
        if (
            not artifact.get("complete")
            or artifact["prepared_sha256"] != manifest["requests_sha256"]
        ):
            raise ValueError("Artifacts must be complete and belong to the same prepared requests.")
    if digest(Path(annotations) / "concepts.npy") != waf["concepts_sha256"]:
        raise ValueError("Concept annotations changed after replay.")
    for name, checksum in enc.get("array_sha256", {}).items():
        if digest(Path(encoded) / name) != checksum:
            raise ValueError(f"Encoded array changed: {name}")
    out = new_directory(output)
    x = np.load(Path(encoded) / "embeddings.npy", mmap_mode="r")
    y = np.load(Path(annotations) / "concepts.npy", mmap_mode="r")
    splits = np.load(Path(prepared) / "splits.npy", mmap_mode="r")
    hashes = np.load(Path(prepared) / "input_hashes.npy", mmap_mode="r")
    labels = np.load(Path(prepared) / "labels.npy", mmap_mode="r")
    row_ids = np.load(Path(prepared) / "row_ids.npy", mmap_mode="r")
    if len(x) != manifest["rows"] or len(y) != len(x) or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Embedding/annotation shapes or concept labels are invalid.")
    n_concepts, dimensions = y.shape[1], x.shape[1]
    weights = np.zeros((n_concepts, dimensions), np.float64)
    intercept = np.zeros(n_concepts, np.float64)
    thresholds = np.zeros(n_concepts, np.float64)
    reference = np.zeros_like(weights)
    directions = np.zeros((n_concepts, repeats, dimensions), np.float32)
    valid = np.zeros(n_concepts, bool)
    results = {}
    start = time.perf_counter()
    for i, name in enumerate(tqdm(waf["concept_names"], desc="Fit concept probes")):
        result, bank = _one(
            x, y[:, i], splits, hashes, seed + i * 100, cap, permutations, repeats, min_support
        )
        results[name] = result
        if "example_positions" in result:
            result["example_row_ids"] = {
                key: row_ids[positions].tolist()
                for key, positions in result["example_positions"].items()
            }
        if bank is not None:
            weights[i], intercept[i], thresholds[i], reference[i], directions[i] = bank
            valid[i] = True
        if conditional and name in waf["mapping"]["concepts"]:
            result["conditional_on_attack"] = {}
            for k, label in enumerate(manifest["label_names"]):
                conditional_splits = np.where(labels[:, k] > 0, splits, -1)
                subreport, _ = _one(
                    x,
                    y[:, i],
                    conditional_splits,
                    hashes,
                    seed + i * 100 + k + 50,
                    cap,
                    0,
                    1,
                    min_support,
                )
                result["conditional_on_attack"][label] = subreport
    np.savez_compressed(
        out / "concept_bank.npz",
        weights=weights,
        intercept=intercept,
        thresholds=thresholds,
        negative_reference=reference,
        directions=directions,
        valid=valid,
        concept_names=np.asarray(waf["concept_names"], dtype=str),
    )
    train_known = np.flatnonzero((splits == 0) & np.all(y >= 0, axis=1))
    rng = np.random.default_rng(seed)
    if len(train_known) > cap:
        train_known = rng.choice(train_known, cap, replace=False)
    correlation = []
    for j in range(n_concepts):
        row = []
        for k in range(n_concepts):
            a, b = y[train_known, j], y[train_known, k]
            value = (
                float(np.corrcoef(a, b)[0, 1])
                if len(a) > 1 and a.std() > 0 and b.std() > 0
                else None
            )
            row.append(value)
        correlation.append(row)
    report = {
        "concepts": results,
        "rows": manifest["rows"],
        "sampled": manifest["sampled"],
        "synthetic": bool(manifest.get("synthetic", False)),
        "seed": seed,
        "max_training_rows_per_concept": cap,
        "min_support": min_support,
        "permutations": permutations,
        "repeats": repeats,
        "prepared_sha256": manifest["requests_sha256"],
        "checkpoint_sha256": enc["checkpoint_sha256"],
        "head_sha256": enc["head_sha256"],
        "concepts_sha256": waf["concepts_sha256"],
        "seconds": time.perf_counter() - start,
        "annotation_correlations": correlation,
        "correlation_training_samples": len(train_known),
        "interpretation": "Linear decodability of CRS proxy labels, not proof of causality.",
        "permutation_note": "Empirical diagnostic conditional on sampled probe data; "
        "tail fractions are not corrected multiple-testing significance.",
        "split_note": "Known content duplicates are removed from probe fitting/validation; "
        "clean test excludes training/validation input hashes and test duplicates.",
    }
    write_json(out / "probe_report.json", report)
    return report
