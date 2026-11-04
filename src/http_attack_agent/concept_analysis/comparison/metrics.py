from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def divide(a, b):
    return float(a / b) if b else None


def from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))
    return {
        "samples": tp + fp + fn + tn,
        "attack_support": tp + fn,
        "normal_support": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "false_negative_rate": divide(fn, fn + tp),
        "accuracy": divide(tp + tn, tp + fp + fn + tn),
    }


def binary_metrics(y, predicted, scores=None):
    y, predicted = np.asarray(y, bool), np.asarray(predicted, bool)
    result = from_counts(
        (y & predicted).sum(),
        (~y & predicted).sum(),
        (y & ~predicted).sum(),
        (~y & ~predicted).sum(),
    )
    both = bool(len(y) and y.any() and (~y).any())
    result["roc_auc"] = float(roc_auc_score(y, scores)) if scores is not None and both else None
    result["average_precision"] = (
        float(average_precision_score(y, scores)) if scores is not None and both else None
    )
    return result


def threshold_for_f1(y, scores):
    if len(np.unique(y)) < 2:
        return None
    precision, recall, thresholds = precision_recall_curve(y, scores)
    values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(values)])


def threshold_for_fpr(y, scores, target):
    """Choose the lowest threshold satisfying the validation normal FPR cap, including ties."""
    if not 0 <= target < 1:
        raise ValueError("Target FPR must be in [0, 1).")
    normals = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    if not len(normals):
        return None
    allowed = int(np.floor(target * len(normals)))
    # For >= decisions, exclude the boundary score and every tie at that score.
    return float(np.nextafter(normals[len(normals) - allowed - 1], np.inf))


def label_metrics(y, predicted, names, min_support=20):
    values = {}
    counts = []
    for j, name in enumerate(names):
        metric = binary_metrics(y[:, j], predicted[:, j])
        metric["reliable_support"] = metric["attack_support"] >= min_support
        values[name] = metric
        counts.append([metric[k] for k in ("tp", "fp", "fn", "tn")])
    supported = [v["f1"] for v in values.values() if v["reliable_support"] and v["f1"] is not None]
    defined = [v["f1"] for v in values.values() if v["attack_support"] > 0 and v["f1"] is not None]
    totals = np.asarray(counts).sum(axis=0) if counts else [0, 0, 0, 0]
    return {
        "micro_f1": from_counts(*totals)["f1"],
        "macro_f1_supported": float(np.mean(supported)) if supported else None,
        "macro_f1_defined": float(np.mean(defined)) if defined else None,
        "subset_accuracy": float(np.mean(np.all(y == predicted, axis=1)))
        if len(y) and len(names)
        else None,
        "per_label": values,
    }


def paired_intervals(y, crs, model, hashes, runs=500, seed=14):
    """Paired percentile intervals, resampling exact-input groups rather than duplicate rows."""
    if runs <= 0 or not len(y):
        return {"status": "disabled_or_empty", "runs": 0}
    _, groups = np.unique(hashes, return_inverse=True)
    group_n = int(groups.max()) + 1
    rows = []
    for predictions in (crs, model):
        masks = [y & predictions, ~y & predictions, y & ~predictions, ~y & ~predictions]
        rows.extend(np.bincount(groups, weights=mask, minlength=group_n) for mask in masks)
    counts = np.asarray(rows).T.astype(np.int64)
    metrics = ("f1", "recall", "precision", "false_positive_rate", "accuracy")
    deltas = {name: [] for name in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(runs):
        totals = counts[rng.integers(0, group_n, size=group_n)].sum(axis=0)
        left, right = from_counts(*totals[:4]), from_counts(*totals[4:])
        for metric in metrics:
            if left[metric] is not None and right[metric] is not None:
                deltas[metric].append(right[metric] - left[metric])
    original_crs = binary_metrics(y, crs)
    original_model = binary_metrics(y, model)
    result = {
        "status": "ok",
        "runs": runs,
        "input_groups": group_n,
        "method": "Paired exact-input-cluster percentile bootstrap; model minus CRS.",
        "deltas": {},
    }
    for metric, values in deltas.items():
        result["deltas"][metric] = {
            "point": (original_model[metric] - original_crs[metric])
            if original_model[metric] is not None and original_crs[metric] is not None
            else None,
            "lower_95": float(np.quantile(values, 0.025)) if values else None,
            "upper_95": float(np.quantile(values, 0.975)) if values else None,
            "valid_resamples": len(values),
        }
    return result
