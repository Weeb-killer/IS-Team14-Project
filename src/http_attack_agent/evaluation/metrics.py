"""Multi-label metrics for one set of predictions, independent of the training loop.

``multilabel_metrics`` is a superset of the report ``training.py`` builds internally: every key
that module already emits is produced here with the same name and meaning, so a saved
``metadata.json`` stays readable, and the additions answer questions the original report cannot.

Three of those additions matter for an imbalanced multi-label benchmark:

* raw confusion counts per label, because a precision of 0.5 means something different at four
  positives than at forty thousand;
* the ``normal`` class as a class, since "no attack label crosses its threshold" is a prediction
  this project makes and should be scored directly rather than through accuracy alone;
* ``macro_f1_supported``, which averages only over labels with enough test examples to be worth
  averaging. SR-BH 2020 contains a label with a single labeled request, and an unfiltered macro
  average lets that one request move the headline number as much as a label with 250,000.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..visualization.profile import MIN_RELIABLE_SUPPORT


def _ranking_score(
    function: Any, truth: np.ndarray, score: np.ndarray
) -> float | None:
    """Return None where a ranking score is undefined, instead of a number that looks real.

    The degenerate cases are tested directly rather than caught, because scikit-learn does not
    signal them consistently. For a label with no positive example ``average_precision_score``
    returns 0.0, and for one whose every row is positive it returns 1.0; neither raises.
    ``roc_auc_score`` returns NaN. A caller that only guards against ``ValueError`` therefore
    records an absent label as a genuine score.
    """

    positives = int(np.asarray(truth).sum())
    if positives == 0 or positives == len(truth):
        return None
    try:
        value = float(function(truth, score))
    except ValueError:
        return None
    return None if np.isnan(value) else value


def _binary_scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def normal_class_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    """Score "no attack label crosses its threshold" as its own class.

    Accuracy alone is misleading here: on SR-BH 2020 roughly 58% of requests are normal, so a
    model that never predicts an attack already scores 0.58.
    """

    truth = targets.sum(axis=1) == 0
    predicted = predictions.sum(axis=1) == 0
    result: dict[str, Any] = {
        "support": int(truth.sum()),
        "predicted": int(predicted.sum()),
        "accuracy": float(np.mean(truth == predicted)),
    }
    result.update(_binary_scores(truth, predicted))
    return result


def attack_ranking_score(
    targets: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Rank quality among requests that carry at least one attack label.

    Label-ranking average precision is undefined for a request with no true label, and most of
    this dataset is normal traffic, so the score is restricted to attack requests. It answers a
    question the thresholded scores cannot: given that a request is an attack, does the correct
    attack type rank above the others?
    """

    attacks = targets.sum(axis=1) > 0
    if not attacks.any():
        return {"samples": 0, "label_ranking_average_precision": None}
    try:
        score = float(
            label_ranking_average_precision_score(
                targets[attacks], probabilities[attacks]
            )
        )
    except ValueError:
        score = float("nan")
    return {
        "samples": int(attacks.sum()),
        "label_ranking_average_precision": None if np.isnan(score) else score,
    }


def per_label_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
    min_support: int,
) -> dict[str, Any]:
    """Per-label scores plus the raw counts they were derived from."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    confusion = multilabel_confusion_matrix(targets, predictions)
    report: dict[str, Any] = {}
    for index, name in enumerate(label_names):
        true_negative, false_positive, false_negative, true_positive = confusion[
            index
        ].ravel()
        support = int(targets[:, index].sum())
        report[name] = {
            "support": support,
            "predicted_positives": int(predictions[:, index].sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "average_precision": _ranking_score(
                average_precision_score, targets[:, index], probabilities[:, index]
            ),
            "roc_auc": _ranking_score(
                roc_auc_score, targets[:, index], probabilities[:, index]
            ),
            "true_positives": int(true_positive),
            "false_positives": int(false_positive),
            "false_negatives": int(false_negative),
            "true_negatives": int(true_negative),
            "reliable": support >= min_support,
        }
    return report


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | float,
    label_names: Sequence[str],
    *,
    min_support: int = MIN_RELIABLE_SUPPORT,
) -> dict[str, Any]:
    """Score one set of multi-label predictions against its targets.

    ``thresholds`` is one decision threshold per label, or a single value applied to every label.
    Scores that are mathematically undefined are reported as ``None`` instead of 0.0, so an
    absent label is never mistaken for a model that missed everything.
    """

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.shape != probabilities.shape:
        raise ValueError(
            f"targets {targets.shape} and probabilities {probabilities.shape} must have "
            "the same shape"
        )
    if targets.ndim != 2:
        raise ValueError(f"targets must be two-dimensional, got shape {targets.shape}")
    if targets.shape[1] != len(label_names):
        raise ValueError(
            f"targets has {targets.shape[1]} columns but {len(label_names)} label names "
            "were provided"
        )

    threshold_array = np.broadcast_to(
        np.asarray(thresholds, dtype=np.float64), (targets.shape[1],)
    )
    targets = (targets > 0).astype(np.int8)
    predictions = (probabilities >= threshold_array).astype(np.int8)

    per_label = per_label_metrics(
        targets, probabilities, predictions, label_names, min_support
    )
    supported = [name for name, item in per_label.items() if item["reliable"]]
    macro_f1_supported = (
        float(np.mean([per_label[name]["f1"] for name in supported])) if supported else None
    )
    normal = normal_class_metrics(targets, predictions)
    defined_ap = [
        per_label[name]["average_precision"]
        for name in supported
        if per_label[name]["average_precision"] is not None
    ]
    macro_ap_supported = float(np.mean(defined_ap)) if defined_ap else None

    # ``average="samples"`` is deliberately absent. With zero_division=0 a request that is
    # correctly predicted to carry no attack scores 0, and most of this dataset is normal
    # traffic, so the sample average would penalise exactly the behaviour it should reward.
    return {
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "macro_f1_supported": macro_f1_supported,
        # Matches the value training.py already writes whenever any label has a positive
        # example, and reports None instead of 0.0 for a target matrix with none at all.
        # scikit-learn scores an absent label as 0.0 and folds it into this average, so read
        # the _supported variant alongside it.
        "macro_average_precision": float(
            average_precision_score(targets, probabilities, average="macro")
        )
        if targets.any()
        else None,
        "macro_average_precision_supported": macro_ap_supported,
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "subset_accuracy": float(np.mean((targets == predictions).all(axis=1))),
        "normal_accuracy": normal["accuracy"],
        "normal": normal,
        "attack_ranking": attack_ranking_score(targets, probabilities),
        "min_support": int(min_support),
        "supported_labels": supported,
        "insufficient_support_labels": [
            name for name in label_names if name not in supported
        ],
        "thresholds": {
            name: float(value) for name, value in zip(label_names, threshold_array)
        },
        "per_label": per_label,
    }
