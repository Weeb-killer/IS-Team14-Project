import numpy as np
import pytest

from http_attack_agent.evaluation.metrics import multilabel_metrics

LABELS = ["alpha", "beta"]

# Hand-checked fixture. Only one cell is predicted wrong: alpha on row 4, whose probability
# of 0.4 falls below the 0.5 threshold even though the row is a true positive.
TARGETS = np.array(
    [[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 0]], dtype=np.int8
)
PROBABILITIES = np.array(
    [[0.9, 0.1], [0.8, 0.7], [0.2, 0.6], [0.1, 0.2], [0.4, 0.1], [0.05, 0.05]]
)
THRESHOLDS = np.array([0.5, 0.5])


def _metrics(**kwargs):
    return multilabel_metrics(TARGETS, PROBABILITIES, THRESHOLDS, LABELS, **kwargs)


def test_every_key_training_already_emits_is_present():
    """training.py must be able to swap to this function without losing a field."""

    result = _metrics()

    for key in (
        "micro_f1",
        "macro_f1",
        "thresholds",
        "macro_average_precision",
        "normal_accuracy",
        "per_label",
    ):
        assert key in result, key
    for key in ("support", "precision", "recall", "f1", "average_precision"):
        assert key in result["per_label"]["alpha"], key


def test_aggregate_scores_match_hand_computation():
    result = _metrics()

    assert result["macro_f1"] == pytest.approx(0.9)  # (0.8 + 1.0) / 2
    assert result["micro_f1"] == pytest.approx(8 / 9)  # 4 TP, 0 FP, 1 FN
    assert result["hamming_loss"] == pytest.approx(1 / 12)
    assert result["subset_accuracy"] == pytest.approx(5 / 6)


def test_per_label_confusion_counts_are_exact():
    alpha = _metrics()["per_label"]["alpha"]

    assert (alpha["true_positives"], alpha["false_positives"]) == (2, 0)
    assert (alpha["false_negatives"], alpha["true_negatives"]) == (1, 3)
    assert alpha["support"] == 3
    assert alpha["predicted_positives"] == 2
    assert alpha["precision"] == pytest.approx(1.0)
    assert alpha["recall"] == pytest.approx(2 / 3)


def test_confusion_counts_cover_every_row():
    result = _metrics()

    for name in LABELS:
        item = result["per_label"][name]
        total = (
            item["true_positives"]
            + item["false_positives"]
            + item["false_negatives"]
            + item["true_negatives"]
        )
        assert total == len(TARGETS)


def test_normal_is_scored_as_its_own_class():
    """Rows 3 and 5 are normal; row 4 is wrongly predicted normal."""

    normal = _metrics()["normal"]

    assert normal["support"] == 2
    assert normal["predicted"] == 3
    assert normal["recall"] == pytest.approx(1.0)
    assert normal["precision"] == pytest.approx(2 / 3)
    assert normal["f1"] == pytest.approx(0.8)
    assert normal["accuracy"] == pytest.approx(5 / 6)


def test_normal_accuracy_still_matches_the_nested_value():
    result = _metrics()

    assert result["normal_accuracy"] == result["normal"]["accuracy"]


def test_perfect_predictions_score_one():
    probabilities = TARGETS.astype(float)

    result = multilabel_metrics(TARGETS, probabilities, THRESHOLDS, LABELS)

    assert result["micro_f1"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["subset_accuracy"] == pytest.approx(1.0)
    assert result["hamming_loss"] == pytest.approx(0.0)
    assert result["normal"]["f1"] == pytest.approx(1.0)


def test_macro_f1_supported_drops_labels_below_the_threshold():
    both = _metrics(min_support=2)
    alpha_only = _metrics(min_support=3)

    assert both["supported_labels"] == ["alpha", "beta"]
    assert both["macro_f1_supported"] == pytest.approx(0.9)

    assert alpha_only["supported_labels"] == ["alpha"]
    assert alpha_only["insufficient_support_labels"] == ["beta"]
    assert alpha_only["macro_f1_supported"] == pytest.approx(0.8)


def test_macro_f1_supported_is_none_when_no_label_qualifies():
    result = _metrics(min_support=99)

    assert result["macro_f1_supported"] is None
    assert result["supported_labels"] == []


def test_undefined_scores_are_none_rather_than_zero():
    """A label with no positive example must not be reported as a score of 0.0."""

    targets = np.zeros((6, 2), dtype=np.int8)
    targets[:, 0] = TARGETS[:, 0]

    result = multilabel_metrics(targets, PROBABILITIES, THRESHOLDS, LABELS)
    beta = result["per_label"]["beta"]

    assert beta["support"] == 0
    assert beta["average_precision"] is None
    assert beta["roc_auc"] is None
    assert beta["f1"] == 0.0, "scikit-learn still reports 0.0 here, which is why support matters"


def test_label_with_every_row_positive_is_not_reported_as_perfect():
    """average_precision_score returns 1.0 for an all-positive column; it is not a real score."""

    targets = np.ones((6, 2), dtype=np.int8)

    result = multilabel_metrics(targets, PROBABILITIES, THRESHOLDS, LABELS)
    alpha = result["per_label"]["alpha"]

    assert alpha["average_precision"] is None
    assert alpha["roc_auc"] is None


def test_macro_average_precision_supported_excludes_undefined_labels():
    """scikit-learn folds an absent label in as 0.0; the supported variant leaves it out."""

    targets = TARGETS.copy()
    targets[:, 1] = 0

    result = multilabel_metrics(
        targets, PROBABILITIES, THRESHOLDS, LABELS, min_support=1
    )
    alpha_ap = result["per_label"]["alpha"]["average_precision"]

    assert result["per_label"]["beta"]["average_precision"] is None
    assert result["macro_average_precision_supported"] == pytest.approx(alpha_ap)
    assert result["macro_average_precision"] < result["macro_average_precision_supported"]


def test_scalar_threshold_is_broadcast_to_every_label():
    result = multilabel_metrics(TARGETS, PROBABILITIES, 0.5, LABELS)

    assert result["thresholds"] == {"alpha": 0.5, "beta": 0.5}
    assert result["macro_f1"] == pytest.approx(0.9)


def test_thresholds_are_applied_per_label():
    """Lowering only alpha's threshold recovers the row 4 positive."""

    result = multilabel_metrics(TARGETS, PROBABILITIES, np.array([0.3, 0.5]), LABELS)

    assert result["per_label"]["alpha"]["recall"] == pytest.approx(1.0)
    assert result["per_label"]["beta"]["recall"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_attack_ranking_uses_only_requests_that_carry_a_label():
    result = _metrics()

    assert result["attack_ranking"]["samples"] == 4
    assert 0.0 <= result["attack_ranking"]["label_ranking_average_precision"] <= 1.0


def test_attack_ranking_is_none_without_any_attack():
    targets = np.zeros((6, 2), dtype=np.int8)

    result = multilabel_metrics(targets, PROBABILITIES, THRESHOLDS, LABELS)

    assert result["attack_ranking"] == {
        "samples": 0,
        "label_ranking_average_precision": None,
    }


@pytest.mark.parametrize(
    "targets, probabilities, labels",
    [
        (TARGETS, PROBABILITIES[:3], LABELS),
        (TARGETS[:, 0], PROBABILITIES[:, 0], LABELS),
        (TARGETS, PROBABILITIES, ["alpha"]),
    ],
    ids=["shape_mismatch", "not_2d", "label_count_mismatch"],
)
def test_malformed_input_raises(targets, probabilities, labels):
    with pytest.raises(ValueError):
        multilabel_metrics(targets, probabilities, THRESHOLDS, labels)
