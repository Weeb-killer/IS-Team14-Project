import json

import numpy as np
import pytest

from http_attack_agent.evaluation.split_audit import audit_split

LABELS = ["alpha", "beta"]


def _targets(rows: int = 30) -> np.ndarray:
    """Both labels appear at a stable rate, so every split stays representative."""

    matrix = np.zeros((rows, 2), dtype=np.float32)
    matrix[::2, 0] = 1.0
    matrix[::3, 1] = 1.0
    return matrix


def _healthy_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.arange(0, 18), np.arange(18, 24), np.arange(24, 30)


def test_healthy_split_reports_no_errors():
    audit = audit_split(_targets(), _healthy_split(), LABELS, min_support=1)

    assert audit.ok
    assert audit.errors == ()
    assert audit.warnings == ()
    assert audit.split_sizes == {"train": 18, "validation": 6, "test": 6}


def test_counts_and_prevalence_match_the_split():
    audit = audit_split(_targets(), _healthy_split(), LABELS, min_support=1)

    assert audit.label_counts["alpha"] == {"train": 9, "validation": 3, "test": 3}
    assert audit.label_counts["beta"] == {"train": 6, "validation": 2, "test": 2}
    assert audit.label_prevalence["alpha"]["train"] == pytest.approx(0.5)
    assert audit.label_prevalence["beta"]["test"] == pytest.approx(2 / 6)


def test_rows_without_any_label_are_counted_as_normal():
    targets = np.zeros((10, 2), dtype=np.float32)
    targets[0, 0] = 1.0
    split = (np.arange(0, 6), np.arange(6, 8), np.arange(8, 10))

    audit = audit_split(targets, split, LABELS, min_support=1)

    assert audit.normal_counts == {"train": 5, "validation": 2, "test": 2}


def test_label_absent_from_test_is_an_error():
    targets = _targets()
    targets[24:, 0] = 0.0  # alpha disappears from the test split only

    audit = audit_split(targets, _healthy_split(), LABELS, min_support=1)

    assert not audit.ok
    assert audit.labels_with("no_positives_in_test") == ["alpha"]
    assert "undefined" in audit.errors[0].message


def test_label_absent_from_validation_is_an_error():
    targets = _targets()
    targets[18:24, 1] = 0.0  # beta disappears from the validation split only

    audit = audit_split(targets, _healthy_split(), LABELS, min_support=1)

    assert audit.labels_with("no_positives_in_validation") == ["beta"]
    assert "0.5" in audit.errors[0].message


def test_test_split_without_negatives_is_an_error():
    targets = _targets()
    targets[24:, 0] = 1.0  # every test row is positive for alpha

    audit = audit_split(targets, _healthy_split(), LABELS, min_support=1)

    assert audit.labels_with("no_negatives_in_test") == ["alpha"]


def test_low_support_is_a_warning_and_does_not_invalidate_the_split():
    targets = _targets()
    targets[24:, 0] = 0.0
    targets[24, 0] = 1.0  # exactly one alpha example survives in test

    audit = audit_split(targets, _healthy_split(), LABELS, min_support=20)

    assert audit.ok, "a warning must not mark the split as unusable"
    assert "alpha" in audit.labels_with("low_support_in_test")
    alpha_warning = next(
        item
        for item in audit.warnings
        if item.code == "low_support_in_test" and item.label == "alpha"
    )
    assert "only 1 positive example(s) in test" in alpha_warning.message
    assert "reorder models" in alpha_warning.message


def test_prevalence_shift_is_reported_when_splits_disagree():
    targets = np.zeros((120, 2), dtype=np.float32)
    targets[:80, 0] = 1.0  # 100% of train, far above the later splits
    targets[80, 0] = 1.0
    targets[100, 0] = 1.0
    targets[:, 1] = 1.0
    split = (np.arange(0, 80), np.arange(80, 100), np.arange(100, 120))

    audit = audit_split(targets, split, LABELS, min_support=1)

    assert audit.labels_with("prevalence_shift") == ["alpha"]


def test_overlapping_splits_are_reported_as_leakage():
    split = (np.arange(0, 20), np.arange(18, 24), np.arange(24, 30))

    audit = audit_split(_targets(), split, LABELS, min_support=1)

    assert not audit.ok
    overlap = [item for item in audit.errors if item.code == "split_overlap"]
    assert len(overlap) == 1
    assert "leakage" in overlap[0].message


def test_empty_split_is_reported():
    split = (np.arange(0, 24), np.empty(0, dtype=int), np.arange(24, 30))

    audit = audit_split(_targets(), split, LABELS, min_support=1)

    assert [item.code for item in audit.errors] == ["empty_split"]


def test_srbh_failure_mode_flags_every_label():
    """All attacks land in train, mirroring a temporal split on SR-BH 2020."""

    targets = np.zeros((100, 2), dtype=np.float32)
    targets[:60, 0] = 1.0
    targets[:60, 1] = 1.0
    split = (np.arange(0, 70), np.arange(70, 85), np.arange(85, 100))

    audit = audit_split(targets, split, LABELS, min_support=20)

    assert not audit.ok
    assert set(audit.labels_with("no_positives_in_test")) == set(LABELS)
    assert set(audit.labels_with("no_positives_in_validation")) == set(LABELS)


def test_report_is_serializable_and_renders():
    audit = audit_split(_targets(), _healthy_split(), LABELS, min_support=1)

    payload = json.loads(json.dumps(audit.to_dict()))
    assert payload["ok"] is True
    assert payload["labels"]["alpha"]["counts"]["test"] == 3

    report = audit.format_report()
    assert "Label support by split" in report
    assert "alpha" in report


@pytest.mark.parametrize(
    "targets, split, labels",
    [
        (np.zeros(10, dtype=np.float32), _healthy_split(), LABELS),
        (np.zeros((30, 3), dtype=np.float32), _healthy_split(), LABELS),
        (np.zeros((30, 2), dtype=np.float32), (np.arange(30), np.arange(0)), LABELS),
        (
            np.zeros((30, 2), dtype=np.float32),
            (np.arange(0, 18), np.arange(18, 24), np.arange(24, 40)),
            LABELS,
        ),
    ],
    ids=["not_2d", "label_count_mismatch", "wrong_split_count", "index_out_of_range"],
)
def test_malformed_input_raises(targets, split, labels):
    with pytest.raises(ValueError):
        audit_split(targets, split, labels)
