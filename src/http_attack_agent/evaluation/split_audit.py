"""Report whether a train/validation/test split can support the reported metrics.

A multi-label split can silently invalidate an entire benchmark. When a label has no
positive example in the test split, scikit-learn returns ``0.0`` under
``zero_division=0``, which is indistinguishable from a model that found nothing. When a
label has no positive example in the validation split, threshold calibration skips it
and the default 0.5 is reported as if it had been tuned. This module makes both
conditions explicit before any training time is spent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from ..visualization.profile import MIN_RELIABLE_SUPPORT

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")

# Ratio between the highest and lowest non-zero prevalence of one label across splits.
# Thresholds calibrated on validation do not transfer to a test split that was drawn
# from a visibly different distribution.
DEFAULT_PREVALENCE_RATIO = 10.0

ERROR = "error"
WARNING = "warning"

# Short consequence shown once per finding code in the terminal report. Per-label
# messages stay on each Finding for the JSON output.
FINDING_SUMMARIES: dict[str, str] = {
    "no_positives_in_validation": (
        "threshold calibration is skipped; the default 0.5 is reported as if tuned"
    ),
    "no_positives_in_test": (
        "precision, recall and F1 are undefined and are reported as 0.0"
    ),
    "no_negatives_in_test": "average precision and ROC AUC are undefined",
    "low_support_in_test": "one example is enough to reorder the compared models",
    "prevalence_shift": "thresholds tuned on validation may not transfer to test",
}


@dataclass(frozen=True)
class Finding:
    """One problem detected in a split."""

    severity: str
    code: str
    message: str
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "label": self.label,
        }


@dataclass(frozen=True)
class SplitAudit:
    """Per-label support for every split, plus the problems that support implies."""

    split_sizes: dict[str, int]
    normal_counts: dict[str, int]
    label_counts: dict[str, dict[str, int]]
    label_prevalence: dict[str, dict[str, float]]
    findings: tuple[Finding, ...]
    min_support: int

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.severity == ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.severity == WARNING)

    @property
    def ok(self) -> bool:
        """True when no error was found. Warnings do not invalidate a split."""

        return not self.errors

    def labels_with(self, code: str) -> list[str]:
        """Labels flagged by one finding code, in the order they were reported."""

        return [item.label for item in self.findings if item.code == code and item.label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_sizes": dict(self.split_sizes),
            "normal_counts": dict(self.normal_counts),
            "min_support": self.min_support,
            "labels": {
                label: {
                    "counts": dict(self.label_counts[label]),
                    "prevalence": dict(self.label_prevalence[label]),
                }
                for label in self.label_counts
            },
            "findings": [item.to_dict() for item in self.findings],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "ok": self.ok,
        }

    def format_report(self) -> str:
        return _format_report(self)


def _format_prevalence(value: float) -> str:
    if value <= 0.0:
        return "0%"
    if value < 0.0001:
        return "<0.01%"
    return f"{value * 100:.2f}%"


def _format_cell(count: int, prevalence: float) -> str:
    return f"{count:,} ({_format_prevalence(prevalence)})"


def _status_by_label(audit: SplitAudit) -> dict[str, str]:
    status = {label: "ok" for label in audit.label_counts}
    for finding in audit.findings:
        if finding.label is None:
            continue
        if finding.severity == ERROR:
            status[finding.label] = "ERROR"
        elif status.get(finding.label) == "ok":
            status[finding.label] = "warn"
    return status


def _format_report(audit: SplitAudit) -> str:
    lines: list[str] = ["Split sizes"]
    total = sum(audit.split_sizes.values())
    for name in SPLIT_NAMES:
        size = audit.split_sizes[name]
        share = f"{size / total * 100:.1f}%" if total else "n/a"
        normal = audit.normal_counts[name]
        lines.append(f"  {name:<12}{size:>12,}  ({share})   normal: {normal:,}")
    lines.append("")

    status = _status_by_label(audit)
    header = ["label", *SPLIT_NAMES, "status"]
    rows: list[list[str]] = [header]
    for label, counts in audit.label_counts.items():
        prevalence = audit.label_prevalence[label]
        rows.append(
            [
                label,
                *[_format_cell(counts[name], prevalence[name]) for name in SPLIT_NAMES],
                status[label],
            ]
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]

    lines.append("Label support by split")
    for index, row in enumerate(rows):
        lines.append("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            lines.append("  " + "  ".join("-" * width for width in widths))
    lines.append("")

    lines.append(f"{len(audit.errors)} error(s), {len(audit.warnings)} warning(s)")
    lines.extend(_format_findings(audit.findings))
    return "\n".join(lines)


def _format_findings(findings: Sequence[Finding], max_labels: int = 6) -> list[str]:
    """Collapse per-label findings that share a code, which keeps a total failure short."""

    lines: list[str] = []
    grouped: dict[tuple[str, str], list[str]] = {}
    for finding in findings:
        prefix = "ERROR" if finding.severity == ERROR else "warn "
        if finding.label is None:
            lines.append(f"  [{prefix}] {finding.message}")
            continue
        grouped.setdefault((finding.severity, finding.code), []).append(finding.label)

    for (severity, code), labels in grouped.items():
        prefix = "ERROR" if severity == ERROR else "warn "
        shown = ", ".join(labels[:max_labels])
        if len(labels) > max_labels:
            shown += f", ... (+{len(labels) - max_labels} more)"
        lines.append(f"  [{prefix}] {code}: {len(labels)} label(s): {shown}")
        summary = FINDING_SUMMARIES.get(code)
        if summary:
            lines.append(f"          -> {summary}")
    return lines


def _validate_inputs(
    targets: np.ndarray,
    split_indices: Sequence[np.ndarray],
    label_names: Sequence[str],
) -> None:
    if targets.ndim != 2:
        raise ValueError(f"targets must be two-dimensional, got shape {targets.shape}")
    if targets.shape[1] != len(label_names):
        raise ValueError(
            f"targets has {targets.shape[1]} columns but {len(label_names)} label "
            "names were provided"
        )
    if len(split_indices) != len(SPLIT_NAMES):
        raise ValueError(
            f"split_indices must contain exactly {len(SPLIT_NAMES)} arrays "
            f"({', '.join(SPLIT_NAMES)}), got {len(split_indices)}"
        )
    for name, indices in zip(SPLIT_NAMES, split_indices):
        array = np.asarray(indices)
        if array.size and (array.min() < 0 or array.max() >= len(targets)):
            raise ValueError(
                f"The {name} split references rows outside the target matrix of "
                f"{len(targets)} rows"
            )


def _overlap_findings(split_indices: Sequence[np.ndarray]) -> list[Finding]:
    """Detect leakage between splits, which slicing bugs can silently introduce."""

    findings: list[Finding] = []
    for first in range(len(SPLIT_NAMES)):
        for second in range(first + 1, len(SPLIT_NAMES)):
            shared = np.intersect1d(
                np.asarray(split_indices[first]), np.asarray(split_indices[second])
            )
            if shared.size:
                findings.append(
                    Finding(
                        severity=ERROR,
                        code="split_overlap",
                        message=(
                            f"The {SPLIT_NAMES[first]} and {SPLIT_NAMES[second]} splits "
                            f"share {shared.size:,} row(s). Any metric computed on this "
                            "split is contaminated by leakage."
                        ),
                    )
                )
    return findings


def _prevalence_summary(prevalence: dict[str, float]) -> str:
    return ", ".join(
        f"{name} {_format_prevalence(prevalence[name])}" for name in SPLIT_NAMES
    )


def audit_split(
    targets: np.ndarray,
    split_indices: Sequence[np.ndarray],
    label_names: Sequence[str],
    *,
    min_support: int = MIN_RELIABLE_SUPPORT,
    prevalence_ratio: float = DEFAULT_PREVALENCE_RATIO,
) -> SplitAudit:
    """Measure per-label support in each split and report what it invalidates.

    ``targets`` is the binary label matrix for the complete table and ``split_indices``
    holds positional row indices, exactly as ``deterministic_split_indices`` returns
    them. Nothing is modified; the result only describes the split that was passed in.
    """

    targets = np.asarray(targets)
    _validate_inputs(targets, split_indices, label_names)
    positives = targets > 0

    split_rows = {
        name: np.asarray(indices, dtype=int)
        for name, indices in zip(SPLIT_NAMES, split_indices)
    }
    split_sizes = {name: int(rows.size) for name, rows in split_rows.items()}
    normal_counts = {
        name: int((~positives[rows].any(axis=1)).sum()) if rows.size else 0
        for name, rows in split_rows.items()
    }

    label_counts: dict[str, dict[str, int]] = {}
    label_prevalence: dict[str, dict[str, float]] = {}
    for column, label in enumerate(label_names):
        counts: dict[str, int] = {}
        prevalence: dict[str, float] = {}
        for name, rows in split_rows.items():
            count = int(positives[rows, column].sum()) if rows.size else 0
            counts[name] = count
            prevalence[name] = count / rows.size if rows.size else 0.0
        label_counts[label] = counts
        label_prevalence[label] = prevalence

    findings: list[Finding] = []
    for name in SPLIT_NAMES:
        if split_sizes[name] == 0:
            findings.append(
                Finding(
                    severity=ERROR,
                    code="empty_split",
                    message=f"The {name} split is empty.",
                )
            )
    findings.extend(_overlap_findings(split_indices))

    for label in label_names:
        counts = label_counts[label]
        test_count = counts["test"]
        test_size = split_sizes["test"]

        if counts["validation"] == 0 and split_sizes["validation"]:
            findings.append(
                Finding(
                    severity=ERROR,
                    code="no_positives_in_validation",
                    label=label,
                    message=(
                        f"{label}: no positive example in validation. Threshold "
                        "calibration skips this label and reports the default 0.5 as "
                        "if it had been tuned."
                    ),
                )
            )

        if test_size == 0:
            continue

        if test_count == 0:
            findings.append(
                Finding(
                    severity=ERROR,
                    code="no_positives_in_test",
                    label=label,
                    message=(
                        f"{label}: no positive example in test. Precision, recall and "
                        "F1 are undefined and will be reported as 0.0."
                    ),
                )
            )
        elif test_count == test_size:
            findings.append(
                Finding(
                    severity=ERROR,
                    code="no_negatives_in_test",
                    label=label,
                    message=(
                        f"{label}: every test row is positive. Average precision and "
                        "ROC AUC are undefined for this label."
                    ),
                )
            )
        elif test_count < min_support:
            findings.append(
                Finding(
                    severity=WARNING,
                    code="low_support_in_test",
                    label=label,
                    message=(
                        f"{label}: only {test_count} positive example(s) in test, below "
                        f"the reliable-support threshold of {min_support}. A single "
                        "example moves the score enough to reorder models."
                    ),
                )
            )

        prevalence = label_prevalence[label]
        observed = [
            prevalence[name]
            for name in SPLIT_NAMES
            if split_sizes[name] and prevalence[name] > 0.0
        ]
        if len(observed) == len(SPLIT_NAMES):
            ratio = max(observed) / min(observed)
            if ratio > prevalence_ratio:
                findings.append(
                    Finding(
                        severity=WARNING,
                        code="prevalence_shift",
                        label=label,
                        message=(
                            f"{label}: prevalence differs by {ratio:.0f}x across splits "
                            f"({_prevalence_summary(prevalence)}). Thresholds "
                            "calibrated on validation may not transfer to test."
                        ),
                    )
                )

    return SplitAudit(
        split_sizes=split_sizes,
        normal_counts=normal_counts,
        label_counts=label_counts,
        label_prevalence=label_prevalence,
        findings=tuple(findings),
        min_support=min_support,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report per-label support for the configured train/validation/test split "
            "before any training time is spent."
        )
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.srbh2020.yaml",
        help="Path to the dataset YAML configuration.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=MIN_RELIABLE_SUPPORT,
        help=(
            "Positive examples required in the test split before a per-label score is "
            f"treated as reliable (default: {MIN_RELIABLE_SUPPORT})."
        ),
    )
    parser.add_argument(
        "--prevalence-ratio",
        type=float,
        default=DEFAULT_PREVALENCE_RATIO,
        help=(
            "Highest tolerated ratio between the largest and smallest non-zero "
            "prevalence of one label across splits "
            f"(default: {DEFAULT_PREVALENCE_RATIO})."
        ),
    )
    parser.add_argument("--json", help="Optional path for the audit as a JSON file.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status code 1 when the audit reports an error.",
    )
    args = parser.parse_args()

    if args.min_support < 0:
        parser.error("--min-support cannot be negative")
    if args.prevalence_ratio < 1.0:
        parser.error("--prevalence-ratio must be at least 1.0")

    # Imported here so the pure audit functions stay importable without pandas.
    from ..data import (
        DatasetConfig,
        build_targets,
        deterministic_split_indices,
        load_table,
    )

    config = DatasetConfig.from_yaml(args.dataset_config)
    frame = load_table(config)
    targets = build_targets(frame, config.label_columns)
    audit = audit_split(
        targets,
        deterministic_split_indices(frame, config),
        list(config.label_columns),
        min_support=args.min_support,
        prevalence_ratio=args.prevalence_ratio,
    )
    print(audit.format_report())

    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
        print(f"\nAudit written to {output_path}")

    if args.strict and not audit.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
