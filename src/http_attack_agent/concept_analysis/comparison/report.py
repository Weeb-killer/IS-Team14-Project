from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..common import read_json


def number(value):
    return "N/A" if value is None else f"{value:.4f}"


def render_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output)
    report = read_json(root / "comparison.json")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    binary_rows, per_label_rows = [], []
    for subset, policies in report["binary_results"].items():
        for policy, detectors in policies.items():
            for detector, values in detectors.items():
                binary_rows.append(
                    {"subset": subset, "policy": policy, "detector": detector, **values}
                )
    for subset, scopes in report["classification"].items():
        for scope, detectors in scopes.items():
            for detector, values in detectors.items():
                for label, scores in values["per_label"].items():
                    per_label_rows.append(
                        {
                            "subset": subset,
                            "scope": scope,
                            "detector": detector,
                            "label": label,
                            "declared_mapped_crs_rule_count": len(
                                report["declared_mapped_rules"][label]
                            ),
                            **scores,
                        }
                    )
    for name, rows in [
        ("binary_metrics.csv", binary_rows),
        ("per_label_metrics.csv", per_label_rows),
    ]:
        with (root / name).open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    primary = report["binary_results"]["common_test"]["saved_policy"]
    for ax, key, title in zip(
        axes,
        ("f1", "recall", "false_positive_rate"),
        ("Attack-detection F1", "Attack recall", "Normal false-positive rate"),
    ):
        values = [primary[d][key] for d in ("CRS", "SecBERT")]
        ax.bar(
            ["CRS", "SecBERT"],
            [v if v is not None else np.nan for v in values],
            color=["#e29842", "#197d9c"],
        )
        ax.set_ylim(0, 1.06)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for j, value in enumerate(values):
            if value is not None:
                ax.text(j, value + 0.02, f"{value:.4f}", ha="center")
    fig.suptitle(
        "Synthetic demonstration" if report["synthetic"] else "Held-out detection: saved policies"
    )
    fig.tight_layout()
    fig.savefig(figures / "binary_detection.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.4 * len(report["label_names"]))))
    all_metrics = report["classification"]["common_test"]["all_labels"]
    for ax, key, title in zip(axes, ("f1", "recall"), ("Per-label F1", "Per-label recall")):
        y = np.arange(len(report["label_names"]))
        for offset, detector, color in [(-0.18, "CRS", "#e29842"), (0.18, "SecBERT", "#197d9c")]:
            values = [
                all_metrics[detector]["per_label"][label][key] for label in report["label_names"]
            ]
            ax.barh(
                y + offset,
                [v if v is not None else np.nan for v in values],
                height=0.34,
                label=detector,
                color=color,
            )
        ax.set_yticks(
            y,
            [
                label + (" [unmapped CRS]" if label in report["unmapped_labels"] else "")
                for label in report["label_names"]
            ],
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 1.03)
        ax.set_title(title + " (CRS labels are a taxonomy projection)")
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "per_label_comparison.png", dpi=180)
    plt.close(fig)
    title = "# Synthetic CRS-versus-model demo" if report["synthetic"] else "# CRS versus SecBERT"
    coverage = report["coverage"]
    lines = [
        title,
        "",
        "**Synthetic fixture only; these numbers are not real CRS/SecBERT results.**"
        if report["synthetic"]
        else "Evaluation of independent detectors against the original dataset attack labels.",
        "",
        (
            f"Sampled preparation: **{report['sampled']}**. "
            f"Held-out rows: **{coverage['test_rows']:,}**. "
            f"Completed CRS comparisons: **{coverage['common_test_rows']:,}** "
            f"({coverage['crs_evaluation_fraction']:.2%}). "
            f"Distinct content-disjoint common test rows: **{coverage['clean_common_test_rows']:,}**."
        ),
        "",
        "## Primary task: normal versus any attack",
        "",
        (
            "Both detectors are scored on exactly the same completed held-out requests. "
            "CRS uses its recorded final blocking anomaly score and threshold in DetectionOnly mode "
            "(would-block decision). SecBERT uses its saved validation-selected class thresholds. "
            "Any rule match by itself is not counted as a CRS blocking decision."
        ),
        "",
        "![Primary detection comparison](figures/binary_detection.png)",
        "",
        "| Subset | Policy | Detector | N | Precision | Recall | F1 | FPR | FNR | ROC AUC | AP |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in binary_rows:
        lines.append(
            f"| {row['subset']} | {row['policy']} | {row['detector']} | {row['samples']:,} | "
            + " | ".join(
                number(row[key])
                for key in (
                    "precision",
                    "recall",
                    "f1",
                    "false_positive_rate",
                    "false_negative_rate",
                    "roc_auc",
                    "average_precision",
                )
            )
            + " |"
        )
    lines += [
        "",
        "[Full binary metrics and confusion counts](binary_metrics.csv)",
        "",
        (
            "The continuous model score is the maximum logit margin over saved class thresholds. "
            "CRS uses the actual aggregate anomaly score; neither score is a calibrated common "
            "probability scale. AUC/AP compare rankings, not score magnitudes."
        ),
        "",
        "## Validation-only operating points",
        "",
        (
            f"Calibration uses {report['calibration_samples']:,} common validation inputs, "
            "excluding exact training inputs and repeated validation inputs. No test labels "
            "select thresholds. validation_f1 maximizes validation detection F1; validation_fpr "
            f"targets a validation normal FPR at most {report['target_validation_fpr']:.2%}. "
            "Discrete CRS scores and ties may produce a lower achieved FPR; the test FPR is not guaranteed."
        ),
        "",
        "| Detector | Policy | Threshold | Validation FPR | Validation recall |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for detector, policies in report["calibration"].items():
        for policy, result in policies.items():
            lines.append(
                f"| {detector} | {policy} | {format(result['threshold'], '.17g')} | "
                f"{number(result['validation']['false_positive_rate'])} | "
                f"{number(result['validation']['recall'])} |"
            )
    lines += [
        "",
        (
            "Thresholds use full precision here: a value just above a tied CRS score "
            "can change a >= decision even when four-decimal rounding looks identical. "
            "Alternative policies are omitted if validation has fewer than four cases "
            "or lacks either normal or attack examples."
        ),
        "",
    ]
    lines += [
        "",
        "## Paired uncertainty",
        "",
        (
            "Differences below are **SecBERT minus CRS**. Positive F1/recall and negative FPR "
            "favor SecBERT. Intervals resample exact-input groups jointly for both detectors. "
            "They describe this test sample and fixed model; they are not a general deployment "
            "guarantee or a multiple-comparison significance claim."
        ),
        "",
        "| Subset | Policy | Metric | Difference | Lower 95% | Upper 95% |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for subset, policies in report["paired_intervals"].items():
        for policy, results in policies.items():
            for metric, interval in results.get("deltas", {}).items():
                lines.append(
                    f"| {subset} | {policy} | {metric} | {number(interval['point'])} | "
                    f"{number(interval['lower_95'])} | {number(interval['upper_95'])} |"
                )
    lines += [
        "",
        "## Attack-category comparison and coverage",
        "",
        report["classification_policy"],
        "",
        (
            "Default mapping uses only the **terminal CAPEC ID** of a rule tag. For example, "
            "capec/1000/152/248/66 maps to SQL injection (66), not additionally to its ancestor "
            "command injection (248). This conservative mapping can leave real semantic overlap "
            "unmapped. The complete mapping is recorded in comparison.json."
        ),
        "",
        "| Label | Declared mapped rules | CRS F1 | SecBERT F1 | Test positives |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in report["label_names"]:
        left = all_metrics["CRS"]["per_label"][label]
        right = all_metrics["SecBERT"]["per_label"][label]
        lines.append(
            f"| {label} | {len(report['declared_mapped_rules'][label])} | "
            f"{number(left['f1'])} | {number(right['f1'])} | {left['attack_support']} |"
        )
    lines += [
        "",
        "![Per-category metrics](figures/per_label_comparison.png)",
        "",
        "| Subset | Scope | Detector | Micro F1 | Macro F1 (support >= 20) | Subset accuracy |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for subset, scopes in report["classification"].items():
        for scope, detectors in scopes.items():
            for detector, result in detectors.items():
                lines.append(
                    f"| {subset} | {scope} | {detector} | {number(result['micro_f1'])} | "
                    f"{number(result['macro_f1_supported'])} | {number(result['subset_accuracy'])} |"
                )
    lines += [
        "",
        "[Full per-category tables and confusion counts](per_label_metrics.csv)",
        "",
        (
            "**Full-task scores include unmapped CRS labels as zero predictions.** "
            "Do not claim better common-category detection solely from those scores. "
            "Mapped-label results are a separate view. A declared rule is not a guarantee "
            "that it executes for every request or covers every dataset attack."
        ),
        "",
        "## Disagreements at saved policies",
        "",
        "| Case | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {count:,} |" for name, count in report["disagreement_counts"].items())
    lines += [
        "",
        (
            "[All held-out predictions](predictions.csv) · "
            "[Seeded examples with actual matched rule IDs](disagreement_examples.json)"
        ),
        "",
        "## Missing WAF inspections",
        "",
        (
            f"Excluded unknown CRS requests: {coverage['excluded_unknown_crs_rows']:,}; "
            f"attacks: {coverage['excluded_attacks']:,}; normals: {coverage['excluded_normals']:,}. "
            "They are excluded from both detectors in the paired tables, not labeled normal."
        ),
        "",
        "| Failure treatment on all test rows | CRS F1 | CRS FPR | CRS recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in report["unknown_crs_scenarios"].items():
        lines.append(
            f"| {name} | {number(result['f1'])} | {number(result['false_positive_rate'])} | "
            f"{number(result['recall'])} |"
        )
    lines += [
        "",
        (
            "These are explicit failure-policy scenarios, not observed WAF predictions. "
            "Model metrics on all test rows are also retained in comparison.json. Large or "
            "class-dependent WAF failures prevent a strong comparative conclusion."
        ),
        "",
        "## Observed processing cost",
        "",
        "| Stage | Rows | Seconds | Requests/second | Configuration |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for detector, values in report["runtime"].items():
        if not isinstance(values, dict):
            continue
        settings = (
            f"CPU workers={values.get('workers')}"
            if detector == "CRS"
            else f"device={values.get('device')}, batch={values.get('batch_size')}"
        )
        lines.append(
            f"| {detector} | {values.get('rows', 'N/A')} | {number(values.get('seconds'))} | "
            f"{number(values.get('rows_per_second'))} | {settings} |"
        )
    lines += [
        "",
        report["runtime"]["note"],
        "",
        (
            "These measurements include different export/logging work and hardware allocation. "
            "They must not be advertised as matched online latency or an inherent speed advantage."
        ),
        "",
        "## Provenance and limits",
        "",
        f"- CRS engine: {report['engine']}; paranoia level: {report['paranoia_level']}.",
        f"- CRS tree SHA-256: `{report['crs_tree_sha256']}`.",
        f"- Model checkpoint SHA-256: `{report['checkpoint_sha256']}`.",
        f"- Model max length: {report['max_length']}.",
        "",
    ]
    lines.extend(f"- {text}" for text in report["limitations"])
    lines += ["", "[Complete machine-readable comparison](comparison.json)", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
