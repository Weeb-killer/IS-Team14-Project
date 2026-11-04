from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import read_json


def _number(value):
    return "N/A" if value is None else f"{value:.4f}"


def report(bank_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(bank_dir)
    probe = read_json(path / "probe_report.json")
    names = list(probe["concepts"])
    rows = []
    for name, result in probe["concepts"].items():
        row = {"concept": name, "status": result["status"]}
        for split, counts in result.get("support", {}).items():
            for key, value in counts.items():
                row[f"{split}_{key}"] = value
        for split in ("test", "clean_test"):
            for metric in ("roc_auc", "average_precision", "balanced_accuracy", "f1", "prevalence"):
                row[f"{split}_{metric}"] = result.get(split, {}).get(metric)
        row["permutation_auc_mean"] = result.get("permutation_auc_mean")
        row["permutation_tail_fraction"] = result.get("permutation_tail_fraction")
        row["clean_test_reliable_support"] = result.get("clean_test_reliable_support", False)
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (path / "probe_metrics.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    figure_dir = path / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.35 * len(names))), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("roc_auc", "average_precision"),
        ("Held-out concept ROC AUC", "Held-out average precision"),
    ):
        values = [probe["concepts"][n].get("clean_test", {}).get(metric, np.nan) for n in names]
        ax.barh(np.arange(len(names)), values, color="#167d9a")
        if metric == "roc_auc":
            ax.axvline(0.5, color="gray", ls="--", label="Chance ROC AUC")
        else:
            prevalence = [
                probe["concepts"][n].get("clean_test", {}).get("prevalence", np.nan) for n in names
            ]
            ax.scatter(
                prevalence,
                np.arange(len(names)),
                marker="|",
                color="#d76b30",
                label="Positive prevalence",
            )
        ax.set_xlim(0, 1.02)
        ax.set_title(title)
        ax.set_yticks(np.arange(len(names)), names)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(figure_dir / "probe_scores.png", dpi=170)
    plt.close(fig)
    lines = [
        "# Synthetic concept-analysis demo"
        if probe.get("synthetic")
        else "# SecBERT concept analysis",
        "",
        (
            "**Synthetic data and a toy classifier only. No real SecBERT or CRS benchmark "
            "was performed.**"
            if probe.get("synthetic")
            else "Post-hoc analysis of the supplied trained checkpoint."
        ),
        "",
        (
            "This report uses PCBM-style concept banks and linear probes. The original "
            "classifier supplies predictions; no concept-to-attack classifier is trained."
        ),
        "",
        (
            f"Prepared requests: {probe['rows']:,}. Sampled run: **{probe['sampled']}**. "
            f"Probe training cap per concept: {probe['max_training_rows_per_concept']:,}. "
            f"Random-label runs: {probe['permutations']}. Repeated fits: {probe['repeats']}."
        ),
        "",
        (
            "CRS annotations are version-dependent detection proxies, not infallible semantic "
            "ground truth. Unknown/failed inspections are excluded from probe targets."
        ),
        "",
        "## Held-out concept decoding",
        "",
        (
            "Clean test excludes exact inputs present in training/validation and repeated test "
            "inputs. This does not undo exposure during historical model training or detect "
            "near duplicates. Low-support results are exploratory."
        ),
        "",
        "| Concept | Status | Test + / - | Clean AUC | Clean AP | Clean F1 | Reliable clean support |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['concept']} | {row['status']} | "
            f"{row.get('test_positives', 0)} / {row.get('test_negatives', 0)} | "
            f"{_number(row.get('clean_test_roc_auc'))} | "
            f"{_number(row.get('clean_test_average_precision'))} | "
            f"{_number(row.get('clean_test_f1'))} | {row['clean_test_reliable_support']} |"
        )
    lines += [
        "",
        "![Concept probe performance](figures/probe_scores.png)",
        "",
        "[Complete metric table](probe_metrics.csv) · [Full probe results](probe_report.json)",
        "",
        (
            "A high probe score indicates linear decodability. It does not establish that the "
            "original classifier uses that concept. Permutation tail fractions are empirical "
            "diagnostics, without a corrected multiple-testing significance claim."
        ),
        "",
    ]
    if "annotation_correlations" in probe:
        values = np.asarray(probe["annotation_correlations"], dtype=float)
        fig, ax = plt.subplots(figsize=(max(6, 0.36 * len(names)), max(5, 0.32 * len(names))))
        mesh = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(names)), names, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(names)), names, fontsize=8)
        ax.set_title("Training-set CRS annotation correlations (not causal relationships)")
        fig.colorbar(mesh, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(figure_dir / "concept_correlations.png", dpi=170)
        plt.close(fig)
        lines += [
            "## Concept co-occurrence",
            "",
            "![Training concept correlations](figures/concept_correlations.png)",
            "",
            "Correlated detection concepts can make individual direction interpretations ambiguous.",
            "",
        ]
    conditional_rows = []
    for name, result in probe["concepts"].items():
        for label, value in result.get("conditional_on_attack", {}).items():
            metrics = value.get("clean_test", {})
            conditional_rows.append(
                {
                    "concept": name,
                    "attack": label,
                    "status": value["status"],
                    **{
                        k: metrics.get(k)
                        for k in ("positives", "negatives", "roc_auc", "average_precision", "f1")
                    },
                }
            )
    if conditional_rows:
        with (path / "conditional_metrics.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(conditional_rows[0]))
            writer.writeheader()
            writer.writerows(conditional_rows)
        lines += [
            "## Within-attack concept decoding",
            "",
            "| Concept | Attack | Status | Clean AUC | Clean AP | Clean F1 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for row in conditional_rows:
            lines.append(
                f"| {row['concept']} | {row['attack']} | {row['status']} | "
                f"{_number(row['roc_auc'])} | {_number(row['average_precision'])} | "
                f"{_number(row['f1'])} |"
            )
        lines += ["", "[Complete conditional table](conditional_metrics.csv)", ""]
    lines += [
        "## Held-out request examples",
        "",
        "| Concept | High-score observed positives | High-score false positives |",
        "| --- | --- | --- |",
    ]
    for name, result in probe["concepts"].items():
        examples = result.get("example_row_ids", {})
        lines.append(
            f"| {name} | {', '.join(examples.get('high_score_positives', [])) or 'N/A'} | "
            f"{', '.join(examples.get('high_score_false_positives', [])) or 'N/A'} |"
        )
    lines += [
        "",
        (
            "These are score-ranked held-out examples, not a claim that all examples "
            "share the same semantic mechanism."
        ),
        "",
    ]
    sensitivity_path = path / "sensitivity_report.json"
    if sensitivity_path.is_file():
        sensitivity = read_json(sensitivity_path)
        labels = list(next((v for v in sensitivity["concepts"].values() if v), {}))
        if labels:
            fig, axes = plt.subplots(1, 2, figsize=(18, max(4, 0.35 * len(names))))
            for ax, key, title, cmap in zip(
                axes,
                ("tcav_score", "mean_logit_drop"),
                (
                    "Original-head TCAV (fraction with positive derivative)",
                    "Original-head mean logit drop after direction intervention",
                ),
                ("viridis", "coolwarm"),
            ):
                values = np.asarray(
                    [
                        [
                            sensitivity["concepts"].get(n, {}).get(label, {}).get(key, np.nan)
                            for label in labels
                        ]
                        for n in names
                    ],
                    dtype=float,
                )
                options = {"vmin": 0, "vmax": 1} if key == "tcav_score" else {}
                finite = values[np.isfinite(values)]
                if key != "tcav_score" and len(finite):
                    bound = max(float(np.abs(finite).max()), 0.001)
                    options = {"vmin": -bound, "vmax": bound}
                mesh = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, **options)
                ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
                ax.set_yticks(np.arange(len(names)), names)
                ax.set_title(title, fontsize=10)
                fig.colorbar(mesh, ax=ax, shrink=0.7)
            fig.tight_layout()
            fig.savefig(figure_dir / "head_sensitivity.png", dpi=170)
            plt.close(fig)
            lines += [
                "## Original classifier sensitivity",
                "",
                (
                    f"Analyzed {sensitivity['samples']:,} distinct held-out inputs. "
                    "A capped, label-aware sample supports rare-class inspection; these are "
                    "sensitivity measurements, not population classification metrics."
                ),
                "",
                "![TCAV and direction intervention](figures/head_sensitivity.png)",
                "",
                "| Concept | Attack | Samples | TCAV | Mean logit drop | Random matched drop |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
            for name in names:
                for label, result in sensitivity["concepts"].get(name, {}).items():
                    lines.append(
                        f"| {name} | {label} | {result['samples']} | "
                        f"{_number(result.get('tcav_score'))} | "
                        f"{_number(result.get('mean_logit_drop'))} | "
                        f"{_number(result.get('random_norm_matched_mean_logit_drop'))} |"
                    )
            lines += [
                "",
                "[Full sensitivity results](sensitivity_report.json)",
                "",
                (
                    "Positive logit drop means that this intervention reduced the original "
                    "class score. It is not a percentage contribution or a causal guarantee. "
                    "Correlated concepts and off-distribution embeddings limit interpretation."
                ),
                "",
            ]
    lines += [
        "## Provenance",
        "",
        f"- Checkpoint SHA-256: `{probe['checkpoint_sha256']}`",
        f"- Prepared requests SHA-256: `{probe['prepared_sha256']}`",
        f"- Concept annotations SHA-256: `{probe['concepts_sha256']}`",
        "",
        (
            "Field visibility is recorded separately from exact evidence-span visibility. "
            "Partial/truncated fields do not establish that the model saw the WAF match. "
            "The original model may use information outside the concept bank."
        ),
        "",
    ]
    (path / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(path / "README.md"), "concepts": len(names)}
