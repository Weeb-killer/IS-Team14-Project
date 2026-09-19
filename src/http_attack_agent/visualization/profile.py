from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import DatasetConfig, resolve_dataset_path

MIN_RELIABLE_SUPPORT = 20
FIGURE_NAMES = (
    "01_dataset_overview",
    "02_attack_label_prevalence",
    "03_label_cooccurrence",
    "04_http_method_by_attack",
    "05_request_length_distribution",
    "06_attack_timeline",
)
FIGURE_TITLES = (
    "Dataset overview",
    "Attack label prevalence",
    "Label co-occurrence",
    "HTTP method by attack",
    "Request length distribution",
    "Attack timeline",
)


@dataclass
class DatasetProfile:
    """Aggregate-only statistics used to build the public dataset profile."""

    label_keys: tuple[str, ...]
    label_columns: tuple[str, ...]
    display_labels: tuple[str, ...]
    row_count: int
    normal_count: int
    label_counts: np.ndarray
    cooccurrence: np.ndarray
    cardinality_counts: dict[int, int]
    method_counts: dict[str, int]
    label_method_counts: tuple[dict[str, int], ...]
    weekly_row_counts: dict[str, int]
    weekly_label_counts: dict[str, tuple[int, ...]]
    length_sample: pd.DataFrame
    timestamp_start: str | None
    timestamp_end: str | None

    @property
    def attack_count(self) -> int:
        return self.row_count - self.normal_count


class _StratifiedReservoir:
    def __init__(self, sample_size: int, seed: int) -> None:
        normal_capacity = sample_size // 2
        self.capacities = {False: normal_capacity, True: sample_size - normal_capacity}
        self.samples: dict[bool, list[tuple[int, int]]] = {False: [], True: []}
        self.seen = {False: 0, True: 0}
        self.rng = np.random.default_rng(seed)

    def add_many(
        self,
        target_lengths: np.ndarray,
        body_lengths: np.ndarray,
        is_attack: np.ndarray,
    ) -> None:
        for target_length, body_length, attack in zip(
            target_lengths, body_lengths, is_attack
        ):
            group = bool(attack)
            self.seen[group] += 1
            capacity = self.capacities[group]
            if capacity == 0:
                continue
            item = (int(target_length), int(body_length))
            if len(self.samples[group]) < capacity:
                self.samples[group].append(item)
                continue
            replacement = int(self.rng.integers(0, self.seen[group]))
            if replacement < capacity:
                self.samples[group][replacement] = item

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for attack in (False, True):
            status = "Attack" if attack else "Normal"
            rows.extend(
                {
                    "request_target_length": target_length,
                    "body_length": body_length,
                    "status": status,
                }
                for target_length, body_length in self.samples[attack]
            )
        return pd.DataFrame(
            rows,
            columns=["request_target_length", "body_length", "status"],
        )


def _display_label(column: str) -> str:
    match = re.match(r"^(\d+)\s*-\s*(.+)$", column)
    if match:
        return f"CAPEC-{match.group(1)} {match.group(2)}"
    return column


def _required_columns(config: DatasetConfig) -> tuple[list[str], str, str, str, str]:
    if not config.label_columns:
        raise ValueError("Dataset visualization requires at least one attack label.")
    logical_fields = {
        "method": config.text_columns.get("method"),
        "request_target": config.text_columns.get("request_target"),
        "body": config.text_columns.get("body"),
    }
    missing_fields = [name for name, column in logical_fields.items() if not column]
    if missing_fields:
        raise ValueError(
            "Dataset visualization requires these logical text fields: "
            + ", ".join(missing_fields)
        )
    time_column = config.split.time_column
    if not time_column:
        raise ValueError("Dataset visualization requires split.time_column for the timeline.")
    method_column = str(logical_fields["method"])
    target_column = str(logical_fields["request_target"])
    body_column = str(logical_fields["body"])
    columns = list(
        dict.fromkeys(
            [
                *config.label_columns.values(),
                time_column,
                method_column,
                target_column,
                body_column,
            ]
        )
    )
    return columns, time_column, method_column, target_column, body_column


def _iter_table_chunks(
    config: DatasetConfig, columns: list[str], chunk_size: int
) -> Iterator[pd.DataFrame]:
    dataset_path = resolve_dataset_path(config)
    if config.format in {"csv", "csv.gz"}:
        yield from pd.read_csv(
            dataset_path,
            usecols=columns,
            dtype={
                column: str
                for column in config.text_columns.values()
                if column in columns
            },
            chunksize=chunk_size,
            low_memory=False,
        )
        return
    if config.format in {"parquet", "pq"}:
        frame = pd.read_parquet(dataset_path, columns=columns)
        for start in range(0, len(frame), chunk_size):
            yield frame.iloc[start : start + chunk_size]
        return
    raise ValueError(f"Unsupported dataset format: {config.format}")


def _update_weekly_counts(
    weekly_rows: Counter[str],
    weekly_labels: dict[str, np.ndarray],
    timestamps: pd.Series,
    targets: np.ndarray,
    time_format: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    parsed = pd.to_datetime(timestamps, format=time_format, errors="coerce", utc=True)
    valid = parsed.notna().to_numpy()
    if not valid.any():
        return None, None
    valid_times = parsed[valid]
    days = valid_times.dt.normalize()
    week_starts = days - pd.to_timedelta(days.dt.dayofweek, unit="D")
    week_keys = week_starts.dt.strftime("%Y-%m-%d")
    valid_targets = targets[valid].astype(np.int64, copy=False)
    for week, positions in pd.Series(np.arange(len(week_keys)), index=week_keys).groupby(
        level=0
    ):
        indices = positions.to_numpy(dtype=int)
        weekly_rows[str(week)] += len(indices)
        counts = valid_targets[indices].sum(axis=0)
        if week not in weekly_labels:
            weekly_labels[str(week)] = np.zeros(targets.shape[1], dtype=np.int64)
        weekly_labels[str(week)] += counts
    return valid_times.min(), valid_times.max()


def collect_dataset_profile(
    config: DatasetConfig,
    *,
    chunk_size: int = 50_000,
    scatter_sample_size: int = 10_000,
    seed: int = 14,
) -> DatasetProfile:
    """Scan the configured table in chunks and retain only aggregate statistics."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if scatter_sample_size <= 1:
        raise ValueError("scatter_sample_size must be greater than one")

    columns, time_column, method_column, target_column, body_column = _required_columns(
        config
    )
    label_keys = tuple(config.label_columns)
    label_columns = tuple(config.label_columns.values())
    n_labels = len(label_columns)
    label_counts = np.zeros(n_labels, dtype=np.int64)
    cooccurrence = np.zeros((n_labels, n_labels), dtype=np.int64)
    cardinality_counts: Counter[int] = Counter()
    method_counts: Counter[str] = Counter()
    label_method_counts = tuple(Counter() for _ in label_columns)
    weekly_rows: Counter[str] = Counter()
    weekly_labels: dict[str, np.ndarray] = {}
    reservoir = _StratifiedReservoir(scatter_sample_size, seed)
    row_count = 0
    normal_count = 0
    timestamp_start: pd.Timestamp | None = None
    timestamp_end: pd.Timestamp | None = None

    for chunk in _iter_table_chunks(config, columns, chunk_size):
        numeric_labels = chunk[list(label_columns)].apply(pd.to_numeric, errors="coerce")
        targets = (numeric_labels.fillna(0).to_numpy() > 0).astype(np.int64)
        cardinality = targets.sum(axis=1)
        is_attack = cardinality > 0
        row_count += len(chunk)
        normal_count += int((~is_attack).sum())
        label_counts += targets.sum(axis=0)
        cooccurrence += targets.T @ targets
        cardinality_counts.update(int(value) for value in cardinality)

        methods = (
            chunk[method_column].fillna("<missing>").astype(str).str.strip().str.upper()
        )
        methods = methods.mask(methods.eq(""), "<missing>")
        method_counts.update(methods.value_counts().to_dict())
        method_values = methods.to_numpy()
        for index in range(n_labels):
            selected = method_values[targets[:, index].astype(bool)]
            label_method_counts[index].update(Counter(selected))

        chunk_start, chunk_end = _update_weekly_counts(
            weekly_rows,
            weekly_labels,
            chunk[time_column],
            targets,
            config.split.time_format,
        )
        if chunk_start is not None:
            timestamp_start = (
                chunk_start
                if timestamp_start is None
                else min(timestamp_start, chunk_start)
            )
            timestamp_end = (
                chunk_end if timestamp_end is None else max(timestamp_end, chunk_end)
            )

        target_lengths = chunk[target_column].fillna("").astype(str).str.len().to_numpy()
        body_lengths = chunk[body_column].fillna("").astype(str).str.len().to_numpy()
        reservoir.add_many(target_lengths, body_lengths, is_attack)

    if row_count == 0:
        raise ValueError("The configured dataset is empty.")

    return DatasetProfile(
        label_keys=label_keys,
        label_columns=label_columns,
        display_labels=tuple(_display_label(column) for column in label_columns),
        row_count=row_count,
        normal_count=normal_count,
        label_counts=label_counts,
        cooccurrence=cooccurrence,
        cardinality_counts=dict(sorted(cardinality_counts.items())),
        method_counts=dict(method_counts),
        label_method_counts=tuple(dict(counter) for counter in label_method_counts),
        weekly_row_counts=dict(sorted(weekly_rows.items())),
        weekly_label_counts={
            week: tuple(int(value) for value in weekly_labels[week])
            for week in sorted(weekly_labels)
        },
        length_sample=reservoir.to_frame(),
        timestamp_start=timestamp_start.isoformat() if timestamp_start is not None else None,
        timestamp_end=timestamp_end.isoformat() if timestamp_end is not None else None,
    )


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Matplotlib is required for dataset visualization. Install project dependencies."
        ) from exc
    return plt


def _style_context(plt):
    return plt.rc_context(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "font.size": 10,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")


def _plot_overview(profile: DatasetProfile, path: Path) -> None:
    plt = _load_pyplot()
    with _style_context(plt):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        statuses = ["Normal", "Attack"]
        values = [profile.normal_count, profile.attack_count]
        colors = ["#6B7280", "#D55E00"]
        bars = axes[0].barh(statuses, values, color=colors)
        axes[0].set_title("Request classification overview")
        axes[0].set_xlabel("Number of requests")
        axes[0].grid(axis="x")
        axes[0].bar_label(
            bars,
            labels=[
                f"{value:,} ({value / profile.row_count:.1%})" for value in values
            ],
            padding=4,
        )
        axes[0].set_xlim(0, max(values) * 1.25)

        cardinalities = sorted(profile.cardinality_counts)
        counts = [profile.cardinality_counts[value] for value in cardinalities]
        bars = axes[1].bar(cardinalities, counts, color="#0072B2")
        axes[1].set_title("Labels assigned per request")
        axes[1].set_xlabel("Number of attack labels")
        axes[1].set_ylabel("Number of requests")
        axes[1].set_xticks(cardinalities)
        axes[1].grid(axis="y")
        axes[1].bar_label(bars, labels=[f"{value:,}" for value in counts], padding=3)
        axes[1].set_ylim(0, max(counts) * 1.18)
        fig.suptitle("SR-BH 2020 dataset overview", fontsize=16, fontweight="bold")
        fig.tight_layout()
        _save_figure(fig, path)
        plt.close(fig)


def _plot_prevalence(profile: DatasetProfile, path: Path) -> None:
    plt = _load_pyplot()
    order = np.argsort(profile.label_counts)
    counts = profile.label_counts[order]
    labels = [profile.display_labels[index] for index in order]
    plotted = np.maximum(counts, 0.5)
    with _style_context(plt):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        bars = ax.barh(labels, plotted, color="#D55E00")
        ax.set_xscale("log")
        ax.set_title("Attack label prevalence")
        ax.set_xlabel("Number of labeled requests (log scale; zero is shown at the axis floor)")
        ax.grid(axis="x", which="both")
        annotations = [
            f"{int(count):,} ({count / profile.row_count:.3%})" for count in counts
        ]
        ax.bar_label(bars, labels=annotations, padding=4, fontsize=9)
        ax.set_xlim(0.5, max(float(plotted.max()) * 8, 5))
        fig.tight_layout()
        _save_figure(fig, path)
        plt.close(fig)


def _jaccard_matrix(profile: DatasetProfile) -> np.ndarray:
    counts = profile.label_counts.astype(float)
    union = counts[:, None] + counts[None, :] - profile.cooccurrence
    result = np.divide(
        profile.cooccurrence,
        union,
        out=np.zeros_like(profile.cooccurrence, dtype=float),
        where=union > 0,
    )
    np.fill_diagonal(result, np.nan)
    return result


def _plot_cooccurrence(profile: DatasetProfile, path: Path) -> None:
    plt = _load_pyplot()
    matrix = _jaccard_matrix(profile)
    with _style_context(plt):
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_title("Attack label co-occurrence (Jaccard similarity)")
        positions = np.arange(len(profile.display_labels))
        ax.set_xticks(positions, profile.display_labels, rotation=55, ha="right")
        ax.set_yticks(positions, profile.display_labels)
        for row in range(len(positions)):
            for column in range(len(positions)):
                if row == column:
                    text = "—"
                    color = "#374151"
                else:
                    count = int(profile.cooccurrence[row, column])
                    text = f"{count:,}" if count else ""
                    value = matrix[row, column]
                    color = "white" if np.isfinite(value) and value < 0.35 else "#111827"
                ax.text(column, row, text, ha="center", va="center", fontsize=6, color=color)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Jaccard similarity; cells show shared request count")
        fig.tight_layout()
        _save_figure(fig, path)
        plt.close(fig)


def _method_matrix(profile: DatasetProfile) -> tuple[list[str], np.ndarray]:
    top_methods = [
        method
        for method, _ in sorted(
            profile.method_counts.items(), key=lambda item: (-item[1], item[0])
        )[:8]
    ]
    columns = [*top_methods, "OTHER"]
    matrix = np.zeros((len(profile.label_counts), len(columns)), dtype=float)
    for label_index, method_counts in enumerate(profile.label_method_counts):
        support = int(profile.label_counts[label_index])
        if support == 0:
            continue
        accounted = 0
        for method_index, method in enumerate(top_methods):
            count = int(method_counts.get(method, 0))
            accounted += count
            matrix[label_index, method_index] = count / support * 100
        matrix[label_index, -1] = max(0, support - accounted) / support * 100
    return columns, matrix


def _plot_methods(profile: DatasetProfile, path: Path) -> None:
    plt = _load_pyplot()
    methods, matrix = _method_matrix(profile)
    labels = [
        label + (" *" if count < MIN_RELIABLE_SUPPORT else "")
        for label, count in zip(profile.display_labels, profile.label_counts)
    ]
    with _style_context(plt):
        fig, ax = plt.subplots(figsize=(12, 8))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_title("HTTP method distribution within each attack label")
        ax.set_xticks(np.arange(len(methods)), methods, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if value >= 0.5:
                    color = "white" if value >= 55 else "#111827"
                    ax.text(
                        column,
                        row,
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Share of label requests (%)")
        fig.text(
            0.01,
            0.01,
            f"* Fewer than {MIN_RELIABLE_SUPPORT} labeled requests; interpret cautiously.",
            fontsize=9,
            color="#4B5563",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        _save_figure(fig, path)
        plt.close(fig)


def _plot_lengths(profile: DatasetProfile, path: Path) -> None:
    plt = _load_pyplot()
    with _style_context(plt):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, status in zip(axes, ("Normal", "Attack")):
            group = profile.length_sample[profile.length_sample["status"] == status]
            x = np.log1p(group["request_target_length"].to_numpy(dtype=float))
            y = np.log1p(group["body_length"].to_numpy(dtype=float))
            if len(group) >= 20:
                mappable = ax.hexbin(x, y, gridsize=38, mincnt=1, cmap="viridis")
                colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
                colorbar.set_label("Sample points per hexagon")
            elif len(group):
                ax.scatter(x, y, s=24, alpha=0.75, color="#0072B2")
            else:
                ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center")
            ax.set_title(f"{status} requests (n={len(group):,})")
            ax.set_xlabel("log1p(request-target length [characters])")
            ax.grid(alpha=0.35)
        axes[0].set_ylabel("log1p(request-body length [characters])")
        fig.suptitle("Request length density from a deterministic stratified sample", fontsize=14)
        fig.subplots_adjust(left=0.08, right=0.96, bottom=0.14, top=0.84, wspace=0.22)
        _save_figure(fig, path)
        plt.close(fig)


def _plot_timeline(profile: DatasetProfile, path: Path) -> None:
    plt = _load_pyplot()
    weeks = list(profile.weekly_row_counts)
    top_indices = np.argsort(profile.label_counts)[::-1][:5]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    with _style_context(plt):
        fig, ax = plt.subplots(figsize=(12, 6))
        if weeks:
            positions = np.arange(len(weeks))
            totals = np.asarray([profile.weekly_row_counts[week] for week in weeks])
            weekly = np.asarray([profile.weekly_label_counts[week] for week in weeks])
            for color, index in zip(colors, top_indices):
                rates = weekly[:, index] / totals * 100
                ax.plot(
                    positions,
                    rates,
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                    color=color,
                    label=profile.display_labels[int(index)],
                )
            ax.set_xticks(positions, weeks, rotation=30, ha="right")
            ax.legend(ncol=2, loc="upper left")
            ax.text(
                0.99,
                0.96,
                "Only weeks containing requests are shown; calendar gaps are omitted.",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#4B5563",
            )
        else:
            ax.text(0.5, 0.5, "No valid timestamps", transform=ax.transAxes, ha="center")
        ax.set_title("Weekly prevalence of the five most frequent attack labels")
        ax.set_xlabel("Observed week starting Monday (UTC)")
        ax.set_ylabel("Labeled requests / all requests that week (%)")
        ax.grid()
        fig.tight_layout()
        _save_figure(fig, path)
        plt.close(fig)


def _summary(profile: DatasetProfile, sample_size: int, seed: int) -> dict[str, object]:
    return {
        "row_count": profile.row_count,
        "normal_requests": profile.normal_count,
        "attack_requests": profile.attack_count,
        "normal_prevalence": profile.normal_count / profile.row_count,
        "attack_prevalence": profile.attack_count / profile.row_count,
        "timestamp_start": profile.timestamp_start,
        "timestamp_end": profile.timestamp_end,
        "observed_weeks": len(profile.weekly_row_counts),
        "label_cardinality": {
            str(cardinality): count
            for cardinality, count in profile.cardinality_counts.items()
        },
        "length_plot_sample": {
            "requested_size": sample_size,
            "actual_size": len(profile.length_sample),
            "seed": seed,
            "normal_samples": int((profile.length_sample["status"] == "Normal").sum()),
            "attack_samples": int((profile.length_sample["status"] == "Attack").sum()),
        },
    }


def _write_label_statistics(profile: DatasetProfile, path: Path) -> None:
    frame = pd.DataFrame(
        {
            "label_key": profile.label_keys,
            "dataset_column": profile.label_columns,
            "display_label": profile.display_labels,
            "count": profile.label_counts,
            "prevalence": profile.label_counts / profile.row_count,
            "support_warning": [
                "insufficient support" if count < MIN_RELIABLE_SUPPORT else ""
                for count in profile.label_counts
            ],
        }
    ).sort_values("count", ascending=False)
    frame.to_csv(path, index=False)


def _write_readme(profile: DatasetProfile, output_dir: Path, image_format: str) -> None:
    top_index = int(np.argmax(profile.label_counts))
    multi_label = sum(
        count for cardinality, count in profile.cardinality_counts.items() if cardinality > 1
    )
    rare_labels = [
        f"{label} ({int(count):,} request{'s' if count != 1 else ''})"
        for label, count in zip(profile.display_labels, profile.label_counts)
        if count < MIN_RELIABLE_SUPPORT
    ]
    rare_text = ", ".join(rare_labels) if rare_labels else "None"
    figures = "\n\n".join(
        f"### {index}. {title}\n\n"
        f"![{name.replace('_', ' ')}](figures/{name}.{image_format})"
        for index, (name, title) in enumerate(
            zip(FIGURE_NAMES, FIGURE_TITLES), start=1
        )
    )
    first_week = next(iter(profile.weekly_row_counts), None)
    content = f"""# SR-BH 2020 Security Attack Profile

This report is generated from aggregate statistics over **{profile.row_count:,} HTTP requests**.
It contains no raw URLs, IP addresses, cookies, headers, user agents, or request bodies.

## Key observations

- **{profile.attack_count:,} requests ({profile.attack_count / profile.row_count:.1%})** have at least one configured attack label; the remaining **{profile.normal_count:,}** are treated as normal.
- The most frequent label is **{profile.display_labels[top_index]}**, with **{int(profile.label_counts[top_index]):,} requests ({profile.label_counts[top_index] / profile.row_count:.1%})**.
- **{multi_label:,} requests** have more than one attack label, so co-occurrence must be considered during evaluation.
- Labels below {MIN_RELIABLE_SUPPORT} observations are marked as insufficient support. Current low-support labels: **{rare_text}**.
- Configured attack labels are strongly concentrated in the earliest observed collection week ({first_week or 'unavailable'}). Treat the timeline as evidence of collection or campaign shift, not a general population trend.

## Reproduce the report

From the repository root, install the project dependencies and run:

```powershell
py -m pip install -r requirements.txt
http-attack-visualize --dataset-config configs/dataset.srbh2020.yaml --output reports/dataset-profile
```

The CSV is processed in chunks. The request-length figure uses a deterministic, stratified sample so repeated runs with the same seed produce the same sample.

## Figures

{figures}

## Interpretation limits

- Normal means that every configured attack target is zero; it is not an independent model prediction.
- Label counts overlap because this is a multi-label dataset. Percentages across labels therefore do not sum to 100%.
- Very small classes cannot support reliable comparisons. In particular, any label with fewer than {MIN_RELIABLE_SUPPORT} observations should be treated as descriptive only.
- SR-BH labels originated from ModSecurity/OWASP CRS signals and were subsequently reviewed manually and semi-automatically. They are not fully rule-independent human ground truth.
- The length-density plot describes a deterministic sample. All other figures use aggregate statistics from the complete dataset.
- The timeline reports weekly prevalence rather than raw volume, reducing distortion from changes in total traffic.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def generate_dataset_profile(
    config: DatasetConfig,
    output: str | Path,
    *,
    chunk_size: int = 50_000,
    scatter_sample_size: int = 10_000,
    seed: int = 14,
    image_format: str = "png",
) -> Path:
    """Generate the aggregate tables, English report, and six static figures."""

    if image_format not in {"png", "pdf"}:
        raise ValueError("image_format must be either 'png' or 'pdf'")
    output_dir = Path(output)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    profile = collect_dataset_profile(
        config,
        chunk_size=chunk_size,
        scatter_sample_size=scatter_sample_size,
        seed=seed,
    )

    summary = _summary(profile, scatter_sample_size, seed)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_label_statistics(profile, output_dir / "label_statistics.csv")

    plotters: Iterable = (
        _plot_overview,
        _plot_prevalence,
        _plot_cooccurrence,
        _plot_methods,
        _plot_lengths,
        _plot_timeline,
    )
    for name, plotter in zip(FIGURE_NAMES, plotters):
        plotter(profile, figures_dir / f"{name}.{image_format}")
    _write_readme(profile, output_dir, image_format)
    return output_dir
