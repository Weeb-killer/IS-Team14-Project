from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from http_attack_agent.data import DatasetConfig, SplitConfig
from http_attack_agent.visualization.profile import (
    FIGURE_NAMES,
    collect_dataset_profile,
    generate_dataset_profile,
)


def _write_dataset(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "01/Jan/2024:10:00:00 +0000",
                "02/Jan/2024:10:00:00 +0000",
                "08/Jan/2024:10:00:00 +0000",
                "09/Jan/2024:10:00:00 +0000",
                "15/Jan/2024:10:00:00 +0000",
                "16/Jan/2024:10:00:00 +0000",
            ],
            "method": ["GET", "POST", "GET", "TRACE", "POST", "GET"],
            "target": [
                "/TOP_SECRET_URL",
                "/login",
                "/search?q=x",
                "/trace",
                "/submit",
                "/home",
            ],
            "body": ["", "TOP_SECRET_BODY", "", "", "name=test", ""],
            "label_a": [0, 1, 1, 1, 0, 0],
            "label_b": [0, 0, 1, 0, 1, 0],
        }
    )
    frame.to_csv(path, index=False)


def _config(path: Path) -> DatasetConfig:
    return DatasetConfig(
        path=path,
        format="csv",
        id_column=None,
        text_columns={"method": "method", "request_target": "target", "body": "body"},
        label_columns={"attack_a": "1 - Attack A", "attack_b": "2 - Attack B"},
        waf_concept_columns={},
        split=SplitConfig(
            time_column="timestamp",
            time_format="%d/%b/%Y:%H:%M:%S %z",
        ),
    )


def _configured_csv(path: Path) -> DatasetConfig:
    config = _config(path)
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"label_a": "1 - Attack A", "label_b": "2 - Attack B"})
    frame.to_csv(path, index=False)
    return config


def test_chunked_aggregation_matches_single_chunk(tmp_path):
    csv_path = tmp_path / "requests.csv"
    _write_dataset(csv_path)
    config = _configured_csv(csv_path)

    chunked = collect_dataset_profile(
        config, chunk_size=2, scatter_sample_size=4, seed=7
    )
    single = collect_dataset_profile(
        config, chunk_size=100, scatter_sample_size=4, seed=7
    )

    assert chunked.row_count == single.row_count == 6
    assert chunked.normal_count == single.normal_count == 2
    assert chunked.cardinality_counts == single.cardinality_counts == {0: 2, 1: 3, 2: 1}
    np.testing.assert_array_equal(chunked.label_counts, [3, 2])
    np.testing.assert_array_equal(chunked.label_counts, single.label_counts)
    np.testing.assert_array_equal(chunked.cooccurrence, [[3, 1], [1, 2]])
    np.testing.assert_array_equal(chunked.cooccurrence, single.cooccurrence)
    assert chunked.method_counts == single.method_counts
    assert chunked.weekly_row_counts == single.weekly_row_counts
    assert chunked.weekly_label_counts == single.weekly_label_counts


def test_stratified_sample_is_deterministic_across_chunk_sizes(tmp_path):
    csv_path = tmp_path / "requests.csv"
    _write_dataset(csv_path)
    config = _configured_csv(csv_path)

    first = collect_dataset_profile(config, chunk_size=2, scatter_sample_size=4, seed=14)
    second = collect_dataset_profile(config, chunk_size=5, scatter_sample_size=4, seed=14)

    pd.testing.assert_frame_equal(first.length_sample, second.length_sample)
    assert first.length_sample["status"].value_counts().to_dict() == {
        "Normal": 2,
        "Attack": 2,
    }


def test_report_generation_writes_six_figures_and_no_raw_payloads(tmp_path):
    csv_path = tmp_path / "requests.csv"
    _write_dataset(csv_path)
    config = _configured_csv(csv_path)
    output = tmp_path / "report"

    result = generate_dataset_profile(
        config,
        output,
        chunk_size=2,
        scatter_sample_size=4,
        seed=14,
    )

    assert result == output
    expected = [output / "figures" / f"{name}.png" for name in FIGURE_NAMES]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    assert (output / "README.md").is_file()
    assert (output / "label_statistics.csv").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 6
    assert summary["attack_requests"] == 4

    public_bytes = b"\n".join(
        path.read_bytes() for path in output.rglob("*") if path.is_file()
    )
    assert b"TOP_SECRET_URL" not in public_bytes
    assert b"TOP_SECRET_BODY" not in public_bytes
