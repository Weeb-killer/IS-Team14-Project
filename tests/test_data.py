import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest

from http_attack_agent.data import (
    DatasetConfig,
    DatasetSourceConfig,
    SplitConfig,
    build_targets,
    deterministic_split_indices,
    load_table,
    resolve_dataset_path,
    serialize_http_request,
)


def test_request_serialization_keeps_field_identity():
    row = {"verb": "POST", "uri": "/login", "payload": "' OR 1=1 --"}
    text = serialize_http_request(
        row, {"method": "verb", "path": "uri", "body": "payload"}
    )
    assert "[METHOD]\nPOST" in text
    assert "[PATH]\n/login" in text
    assert "[BODY]\n' OR 1=1 --" in text


def test_targets_are_multilabel_binary():
    frame = pd.DataFrame({"a": [0, 2], "b": [1, 0]})
    result = build_targets(frame, {"x": "a", "y": "b"})
    np.testing.assert_array_equal(result, [[0, 1], [1, 0]])


def test_split_is_deterministic(tmp_path):
    frame = pd.DataFrame({"id": range(100)})
    config = DatasetConfig(
        path=tmp_path / "unused.csv",
        format="csv",
        id_column="id",
        text_columns={},
        label_columns={},
        waf_concept_columns={},
        split=SplitConfig(random_seed=7),
    )
    first = deterministic_split_indices(frame, config)
    second = deterministic_split_indices(frame, config)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_csv_loader_preserves_mixed_http_field_values(tmp_path):
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text("request_cookie\n00123\nabc\n", encoding="utf-8")
    config = DatasetConfig(
        path=csv_path,
        format="csv",
        id_column=None,
        text_columns={"cookie": "request_cookie"},
        label_columns={},
        waf_concept_columns={},
        split=SplitConfig(),
    )

    frame = load_table(config)

    assert frame["request_cookie"].tolist() == ["00123", "abc"]


def test_temporal_split_sorts_mixed_offsets_by_instant(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": [
                "18/Jul/2020:12:00:00 +0000",
                "18/Jul/2020:13:00:00 +0200",
                "18/Jul/2020:12:00:00 +0200",
                "18/Jul/2020:13:00:00 +0000",
            ]
        }
    )
    config = DatasetConfig(
        path=tmp_path / "unused.csv",
        format="csv",
        id_column=None,
        text_columns={},
        label_columns={},
        waf_concept_columns={},
        split=SplitConfig(
            test_size=0.25,
            validation_size=0.25,
            time_column="timestamp",
            time_format="%d/%b/%Y:%H:%M:%S %z",
        ),
    )

    train, validation, test = deterministic_split_indices(frame, config)

    np.testing.assert_array_equal(train, [2, 1])
    np.testing.assert_array_equal(validation, [0])
    np.testing.assert_array_equal(test, [3])


def test_multilabel_stratification_preserves_normal_and_attack_proportions(tmp_path):
    frame = pd.DataFrame(
        {
            "sql": [0] * 120 + [1] * 50 + [0] * 20 + [1] * 10,
            "xss": [0] * 120 + [0] * 50 + [1] * 20 + [1] * 10,
            "timestamp": range(200),
        }
    )
    config = DatasetConfig(
        path=tmp_path / "unused.csv",
        format="csv",
        id_column=None,
        text_columns={},
        label_columns={"sql": "sql", "xss": "xss"},
        waf_concept_columns={},
        split=SplitConfig(
            test_size=0.15,
            validation_size=0.15,
            random_seed=14,
            strategy="multilabel_stratified",
            time_column="timestamp",
        ),
    )

    first = deterministic_split_indices(frame, config)
    second = deterministic_split_indices(frame, config)

    assert [len(rows) for rows in first] == [140, 30, 30]
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(len(frame)))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    for column in ("sql", "xss"):
        total_positives = int(frame[column].sum())
        for rows, share in zip(first, (0.70, 0.15, 0.15)):
            observed = int(frame.iloc[rows][column].sum())
            assert observed > 0
            assert abs(observed - total_positives * share) <= 1
    normal = (frame[["sql", "xss"]].sum(axis=1) == 0).to_numpy()
    for rows, share in zip(first, (0.70, 0.15, 0.15)):
        assert abs(int(normal[rows].sum()) - int(normal.sum()) * share) <= 1


def test_multilabel_stratification_keeps_singleton_in_training(tmp_path):
    frame = pd.DataFrame(
        {
            "common": [0] * 30 + [1] * 15 + [0],
            "rare": [0] * 45 + [1],
        }
    )
    config = DatasetConfig(
        path=tmp_path / "unused.csv",
        format="csv",
        id_column=None,
        text_columns={},
        label_columns={"common": "common", "rare": "rare"},
        waf_concept_columns={},
        split=SplitConfig(
            test_size=0.15, validation_size=0.15, strategy="multilabel_stratified"
        ),
    )

    train, validation, test = deterministic_split_indices(frame, config)

    assert 45 in train
    assert 45 not in validation
    assert 45 not in test
    assert all(frame.iloc[rows]["common"].sum() > 0 for rows in (train, validation, test))


def _source_config(tmp_path, path, source):
    return DatasetConfig(
        path=path,
        format="csv",
        id_column=None,
        text_columns={},
        label_columns={},
        waf_concept_columns={},
        split=SplitConfig(),
        project_root=tmp_path,
        source=source,
    )


def test_yaml_mode_selects_the_matching_source_block(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "dataset.yaml"
    config_file.write_text(
        """
source:
  mode: remote
  local:
    path: data/local.csv
  remote:
    url: "https://example.test/dataset.zip"
    path: data/downloads/remote.csv
    archive: zip
    member: dataset.csv
format: csv
id_column: null
text_columns: {}
label_columns: {}
waf_concept_columns: {}
split: {}
""".strip(),
        encoding="utf-8",
    )

    config = DatasetConfig.from_yaml(config_file)

    assert config.source.mode == "remote"
    assert config.source.url == "https://example.test/dataset.zip"
    assert config.source.member == "dataset.csv"
    assert config.path == (tmp_path / "data" / "downloads" / "remote.csv").resolve()


def test_local_source_falls_back_to_unique_glob(tmp_path):
    local_file = tmp_path / "data" / "original-srbh-name.csv"
    local_file.parent.mkdir()
    local_file.write_text("column\nvalue\n", encoding="utf-8")
    config = _source_config(
        tmp_path,
        tmp_path / "data" / "data_capec_multilabel.csv",
        DatasetSourceConfig(mode="local", glob="data/**/*.csv"),
    )

    assert resolve_dataset_path(config) == local_file.resolve()


def test_local_source_rejects_ambiguous_glob(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "one.csv").write_text("x\n1\n", encoding="utf-8")
    (data_dir / "two.csv").write_text("x\n2\n", encoding="utf-8")
    config = _source_config(
        tmp_path,
        data_dir / "missing.csv",
        DatasetSourceConfig(mode="local", glob="data/**/*.csv"),
    )

    with pytest.raises(ValueError, match="more than one file"):
        resolve_dataset_path(config)


def test_remote_source_downloads_and_reuses_cached_file(tmp_path):
    source_file = tmp_path / "source.csv"
    payload = b"column\nvalue\n"
    source_file.write_bytes(payload)
    target = tmp_path / "data" / "cached.csv"
    config = _source_config(
        tmp_path,
        target,
        DatasetSourceConfig(
            mode="remote",
            url=source_file.as_uri(),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )

    assert resolve_dataset_path(config) == target
    assert target.read_bytes() == payload
    source_file.unlink()
    assert resolve_dataset_path(config) == target


def test_remote_source_extracts_named_csv_from_zip(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    payload = b"column\nvalue\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("release/data_capec_multilabel.csv", payload)
    target = tmp_path / "data" / "data_capec_multilabel.csv"
    config = _source_config(
        tmp_path,
        target,
        DatasetSourceConfig(
            mode="remote",
            url=archive_path.as_uri(),
            archive="zip",
            member="data_capec_multilabel.csv",
        ),
    )

    assert resolve_dataset_path(config) == target
    assert target.read_bytes() == payload
