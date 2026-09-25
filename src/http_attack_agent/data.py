from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 14
    time_column: str | None = None
    time_format: str | None = None
    group_column: str | None = None
    strategy: str | None = None


@dataclass(frozen=True)
class DatasetSourceConfig:
    """Settings for locating a local dataset or acquiring a remote one."""

    mode: str = "local"
    glob: str | None = None
    url: str | None = None
    sha256: str | None = None
    archive: str = "none"
    member: str | None = None
    timeout_seconds: int = 300


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    format: str
    id_column: str | None
    text_columns: dict[str, str]
    label_columns: dict[str, str]
    waf_concept_columns: dict[str, str]
    split: SplitConfig
    project_root: Path = Path(".")
    source: DatasetSourceConfig = field(default_factory=DatasetSourceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyYAML is required to read dataset configuration") from exc
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parent.parent.resolve()

        source_raw = raw.get("source")
        if source_raw is None:
            # Backwards compatibility with the original top-level `path` format.
            source = DatasetSourceConfig(mode="local")
            raw_path = raw.get("path")
            if not raw_path:
                raise ValueError("Dataset configuration must define 'path' or 'source'.")
        else:
            if not isinstance(source_raw, dict):
                raise ValueError("'source' must be a YAML mapping.")
            mode = str(source_raw.get("mode", "local")).lower()
            if mode not in {"local", "remote"}:
                raise ValueError("source.mode must be either 'local' or 'remote'.")
            selected = source_raw.get(mode, {})
            if not isinstance(selected, dict):
                raise ValueError(f"source.{mode} must be a YAML mapping.")
            raw_path = selected.get("path")
            if not raw_path:
                raise ValueError(f"source.{mode}.path is required.")

            archive = str(selected.get("archive", "none")).lower()
            if archive not in {"none", "zip"}:
                raise ValueError("source.remote.archive must be 'none' or 'zip'.")
            source = DatasetSourceConfig(
                mode=mode,
                glob=selected.get("glob"),
                url=selected.get("url"),
                sha256=selected.get("sha256"),
                archive=archive,
                member=selected.get("member"),
                timeout_seconds=int(selected.get("timeout_seconds", 300)),
            )

        data_path = Path(raw_path)
        if not data_path.is_absolute():
            data_path = (project_root / data_path).resolve()
        return cls(
            path=data_path,
            format=str(raw.get("format", data_path.suffix.lstrip("."))).lower(),
            id_column=raw.get("id_column"),
            text_columns=dict(raw["text_columns"]),
            label_columns=dict(raw["label_columns"]),
            waf_concept_columns=dict(raw.get("waf_concept_columns", {})),
            split=SplitConfig(**raw.get("split", {})),
            project_root=project_root,
            source=source,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected.lower()}, got {actual.lower()}."
        )


def _resolve_local_path(config: DatasetConfig) -> Path:
    if config.path.is_file():
        return config.path

    if config.source.glob:
        matches = sorted(
            candidate.resolve()
            for candidate in config.project_root.glob(config.source.glob)
            if candidate.is_file()
        )
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Local dataset was not found at {config.path}, and glob "
                f"'{config.source.glob}' matched no files under {config.project_root}."
            )
        listed = "\n  - ".join(str(match) for match in matches)
        raise ValueError(
            "The local dataset glob matched more than one file. Set "
            f"source.local.path to the intended CSV:\n  - {listed}"
        )

    raise FileNotFoundError(
        f"Local dataset was not found at {config.path}. Update source.local.path "
        "or configure source.local.glob."
    )


def _zip_member_name(archive: zipfile.ZipFile, requested: str | None) -> str:
    files = [name for name in archive.namelist() if not name.endswith("/")]
    if requested:
        if requested in files:
            return requested
        basename_matches = [name for name in files if Path(name).name == requested]
        if len(basename_matches) == 1:
            return basename_matches[0]
        if not basename_matches:
            raise FileNotFoundError(
                f"Archive member '{requested}' was not found in the downloaded ZIP."
            )
        raise ValueError(
            f"Archive member name '{requested}' is ambiguous: {basename_matches}."
        )

    csv_files = [name for name in files if name.lower().endswith(".csv")]
    if len(csv_files) == 1:
        return csv_files[0]
    raise ValueError(
        "The downloaded ZIP must contain exactly one CSV when source.remote.member "
        "is not configured."
    )


def _download_remote(config: DatasetConfig) -> Path:
    target = config.path
    if target.is_file():
        _verify_checksum(target, config.source.sha256)
        return target
    if not config.source.url:
        raise ValueError("source.remote.url is required when source.mode is 'remote'.")

    target.parent.mkdir(parents=True, exist_ok=True)
    download_path: Path | None = None
    extracted_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="srbh2020-download-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as download_handle:
            download_path = Path(download_handle.name)
            request = urllib.request.Request(
                config.source.url,
                headers={"User-Agent": "http-attack-agent/0.1"},
            )
            with urllib.request.urlopen(
                request, timeout=config.source.timeout_seconds
            ) as response:
                shutil.copyfileobj(response, download_handle)

        candidate = download_path
        if config.source.archive == "zip":
            with zipfile.ZipFile(download_path) as archive:
                member = _zip_member_name(archive, config.source.member)
                with archive.open(member) as source_handle, tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix="srbh2020-extracted-",
                    suffix=target.suffix,
                    dir=target.parent,
                    delete=False,
                ) as extracted_handle:
                    extracted_path = Path(extracted_handle.name)
                    shutil.copyfileobj(source_handle, extracted_handle)
            candidate = extracted_path

        _verify_checksum(candidate, config.source.sha256)
        candidate.replace(target)
        if candidate == download_path:
            download_path = None
        else:
            extracted_path = None
        return target
    finally:
        for temporary_path in (download_path, extracted_path):
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def resolve_dataset_path(config: DatasetConfig) -> Path:
    """Return a usable dataset path, downloading it when configured to do so."""

    if config.source.mode == "local":
        return _resolve_local_path(config)
    if config.source.mode == "remote":
        return _download_remote(config)
    raise ValueError(f"Unsupported dataset source mode: {config.source.mode}")


def load_table(config: DatasetConfig) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(config)
    if config.format in {"parquet", "pq"}:
        frame = pd.read_parquet(dataset_path)
    elif config.format in {"csv", "csv.gz"}:
        # HTTP fields are text even when a CSV chunk happens to contain only digits.
        frame = pd.read_csv(
            dataset_path,
            dtype={column: str for column in config.text_columns.values()},
        )
    else:
        raise ValueError(f"Unsupported dataset format: {config.format}")
    validate_columns(frame, config)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve the configured local dataset or download the remote dataset."
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.srbh2020.yaml",
        help="Path to the dataset YAML configuration.",
    )
    args = parser.parse_args()
    print(resolve_dataset_path(DatasetConfig.from_yaml(args.dataset_config)))


def validate_columns(frame: pd.DataFrame, config: DatasetConfig) -> None:
    required = {*config.text_columns.values(), *config.label_columns.values()}
    if config.id_column:
        required.add(config.id_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Dataset configuration references missing columns: " + ", ".join(missing)
        )


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\x00", "<NUL>")


def serialize_http_request(row: Mapping[str, Any], text_columns: Mapping[str, str]) -> str:
    """Serialize heterogeneous HTTP fields while retaining their field identity."""

    parts: list[str] = []
    for logical_name, physical_column in text_columns.items():
        marker = logical_name.upper().replace(" ", "_")
        parts.append(f"[{marker}]\n{_safe_text(row.get(physical_column))}")
    return "\n".join(parts)


def build_texts(frame: pd.DataFrame, config: DatasetConfig) -> list[str]:
    return [serialize_http_request(row, config.text_columns) for row in frame.to_dict("records")]


def build_targets(frame: pd.DataFrame, columns: Mapping[str, str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float32)
    values = frame[list(columns.values())].fillna(0).astype(np.float32).to_numpy()
    return (values > 0).astype(np.float32)


def _multilabel_stratified_indices(
    frame: pd.DataFrame, config: DatasetConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split complete label combinations, including the all-zero normal class."""

    if not config.label_columns:
        raise ValueError("Multilabel stratification requires at least one label column.")
    split = config.split
    n_rows = len(frame)
    n_test = round(n_rows * split.test_size)
    n_valid = round(n_rows * split.validation_size)
    target_sizes = np.array([n_rows - n_test - n_valid, n_valid, n_test], dtype=int)
    if (
        not 0 < split.test_size < 1
        or not 0 < split.validation_size < 1
        or split.test_size + split.validation_size >= 1
        or np.any(target_sizes == 0)
    ):
        raise ValueError(
            "Multilabel stratification needs nonempty train, validation, and test "
            "splits with test_size + validation_size < 1."
        )

    # Packed label combinations keep this practical even for a large HTTP table.
    labels = build_targets(frame, config.label_columns).astype(np.uint8)
    patterns = np.packbits(labels, axis=1, bitorder="little")
    _, group_ids, group_sizes = np.unique(
        patterns, axis=0, return_inverse=True, return_counts=True
    )
    grouped_rows = np.argsort(group_ids, kind="stable")
    group_offsets = np.concatenate(([0], np.cumsum(group_sizes)))
    ratios = target_sizes / n_rows
    remaining = target_sizes.copy()
    selections: list[list[np.ndarray]] = [[], [], []]
    rng = np.random.default_rng(split.random_seed)

    # Allocate rare combinations first; the largest group absorbs rounding to
    # preserve the exact requested split sizes.
    for group in np.argsort(group_sizes, kind="stable"):
        size = int(group_sizes[group])
        desired = size * ratios
        minimum = np.ones(3, dtype=int) if size >= 3 else np.zeros(3, dtype=int)
        allocation = np.minimum(np.maximum(np.floor(desired).astype(int), minimum), remaining)
        while allocation.sum() > size:
            removable = allocation > np.minimum(minimum, remaining)
            excess = np.where(removable, allocation - desired, -np.inf)
            allocation[int(np.argmax(excess))] -= 1
        while allocation.sum() < size:
            available = allocation < remaining
            shortfall = np.where(available, desired - allocation, -np.inf)
            # Prefer the larger target split when fractional shortfalls tie.
            choice = int(np.argmax(shortfall + ratios * 1e-8))
            allocation[choice] += 1

        rows = grouped_rows[group_offsets[group] : group_offsets[group + 1]].copy()
        rng.shuffle(rows)
        offset = 0
        for split_number, count in enumerate(allocation):
            if count:
                selections[split_number].append(rows[offset : offset + count])
            offset += int(count)
        remaining -= allocation

    result = tuple(
        np.concatenate(parts) if parts else np.empty(0, dtype=int)
        for parts in selections
    )
    for indices in result:
        rng.shuffle(indices)
    return result


def deterministic_split_indices(
    frame: pd.DataFrame, config: DatasetConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a shared split for every model in the zoo."""

    split = config.split
    strategy = split.strategy or (
        "time" if split.time_column else "group" if split.group_column else "random"
    )
    if strategy == "multilabel_stratified":
        return _multilabel_stratified_indices(frame, config)
    if strategy not in {"time", "group", "random"}:
        raise ValueError(f"Unknown split strategy: {strategy}")
    indices = np.arange(len(frame))
    if strategy == "time":
        if not split.time_column:
            raise ValueError("split.time_column is required for the time strategy.")
        if split.time_column not in frame:
            raise ValueError(f"Missing time_column: {split.time_column}")
        parsed_time = pd.to_datetime(
            frame[split.time_column], format=split.time_format, errors="raise", utc=True
        )
        indices = np.argsort(parsed_time.to_numpy())
    elif strategy == "group":
        if not split.group_column:
            raise ValueError("split.group_column is required for the group strategy.")
        if split.group_column not in frame:
            raise ValueError(f"Missing group_column: {split.group_column}")
        groups = frame[split.group_column].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(split.random_seed)
        rng.shuffle(unique_groups)
        rank = {group: i for i, group in enumerate(unique_groups)}
        indices = np.argsort(np.asarray([rank[group] for group in groups]))
    else:
        rng = np.random.default_rng(split.random_seed)
        rng.shuffle(indices)

    n_total = len(indices)
    n_test = round(n_total * split.test_size)
    n_valid = round(n_total * split.validation_size)
    test = indices[-n_test:] if n_test else np.empty(0, dtype=int)
    valid_end = n_total - n_test if n_test else n_total
    valid = indices[valid_end - n_valid : valid_end] if n_valid else np.empty(0, dtype=int)
    train = indices[: valid_end - n_valid]
    return train, valid, test


class FrameDataset:
    """A small torch-compatible dataset without importing torch at module import time."""

    def __init__(
        self,
        frame: pd.DataFrame,
        text_columns: Mapping[str, str],
        targets: np.ndarray,
        row_ids: Iterable[Any],
        indices: np.ndarray,
        concepts: np.ndarray | None = None,
    ) -> None:
        # Keep zero-copy column arrays and render only the current minibatch. Materializing
        # 907k serialized requests up front can otherwise require several extra GB of RAM.
        self.fields = {
            logical_name: frame[physical_column].to_numpy(copy=False)
            for logical_name, physical_column in text_columns.items()
        }
        self.targets = targets
        self.row_ids = np.asarray(list(row_ids), dtype=str)
        self.indices = np.asarray(indices)
        self.concepts = concepts

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        text = "\n".join(
            f"[{logical_name.upper().replace(' ', '_')}]\n{_safe_text(values[index])}"
            for logical_name, values in self.fields.items()
        )
        result: dict[str, Any] = {
            "text": text,
            "labels": self.targets[index],
            "row_id": self.row_ids[index],
        }
        if self.concepts is not None:
            result["concepts"] = self.concepts[index]
        return result
