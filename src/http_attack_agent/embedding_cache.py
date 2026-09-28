"""Compute frozen-backbone embeddings once, then train the classifier head on them.

A frozen encoder in evaluation mode is a fixed function of its input, so every epoch recomputes
the same vectors. Caching them turns the expensive part of a model-zoo comparison into a single
forward pass per backbone, after which the head trains on a plain feature matrix. Re-running with
different head hyperparameters, thresholds or epoch counts then costs seconds instead of hours.

The cache is only valid while nothing that determines an embedding has changed. The manifest
records the backbone, the input length, the serialized fields and a content hash of those columns,
and a cache whose manifest does not match is recomputed rather than reused.

This module owns the optimization. Threshold calibration, metric reporting and epoch selection
stay in `training.py`, which consumes :func:`train_head_epochs` and therefore reports exactly the
same fields whichever path produced the predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .data import DatasetConfig
from .evaluation.predict import predict_dataset, resolve_device

CACHE_VERSION = 1
EMBEDDINGS_FILE = "embeddings.npy"
ROW_IDS_FILE = "row_ids.npy"
MANIFEST_FILE = "manifest.json"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc
    return torch


def content_digest(frame: Any, text_columns: Mapping[str, str]) -> str:
    """Hash the columns that are serialized into model input.

    Row identifiers are not enough on their own: with ``id_column: null`` they are positional
    and would match after the underlying table changed. Hashing is vectorized by pandas, so it
    costs a fraction of the forward pass it protects.
    """

    from pandas.util import hash_pandas_object

    columns = [text_columns[name] for name in sorted(text_columns)]
    hashed = hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def build_manifest(
    *,
    model_id: str,
    max_length: int,
    text_columns: Mapping[str, str],
    rows: int,
    digest: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "model_id": model_id,
        "max_length": int(max_length),
        "text_columns": dict(text_columns),
        "rows": int(rows),
        "content_digest": digest,
        "dtype": dtype,
    }


@dataclass(frozen=True)
class EmbeddingCache:
    """Embeddings for every row of one table, produced by one frozen backbone."""

    embeddings: np.ndarray
    row_ids: np.ndarray
    manifest: dict[str, Any]

    @property
    def hidden_size(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.embeddings)


def cache_directory(root: str | Path, model_name: str) -> Path:
    return Path(root) / model_name


def save_cache(
    directory: str | Path,
    embeddings: np.ndarray,
    row_ids: np.ndarray,
    manifest: Mapping[str, Any],
) -> Path:
    """Write the cache, leaving no partially written directory behind on failure."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / EMBEDDINGS_FILE, embeddings)
    np.save(path / ROW_IDS_FILE, np.asarray(row_ids, dtype=str))
    # The manifest is written last so an interrupted run cannot leave one that claims
    # a complete cache.
    (path / MANIFEST_FILE).write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    return path


def load_cache(
    directory: str | Path, expected: Mapping[str, Any]
) -> EmbeddingCache | None:
    """Return the cache when it still matches, otherwise None so the caller recomputes."""

    path = Path(directory)
    manifest_path = path / MANIFEST_FILE
    embeddings_path = path / EMBEDDINGS_FILE
    row_ids_path = path / ROW_IDS_FILE
    if not (manifest_path.is_file() and embeddings_path.is_file() and row_ids_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest != dict(expected):
        return None
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if len(embeddings) != manifest["rows"]:
        return None
    return EmbeddingCache(
        embeddings=embeddings,
        row_ids=np.load(row_ids_path, allow_pickle=False),
        manifest=manifest,
    )


def compute_cache(
    model: Any,
    tokenizer: Any,
    frame: Any,
    config: DatasetConfig,
    *,
    model_id: str,
    max_length: int,
    batch_size: int = 64,
    device: Any | None = None,
    dtype: str = "float32",
    progress: bool = True,
    description: str = "Encoding requests",
) -> EmbeddingCache:
    """Run the frozen backbone over every row exactly once."""

    if dtype not in {"float32", "float16"}:
        raise ValueError("dtype must be either 'float32' or 'float16'")
    prediction = predict_dataset(
        model,
        tokenizer,
        frame,
        config,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        return_embeddings=True,
        progress=progress,
        description=description,
    )
    if prediction.embeddings is None:  # pragma: no cover - guarded by return_embeddings
        raise RuntimeError("The model did not return embeddings")
    manifest = build_manifest(
        model_id=model_id,
        max_length=max_length,
        text_columns=config.text_columns,
        rows=len(frame),
        digest=content_digest(frame, config.text_columns),
        dtype=dtype,
    )
    return EmbeddingCache(
        embeddings=prediction.embeddings.astype(dtype),
        row_ids=prediction.row_ids,
        manifest=manifest,
    )


def load_or_compute_cache(
    directory: str | Path,
    model: Any,
    tokenizer: Any,
    frame: Any,
    config: DatasetConfig,
    *,
    model_id: str,
    max_length: int,
    batch_size: int = 64,
    device: Any | None = None,
    dtype: str = "float32",
    progress: bool = True,
    refresh: bool = False,
) -> tuple[EmbeddingCache, bool]:
    """Reuse a matching cache, otherwise encode and store one.

    Returns the cache and whether it was reused.
    """

    expected = build_manifest(
        model_id=model_id,
        max_length=max_length,
        text_columns=config.text_columns,
        rows=len(frame),
        digest=content_digest(frame, config.text_columns),
        dtype=dtype,
    )
    if not refresh:
        cached = load_cache(directory, expected)
        if cached is not None:
            return cached, True
    cache = compute_cache(
        model,
        tokenizer,
        frame,
        config,
        model_id=model_id,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        progress=progress,
    )
    save_cache(directory, cache.embeddings, cache.row_ids, cache.manifest)
    return cache, False


def _batched(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _embedding_batch(embeddings: np.ndarray, rows: np.ndarray, device: Any) -> Any:
    """Materialize one batch out of a possibly memory-mapped embedding matrix."""

    torch = _require_torch()
    block = np.asarray(embeddings[rows], dtype=np.float32)
    return torch.as_tensor(block, dtype=torch.float32, device=device)


def head_predict(
    head: Any,
    embeddings: np.ndarray,
    indices: np.ndarray,
    *,
    device: Any | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Probabilities for the selected rows, in the order the indices were given."""

    torch = _require_torch()
    device = device if device is not None else resolve_device()
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        raise ValueError("head_predict needs at least one row")
    head.eval()
    head.to(device)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batched(indices, batch_size):
            block = _embedding_batch(embeddings, batch, device)
            chunks.append(torch.sigmoid(head(block)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def train_head_epochs(
    head: Any,
    embeddings: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pos_weight: np.ndarray,
    gradient_clip: float,
    device: Any | None = None,
    seed: int = 14,
    progress: Any | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Train the head on cached embeddings, yielding validation probabilities each epoch.

    Yields ``(epoch, mean_train_loss, validation_probabilities)``. Thresholds, metrics and
    epoch selection stay with the caller, so a cached run reports exactly the fields a
    conventional run reports.
    """

    torch = _require_torch()
    device = device if device is not None else resolve_device()
    head.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    train_indices = np.asarray(train_indices, dtype=int)
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        head.train()
        order = train_indices.copy()
        rng.shuffle(order)
        losses: list[float] = []
        batches = _batched(order, batch_size)
        if progress is not None:
            batches = progress(batches, epoch, int(np.ceil(len(order) / batch_size)))
        for batch in batches:
            block = _embedding_batch(embeddings, batch, device)
            labels = torch.as_tensor(targets[batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(block), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = head_predict(
            head, embeddings, validation_indices, device=device, batch_size=max(batch_size, 4096)
        )
        yield epoch, float(np.mean(losses)) if losses else 0.0, validation


def estimated_cache_bytes(rows: int, hidden_size: int, dtype: str = "float32") -> int:
    """Disk a cache will occupy. A model-zoo comparison keeps one of these per backbone."""

    width = 2 if dtype == "float16" else 4
    return rows * hidden_size * width


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"  # pragma: no cover - unreachable


__all__ = [
    "EmbeddingCache",
    "build_manifest",
    "cache_directory",
    "compute_cache",
    "content_digest",
    "estimated_cache_bytes",
    "format_bytes",
    "head_predict",
    "load_cache",
    "load_or_compute_cache",
    "save_cache",
    "train_head_epochs",
]
