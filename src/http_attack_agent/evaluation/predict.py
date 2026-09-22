"""Batched inference over a saved model, reusable outside the training loop.

`training.py` already contains this logic, but as private helpers, so the only way to obtain
predictions today is to run a full training job. Exposing it here lets a saved checkpoint be
scored again — on a different split, with different thresholds, or on data it has never seen —
without retraining, and keeps the forward pass in one place once `training.py` adopts it.

Embeddings can be collected during the same pass. `training.py` currently forwards the test split
twice, once for metrics and once to export embeddings for the interpretability report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ..data import DatasetConfig, FrameDataset, build_targets


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc
    return torch


def resolve_device(prefer_cpu: bool = False) -> Any:
    """Return CUDA when it is available and not declined, otherwise CPU."""

    torch = _require_torch()
    if prefer_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


@dataclass(frozen=True)
class Prediction:
    """Model output for one set of rows, in the order the rows were supplied."""

    probabilities: np.ndarray
    row_ids: np.ndarray
    targets: np.ndarray | None = None
    logits: np.ndarray | None = None
    embeddings: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.probabilities)

    @property
    def has_targets(self) -> bool:
        return self.targets is not None and self.targets.shape[1] > 0


class _RequestCollator:
    """Tokenize one minibatch of serialized requests.

    Equivalent to ``training.RequestCollator``. The two should be folded into a single
    implementation in ``data.py`` once ``training.py`` adopts this module.
    """

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        torch = _require_torch()
        encoded = self.tokenizer(
            [row["text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "row_ids": [row["row_id"] for row in rows],
        }
        if rows[0]["labels"].size:
            batch["labels"] = torch.as_tensor(
                np.stack([row["labels"] for row in rows]), dtype=torch.float32
            )
        return batch


def predict_batches(
    model: Any,
    loader: Any,
    *,
    device: Any | None = None,
    return_embeddings: bool = False,
    progress: bool = True,
    description: str = "Prediction",
) -> Prediction:
    """Run the model over every batch and collect probabilities in loader order.

    The loader must not shuffle; row order is what lets predictions be joined back to requests.
    """

    torch = _require_torch()
    device = device if device is not None else resolve_device()
    model.eval()
    model.to(device)

    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    row_ids: list[str] = []

    batches = (
        tqdm(loader, total=len(loader), desc=description, unit="batch", dynamic_ncols=True)
        if progress
        else loader
    )
    with torch.no_grad():
        for batch in batches:
            output = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            logits.append(output.logits.cpu().numpy())
            probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
            row_ids.extend(batch["row_ids"])
            if "labels" in batch:
                targets.append(batch["labels"].numpy())
            if return_embeddings:
                embeddings.append(output.embeddings.cpu().numpy())

    if not probabilities:
        raise ValueError("The loader yielded no batches, so there is nothing to predict")

    return Prediction(
        probabilities=np.concatenate(probabilities).astype(np.float32),
        row_ids=np.asarray(row_ids, dtype=str),
        targets=np.concatenate(targets).astype(np.float32) if targets else None,
        logits=np.concatenate(logits).astype(np.float32),
        embeddings=(
            np.concatenate(embeddings).astype(np.float32) if embeddings else None
        ),
    )


def build_loader(
    frame: Any,
    tokenizer: Any,
    *,
    text_columns: Mapping[str, str],
    targets: np.ndarray,
    row_ids: Sequence[Any],
    indices: np.ndarray,
    max_length: int,
    batch_size: int,
) -> Any:
    """Build a non-shuffling DataLoader over the selected rows."""

    torch = _require_torch()
    dataset = FrameDataset(frame, text_columns, targets, row_ids, indices)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_RequestCollator(tokenizer, max_length),
        num_workers=0,
    )


def predict_dataset(
    model: Any,
    tokenizer: Any,
    frame: Any,
    config: DatasetConfig,
    *,
    max_length: int,
    indices: np.ndarray | None = None,
    batch_size: int = 8,
    device: Any | None = None,
    return_embeddings: bool = False,
    progress: bool = True,
    description: str = "Prediction",
) -> Prediction:
    """Score the selected rows of a loaded table.

    ``indices`` selects positional rows, matching ``deterministic_split_indices``. Omitting it
    scores the whole table. Labels are attached when the configuration declares them, so the same
    call serves both evaluation and inference over unlabeled requests.
    """

    selected = (
        np.arange(len(frame)) if indices is None else np.asarray(indices, dtype=int)
    )
    targets = build_targets(frame, config.label_columns)
    row_ids = (
        frame[config.id_column].astype(str)
        if config.id_column
        else frame.index.to_series().map(lambda value: f"row-{value}")
    )
    loader = build_loader(
        frame,
        tokenizer,
        text_columns=config.text_columns,
        targets=targets,
        row_ids=row_ids,
        indices=selected,
        max_length=max_length,
        batch_size=batch_size,
    )
    return predict_batches(
        model,
        loader,
        device=device,
        return_embeddings=return_embeddings,
        progress=progress,
        description=description,
    )
