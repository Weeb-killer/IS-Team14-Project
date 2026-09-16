from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)

from .data import (
    DatasetConfig,
    FrameDataset,
    build_targets,
    deterministic_split_indices,
    load_table,
)
from .explain.analysis import EmbeddingBundle
from .models.hf_classifier import build_classifier, load_tokenizer
from .models.zoo import get_model_spec


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc
    return torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RequestCollator:
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
            "labels": torch.as_tensor(
                np.stack([row["labels"] for row in rows]), dtype=torch.float32
            ),
            "row_ids": [row["row_id"] for row in rows],
        }
        if "concepts" in rows[0]:
            batch["concepts"] = np.stack([row["concepts"] for row in rows])
        return batch


def _make_loader(dataset: FrameDataset, collator: RequestCollator, batch_size: int, shuffle: bool):
    torch = _require_torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
    )


def _optimal_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    candidates = np.linspace(0.05, 0.95, 19)
    for index in range(targets.shape[1]):
        if targets[:, index].sum() == 0:
            continue
        scores = [
            f1_score(
                targets[:, index],
                probabilities[:, index] >= threshold,
                zero_division=0,
            )
            for threshold in candidates
        ]
        thresholds[index] = candidates[int(np.argmax(scores))]
    return thresholds


def _metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    predictions = (probabilities >= thresholds).astype(np.int8)
    precision, recall, per_label_f1, _ = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    result: dict[str, Any] = {
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "thresholds": {
            name: float(value) for name, value in zip(label_names, thresholds)
        },
    }
    try:
        result["macro_average_precision"] = float(
            average_precision_score(targets, probabilities, average="macro")
        )
    except ValueError:
        result["macro_average_precision"] = None
    result["normal_accuracy"] = float(
        np.mean((targets.sum(axis=1) == 0) == (predictions.sum(axis=1) == 0))
    )
    per_label: dict[str, Any] = {}
    for index, name in enumerate(label_names):
        try:
            ap = float(average_precision_score(targets[:, index], probabilities[:, index]))
        except ValueError:
            ap = None
        per_label[name] = {
            "support": int(targets[:, index].sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_label_f1[index]),
            "average_precision": ap,
        }
    result["per_label"] = per_label
    return result


def _predict(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:
    torch = _require_torch()
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            all_targets.append(batch["labels"].numpy())
            all_probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probabilities)


def _export_embeddings(
    model: Any,
    loader: Any,
    device: Any,
    output_path: Path,
    label_names: list[str],
    concept_names: list[str],
) -> None:
    torch = _require_torch()
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    concepts: list[np.ndarray] = []
    row_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            embeddings.append(output.embeddings.cpu().numpy())
            logits.append(output.logits.cpu().numpy())
            labels.append(batch["labels"].numpy())
            row_ids.extend(batch["row_ids"])
            if "concepts" in batch:
                concepts.append(batch["concepts"])
    concept_matrix = (
        np.concatenate(concepts).astype(np.int8)
        if concepts
        else np.empty((len(row_ids), 0), dtype=np.int8)
    )
    EmbeddingBundle(
        embeddings=np.concatenate(embeddings).astype(np.float32),
        logits=np.concatenate(logits).astype(np.float32),
        labels=np.concatenate(labels).astype(np.int8),
        concepts=concept_matrix,
        row_ids=np.asarray(row_ids, dtype=str),
        label_names=label_names,
        concept_names=concept_names,
    ).save(output_path)


def train_one(
    model_name: str,
    config: DatasetConfig,
    frame: Any,
    targets: np.ndarray,
    concepts: np.ndarray | None,
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch = _require_torch()
    spec = get_model_spec(model_name)
    output_dir = output_root / model_name if len(args.model) > 1 else output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(spec.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = build_classifier(spec.model_id, targets.shape[1], dropout=args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    train_idx, valid_idx, test_idx = split_indices
    row_ids = (
        frame[config.id_column].astype(str)
        if config.id_column
        else frame.index.to_series().map(lambda value: f"row-{value}")
    )
    train_data = FrameDataset(
        frame, config.text_columns, targets, row_ids, train_idx, concepts
    )
    valid_data = FrameDataset(
        frame, config.text_columns, targets, row_ids, valid_idx, concepts
    )
    test_data = FrameDataset(
        frame, config.text_columns, targets, row_ids, test_idx, concepts
    )
    collator = RequestCollator(tokenizer, min(spec.max_length, args.max_length))
    train_loader = _make_loader(train_data, collator, args.batch_size, True)
    valid_loader = _make_loader(valid_data, collator, args.batch_size, False)
    test_loader = _make_loader(test_data, collator, args.batch_size, False)

    train_targets = targets[train_idx]
    positives = train_targets.sum(axis=0)
    negatives = len(train_targets) - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1), 1.0, args.max_pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    best_thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    label_names = list(config.label_columns)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            loss = criterion(output.logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_targets, validation_probabilities = _predict(model, valid_loader, device)
        epoch_thresholds = _optimal_thresholds(validation_targets, validation_probabilities)
        validation = _metrics(
            validation_targets, validation_probabilities, epoch_thresholds, label_names
        )
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(record)
        print(json.dumps({"model": model_name, **record}))
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_thresholds = epoch_thresholds

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)
    test_targets, test_probabilities = _predict(model, test_loader, device)
    test_metrics = _metrics(
        test_targets, test_probabilities, best_thresholds, label_names
    )

    torch.save(best_state, output_dir / "model.pt")
    torch.save(model.classifier.state_dict(), output_dir / "embedding_head.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    metadata = {
        "model": asdict(spec),
        "label_names": label_names,
        "concept_names": list(config.waf_concept_columns) if concepts is not None else [],
        "hidden_size": model.hidden_size,
        "dropout": args.dropout,
        "test_metrics": test_metrics,
        "history": history,
        "split": {
            "train": len(train_idx),
            "validation": len(valid_idx),
            "test": len(test_idx),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _export_embeddings(
        model,
        test_loader,
        device,
        output_dir / "test_embeddings.npz",
        label_names,
        metadata["concept_names"],
    )
    return {"model": model_name, **test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train comparable HTTP attack encoders")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = DatasetConfig.from_yaml(args.dataset_config)
    seed_everything(config.split.random_seed)
    frame = load_table(config)
    targets = build_targets(frame, config.label_columns)
    if targets.shape[1] == 0:
        raise ValueError("At least one attack label is required")

    concept_columns_exist = all(
        column in frame.columns for column in config.waf_concept_columns.values()
    )
    concepts = (
        build_targets(frame, config.waf_concept_columns)
        if config.waf_concept_columns and concept_columns_exist
        else None
    )
    split_indices = deterministic_split_indices(frame, config)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in args.model:
        results.append(
            train_one(
                model_name,
                config,
                frame,
                targets,
                concepts,
                split_indices,
                output_root,
                args,
            )
        )
    (output_root / "benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
