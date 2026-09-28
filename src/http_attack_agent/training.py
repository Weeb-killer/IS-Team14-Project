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
from tqdm.auto import tqdm

from .data import (
    DatasetConfig,
    FrameDataset,
    build_targets,
    deterministic_split_indices,
    load_table,
)
from .explain.analysis import EmbeddingBundle
from .models.checkpoint import save_local_checkpoint
from .models.hf_classifier import (
    build_classifier,
    freeze_backbone,
    load_tokenizer,
    parameter_counts,
    set_train_mode,
)
from .embedding_cache import (
    cache_directory,
    estimated_cache_bytes,
    format_bytes,
    head_predict,
    load_or_compute_cache,
    train_head_epochs,
)
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


def _cpu_state_snapshot(model: Any) -> dict[str, Any]:
    """Keep the best epoch independent of later in-place parameter updates."""

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


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


def _progress_batches(loader: Any, description: str, leave: bool = False) -> Any:
    return tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=leave,
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


def _predict(
    model: Any, loader: Any, device: Any, description: str = "Evaluation"
) -> tuple[np.ndarray, np.ndarray]:
    torch = _require_torch()
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _progress_batches(loader, description):
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
    description: str = "Export test embeddings",
) -> None:
    torch = _require_torch()
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    concepts: list[np.ndarray] = []
    row_ids: list[str] = []
    with torch.no_grad():
        for batch in _progress_batches(loader, description):
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

    tqdm.write(f"Loading pretrained model {spec.model_id} for {model_name}...")
    tokenizer = load_tokenizer(spec.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = build_classifier(spec.model_id, targets.shape[1], dropout=args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    if args.freeze_backbone:
        freeze_backbone(model)
    counts = parameter_counts(model)
    tqdm.write(
        json.dumps(
            {
                "model": model_name,
                "freeze_backbone": bool(args.freeze_backbone),
                **counts,
            }
        )
    )

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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    cache = None
    head_epochs = None
    if args.cache_embeddings:
        if not args.freeze_backbone:
            raise ValueError(
                "--cache-embeddings requires --freeze-backbone: a backbone that still "
                "trains produces different embeddings every epoch."
            )
        cache_dir = cache_directory(
            args.embedding_cache or (output_root / "embedding-cache"), model_name
        )
        cache, reused = load_or_compute_cache(
            cache_dir,
            model,
            tokenizer,
            frame,
            config,
            model_id=spec.model_id,
            max_length=min(spec.max_length, args.max_length),
            batch_size=args.encode_batch_size,
            device=device,
            dtype=args.cache_dtype,
            refresh=args.refresh_cache,
        )
        tqdm.write(
            json.dumps(
                {
                    "model": model_name,
                    "embedding_cache": str(cache_dir),
                    "reused": reused,
                    "rows": len(cache),
                    "hidden_size": cache.hidden_size,
                    "size": format_bytes(
                        estimated_cache_bytes(
                            len(cache), cache.hidden_size, args.cache_dtype
                        )
                    ),
                }
            )
        )
        head_epochs = train_head_epochs(
            model.classifier,
            cache.embeddings,
            targets,
            train_idx,
            valid_idx,
            epochs=args.epochs,
            batch_size=args.head_batch_size,
            learning_rate=args.learning_rate,
            pos_weight=pos_weight,
            gradient_clip=args.gradient_clip,
            device=device,
            seed=config.split.random_seed,
            progress=lambda batches, epoch, total: _progress_batches(
                list(batches), f"{model_name} epoch {epoch}/{args.epochs} head", leave=True
            ),
        )

    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    best_thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    label_names = list(config.label_columns)
    for epoch in range(1, args.epochs + 1):
        if head_epochs is not None:
            # The backbone is frozen and already encoded, so only the head is trained.
            _, train_loss, validation_probabilities = next(head_epochs)
            validation_targets = targets[valid_idx]
        else:
            set_train_mode(model, args.freeze_backbone)
            losses: list[float] = []
            running_loss = 0.0
            progress = _progress_batches(
                train_loader,
                f"{model_name} epoch {epoch}/{args.epochs} train",
                leave=True,
            )
            for batch_number, batch in enumerate(progress, start=1):
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
                loss = criterion(output.logits, batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
                optimizer.step()
                batch_loss = float(loss.detach().cpu())
                losses.append(batch_loss)
                running_loss += batch_loss
                if batch_number == 1 or batch_number % 25 == 0:
                    progress.set_postfix(loss=f"{running_loss / batch_number:.4f}", refresh=False)
            train_loss = float(np.mean(losses))
            validation_targets, validation_probabilities = _predict(
                model,
                valid_loader,
                device,
                f"{model_name} epoch {epoch}/{args.epochs} validation",
            )
        epoch_thresholds = _optimal_thresholds(validation_targets, validation_probabilities)
        validation = _metrics(
            validation_targets, validation_probabilities, epoch_thresholds, label_names
        )
        record = {"epoch": epoch, "train_loss": train_loss, **validation}
        history.append(record)
        tqdm.write(json.dumps({"model": model_name, **record}))
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_state = _cpu_state_snapshot(model)
            best_thresholds = epoch_thresholds

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)
    if cache is not None:
        test_targets = targets[test_idx]
        test_probabilities = head_predict(
            model.classifier, cache.embeddings, test_idx, device=device
        )
    else:
        test_targets, test_probabilities = _predict(
            model, test_loader, device, f"{model_name} test"
        )
    test_metrics = _metrics(
        test_targets, test_probabilities, best_thresholds, label_names
    )

    metadata = {
        "model": asdict(spec),
        "label_names": label_names,
        "concept_names": list(config.waf_concept_columns) if concepts is not None else [],
        "hidden_size": model.hidden_size,
        "dropout": args.dropout,
        "freeze_backbone": bool(args.freeze_backbone),
        "cached_embeddings": cache is not None,
        "parameters": counts,
        "max_length": min(spec.max_length, args.max_length),
        "text_columns": dict(config.text_columns),
        "dataset_rows": len(frame),
        "smoke_sample": args.max_rows is not None,
        "test_metrics": test_metrics,
        "history": history,
        "split": {
            "train": len(train_idx),
            "validation": len(valid_idx),
            "test": len(test_idx),
        },
    }
    save_local_checkpoint(model, tokenizer, output_dir, metadata)
    _export_embeddings(
        model,
        test_loader,
        device,
        output_dir / "test_embeddings.npz",
        label_names,
        metadata["concept_names"],
        f"{model_name} export test embeddings",
    )
    weights_path = (output_dir / "model.pt").resolve()
    print(json.dumps({"model": model_name, "saved_weights": str(weights_path)}))
    return {
        "model": model_name,
        "saved_weights": str(weights_path),
        **test_metrics,
    }


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
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Random row cap for a quick training smoke run; omit for the full experiment",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Train only the classifier head and keep the pretrained encoder fixed in "
            "evaluation mode. Recommended when comparing pretrained representations."
        ),
    )
    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        help=(
            "Encode every request once with the frozen backbone and train the head on the "
            "cached vectors. Requires --freeze-backbone."
        ),
    )
    parser.add_argument(
        "--embedding-cache",
        help="Directory for cached embeddings; defaults to <output>/embedding-cache.",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Stored precision. float16 halves the cache on disk (default: float32).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-encode even when a matching cache exists.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=64,
        help="Batch size for the one-off encoding pass (default: 64).",
    )
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=256,
        help="Batch size for head training on cached embeddings (default: 256).",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 3:
        parser.error("--max-rows must be at least 3")
    if args.cache_embeddings and not args.freeze_backbone:
        parser.error("--cache-embeddings requires --freeze-backbone")
    for model_name in args.model:
        get_model_spec(model_name)

    config = DatasetConfig.from_yaml(args.dataset_config)
    seed_everything(config.split.random_seed)
    tqdm.write(f"Loading dataset from {config.source.mode} source...")
    frame = load_table(config)
    total_rows = len(frame)
    tqdm.write(f"Loaded {total_rows:,} requests")
    if args.max_rows is not None and total_rows > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=config.split.random_seed)
        frame = frame.sort_index()
        print(
            json.dumps(
                {
                    "training_smoke_sample": len(frame),
                    "dataset_rows": total_rows,
                    "warning": "Do not use --max-rows for formal benchmark results",
                }
            )
        )
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
