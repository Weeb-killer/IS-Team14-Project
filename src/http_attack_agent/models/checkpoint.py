from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hf_classifier import build_classifier, load_tokenizer


def save_local_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Save one complete inference checkpoint without duplicating backbone weights."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required to save model weights") from exc

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.config.save_pretrained(directory / "backbone_config")
    tokenizer.save_pretrained(directory / "tokenizer")
    torch.save(model.state_dict(), directory / "model.pt")
    torch.save(model.classifier.state_dict(), directory / "embedding_head.pt")
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory


def load_local_checkpoint(
    artifact_dir: str | Path,
    device: str = "cpu",
) -> tuple[Any, Any, dict[str, Any]]:
    """Rebuild a saved classifier and tokenizer using local files only."""

    try:
        import torch
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install neural dependencies before loading a checkpoint") from exc

    directory = Path(artifact_dir)
    metadata_path = directory / "metadata.json"
    config_dir = directory / "backbone_config"
    weights_path = directory / "model.pt"
    tokenizer_dir = directory / "tokenizer"
    for required in (metadata_path, config_dir / "config.json", weights_path, tokenizer_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"Incomplete local checkpoint: {required} is missing. "
                "Retrain with the current version to save all required files."
            )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    backbone_config = AutoConfig.from_pretrained(
        str(config_dir), local_files_only=True, trust_remote_code=False
    )
    model = build_classifier(
        model_id=str(metadata["model"]["model_id"]),
        num_labels=len(metadata["label_names"]),
        dropout=float(metadata.get("dropout", 0.1)),
        backbone_config=backbone_config,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    tokenizer = load_tokenizer(str(tokenizer_dir), local_files_only=True)
    return model, tokenizer, metadata
