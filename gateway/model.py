from __future__ import annotations

import json
import math
from pathlib import Path

from http_attack_agent.concept_analysis.modsecurity import HEADERS
from http_attack_agent.models.zoo import MODEL_ZOO


def checkpoint_settings(metadata):
    spec = MODEL_ZOO["secbert"]
    if metadata["model"]["model_id"] != spec.model_id or metadata.get("smoke_sample") is not False:
        raise ValueError("Use the full SR-BH 2020 SecBERT checkpoint.")
    labels = metadata["label_names"]
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
        or "normal" in labels
    ):
        raise ValueError("Invalid ordered attack labels in metadata.json.")
    saved = metadata.get("test_metrics", {}).get("thresholds")
    if isinstance(saved, dict) and set(saved) == set(labels):
        saved = [saved[label] for label in labels]
    if not isinstance(saved, list) or len(saved) != len(labels):
        raise ValueError("Saved per-label thresholds in metadata.json are required.")
    if any(
        type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
        for value in saved
    ):
        raise ValueError("Saved attack thresholds must be finite probabilities.")
    thresholds = [float(value) for value in saved]
    length = metadata["max_length"]
    if type(length) is not int or not 1 <= length <= spec.max_length:
        raise ValueError("Invalid saved tokenizer length.")
    fields = list(metadata["text_columns"])
    expected_fields = {"method", "request_target", "protocol", "body", *HEADERS}
    if len(fields) != len(expected_fields) or set(fields) != expected_fields:
        raise ValueError("Checkpoint HTTP fields differ from the live SR-BH 2020 schema.")
    for name in ("hidden_size", "dataset_rows"):
        if type(metadata.get(name)) is not int or metadata[name] <= 0:
            raise ValueError(f"Invalid saved {name}.")
    return labels, thresholds, length, fields


class ProbeBank:
    def __init__(self, path, checkpoint, metadata):
        import numpy as np
        from http_attack_agent.concept_analysis.common import digest

        self.report = json.loads((path / "probe_report.json").read_text(encoding="utf-8"))
        if self.report.get("synthetic") is not False:
            raise ValueError("A concept bank trained with the original checkpoint is required.")
        if self.report.get("rows") != metadata["dataset_rows"] and not self.report.get("sampled"):
            raise ValueError("Checkpoint and probe report dataset sizes differ.")
        self.checkpoint_sha = digest(checkpoint / "model.pt")
        if self.report.get("checkpoint_sha256") != self.checkpoint_sha:
            raise ValueError("model.pt SHA-256 does not match the concept probe report.")
        if self.report.get("head_sha256") != digest(checkpoint / "embedding_head.pt"):
            raise ValueError("embedding_head.pt SHA-256 does not match the concept probe report.")
        bank_file = path / "concept_bank.npz"
        self.sha = digest(bank_file)
        with np.load(bank_file, allow_pickle=False) as bank:
            names = bank["concept_names"].astype(str)
            n = len(names)
            valid = bank["valid"]
            weights, intercept, thresholds = (
                bank[k] for k in ("weights", "intercept", "thresholds")
            )
            if (
                names.shape != (n,)
                or len(set(names)) != n
                or weights.shape != (n, metadata["hidden_size"])
                or intercept.shape != (n,)
                or thresholds.shape != (n,)
                or valid.shape != (n,)
                or not np.isin(valid, [0, 1]).all()
            ):
                raise ValueError("Concept bank shape or validity mask is incompatible.")
            if set(names) != set(self.report["concepts"]):
                raise ValueError("Concept bank and report contain different concepts.")
            keep = valid.astype(bool)
            self.total = n
            self.names = names[keep].tolist()
            self.weights, self.intercept, self.thresholds = (
                weights[keep],
                intercept[keep],
                thresholds[keep],
            )
            if not self.names or not all(
                np.isfinite(v).all()
                for v in (
                    self.weights,
                    self.intercept,
                    self.thresholds,
                )
            ):
                raise ValueError("Concept bank has no usable finite probes.")

    def score(self, embedding):
        import numpy as np

        scores = self.weights @ embedding + self.intercept
        if not np.isfinite(scores).all():
            return {"status": "unknown", "error": "non_finite_probe_scores", "scores": []}
        return {
            "status": "complete",
            "valid_count": len(self.names),
            "total_count": self.total,
            "scores": [
                {
                    "name": name,
                    "score": float(score),
                    "threshold": float(limit),
                    "active": bool(score >= limit),
                }
                for name, score, limit in zip(self.names, scores, self.thresholds, strict=True)
            ],
        }


class SecBert:
    def __init__(self, checkpoint: Path, bank: Path | None, device: str):
        from http_attack_agent.concept_analysis.common import digest
        from http_attack_agent.models.checkpoint import load_local_checkpoint

        saved_metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        self.labels, self.thresholds, self.max_length, fields = checkpoint_settings(saved_metadata)
        self.columns = {name: name for name in fields}
        self.probes = ProbeBank(bank, checkpoint, saved_metadata) if bank else None
        self.model, self.tokenizer, metadata = load_local_checkpoint(checkpoint, device=device)
        if metadata != saved_metadata or self.model.hidden_size != metadata["hidden_size"]:
            raise ValueError("Loaded SecBERT metadata or embedding width changed.")
        self.device = device
        self.identity = {
            "model_id": metadata["model"]["model_id"],
            "checkpoint_sha256": self.probes.checkpoint_sha
            if self.probes
            else digest(checkpoint / "model.pt"),
            "probe_bank_sha256": self.probes.sha if self.probes else None,
            "max_length": self.max_length,
            "dataset_rows": metadata["dataset_rows"],
            "device": device,
        }

    def inspect(self, request):
        import numpy as np
        import torch
        from http_attack_agent.data import serialize_http_request

        text = serialize_http_request(request.model_fields(HEADERS), self.columns)
        inputs = self.tokenizer(
            [text], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        with torch.inference_mode():
            result = self.model(
                inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)
            )
            probabilities = torch.sigmoid(result.logits[0]).float().cpu().numpy()
            embedding = result.embeddings[0].float().cpu().numpy()
        if probabilities.shape != (len(self.labels),) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid model probabilities.")
        if embedding.shape != (self.model.hidden_size,) or not np.isfinite(embedding).all():
            raise ValueError("Invalid model embedding.")
        scores = [
            {
                "label": label,
                "probability": float(prob),
                "threshold": float(limit),
                "active": bool(prob >= limit),
            }
            for label, prob, limit in zip(self.labels, probabilities, self.thresholds, strict=True)
        ]
        return {
            "status": "complete",
            "scores": scores,
            "would_block": any(score["active"] for score in scores),
            "concepts": self.probes.score(embedding)
            if self.probes
            else {
                "status": "unknown",
                "error": "not_configured",
                "scores": [],
            },
        }
