from __future__ import annotations

from typing import Any

import numpy as np


def tcav_report(
    head: Any,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    cavs: dict[str, np.ndarray],
    random_directions: int = 20,
    seed: int = 14,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Compute directional sensitivity of every class logit to each CAV."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc

    head.eval()
    device = next(head.parameters()).device
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {}

    for concept_name, cav in cavs.items():
        cav_tensor = torch.as_tensor(cav, dtype=torch.float32, device=device)
        class_reports: dict[str, Any] = {}
        for class_index, class_name in enumerate(label_names):
            selected = np.flatnonzero(labels[:, class_index] > 0)
            if not len(selected):
                selected = np.arange(len(embeddings))
            derivatives: list[np.ndarray] = []
            random_scores = np.zeros(random_directions, dtype=np.float64)
            random_vectors = rng.normal(size=(random_directions, embeddings.shape[1]))
            random_vectors /= np.maximum(
                np.linalg.norm(random_vectors, axis=1, keepdims=True), 1e-12
            )
            random_tensors = torch.as_tensor(
                random_vectors, dtype=torch.float32, device=device
            )

            for start in range(0, len(selected), batch_size):
                batch_indices = selected[start : start + batch_size]
                batch = torch.as_tensor(
                    embeddings[batch_indices], dtype=torch.float32, device=device
                ).requires_grad_(True)
                logits = head(batch)
                gradients = torch.autograd.grad(logits[:, class_index].sum(), batch)[0]
                directional = (gradients * cav_tensor).sum(dim=1)
                derivatives.append(directional.detach().cpu().numpy())
                random_directional = gradients @ random_tensors.T
                random_scores += (random_directional > 0).sum(dim=0).detach().cpu().numpy()

            values = np.concatenate(derivatives)
            score = float(np.mean(values > 0))
            null_scores = random_scores / len(selected)
            p_value = float(
                (1 + np.sum(null_scores >= score)) / (1 + len(null_scores))
            )
            class_reports[class_name] = {
                "samples": int(len(selected)),
                "tcav_score": score,
                "mean_directional_derivative": float(values.mean()),
                "random_direction_score_mean": float(null_scores.mean()),
                "random_direction_p_value": p_value,
            }
        report[concept_name] = class_reports
    return report


def concept_erasure_report(
    head: Any,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    cavs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Remove the projection on each CAV and measure the resulting logit change."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc

    head.eval()
    device = next(head.parameters()).device
    original = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        original_logits = head(original)
    report: dict[str, Any] = {}
    for concept_name, cav in cavs.items():
        direction = torch.as_tensor(cav, dtype=torch.float32, device=device)
        projection = (original @ direction).unsqueeze(1) * direction.unsqueeze(0)
        erased = original - projection
        with torch.no_grad():
            erased_logits = head(erased)
        delta = (original_logits - erased_logits).cpu().numpy()
        concept_report: dict[str, Any] = {}
        for class_index, class_name in enumerate(label_names):
            selected = labels[:, class_index] > 0
            if not selected.any():
                selected = np.ones(len(labels), dtype=bool)
            values = delta[selected, class_index]
            concept_report[class_name] = {
                "samples": int(selected.sum()),
                "mean_logit_drop": float(values.mean()),
                "median_logit_drop": float(np.median(values)),
                "fraction_prediction_reduced": float(np.mean(values > 0)),
            }
        report[concept_name] = concept_report
    return report
