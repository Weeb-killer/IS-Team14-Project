from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class EmbeddingBundle:
    embeddings: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    concepts: np.ndarray
    row_ids: np.ndarray
    label_names: list[str]
    concept_names: list[str]

    def save(self, path: str | Path) -> None:
        """Save an archive containing only non-pickle NumPy dtypes."""

        np.savez_compressed(
            path,
            embeddings=np.asarray(self.embeddings, dtype=np.float32),
            logits=np.asarray(self.logits, dtype=np.float32),
            labels=np.asarray(self.labels, dtype=np.int8),
            concepts=np.asarray(self.concepts, dtype=np.int8),
            row_ids=np.asarray(self.row_ids.tolist(), dtype=str),
            label_names=np.asarray(self.label_names, dtype=str),
            concept_names=np.asarray(self.concept_names, dtype=str),
        )

    @classmethod
    def load(cls, path: str) -> "EmbeddingBundle":
        raw = np.load(path, allow_pickle=False)
        return cls(
            embeddings=np.asarray(raw["embeddings"], dtype=np.float32),
            logits=np.asarray(raw["logits"], dtype=np.float32),
            labels=np.asarray(raw["labels"], dtype=np.int8),
            concepts=np.asarray(raw["concepts"], dtype=np.int8),
            row_ids=np.asarray(raw["row_ids"], dtype=str),
            label_names=np.asarray(raw["label_names"], dtype=str).tolist(),
            concept_names=np.asarray(raw["concept_names"], dtype=str).tolist(),
        )

    def subset(self, indices: np.ndarray) -> "EmbeddingBundle":
        return EmbeddingBundle(
            embeddings=self.embeddings[indices],
            logits=self.logits[indices],
            labels=self.labels[indices],
            concepts=self.concepts[indices],
            row_ids=self.row_ids[indices],
            label_names=self.label_names,
            concept_names=self.concept_names,
        )


def representative_subsample(
    bundle: EmbeddingBundle, max_samples: int, seed: int = 14
) -> EmbeddingBundle:
    """Cap analysis cost while retaining examples from rare labels/concepts."""

    n_samples = len(bundle.embeddings)
    if max_samples <= 0 or n_samples <= max_samples:
        return bundle
    rng = np.random.default_rng(seed)
    matrices = [bundle.labels]
    if bundle.concepts.shape[1]:
        matrices.append(bundle.concepts)
    annotations = np.concatenate(matrices, axis=1)
    retained: set[int] = set()
    per_concept = max(20, min(1000, max_samples // max(annotations.shape[1], 1)))
    for column in range(annotations.shape[1]):
        positives = np.flatnonzero(annotations[:, column] > 0)
        if len(positives) > per_concept:
            positives = rng.choice(positives, size=per_concept, replace=False)
        retained.update(int(value) for value in positives)
    if len(retained) > max_samples:
        retained = set(rng.choice(list(retained), size=max_samples, replace=False).tolist())
    remaining = max_samples - len(retained)
    if remaining:
        candidates = np.setdiff1d(
            np.arange(n_samples), np.fromiter(retained, dtype=int), assume_unique=False
        )
        retained.update(
            int(value)
            for value in rng.choice(candidates, size=remaining, replace=False)
        )
    return bundle.subset(np.asarray(sorted(retained), dtype=int))


def _safe_fold_count(target: np.ndarray, requested: int) -> int:
    class_counts = np.bincount(target.astype(int), minlength=2)
    return int(min(requested, class_counts.min()))


def fit_concept_probe(
    embeddings: np.ndarray,
    target: np.ndarray,
    folds: int = 5,
    permutations: int = 10,
    seed: int = 14,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Measure linear decodability and return a concept activation vector (CAV)."""

    target = target.astype(int)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    n_splits = _safe_fold_count(target, folds)
    if positives < 2 or negatives < 2 or n_splits < 2:
        return {
            "status": "insufficient_examples",
            "positives": positives,
            "negatives": negatives,
        }, None

    estimator = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        estimator, embeddings, target, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    observed_balanced_accuracy = balanced_accuracy_score(target, predictions)
    observed_auc = roc_auc_score(target, probabilities)

    rng = np.random.default_rng(seed)
    null_scores: list[float] = []
    for _ in range(permutations):
        shuffled = rng.permutation(target)
        shuffled_probabilities = cross_val_predict(
            estimator, embeddings, shuffled, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        null_scores.append(roc_auc_score(shuffled, shuffled_probabilities))
    p_value = (
        1.0 + sum(score >= observed_auc for score in null_scores)
    ) / (1.0 + len(null_scores))

    estimator.fit(embeddings, target)
    cav = estimator.coef_[0].astype(np.float32)
    cav /= max(float(np.linalg.norm(cav)), 1e-12)
    report = {
        "status": "ok",
        "positives": positives,
        "negatives": negatives,
        "folds": n_splits,
        "roc_auc": float(observed_auc),
        "balanced_accuracy": float(observed_balanced_accuracy),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "permutation_auc_mean": float(np.mean(null_scores)),
        "permutation_p_value": float(p_value),
    }
    return report, cav


def prototype_report(
    embeddings: np.ndarray,
    target: np.ndarray,
    row_ids: np.ndarray,
    nearest: int = 5,
) -> dict[str, Any]:
    target = target.astype(bool)
    if target.sum() == 0 or (~target).sum() == 0:
        return {"status": "insufficient_examples"}
    unit = normalize(embeddings)
    positive_centroid = normalize(unit[target].mean(axis=0, keepdims=True))[0]
    negative_centroid = normalize(unit[~target].mean(axis=0, keepdims=True))[0]
    positive_similarity = unit @ positive_centroid
    negative_similarity = unit @ negative_centroid
    margin = positive_similarity - negative_similarity
    prototype_indices = np.flatnonzero(target)[
        np.argsort(positive_similarity[target])[::-1][:nearest]
    ]
    return {
        "status": "ok",
        "centroid_cosine_similarity": float(positive_centroid @ negative_centroid),
        "positive_mean_margin": float(margin[target].mean()),
        "negative_mean_margin": float(margin[~target].mean()),
        "prototype_row_ids": row_ids[prototype_indices].astype(str).tolist(),
    }


def build_concept_matrix(
    bundle: EmbeddingBundle, source: str
) -> tuple[np.ndarray, list[str], list[str]]:
    matrices: list[np.ndarray] = []
    names: list[str] = []
    origins: list[str] = []
    if source in {"waf", "both"} and bundle.concepts.shape[1]:
        matrices.append(bundle.concepts)
        names.extend(bundle.concept_names)
        origins.extend(["waf"] * len(bundle.concept_names))
    if source in {"labels", "both"}:
        matrices.append(bundle.labels)
        names.extend([f"label:{name}" for name in bundle.label_names])
        origins.extend(["dataset_label"] * len(bundle.label_names))
    if not matrices:
        return np.empty((len(bundle.embeddings), 0), dtype=np.int8), [], []
    return np.concatenate(matrices, axis=1), names, origins


def analyze_embeddings(
    bundle: EmbeddingBundle,
    concept_source: str = "both",
    folds: int = 5,
    permutations: int = 10,
    seed: int = 14,
    conditional_probes: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    matrix, names, origins = build_concept_matrix(bundle, concept_source)
    report: dict[str, Any] = {
        "samples": int(len(bundle.embeddings)),
        "embedding_dimensions": int(bundle.embeddings.shape[1]),
        "concept_source": concept_source,
        "interpretation": {
            "probe": "Information is linearly decodable from the embedding.",
            "prototype": "Examples with the concept occupy a coherent cosine-space region.",
            "tcav": "The trained prediction head is sensitive to the learned concept direction.",
        },
        "concepts": {},
    }
    cavs: dict[str, np.ndarray] = {}
    for index, (name, origin) in enumerate(zip(names, origins)):
        target = matrix[:, index]
        probe, cav = fit_concept_probe(
            bundle.embeddings,
            target,
            folds=folds,
            permutations=permutations,
            seed=seed + index,
        )
        concept_report = {
            "origin": origin,
            "probe": probe,
            "prototype": prototype_report(
                bundle.embeddings, target, bundle.row_ids
            ),
        }
        if conditional_probes and origin == "waf":
            conditional: dict[str, Any] = {}
            for label_index, label_name in enumerate(bundle.label_names):
                selected = bundle.labels[:, label_index] > 0
                conditional_target = target[selected]
                if selected.sum() < 40 or len(np.unique(conditional_target)) < 2:
                    continue
                conditional[label_name], _ = fit_concept_probe(
                    bundle.embeddings[selected],
                    conditional_target,
                    folds=min(3, folds),
                    permutations=min(3, permutations),
                    seed=seed + index + label_index + 1000,
                )
            concept_report["conditional_on_attack_label"] = conditional
        report["concepts"][name] = concept_report
        if cav is not None:
            cavs[name] = cav
    return report, cavs
