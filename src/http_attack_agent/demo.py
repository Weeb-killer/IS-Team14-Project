from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from typing import Any
import zipfile


CORE_DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}

FULL_DEPENDENCIES = {
    "pyarrow": "pyarrow",
    "torch": "torch",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "sentencepiece": "sentencepiece",
    "matplotlib": "matplotlib",
    "umap": "umap-learn",
}


class DemoFailure(RuntimeError):
    """A concise, user-facing smoke-demo failure."""


def _check_dependencies(full: bool) -> dict[str, str]:
    requested = dict(CORE_DEPENDENCIES)
    if full:
        requested.update(FULL_DEPENDENCIES)
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name, distribution_name in requested.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # Import can fail because of a missing native library.
            missing.append(f"{distribution_name} ({type(exc).__name__}: {exc})")
            continue
        versions[distribution_name] = str(getattr(module, "__version__", "installed"))
    if missing:
        raise DemoFailure(
            "Missing or unusable dependencies:\n  - "
            + "\n  - ".join(missing)
            + "\nInstall the project environment with: pip install -r requirements.txt"
        )
    return versions


def _synthetic_frame() -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    cases = (
        {
            "kind": "normal",
            "method": "GET",
            "target": "/products?category=books&page={index}",
            "body": "",
            "content_type": "",
            "sqli": 0,
            "traversal": 0,
            "waf_sqli": 0,
            "waf_lfi": 0,
        },
        {
            "kind": "sqli",
            "method": "GET",
            "target": "/products?id=1%27+UNION+SELECT+password+FROM+users--+{index}",
            "body": "",
            "content_type": "",
            "sqli": 1,
            "traversal": 0,
            "waf_sqli": 1,
            "waf_lfi": 0,
        },
        {
            "kind": "traversal",
            "method": "GET",
            "target": "/download?file=../../../../etc/passwd&attempt={index}",
            "body": "",
            "content_type": "",
            "sqli": 0,
            "traversal": 1,
            "waf_sqli": 0,
            "waf_lfi": 1,
        },
        {
            "kind": "mixed",
            "method": "POST",
            "target": "/admin/export?file=../../../etc/passwd",
            "body": "id=1%27+OR+%271%27=%271%27--+{index}",
            "content_type": "application/x-www-form-urlencoded",
            "sqli": 1,
            "traversal": 1,
            "waf_sqli": 1,
            "waf_lfi": 1,
        },
    )
    for case in cases:
        for index in range(24):
            rows.append(
                {
                    "request_id": f"demo-{case['kind']}-{index:02d}",
                    "request_http_method": case["method"],
                    "request_http_request": case["target"].format(index=index),
                    "request_body": case["body"].format(index=index),
                    "request_content_type": case["content_type"],
                    "66 - SQL Injection": case["sqli"],
                    "126 - Path Traversal": case["traversal"],
                    "waf_attack_sqli": case["waf_sqli"],
                    "waf_attack_lfi": case["waf_lfi"],
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=14).reset_index(drop=True)


def _base_dataset_config() -> dict[str, Any]:
    return {
        "format": "csv",
        "id_column": "request_id",
        "text_columns": {
            "method": "request_http_method",
            "request_target": "request_http_request",
            "body": "request_body",
            "content_type": "request_content_type",
        },
        "label_columns": {
            "sql_injection": "66 - SQL Injection",
            "path_traversal": "126 - Path Traversal",
        },
        "waf_concept_columns": {
            "waf_sqli": "waf_attack_sqli",
            "waf_lfi": "waf_attack_lfi",
        },
        "split": {
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_seed": 14,
        },
    }


def _prepare_demo_data(output_dir: Path) -> tuple[Path, Path]:
    frame = _synthetic_frame()
    config_dir = output_dir / "configs"
    source_dir = output_dir / "source"
    config_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    csv_path = source_dir / "synthetic_http.csv"
    frame.to_csv(csv_path, index=False)
    csv_bytes = csv_path.read_bytes()

    local_config = _base_dataset_config()
    local_config["source"] = {
        "mode": "local",
        "local": {
            "path": "source/preferred_name_missing.csv",
            "glob": "source/*.csv",
        },
    }
    local_path = config_dir / "dataset.local.yaml"
    local_path.write_text(json.dumps(local_config, indent=2), encoding="utf-8")

    archive_path = source_dir / "synthetic_http.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("release/synthetic_http.csv", csv_bytes)
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    remote_config = _base_dataset_config()
    remote_config["source"] = {
        "mode": "remote",
        "remote": {
            "url": archive_path.resolve().as_uri(),
            "path": f"cache/downloaded_http_{csv_sha256[:12]}.csv",
            "archive": "zip",
            "member": "synthetic_http.csv",
            "sha256": csv_sha256,
            "timeout_seconds": 30,
        },
    }
    remote_path = config_dir / "dataset.remote.yaml"
    remote_path.write_text(json.dumps(remote_config, indent=2), encoding="utf-8")
    return local_path, remote_path


def _exercise_data_pipeline(output_dir: Path) -> dict[str, Any]:
    import pandas as pd

    from .data import (
        DatasetConfig,
        build_targets,
        build_texts,
        deterministic_split_indices,
        load_table,
        resolve_dataset_path,
    )

    local_path, remote_path = _prepare_demo_data(output_dir)
    local_config = DatasetConfig.from_yaml(local_path)
    remote_config = DatasetConfig.from_yaml(remote_path)
    local_frame = load_table(local_config)
    remote_frame = load_table(remote_config)
    pd.testing.assert_frame_equal(local_frame, remote_frame)

    texts = build_texts(local_frame, local_config)
    labels = build_targets(local_frame, local_config.label_columns)
    concepts = build_targets(local_frame, local_config.waf_concept_columns)
    train, validation, test = deterministic_split_indices(local_frame, local_config)
    if not texts or labels.shape != (len(local_frame), 2):
        raise DemoFailure("Dataset serialization or multi-label target creation failed.")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise DemoFailure("Dataset split indices overlap.")
    if len(train) + len(validation) + len(test) != len(local_frame):
        raise DemoFailure("Dataset split indices do not cover every row.")
    return {
        "config": local_config,
        "frame": local_frame,
        "texts": texts,
        "labels": labels,
        "concepts": concepts,
        "train": train,
        "validation": validation,
        "test": test,
        "local_dataset": str(resolve_dataset_path(local_config)),
        "remote_cache": str(resolve_dataset_path(remote_config)),
    }


def _exercise_model_and_embeddings(data: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier

    from .explain.analysis import EmbeddingBundle, analyze_embeddings

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=512,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform([data["texts"][i] for i in data["train"]])
    all_matrix = vectorizer.transform(data["texts"])
    dimensions = min(16, train_matrix.shape[0] - 1, train_matrix.shape[1] - 1)
    if dimensions < 2:
        raise DemoFailure("The synthetic TF-IDF matrix is unexpectedly too small.")
    reducer = TruncatedSVD(n_components=dimensions, random_state=14)
    reducer.fit(train_matrix)
    embeddings = reducer.transform(all_matrix).astype(np.float32)

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=14)
    )
    classifier.fit(embeddings[data["train"]], data["labels"][data["train"]])
    probabilities = classifier.predict_proba(embeddings[data["test"]])
    predictions = (probabilities >= 0.5).astype(np.int8)
    micro_f1 = float(
        f1_score(data["labels"][data["test"]], predictions, average="micro")
    )
    if micro_f1 < 0.8:
        raise DemoFailure(f"Synthetic classification quality is unexpectedly low: {micro_f1:.3f}")

    logits = classifier.decision_function(embeddings)
    if logits.ndim == 1:
        logits = logits[:, None]
    embedding_path = output_dir / "artifacts" / "demo_embeddings.npz"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = EmbeddingBundle(
        embeddings=embeddings[data["test"]],
        logits=np.asarray(logits[data["test"]], dtype=np.float32),
        labels=data["labels"][data["test"]].astype(np.int8),
        concepts=data["concepts"][data["test"]].astype(np.int8),
        row_ids=np.asarray(
            data["frame"].iloc[data["test"]]["request_id"].astype(str).tolist(),
            dtype=str,
        ),
        label_names=list(data["config"].label_columns),
        concept_names=list(data["config"].waf_concept_columns),
    )
    bundle.save(embedding_path)
    bundle = EmbeddingBundle.load(str(embedding_path))
    explanation, cavs = analyze_embeddings(
        bundle,
        concept_source="both",
        folds=3,
        permutations=2,
        seed=14,
    )
    if not cavs:
        raise DemoFailure("Embedding analysis produced no concept activation vectors.")
    return {
        "micro_f1": micro_f1,
        "embedding_dimensions": dimensions,
        "embedding_path": str(embedding_path),
        "bundle": bundle,
        "explanation": explanation,
        "cavs": cavs,
    }


def _exercise_neural_explanations(model_result: dict[str, Any]) -> dict[str, Any]:
    from .explain.tcav import concept_erasure_report, tcav_report
    from .models.hf_classifier import build_embedding_head

    bundle = model_result["bundle"]
    head = build_embedding_head(
        hidden_size=bundle.embeddings.shape[1],
        num_labels=len(bundle.label_names),
        dropout=0.0,
    )
    tcav = tcav_report(
        head,
        bundle.embeddings,
        bundle.labels,
        bundle.label_names,
        model_result["cavs"],
        random_directions=3,
        seed=14,
    )
    erasure = concept_erasure_report(
        head,
        bundle.embeddings,
        bundle.labels,
        bundle.label_names,
        model_result["cavs"],
    )
    return {"tcav_concepts": sorted(tcav), "erasure_concepts": sorted(erasure)}


def _exercise_waf_pipeline() -> dict[str, Any]:
    from .waf.audit import iter_events
    from .waf.concepts import aggregate_events

    project_root = Path(__file__).resolve().parents[2]
    concept_config = project_root / "configs" / "concepts.yaml"
    if not concept_config.is_file():
        raise DemoFailure(f"WAF concept configuration is missing: {concept_config}")
    alert = (
        "ModSecurity: Warning. Matched \"Operator\" against variable `ARGS:id` "
        '[id "942100"] [msg "SQL Injection Attack"] '
        '[data "Matched Data: union select"] [severity "CRITICAL"] '
        '[tag "attack-sqli"] [tag "OWASP_CRS/ATTACK-SQLI"] '
        '[request_id "demo-sqli-00"] [unique_id "demo-tx-00"] [uri "/products"]'
    )
    events = list(iter_events([alert], concept_config))
    concepts, rows = aggregate_events(events)
    if len(events) != 1 or len(rows) != 1 or "waf_sqli" not in concepts:
        raise DemoFailure("OWASP CRS event parsing or concept aggregation failed.")
    return {
        "events": len(events),
        "requests": len(rows),
        "rule_ids": json.loads(rows[0]["waf_rule_ids"]),
        "concepts": concepts,
    }


def _exercise_repository_interfaces() -> dict[str, Any]:
    from .models.zoo import MODEL_ZOO, get_model_spec

    modules = (
        "http_attack_agent.training",
        "http_attack_agent.explain.cli",
        "http_attack_agent.waf.audit",
        "http_attack_agent.waf.concepts",
    )
    for module in modules:
        importlib.import_module(module)
    specs = [get_model_spec(name, require_enabled=False) for name in MODEL_ZOO]
    return {
        "imported_modules": list(modules),
        "registered_models": [asdict(spec) for spec in specs],
        "enabled_models": [spec.name for spec in specs if spec.enabled],
    }


def run_demo(output_dir: str | Path, full_dependency_check: bool = True) -> dict[str, Any]:
    """Run the repository's fast, offline end-to-end smoke demo."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dependencies = _check_dependencies(full_dependency_check)
    print("[PASS] Dependency imports")
    data = _exercise_data_pipeline(output)
    print("[PASS] Local and remote dataset modes")
    model_result = _exercise_model_and_embeddings(data, output)
    print("[PASS] Multi-label model and embedding analysis")
    neural = (
        _exercise_neural_explanations(model_result)
        if full_dependency_check
        else {"status": "skipped (--core-only)"}
    )
    if full_dependency_check:
        print("[PASS] PyTorch TCAV and concept erasure")
    waf = _exercise_waf_pipeline()
    print("[PASS] OWASP CRS event and concept pipeline")
    interfaces = _exercise_repository_interfaces()
    print("[PASS] Model zoo and command-module imports")

    report = {
        "status": "passed",
        "scope": "full" if full_dependency_check else "core",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependencies,
        "dataset": {
            "rows": len(data["frame"]),
            "labels": list(data["config"].label_columns),
            "concepts": list(data["config"].waf_concept_columns),
            "split_sizes": {
                "train": len(data["train"]),
                "validation": len(data["validation"]),
                "test": len(data["test"]),
            },
            "local_dataset": data["local_dataset"],
            "remote_cache": data["remote_cache"],
        },
        "lightweight_model": {
            "micro_f1": model_result["micro_f1"],
            "embedding_dimensions": model_result["embedding_dimensions"],
            "embedding_path": model_result["embedding_path"],
        },
        "embedding_analysis": {
            "analyzed_concepts": sorted(model_result["explanation"]["concepts"]),
            "concept_vectors": sorted(model_result["cavs"]),
            "neural_explanations": neural,
        },
        "waf": waf,
        "repository_interfaces": interfaces,
        "limitations": [
            "The demo does not download large pretrained model weights.",
            "Model-zoo registration does not guarantee online checkpoint access.",
            "The demo does not train on the full SR-BH 2020 dataset.",
            "The demo parses a representative CRS alert but does not start a WAF engine.",
        ],
    }
    report_path = output / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSMOKE DEMO PASSED\nReport: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fast, offline end-to-end smoke demo for this repository."
    )
    parser.add_argument(
        "--output",
        default="runs/smoke-demo",
        help="Directory for synthetic inputs and the JSON report.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip neural/visualization dependency checks and PyTorch explanations.",
    )
    args = parser.parse_args()
    try:
        run_demo(args.output, full_dependency_check=not args.core_only)
    except Exception as exc:
        print(f"\nSMOKE DEMO FAILED\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
