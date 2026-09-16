from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import EmbeddingBundle, analyze_embeddings, representative_subsample
from .tcav import concept_erasure_report, tcav_report
from ..models.hf_classifier import build_embedding_head


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze security concepts in embeddings")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concept-source", choices=["waf", "labels", "both"], default="both")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--random-directions", type=int, default=20)
    parser.add_argument(
        "--conditional-probes",
        action="store_true",
        help="Probe WAF concepts within each attack label to reduce coarse-label confounding",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Representative cap for expensive probe/TCAV analysis; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument(
        "--artifact-dir",
        help="Directory containing metadata.json and embedding_head.pt; defaults to embedding parent",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_bundle = EmbeddingBundle.load(args.embeddings)
    bundle = representative_subsample(full_bundle, args.max_samples, args.seed)
    report, cavs = analyze_embeddings(
        bundle,
        concept_source=args.concept_source,
        folds=args.folds,
        permutations=args.permutations,
        seed=args.seed,
        conditional_probes=args.conditional_probes,
    )

    if cavs:
        np.savez_compressed(output_dir / "concept_vectors.npz", **cavs)

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else Path(args.embeddings).parent
    metadata_path = artifact_dir / "metadata.json"
    head_path = artifact_dir / "embedding_head.pt"
    if cavs and metadata_path.exists() and head_path.exists():
        try:
            import torch

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            head = build_embedding_head(
                metadata["hidden_size"],
                len(metadata["label_names"]),
                metadata.get("dropout", 0.1),
            )
            head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
            report["tcav"] = tcav_report(
                head,
                bundle.embeddings,
                bundle.labels,
                bundle.label_names,
                cavs,
                random_directions=args.random_directions,
                seed=args.seed,
            )
            report["concept_erasure"] = concept_erasure_report(
                head,
                bundle.embeddings,
                bundle.labels,
                bundle.label_names,
                cavs,
            )
        except ImportError:
            report["tcav_status"] = "skipped: torch is not installed"
    else:
        report["tcav_status"] = "skipped: classifier head artifact or valid CAV is missing"

    (output_dir / "embedding_report.json").write_text(
        json.dumps(report, indent=2, default=_jsonable), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir / "embedding_report.json")}))


if __name__ == "__main__":
    main()
