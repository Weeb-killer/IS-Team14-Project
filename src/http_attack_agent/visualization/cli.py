from __future__ import annotations

import argparse
import json

from ..data import DatasetConfig
from .profile import generate_dataset_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a static, aggregate-only security profile for an HTTP dataset."
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.srbh2020.yaml",
        help="Path to the dataset YAML configuration.",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset-profile",
        help="Directory for the report, aggregate tables, and figures.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="CSV rows processed per chunk (default: 50000).",
    )
    parser.add_argument(
        "--scatter-sample-size",
        type=int,
        default=10_000,
        help="Maximum stratified sample used by the length plot (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument(
        "--image-format",
        choices=["png", "pdf"],
        default="png",
        help="Figure format (default: png).",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")
    if args.scatter_sample_size <= 1:
        parser.error("--scatter-sample-size must be greater than one")

    result = generate_dataset_profile(
        DatasetConfig.from_yaml(args.dataset_config),
        args.output,
        chunk_size=args.chunk_size,
        scatter_sample_size=args.scatter_sample_size,
        seed=args.seed,
        image_format=args.image_format,
    )
    print(json.dumps({"output": str(result), "figures": 6}))


if __name__ == "__main__":
    main()
