from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Held-out CRS-versus-SecBERT baseline comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Compare independent decisions against dataset labels")
    for name in ("prepared", "annotations", "encoded", "checkpoint", "output"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--mapping", help="Optional explicit YAML label mapping override")
    run.add_argument("--target-fpr", type=float, default=0.01)
    run.add_argument("--bootstrap", type=int, default=500)
    run.add_argument("--seed", type=int, default=14)
    run.add_argument(
        "--examples", type=int, default=50, help="Seeded examples per disagreement type"
    )
    demo = commands.add_parser("demo", help="Synthetic fixture, without CRS or model downloads")
    demo.add_argument("--output", default="runs/crs-secbert-comparison-demo")
    report = commands.add_parser("report", help="Rebuild report from comparison.json")
    report.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "run":
        from .evaluate import compare

        result = compare(**args)
    elif command == "demo":
        from .demo import demo

        result = demo(**args)
    else:
        from .report import render_report

        render_report(**args)
        result = {"output": args["output"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
