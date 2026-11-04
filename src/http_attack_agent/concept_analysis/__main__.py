from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Independent PCBM-style concept analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Stream CSV into traceable request records")
    prepare.add_argument("--dataset-config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--max-rows", type=int, default=0, help="0=all; sampled runs are exploratory"
    )
    prepare.add_argument("--seed", type=int, default=14)
    replay = commands.add_parser(
        "replay", help="Offline request inspection with real ModSecurity/CRS"
    )
    replay.add_argument("--prepared", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--crs", required=True, help="Local CRS release root")
    replay.add_argument("--concept-config", help="Optional YAML override of the bundled mapping")
    replay.add_argument("--workers", type=int, default=4)
    replay.add_argument("--paranoia", type=int, default=1)
    replay.add_argument("--library", help="Optional libmodsecurity shared-library path")
    encode = commands.add_parser(
        "encode", help="Export embeddings using an existing local checkpoint"
    )
    encode.add_argument("--prepared", required=True)
    encode.add_argument("--checkpoint", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--batch-size", type=int, default=32)
    encode.add_argument("--device", help="Default: CUDA if available, otherwise CPU")
    fit = commands.add_parser(
        "fit", help="Fit train-only concept probes; evaluate held-out requests"
    )
    for name in ("prepared", "annotations", "encoded", "output"):
        fit.add_argument(f"--{name}", required=True)
    fit.add_argument("--seed", type=int, default=14)
    fit.add_argument("--cap", type=int, default=20000)
    fit.add_argument("--permutations", type=int, default=20)
    fit.add_argument("--repeats", type=int, default=3)
    fit.add_argument("--min-support", type=int, default=20)
    fit.add_argument("--conditional", action="store_true")
    analyze = commands.add_parser("analyze", help="TCAV and interventions on the original head")
    for name in ("prepared", "encoded", "bank-dir", "checkpoint"):
        analyze.add_argument(f"--{name}", required=True)
    analyze.add_argument("--max-samples", type=int, default=5000)
    analyze.add_argument("--random-directions", type=int, default=10)
    analyze.add_argument("--seed", type=int, default=14)
    analyze.add_argument("--device")
    explain = commands.add_parser("explain", help="Inspect one existing prepared request")
    for name in (
        "prepared",
        "annotations",
        "encoded",
        "bank-dir",
        "checkpoint",
        "row-id",
        "output",
    ):
        explain.add_argument(f"--{name}", required=True)
    explain.add_argument("--ablate-field", help="Optional whole logical input-field removal")
    explain.add_argument("--device")
    report = commands.add_parser(
        "report", help="English Markdown, CSV tables, and scientific plots"
    )
    report.add_argument("--bank-dir", required=True)
    demo = commands.add_parser("demo", help="Synthetic smoke demo, without model downloads or CRS")
    demo.add_argument("--output", default="runs/concept-analysis-demo")
    check = commands.add_parser(
        "check-waf", help="Verify real CRS SQLi detection and completed phases"
    )
    check.add_argument("--crs", required=True)
    check.add_argument("--library")
    estimate = commands.add_parser("estimate", help="Extrapolate measured pilot stage timings")
    estimate.add_argument(
        "--pilot", required=True, help="Parent of prepared/, waf/, encoded/, bank/"
    )
    estimate.add_argument("--total-rows", type=int, default=907815)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .prepare import prepare

        result = prepare(**args)
    elif command == "replay":
        from .replay import replay

        result = replay(**args)
    elif command == "encode":
        from .encode import encode

        result = encode(**args)
    elif command == "fit":
        from .probes import fit

        result = fit(**args)
        result = {"concepts": len(result["concepts"]), "seconds": result["seconds"]}
    elif command == "analyze":
        from .sensitivity import analyze

        result = analyze(**args)
        result = {"samples": result["samples"]}
    elif command == "explain":
        from .sensitivity import explain_request

        result = explain_request(**args)
        result = {"row_id": result["row_id"], "output": args["output"]}
    elif command == "report":
        from .report import report

        result = report(**args)
    elif command == "demo":
        from .demo import demo

        result = demo(**args)
    elif command == "check-waf":
        from .modsecurity import ModSecurity, check_engine

        engine = ModSecurity(**args)
        try:
            check_engine(engine)
            result = {"status": "pass", "engine": engine.version, "crs_hash": engine.crs_hash}
        finally:
            engine.close()
    else:
        from .common import read_json

        root = Path(args["pilot"])
        result = {
            "total_rows": args["total_rows"],
            "estimated_minutes": {},
            "note": "Preparation includes a full label scan and hash pass. Only WAF/encode "
            "are extrapolated linearly; probe cost depends on cap and repetitions, "
            "so its pilot duration is reported without linear scaling.",
        }
        for stage in ("prepared", "waf", "encoded"):
            path = root / stage / "manifest.json"
            if path.is_file():
                data = read_json(path)
                if stage == "prepared":
                    result["preparation_pilot_minutes"] = data["seconds"] / 60
                else:
                    rate = data["rows_per_second"]
                    result["estimated_minutes"][stage] = (
                        args["total_rows"] / rate / 60 if rate > 0 else None
                    )
        bank = root / "bank" / "probe_report.json"
        if bank.is_file():
            result["probe_pilot_minutes"] = read_json(bank)["seconds"] / 60
        sensitivity = root / "bank" / "sensitivity_report.json"
        if sensitivity.is_file() and "seconds" in read_json(sensitivity):
            result["head_analysis_pilot_minutes"] = read_json(sensitivity)["seconds"] / 60
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
