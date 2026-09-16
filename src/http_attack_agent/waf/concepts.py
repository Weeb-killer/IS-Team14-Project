from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def aggregate_events(events: Iterable[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Aggregate parsed WAF events into one binary concept row per request."""

    grouped: dict[str, dict[str, Any]] = {}
    concept_names: set[str] = set()
    rules: dict[str, set[str]] = defaultdict(set)
    for event in events:
        request_id = event.get("request_id") or event.get("transaction_id")
        if not request_id:
            continue
        row = grouped.setdefault(str(request_id), {"request_id": str(request_id)})
        for concept in event.get("concepts", []):
            concept_names.add(concept)
            row[concept] = 1
        if event.get("rule_id"):
            rules[str(request_id)].add(str(event["rule_id"]))

    ordered_concepts = sorted(concept_names)
    rows: list[dict[str, Any]] = []
    for request_id, row in grouped.items():
        for concept in ordered_concepts:
            row.setdefault(concept, 0)
        row["waf_rule_ids"] = json.dumps(sorted(rules[request_id]))
        rows.append(row)
    rows.sort(key=lambda row: row["request_id"])
    return ordered_concepts, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate WAF events into concept columns")
    parser.add_argument("--events", required=True, help="JSONL from http-attack-parse-waf")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.events).open("r", encoding="utf-8") as source:
        events = (json.loads(line) for line in source if line.strip())
        concepts, rows = aggregate_events(events)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["request_id", *concepts, "waf_rule_ids"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"requests": len(rows), "concepts": concepts, "output": str(output_path)}))


if __name__ == "__main__":
    main()
