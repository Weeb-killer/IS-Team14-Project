from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

BRACKET_FIELD = re.compile(r'\[([A-Za-z_]+) "((?:\\.|[^"\\])*)"\]')
MATCHED_VARIABLE = re.compile(r"against variable\s+[`']([^`']+)[`']", re.IGNORECASE)


@dataclass(frozen=True)
class WAFEvent:
    request_id: str | None
    transaction_id: str | None
    rule_id: str
    message: str | None
    tags: list[str]
    matched_variable: str | None
    matched_data: str | None
    severity: str | None
    uri: str | None
    raw_line: str


def _unescape_log_value(value: str) -> str:
    return value.replace(r'\"', '"').replace(r"\\", "\\")


def parse_alert_line(line: str) -> WAFEvent | None:
    """Parse common ModSecurity/Coraza error-log alerts.

    Full multipart audit logs should first be converted to one alert per line by the
    chosen WAF integration. Keeping this parser strict prevents fabricated matches.
    """

    fields: dict[str, list[str]] = {}
    for key, value in BRACKET_FIELD.findall(line):
        fields.setdefault(key.lower(), []).append(_unescape_log_value(value))
    rule_ids = fields.get("id")
    if not rule_ids:
        return None
    variable_match = MATCHED_VARIABLE.search(line)
    return WAFEvent(
        request_id=(fields.get("request_id") or [None])[0],
        transaction_id=(fields.get("unique_id") or [None])[0],
        rule_id=rule_ids[0],
        message=(fields.get("msg") or [None])[0],
        tags=fields.get("tag", []),
        matched_variable=variable_match.group(1) if variable_match else None,
        matched_data=(fields.get("data") or [None])[0],
        severity=(fields.get("severity") or [None])[0],
        uri=(fields.get("uri") or [None])[0],
        raw_line=line.rstrip("\r\n"),
    )


def load_concept_config(path: str | Path) -> tuple[dict[str, dict[str, set[str]]], set[str]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required to read concept configuration") from exc
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    concepts = {}
    for name, values in raw.get("concepts", {}).items():
        concepts[name] = {
            "tags": set(values.get("any_tags", [])),
            "rule_ids": {str(value) for value in values.get("any_rule_ids", [])},
            "variable_prefixes": set(values.get("variable_prefixes", [])),
        }
    excluded = {str(value) for value in raw.get("exclude_rule_ids", [])}
    return concepts, excluded


def event_concepts(
    event: WAFEvent, concepts: dict[str, dict[str, set[str]]]
) -> list[str]:
    tags = set(event.tags)
    matches: list[str] = []
    for name, matcher in concepts.items():
        tag_match = bool(tags.intersection(matcher["tags"]))
        rule_match = event.rule_id in matcher["rule_ids"]
        variable_match = bool(
            event.matched_variable
            and any(
                event.matched_variable.startswith(prefix)
                for prefix in matcher["variable_prefixes"]
            )
        )
        if tag_match or rule_match or variable_match:
            matches.append(name)
    return sorted(matches)


def iter_events(
    lines: Iterable[str], concept_config: str | Path | None = None
) -> Iterable[dict]:
    concepts: dict[str, dict[str, set[str]]] = {}
    excluded: set[str] = set()
    if concept_config:
        concepts, excluded = load_concept_config(concept_config)
    for line in lines:
        event = parse_alert_line(line)
        if event is None or event.rule_id in excluded:
            continue
        payload = asdict(event)
        payload["concepts"] = event_concepts(event, concepts)
        yield payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse OWASP CRS alert lines")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concept-config", default="configs/concepts.yaml")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="replace") as source:
        events = list(iter_events(source, args.concept_config))
    with output_path.open("w", encoding="utf-8") as destination:
        for event in events:
            destination.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps({"events": len(events), "output": str(output_path)}))


if __name__ == "__main__":
    main()
