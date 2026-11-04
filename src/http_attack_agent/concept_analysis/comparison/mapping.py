from __future__ import annotations

import re
from pathlib import Path

import yaml

# Match terminal CAPEC IDs only. CAPEC hierarchy ancestors are not extra attack labels.
CAPEC_IDS = {
    "protocol_manipulation": 272,
    "code_injection": 242,
    "os_command_injection": 88,
    "path_traversal": 126,
    "sql_injection": 66,
    "dictionary_password_attack": 16,
    "vulnerable_software_scanning": 310,
    "input_data_manipulation": 153,
    "command_injection": 248,
    "http_verb_tampering": 274,
    "fake_source_data": 194,
    "http_response_splitting": 34,
    "http_request_smuggling": 33,
}


def capec_leaves(tags):
    result = set()
    for tag in tags:
        if re.fullmatch(r"capec/(?:\d+/)*\d+", tag, flags=re.IGNORECASE):
            result.add(int(tag.rsplit("/", 1)[1]))
    return result


def load_mapping(labels, path=None):
    if path:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        mapping = config["labels"]
        if set(mapping) - set(labels):
            raise ValueError("Mapping contains labels absent from this dataset.")
        return {label: mapping.get(label, {}) for label in labels}
    return {
        label: {"capec_leaf": CAPEC_IDS[label]} if label in CAPEC_IDS else {} for label in labels
    }


def matches(rule, definition):
    tags = rule.get("tags", [])
    return bool(
        definition.get("capec_leaf") in capec_leaves(tags)
        or str(rule["rule_id"]) in {str(x) for x in definition.get("rule_ids", [])}
        or set(tags).intersection(definition.get("tags", []))
    )


def catalog_rules(crs, paranoia=1):
    """Inventory declared metadata; per-transaction execution still depends on CRS controls."""
    result = []
    for path in sorted((Path(crs) / "rules").glob("REQUEST-*.conf")):
        logical = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if not logical and (not line.strip() or line.lstrip().startswith("#")):
                continue
            continuation = line.rstrip().endswith("\\")
            logical += line.rstrip()[:-1] if continuation else line
            if continuation:
                continue
            rule_id = re.search(r'["\']id:\s*(\d+)\s*,', logical)
            if rule_id:
                actions = logical[rule_id.start() :]
                tags = re.findall(r"(?:^|,)\s*tag:'([^']+)'", actions)
                levels = [int(t.split("/")[-1]) for t in tags if t.startswith("paranoia-level/")]
                phase = re.search(r"\bphase:\s*(\d+)", actions)
                if (not levels or max(levels) <= paranoia) and phase and int(phase[1]) in (1, 2):
                    result.append(
                        {
                            "rule_id": rule_id[1],
                            "tags": tags,
                            "declared_paranoia_level": max(levels, default=1),
                            "phase": int(phase[1]),
                            "file": path.name,
                        }
                    )
            logical = ""
    return result
