from __future__ import annotations

import re
from dataclasses import asdict
from urllib.parse import parse_qs, urlsplit

import yaml

from ..waf.audit import parse_alert_line


def load_concepts(path):
    if path is None:
        from copy import deepcopy

        from .defaults import DEFAULT_CONCEPTS

        config = deepcopy(DEFAULT_CONCEPTS)
    else:
        with open(path, encoding="utf-8") as source:
            config = yaml.safe_load(source)
    names = list(config["concepts"])
    names += [f"location:{x}" for x in config["locations"]]
    names += [f"{c}@{loc}" for c, loc in config.get("joint_concepts", [])]
    if len(set(names)) != len(names):
        raise ValueError("Concept names must be unique.")
    return config, names


def parse_event(line, rule_metadata=None):
    event = parse_alert_line(line)
    if event is None:
        return None
    result = asdict(event)
    if not result["matched_variable"] and result["matched_data"]:
        # Common native ModSecurity logdata format, also used by libinjection.
        match = re.search(r"found within ([A-Za-z_]+(?::[^\s:]+)?):", result["matched_data"])
        if match:
            result["matched_variable"] = match.group(1)
    # Some native libinjection/multiMatch alerts omit declared rule tags.
    # Enrich only an actual logged hit, using the exact loaded CRS rule ID.
    metadata = (rule_metadata or {}).get(result["rule_id"])
    if metadata:
        logged_tags = result["tags"]
        added = [tag for tag in metadata["tags"] if tag not in logged_tags]
        if added:
            result["logged_tags"] = logged_tags.copy()
            result["tags_from_rule_metadata"] = added
            result["rule_metadata_file"] = metadata["file"]
            result["tags"] = list(dict.fromkeys([*logged_tags, *added]))
    return result


def locate(variable, fields):
    """Resolve WAF collection separately from original CSV field provenance."""
    variable = variable or ""
    collection, _, key = variable.partition(":")
    source = None
    if collection.startswith("ARGS"):
        location = "args"
        query = parse_qs(urlsplit(fields.get("request_target", "")).query, keep_blank_values=True)
        body = (
            parse_qs(fields.get("body", ""), keep_blank_values=True)
            if fields.get("content_type", "").split(";")[0].lower()
            == "application/x-www-form-urlencoded"
            else {}
        )
        if collection.startswith("ARGS_GET") or key in query and key not in body:
            source = "request_target"
        elif collection.startswith("ARGS_POST") or key in body and key not in query:
            source = "body"
    elif collection.startswith("REQUEST_COOKIES"):
        location, source = "cookies", "cookie"
    elif collection.startswith("REQUEST_HEADERS"):
        location = "headers"
        aliases = {
            "user-agent": "user_agent",
            "content-type": "content_type",
            "accept-language": "accept_language",
            "accept-encoding": "accept_encoding",
            "dnt": "do_not_track",
        }
        candidate = aliases.get(key.lower(), key.lower().replace("-", "_"))
        source = candidate if candidate in fields else None
    elif collection in {
        "REQUEST_URI",
        "REQUEST_URI_RAW",
        "REQUEST_FILENAME",
        "REQUEST_BASENAME",
        "QUERY_STRING",
    }:
        location, source = "uri", "request_target"
    elif collection in {"REQUEST_BODY", "XML"}:
        location, source = "body", "body"
    else:
        location = None
    return {
        "location": location,
        "field": source,
        "variable": variable,
        "span_status": "not_proven",
    }


def annotate_events(events, fields, config):
    active = set()
    annotated = []
    for event in events:
        rule_id = str(event["rule_id"])
        if rule_id in config.get("exclude_rule_ids", []):
            continue
        tags = set(event.get("tags", []))
        matched = [
            name
            for name, matcher in config["concepts"].items()
            if tags.intersection(matcher.get("tags", [])) or rule_id in matcher.get("rule_ids", [])
        ]
        position = locate(event.get("matched_variable"), fields)
        location = position["location"]
        # Internal TX counters do not become raw-input location concepts.
        active.update(matched)
        if location in config["locations"]:
            active.add(f"location:{location}")
        for concept, loc in config.get("joint_concepts", []):
            if concept in matched and location == loc:
                active.add(f"{concept}@{loc}")
        annotated.append({**event, "concepts": matched, "position_info": position})
    return active, annotated
