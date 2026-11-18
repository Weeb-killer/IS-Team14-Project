"""Convert stored evidence to plain, localized template data without changing it.

All returned strings remain ordinary text. Templates must keep Django autoescaping
enabled because unknown labels, rule names and error codes may originate in input.
"""

import math

LABEL_NAMES = {
    "protocol_manipulation": "Protocol manipulation",
    "code_injection": "Code injection",
    "os_command_injection": "Operating system command injection",
    "path_traversal": "Path traversal",
    "sql_injection": "SQL injection",
    "dictionary_password_attack": "Dictionary password attack",
    "vulnerable_software_scanning": "Vulnerable software scanning",
    "input_data_manipulation": "Input data manipulation",
    "command_injection": "Command injection",
    "http_verb_tampering": "HTTP verb tampering",
    "fake_source_data": "Source data spoofing",
    "http_response_splitting": "HTTP response splitting",
    "http_request_smuggling": "HTTP request smuggling",
}

CONCEPT_NAMES = {
    "waf_sqli": "SQL injection features",
    "waf_xss": "Cross-site scripting (XSS) features",
    "waf_rce": "Remote code execution features",
    "waf_lfi": "Local file inclusion features",
    "waf_rfi": "Remote file inclusion features",
    "waf_protocol": "HTTP protocol anomaly features",
    "waf_libinjection_sqli": "libinjection SQL injection features",
    "waf_libinjection_xss": "libinjection cross-site scripting features",
    "waf_sqli_database_names": "SQL injection database name features",
    "waf_sqli_sleep_benchmark": "SQL delay and benchmark function features",
}

LOCATION_NAMES = {
    "uri": "Request path and query string",
    "args": "Request parameters",
    "headers": "Request headers",
    "cookies": "Cookie",
    "body": "Request body",
}

ERROR_NAMES = {
    "not_configured": "Not configured. No inspection result is available.",
    "not_inspected": "Inspection has not run, or the request was rejected before inspection.",
    "busy": "The inspection worker was busy. No result is available for this request.",
    "timeout": "Inspection timed out. No result is available for this request.",
    "initialization_timeout": "Loading the model or rule engine timed out.",
    "initialization_failed": "The model or rule engine failed to initialize. Check the gateway logs.",
    "inspection_failed": "Inspection failed. Check the gateway logs.",
    "worker_unavailable": "The inspection worker is unavailable. Check the gateway logs.",
    "non_finite_probe_scores": "The concept probes returned invalid numeric scores.",
    "score_telemetry_unavailable": "CRS did not provide complete anomaly scores. No blocking decision is available.",
    "incomplete_inspection": "CRS did not complete inspection of this request.",
}

MISSING = "Not provided"


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _text(value, default=MISSING):
    return value if isinstance(value, str) and value.strip() else default


def _number(value):
    if type(value) not in {int, float}:
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _number_label(value):
    number = _number(value)
    return f"{number:.4f}" if number is not None else MISSING


def _probability(value):
    number = _number(value)
    return number if number is not None and 0 <= number <= 1 else None


def _probability_label(value):
    number = _probability(value)
    return f"{number:.2%}" if number is not None else MISSING


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _duration(value):
    number = _number(value)
    return f"{number:,.0f} ms" if number is not None and number >= 0 else "Not recorded"


def _error_label(value):
    code = _text(value, "")
    return ERROR_NAMES.get(code, f"Unrecognized status reason: {code}") if code else ""


def _engine(value, *, crs=False):
    source = _mapping(value)
    complete = source.get("status") == "complete" and type(source.get("would_block")) is bool
    would_block = source["would_block"] if complete else None
    if would_block is True:
        decision = (
            "Anomaly score reached the blocking threshold"
            if crs
            else "Attack threshold reached; the model recommends blocking"
        )
    elif would_block is False:
        decision = (
            "Anomaly score below the blocking threshold" if crs else "No attack threshold reached"
        )
    else:
        decision = "No CRS decision available" if crs else "No model decision available"
    error = _error_label(source.get("error"))
    if not complete and not error:
        error = (
            "The record has no explicit inspection decision."
            if source
            else "No inspection result was recorded for this request."
        )
    return {
        "status": "complete" if complete else "unknown",
        "status_label": "Inspection complete" if complete else "No result available",
        "error_label": error,
        "would_block": would_block,
        "decision_label": decision,
        "tone": "danger" if would_block is True else "success" if complete else "muted",
        "duration_label": _duration(source.get("duration_ms")),
    }


def _active(value, score, threshold, complete):
    """Only display a decision supported by finite scores and an explicit flag."""
    if (
        not complete
        or type(value) is not bool
        or score is None
        or threshold is None
        or value != (score >= threshold)
    ):
        return None
    return value


def _rows(source, *, complete, concepts=False):
    scores = source.get("scores")
    if not isinstance(scores, list):
        return []
    rows = []
    for value in scores:
        if not isinstance(value, dict):
            continue
        name = _text(
            value.get("name" if concepts else "label"),
            "Unknown concept" if concepts else "Unknown label",
        )
        number = _number if concepts else _probability
        active = _active(
            value.get("active"),
            number(value.get("score" if concepts else "probability")),
            number(value.get("threshold")),
            complete,
        )
        row = {
            "active": active,
            "threshold_label": (_number_label if concepts else _probability_label)(
                value.get("threshold")
            ),
            "tone": "danger" if active is True else "success" if active is False else "muted",
            "result_label": ("Active" if concepts else "Threshold reached")
            if active is True
            else ("Inactive" if concepts else "Below threshold")
            if active is False
            else "Cannot determine",
        }
        if concepts:
            row.update(
                name=name,
                name_label=concept_name(name),
                score_label=_number_label(value.get("score")),
            )
        else:
            row.update(
                label=name,
                label_name=LABEL_NAMES.get(name, name),
                probability_label=_probability_label(value.get("probability")),
            )
        rows.append(row)
    return rows


def concept_name(name):
    """Translate declared concepts, collection locations and joint concepts."""
    if name.startswith("location:"):
        location = name.partition(":")[2]
        return f"Location concept: {LOCATION_NAMES.get(location, location)}"
    if "@" in name:
        concept, _, location = name.partition("@")
        return f"{CONCEPT_NAMES.get(concept, concept)} / {LOCATION_NAMES.get(location, location)}"
    return CONCEPT_NAMES.get(name, name)


def _identity(value, *, crs=False):
    source = _mapping(value)
    labels = (
        {
            "engine": "Rule engine version",
            "rules_sha256": "Rule files SHA-256",
            "paranoia": "CRS paranoia level",
        }
        if crs
        else {
            "model_id": "Model identifier",
            "checkpoint_sha256": "Checkpoint SHA-256",
            "probe_bank_sha256": "Concept probe bank SHA-256",
            "max_length": "Maximum input length (tokens)",
            "dataset_rows": "Dataset samples",
            "device": "Inference device",
        }
    )
    result = []
    for key, value in source.items():
        if isinstance(value, str):
            text = value or "Not recorded"
            if key == "device" and text == "cpu":
                text = "CPU"
        elif type(value) in {int, float} and _number(value) is not None:
            text = f"{value:,}" if type(value) is int else str(value)
        elif value is None:
            text = "Not recorded"
        else:
            continue
        result.append({"label": labels.get(key, key), "value": text})
    return result


def evidence_context(evidence, *, mode="observe"):
    """Return a fixed view schema for new, old and incomplete audit evidence."""
    source = _mapping(evidence)
    model_source, crs_source = (_mapping(source.get(name)) for name in ("model", "crs"))
    model = _engine(model_source)
    model["scores"] = _rows(model_source, complete=model["status"] == "complete")

    concept_source = _mapping(model_source.get("concepts"))
    concept_rows = _rows(
        concept_source, complete=concept_source.get("status") == "complete", concepts=True
    )
    concept_complete = concept_source.get("status") == "complete" and bool(concept_rows)
    concepts = {
        "status": "complete" if concept_complete else "unknown",
        "status_label": "Probe scoring complete"
        if concept_complete
        else "No probe result available",
        "error_label": _error_label(concept_source.get("error")),
        "valid_count": _count(concept_source.get("valid_count")),
        "total_count": _count(concept_source.get("total_count")),
        "active_count": sum(row["active"] is True for row in concept_rows)
        if concept_complete
        else None,
        "scores": concept_rows,
        "note": "Concept scores are linear probe responses, not probabilities or proof that a concept caused the model decision.",
    }
    if not concept_complete and not concepts["error_label"]:
        concepts["error_label"] = "No usable concept probe scores were recorded for this request."

    crs = _engine(crs_source, crs=True)
    hit_count = _count(crs_source.get("hit_count"))
    crs.update(
        anomaly_score_label=_number_label(crs_source.get("anomaly_score")),
        threshold_label=_number_label(crs_source.get("threshold")),
        hit_count_label=str(hit_count) if hit_count is not None else MISSING,
        hits=[
            {
                "rule_id": _text(hit.get("rule_id")),
                "name": _text(hit.get("name")),
                "field": _text(hit.get("field")),
            }
            for hit in crs_source.get("hits", [])
            if isinstance(hit, dict)
        ]
        if isinstance(crs_source.get("hits"), list)
        else [],
    )

    if model["status"] == crs["status"] == "complete":
        comparison = (
            "The model and CRS agree on whether to block."
            if model["would_block"] == crs["would_block"]
            else "The model and CRS disagree on whether to block."
        )
    else:
        comparison = "Both the model and CRS must complete inspection before their decisions can be compared."
    if model["would_block"] is True:
        title = "The model found attack labels at or above their thresholds"
        active_labels = [row["label_name"] for row in model["scores"] if row["active"] is True]
        text = (
            "Threshold reached: " + ", ".join(active_labels) + ". "
            if active_labels
            else model["decision_label"] + ". "
        )
    elif model["would_block"] is False:
        title = "The model found no attack labels at or above their thresholds"
        text = "No saved label threshold was reached. This does not guarantee that the request is safe. "
    else:
        title = "No model decision is available for this request"
        text = model["error_label"] + " "
    text += (
        "Observe mode records inspection decisions. See the request action above for the actual outcome."
        if mode == "observe"
        else "See the request action above for the actual outcome."
    )
    return {
        "model": model,
        "concepts": concepts,
        "crs": crs,
        "model_identity": _identity(model_source.get("identity")),
        "crs_identity": _identity(crs_source.get("identity"), crs=True),
        "summary": {"title": title, "text": text, "tone": model["tone"], "comparison": comparison},
    }
