from __future__ import annotations

import re

TELEMETRY_RULE_ID = "1000004"
TELEMETRY_DIRECTIVE = """
SecAction "id:1000004,phase:2,pass,log,msg:'CONCEPT_COMPARISON_SCORES',logdata:'total=%{TX.blocking_inbound_anomaly_score};threshold=%{TX.inbound_anomaly_score_threshold}'"
"""


def decision_telemetry(events):
    """Read actual aggregate TX values; never approximate scores by summing alert severity."""
    for event in reversed(events):
        if event["rule_id"] != TELEMETRY_RULE_ID:
            continue
        data = event.get("matched_data") or ""
        match = re.fullmatch(r"total=(\d+(?:\.\d+)?);threshold=(\d+(?:\.\d+)?)", data)
        if match:
            score, threshold = map(float, match.groups())
            if threshold > 0:
                return {
                    "anomaly_score": score,
                    "threshold": threshold,
                    "would_block": score >= threshold,
                }
    return None
