from __future__ import annotations

from http_attack_agent.concept_analysis.comparison.telemetry import decision_telemetry
from http_attack_agent.concept_analysis.events import parse_event
from http_attack_agent.concept_analysis.modsecurity import ModSecurity

INTERNAL_RULES = {"1000000", "1000001", "1000002", "1000003", "1000004", "1000010", "1000011"}
EVENT_LIMIT = 200


class LiveCRS(ModSecurity):
    """Use the training project's CRS config with the live HTTP request bytes."""

    def __init__(self, crs, max_body, paranoia):
        super().__init__(crs, paranoia=paranoia, max_body=max_body)
        self.identity = {
            "engine": self.version,
            "rules_sha256": self.crs_hash,
            "paranoia": paranoia,
        }

    def inspect(self, request):
        self.logs.clear()
        transaction = self.lib.msc_new_transaction(self.engine, self.rules, None)
        if not transaction:
            raise RuntimeError("Could not allocate a ModSecurity transaction.")

        def call(name, *args):
            if getattr(self.lib, name)(transaction, *args) < 0:
                raise RuntimeError(f"{name} failed")

        try:
            call(
                "msc_process_connection",
                request.peer[0].encode(),
                request.peer[1],
                request.local[0].encode(),
                request.local[1],
            )
            call(
                "msc_process_uri",
                request.target,
                request.method,
                request.version.removeprefix("HTTP/").encode("ascii"),
            )
            for name, value in request.headers:
                call("msc_add_n_request_header", name, len(name), value, len(value))
            call("msc_process_request_headers")
            if request.body:
                call("msc_append_request_body", request.body, len(request.body))
            call("msc_process_request_body")
            call("msc_process_logging")
            parsed = [
                event
                for line in self.logs
                if (event := parse_event(line, self.rule_metadata)) is not None
            ]
        finally:
            self.lib.msc_transaction_cleanup(transaction)
            self.logs.clear()

        ids = {event["rule_id"] for event in parsed}
        phases_complete = {"1000001", "1000002"} <= ids and "1000003" not in ids
        decision = decision_telemetry(parsed) if phases_complete else None
        events = [event for event in parsed if event["rule_id"] not in INTERNAL_RULES]
        return {
            "status": "complete" if decision is not None else "unknown",
            "error": None
            if decision is not None
            else ("score_telemetry_unavailable" if phases_complete else "incomplete_inspection"),
            "would_block": decision["would_block"] if decision else None,
            "anomaly_score": decision["anomaly_score"] if decision else None,
            "threshold": decision["threshold"] if decision else None,
            "matched": bool(events) if decision is not None or events else None,
            "hit_count": len(events),
            "hits": [
                {
                    "rule_id": e["rule_id"],
                    "name": (e["message"] or "")[:240],
                    "field": (e["matched_variable"] or "")[:180] or None,
                }
                for e in events[:EVENT_LIMIT]
            ],
        }
