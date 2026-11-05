"""Offline libModSecurity 3 C-API connector. No HTTP server or backend is used."""

from __future__ import annotations

import ctypes as ct
import hashlib
from ctypes.util import find_library
from pathlib import Path

from .comparison.mapping import catalog_rules
from .comparison.telemetry import TELEMETRY_DIRECTIVE, decision_telemetry
from .events import parse_event

HEADERS = {
    "content_type": "Content-Type",
    "host": "Host",
    "user_agent": "User-Agent",
    "referer": "Referer",
    "origin": "Origin",
    "cookie": "Cookie",
    "accept": "Accept",
    "accept_language": "Accept-Language",
    "accept_encoding": "Accept-Encoding",
    "do_not_track": "DNT",
    "connection": "Connection",
}


def check_engine(engine):
    result = engine.inspect(
        {
            "method": "GET",
            "request_target": "/?id=1%20OR%201=1--",
            "protocol": "HTTP/1.1",
            "host": "localhost",
            "user_agent": "ConceptAnalysisCheck",
        }
    )
    if (
        result["status"] != "complete"
        or not any("attack-sqli" in event["tags"] for event in result["events"])
        or not result.get("decision")
        or not result["decision"]["would_block"]
    ):
        raise RuntimeError(f"Real CRS SQLi/logging/phase check failed: {result}")


def rules_configuration(crs, paranoia=1, max_body=4 * 1024 * 1024):
    root = Path(crs).resolve()
    setup = root / "crs-setup.conf.example"
    paths = sorted((root / "rules").glob("REQUEST-*.conf"))
    if not setup.is_file() or not paths:
        raise FileNotFoundError("Expected crs-setup.conf.example and rules/REQUEST-*.conf.")
    if not 1 <= paranoia <= 4:
        raise ValueError("Paranoia level must be 1..4.")
    text = f'''SecRuleEngine DetectionOnly
SecRequestBodyAccess On
SecRequestBodyLimit {max_body}
SecRequestBodyNoFilesLimit {max_body}
SecRequestBodyLimitAction Reject
SecResponseBodyAccess Off
SecAuditEngine Off
SecDebugLogLevel 0
SecTmpDir /tmp
SecDataDir /tmp
Include "{setup.as_posix()}"
SecAction "id:1000000,phase:1,pass,nolog,setvar:tx.blocking_paranoia_level={paranoia},setvar:tx.detection_paranoia_level={paranoia},setvar:tx.early_blocking=0"
SecRule REQUEST_HEADERS:Content-Type "(?i)^application/json" "id:1000010,phase:1,pass,nolog,ctl:requestBodyProcessor=JSON"
SecRule REQUEST_HEADERS:Content-Type "(?i)(?:application|text)/xml" "id:1000011,phase:1,pass,nolog,ctl:requestBodyProcessor=XML"
'''
    text += "\n".join(f'Include "{p.as_posix()}"' for p in paths)
    # End-of-phase sentinels also verify that the log callback is operational.
    # Older libModSecurity 3 releases expose the PCRE limit flag only through TX.
    text += """
SecAction "id:1000001,phase:1,pass,log,msg:'CONCEPT_PHASE_1_COMPLETE'"
SecAction "id:1000002,phase:2,pass,log,msg:'CONCEPT_PHASE_2_COMPLETE'"
SecRule REQBODY_ERROR|REQBODY_PROCESSOR_ERROR|INBOUND_DATA_ERROR|TX:MSC_PCRE_LIMITS_EXCEEDED "@gt 0" "id:1000003,phase:2,t:none,pass,log,msg:'CONCEPT_INCOMPLETE_INSPECTION'"
"""
    text += TELEMETRY_DIRECTIVE
    checksum = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        checksum.update(path.relative_to(root).as_posix().encode())
        checksum.update(path.read_bytes())
    return text, checksum.hexdigest()


class ModSecurity:
    def __init__(self, crs, paranoia=1, library=None, max_body=4 * 1024 * 1024):
        self.max_body = max_body
        self.configuration, self.crs_hash = rules_configuration(crs, paranoia, max_body)
        self.rule_metadata = {rule["rule_id"]: rule for rule in catalog_rules(crs, paranoia)}
        library = library or find_library("modsecurity")
        if not library:
            raise RuntimeError(
                "libModSecurity 3 was not found. Use the included Linux Dockerfile "
                "or install libmodsecurity3. This is not a pip dependency."
            )
        self.lib = ct.CDLL(library)
        p, s, i, z = ct.c_void_p, ct.c_char_p, ct.c_int, ct.c_size_t
        signatures = {
            "msc_init": ([], p),
            "msc_who_am_i": ([p], s),
            "msc_create_rules_set": ([], p),
            "msc_rules_add": ([p, s, ct.POINTER(s)], i),
            "msc_new_transaction": ([p, p, p], p),
            "msc_process_connection": ([p, s, i, s, i], i),
            "msc_process_uri": ([p, s, s, s], i),
            "msc_add_n_request_header": ([p, s, z, s, z], i),
            "msc_process_request_headers": ([p], i),
            "msc_append_request_body": ([p, s, z], i),
            "msc_process_request_body": ([p], i),
            "msc_process_logging": ([p], i),
            "msc_transaction_cleanup": ([p], None),
            "msc_rules_cleanup": ([p], i),
            "msc_cleanup": ([p], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.logs = []
        callback_type = ct.CFUNCTYPE(None, p, p)

        def log_callback(_context, message):
            if message:
                self.logs.append(ct.string_at(message).decode("utf-8", errors="replace"))

        self.callback = callback_type(log_callback)  # keep alive until engine cleanup
        self.lib.msc_set_log_cb.argtypes = [p, callback_type]
        self.lib.msc_set_log_cb.restype = None
        self.engine = self.lib.msc_init()
        self.rules = self.lib.msc_create_rules_set()
        if not self.engine or not self.rules:
            raise RuntimeError("Could not create ModSecurity engine/rules.")
        self.lib.msc_set_log_cb(self.engine, self.callback)
        error = s()
        if self.lib.msc_rules_add(self.rules, self.configuration.encode(), ct.byref(error)) < 0:
            message = error.value.decode(errors="replace") if error.value else "unknown"
            self.close()
            raise RuntimeError(f"CRS configuration failed to load: {message}")
        self.version = self.lib.msc_who_am_i(self.engine).decode()

    def inspect(self, fields):
        self.logs.clear()
        body = fields.get("body", "").encode("utf-8")
        if len(body) > self.max_body:
            return {"status": "unknown", "error": "body_limit", "events": []}
        # URI/method/version are C strings. Do not silently truncate embedded NULs.
        values = [fields.get(k, "") for k in ("request_target", "method", "protocol")]
        if any("\x00" in value for value in values) or not all(values):
            return {
                "status": "unknown",
                "error": "missing_or_unrepresentable_request_line",
                "events": [],
            }
        transaction = self.lib.msc_new_transaction(self.engine, self.rules, None)
        if not transaction:
            raise RuntimeError("Could not allocate a ModSecurity transaction.")
        synthetic_headers = []
        try:

            def call(name, *args):
                if getattr(self.lib, name)(transaction, *args) < 0:
                    raise RuntimeError(f"{name} failed")

            call("msc_process_connection", b"127.0.0.1", 12345, b"127.0.0.1", 80)
            uri, method, protocol = (value.encode() for value in values)
            version = protocol.removeprefix(b"HTTP/")
            call("msc_process_uri", uri, method, version)
            for logical, header in HEADERS.items():
                value = fields.get(logical, "")
                if value:
                    key, value = header.encode(), value.encode("utf-8")
                    call("msc_add_n_request_header", key, len(key), value, len(value))
            if body:
                key, value = b"Content-Length", str(len(body)).encode()
                call("msc_add_n_request_header", key, len(key), value, len(value))
                synthetic_headers.append("Content-Length")
            call("msc_process_request_headers")
            if body:
                call("msc_append_request_body", body, len(body))
            call("msc_process_request_body")
            call("msc_process_logging")
            parsed = [
                e for line in self.logs if (e := parse_event(line, self.rule_metadata)) is not None
            ]
            rule_ids = {e["rule_id"] for e in parsed}
            complete = {"1000001", "1000002"} <= rule_ids and "1000003" not in rule_ids
            events = [e for e in parsed if not e["rule_id"].startswith("10000")]
            return {
                "status": "complete" if complete else "unknown",
                "error": None if complete else "incomplete_phase_or_processing_error",
                "events": events,
                "synthetic_headers": synthetic_headers,
                "decision": decision_telemetry(parsed) if complete else None,
            }
        except (RuntimeError, ValueError, UnicodeError) as exc:
            return {"status": "unknown", "error": str(exc), "events": []}
        finally:
            self.lib.msc_transaction_cleanup(transaction)

    def close(self):
        if getattr(self, "rules", None):
            self.lib.msc_rules_cleanup(self.rules)
            self.rules = None
        if getattr(self, "engine", None):
            self.lib.msc_cleanup(self.engine)
            self.engine = None
