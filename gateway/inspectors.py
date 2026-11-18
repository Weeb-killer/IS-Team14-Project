"""Persistent, serialized workers with killable inference timeouts (Windows spawn)."""

import atexit
import importlib
import json
import logging
import multiprocessing
import threading
import time
from functools import lru_cache
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)


def unknown(error):
    return {"status": "unknown", "error": error, "would_block": None}


def _worker(connection, factory, arguments):
    try:
        module, name = factory.rsplit(".", 1)
        inspector = getattr(importlib.import_module(module), name)(*arguments)
        connection.send({"ready": True, "identity": inspector.identity})
    except Exception:
        logger.exception("Inspector initialization failed")
        connection.close()
        return
    try:
        while True:
            request = connection.recv()
            try:
                result = inspector.inspect(request)
                if not isinstance(result, dict) or result.get("status") not in {
                    "complete",
                    "unknown",
                }:
                    raise ValueError("Invalid inspector result")
                if result["status"] == "complete" and type(result.get("would_block")) is not bool:
                    raise ValueError("Missing inspector decision")
                # Reject non-finite/non-serializable evidence before it reaches JSONField.
                json.dumps(result, allow_nan=False)
            except Exception:
                logger.exception("Inspector execution failed")
                result = unknown("inspection_failed")
            connection.send(result)
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        close = getattr(inspector, "close", None)
        if close:
            close()
        connection.close()


class ProcessInspector:
    def __init__(self, factory, arguments, timeout, startup_timeout):
        self.factory = factory
        self.arguments = arguments
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.lock = threading.Lock()
        self.process = None
        self.connection = None
        self.identity = None
        self.startup_error = None
        atexit.register(self.close)

    def _stop(self):
        if self.process:
            if self.process.pid is not None:
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=0.5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=0.5)
            self.process.close()
        if self.connection:
            self.connection.close()
        self.process = self.connection = None

    def close(self):
        with self.lock:
            self._stop()

    def _start(self):
        self.identity = None
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_worker, args=(child, self.factory, self.arguments), daemon=True
        )
        try:
            self.process.start()
        finally:
            child.close()
        if not self.connection.poll(self.startup_timeout):
            self.startup_error = "initialization_timeout"
            self._stop()
            return False
        ready = self.connection.recv()
        self.identity = ready["identity"]
        return True

    def inspect(self, request):
        started = time.monotonic()
        if not self.lock.acquire(timeout=self.timeout):
            return {**unknown("busy"), "duration_ms": round((time.monotonic() - started) * 1000)}
        try:
            if self.startup_error or self.process is None and not self._start():
                result = unknown(self.startup_error)
            else:
                self.connection.send(request)
                if self.connection.poll(self.timeout):
                    result = self.connection.recv()
                else:
                    result = unknown("timeout")
                    self._stop()
        except (EOFError, BrokenPipeError, OSError, ValueError, RuntimeError):
            result = unknown("worker_unavailable")
            # Failed initialization requires a service restart after config/artifact repair.
            if self.identity is None:
                self.startup_error = "initialization_failed"
            self._stop()
        finally:
            self.lock.release()
        return {
            **result,
            "identity": self.identity,
            "duration_ms": round((time.monotonic() - started) * 1000),
        }


class Inspectors:
    def __init__(self):
        self.model = (
            ProcessInspector(
                "gateway.model.SecBert",
                (
                    Path(settings.SECBERT_CHECKPOINT),
                    Path(settings.SECBERT_PROBE_BANK) if settings.SECBERT_PROBE_BANK else None,
                    settings.SECBERT_DEVICE,
                ),
                settings.MODEL_TIMEOUT,
                settings.MODEL_STARTUP_TIMEOUT,
            )
            if settings.SECBERT_CHECKPOINT
            else None
        )
        self.crs = (
            ProcessInspector(
                "gateway.crs.LiveCRS",
                (settings.OWASP_CRS, settings.GATEWAY_MAX_BODY, settings.CRS_PARANOIA),
                settings.CRS_TIMEOUT,
                settings.CRS_STARTUP_TIMEOUT,
            )
            if settings.OWASP_CRS
            else None
        )

    def inspect(self, request):
        model = self.model.inspect(request) if self.model else unknown("not_configured")
        if self.model is None:
            model["concepts"] = {"status": "unknown", "error": "not_configured", "scores": []}
        crs = self.crs.inspect(request) if self.crs else unknown("not_configured")
        return model, crs


@lru_cache(maxsize=1)
def get_inspectors():
    return Inspectors()
