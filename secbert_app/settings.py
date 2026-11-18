"""Ingress settings. Model artifacts are deliberately unconfigured by default."""

import math
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from django.core.exceptions import ImproperlyConfigured
from django.core.management.utils import get_random_secret_key

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_REPO = Path(
    os.environ.get("TRAINING_REPO", BASE_DIR.parent / "IS-Team14-Project")
).resolve()
if (TRAINING_REPO / "src" / "http_attack_agent").is_dir():
    sys.path.insert(0, str(TRAINING_REPO / "src"))


def positive(name, default, cast=int):
    try:
        value = cast(os.environ.get(name, default))
        if value <= 0 or not math.isfinite(value):
            raise ValueError
        return value
    except (ValueError, OverflowError) as exc:
        raise ImproperlyConfigured(f"{name} must be a positive finite number.") from exc


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or get_random_secret_key()
DEBUG = False
LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("GATEWAY_ALLOWED_HOSTS", ",".join(LOCAL_HOSTS)).split(",")
    if host.strip()
]
ROOT_URLCONF = "secbert_app.ingress_urls"
WSGI_APPLICATION = "secbert_app.wsgi.application"
INSTALLED_APPS = ["gateway"]
MIDDLEWARE = []
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    }
]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("GATEWAY_DATABASE", str(BASE_DIR / "requests.sqlite3")),
        "OPTIONS": {"timeout": 10, "init_command": "PRAGMA journal_mode=WAL;"},
    }
}
USE_TZ = True
TIME_ZONE = "America/New_York"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

GATEWAY_UPSTREAM = os.environ.get("GATEWAY_UPSTREAM", "http://127.0.0.1:9000")
try:
    upstream = urlsplit(GATEWAY_UPSTREAM)
    if (
        upstream.scheme not in {"http", "https"}
        or not upstream.hostname
        or upstream.username is not None
        or upstream.password is not None
        or upstream.path not in {"", "/"}
        or upstream.query
        or upstream.fragment
        or upstream.port == 0
    ):
        raise ValueError
except ValueError as exc:
    raise ImproperlyConfigured(
        "GATEWAY_UPSTREAM must be an HTTP(S) origin without credentials or path."
    ) from exc
GATEWAY_MODE = os.environ.get("GATEWAY_MODE", "observe")
MODEL_FAILURE_POLICY = os.environ.get("MODEL_FAILURE_POLICY", "closed")
if GATEWAY_MODE not in {"observe", "enforce"} or MODEL_FAILURE_POLICY not in {"open", "closed"}:
    raise ImproperlyConfigured(
        "Use GATEWAY_MODE=observe|enforce and MODEL_FAILURE_POLICY=open|closed."
    )


def artifact_path(name):
    value = os.environ.get(name, "").strip()
    return str((BASE_DIR / value).resolve()) if value else ""


SECBERT_CHECKPOINT = artifact_path("SECBERT_CHECKPOINT")
SECBERT_PROBE_BANK = artifact_path("SECBERT_PROBE_BANK")
SECBERT_DEVICE = os.environ.get("SECBERT_DEVICE", "cpu")
if SECBERT_PROBE_BANK and not SECBERT_CHECKPOINT:
    raise ImproperlyConfigured("SECBERT_PROBE_BANK requires SECBERT_CHECKPOINT.")
OWASP_CRS = os.environ.get("OWASP_CRS", "")
CRS_PARANOIA = positive("CRS_PARANOIA", 1)
if CRS_PARANOIA > 4:
    raise ImproperlyConfigured("CRS_PARANOIA must be 1..4.")
GATEWAY_MAX_BODY = positive("GATEWAY_MAX_BODY", 1024 * 1024)
GATEWAY_MAX_RESPONSE = positive("GATEWAY_MAX_RESPONSE", 8 * 1024 * 1024)
GATEWAY_UPSTREAM_TIMEOUT = positive("GATEWAY_UPSTREAM_TIMEOUT", 5.0, float)
MODEL_TIMEOUT = positive("MODEL_TIMEOUT", 5.0, float)
MODEL_STARTUP_TIMEOUT = positive("MODEL_STARTUP_TIMEOUT", 120.0, float)
CRS_TIMEOUT = positive("CRS_TIMEOUT", 2.0, float)
CRS_STARTUP_TIMEOUT = positive("CRS_STARTUP_TIMEOUT", 30.0, float)
DATA_UPLOAD_MAX_MEMORY_SIZE = GATEWAY_MAX_BODY

# Raw detection evidence is a disposable copy; the database remains the audit source.
GATEWAY_EVIDENCE_DIR = (
    BASE_DIR / os.environ.get("GATEWAY_EVIDENCE_DIR", "evidence_cache")
).absolute()
resolved_evidence_dir = GATEWAY_EVIDENCE_DIR.resolve()
protected_dirs = [
    BASE_DIR / name
    for name in (
        "artifacts",
        "gateway",
        "secbert_app",
        ".git",
        ".venv",
        ".agents",
        ".codex",
        ".aws",
    )
]
if (
    resolved_evidence_dir == BASE_DIR
    or not resolved_evidence_dir.is_relative_to(BASE_DIR)
    or any(
        resolved_evidence_dir == protected.resolve()
        or resolved_evidence_dir.is_relative_to(protected.resolve())
        for protected in protected_dirs
    )
):
    raise ImproperlyConfigured(
        "GATEWAY_EVIDENCE_DIR must be a dedicated folder inside the APP directory, "
        "outside source, artifacts, virtual environment and repository metadata."
    )
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"gateway": {"handlers": ["console"], "level": "INFO", "propagate": False}},
}
