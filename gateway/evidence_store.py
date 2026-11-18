"""Disposable JSON copies of request evidence, separate from the audit database."""

import json
import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from django.conf import settings


def _is_reparse(info):
    """Exclude POSIX symlinks and Windows junctions/other reparse points."""
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _root():
    root = Path(settings.GATEWAY_EVIDENCE_DIR).absolute()
    # Inspect lexical ancestors before resolving, so a configured link is never hidden.
    for parent in (*reversed(root.parents), root):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if _is_reparse(info):
            raise OSError("Evidence cache paths must not contain symbolic links or reparse points.")
        if not stat.S_ISDIR(info.st_mode):
            raise NotADirectoryError(str(parent))
    root = root.resolve()
    app = Path(settings.BASE_DIR).resolve()
    artifacts = app / "artifacts"
    if root in (app, artifacts) or root in app.parents or artifacts in root.parents:
        raise OSError("The evidence cache must not contain the application or model artifacts.")
    return root


def _owned_name(name):
    if not name.endswith(".json"):
        return False
    try:
        return str(UUID(name[:-5])) + ".json" == name
    except ValueError:
        return False


def save_evidence(request_id, evidence):
    """Atomically save an evidence mapping; no directory is created until this call."""
    name = f"{UUID(str(request_id))}.json"
    payload = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    _root()  # Recheck the resulting directory before opening the temporary file.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".evidence-",
            suffix=".tmp",
            dir=root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        target = root / name
        os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def cache_stats():
    """Read cache size without creating it; return a safe unavailable state on errors."""
    result = {
        "exists": False,
        "file_count": 0,
        "total_bytes": 0,
        "path": str(Path(settings.GATEWAY_EVIDENCE_DIR).absolute()),
        "error": None,
    }
    try:
        root = _root()
        with os.scandir(root) as entries:
            result["exists"] = True
            for entry in entries:
                if not _owned_name(entry.name):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError:
                    result["error"] = "unavailable"
                    continue
                if stat.S_ISREG(info.st_mode) and not _is_reparse(info):
                    result["file_count"] += 1
                    result["total_bytes"] += info.st_size
    except FileNotFoundError:
        pass
    except OSError:
        result["error"] = "unavailable"
    return result


def clear_evidence():
    """Remove only direct UUID.json regular files; retain folders and all other files."""
    result = {"deleted_count": 0, "failed_count": 0}
    root = _root()
    try:
        entries = os.scandir(root)
    except FileNotFoundError:
        return result
    with entries:
        for entry in entries:
            if not _owned_name(entry.name):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or _is_reparse(info):
                    continue
                # Do not recurse, resolve an entry, or remove the directory itself.
                Path(entry.path).unlink()
                result["deleted_count"] += 1
            except FileNotFoundError:
                continue
            except OSError:
                result["failed_count"] += 1
    return result
