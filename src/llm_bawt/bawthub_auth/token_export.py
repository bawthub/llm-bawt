"""Hand the BawtHub service token to BawtHub through a shared file (TASK-1472).

The app writes the token in effect to ``BAWTHUB_SERVICE_TOKEN_EXPORT_FILE``, a
file in a small volume that BawtHub's frontends mount read-only and read via
their ``BAWTHUB_SERVICE_TOKEN_FILE``. BawtHub gets only the token, never the
master key it is derived from.

* Written at startup (:func:`export_service_token`) and again by the token
  source whenever the token in effect changes, before the new token is used
  (``ServiceTokenSource._publish``).
* The write is atomic (temp file in the same directory, fsync, ``os.replace``)
  and the file is mode 0600. Mount the *directory*: a single-file bind mount
  pins the old inode, so the reader would never see a replacement.
* ``BAWTHUB_SERVICE_TOKEN_EXPORT_UID`` chowns the file to the reader's uid. The
  app runs as root and a root-owned 0600 file is unreadable by a non-root
  reader (BawtHub's prod image runs as its own user).
* With no usable token the stale file is removed, so both sides fail closed
  together instead of BawtHub accepting a token llm-bawt no longer holds.
* Unset export path = nothing to do (a bare dev run without BawtHub).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .service_token import ExportResult, ServiceTokenSource, service_token_source

EXPORT_UID_ENV = "BAWTHUB_SERVICE_TOKEN_EXPORT_UID"
_MODE = 0o600


def export_service_token(source: ServiceTokenSource | None = None) -> ExportResult:
    """App startup: write (or withdraw) the export. Never raises; never logs the value."""
    return (source or service_token_source()).sync(force=True)


def publish(path: Path, token: str | None, environ: Mapping[str, str], *,
            uid_env: str = EXPORT_UID_ENV) -> ExportResult:
    """Write ``token`` to ``path``, or remove ``path`` when there is none.

    ``uid_env`` names the env var holding the reader's uid (the bridge-token
    exports use their own, TASK-1477). Raises ``ValueError`` (bad uid, nothing
    written) or ``OSError``; the caller reports those. Never logs.
    """
    uid = _reader_uid(environ, uid_env)
    if token is None:
        path.unlink(missing_ok=True)
        return ExportResult.REMOVED
    _write_atomically(path, token, uid)
    return ExportResult.WRITTEN


def _reader_uid(environ: Mapping[str, str], uid_env: str) -> int | None:
    raw = (environ.get(uid_env) or "").strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise ValueError(f"{uid_env} must be a numeric uid")
    return int(raw)


def _write_atomically(path: Path, token: str, uid: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates the file 0600 with O_EXCL, so no reader ever sees a
    # wider mode or a partial write under the final name.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, _MODE)
        if uid is not None:
            os.chown(tmp, uid, -1)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
