"""Export each caller's API key into its own directory volume (TASK-2380).

``LLM_BAWT_API_KEY_EXPORTS`` lists ``principal=/dir[:uid[:gid]]`` entries,
comma separated. For each, the app writes ``<dir>/api-key`` holding the full
header value (``<principal>.<key>``), atomically (0600, or 0640 with a gid),
owned by the consumer's uid/gid when given. Each consumer mounts only its own
directory read-only (a directory, never a single-file bind, so a replaced file
is seen) and points ``LLM_BAWT_API_KEY_FILE`` at the file.

A principal that can't be derived right now keeps its last good file. Nothing
here logs a key.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from . import keys

log = logging.getLogger(__name__)

EXPORTS_ENV = "LLM_BAWT_API_KEY_EXPORTS"
FILE_NAME = "api-key"


class ExportSpec:
    __slots__ = ("principal", "directory", "uid", "gid")

    def __init__(self, principal: str, directory: Path, uid: int | None, gid: int | None):
        self.principal = principal
        self.directory = directory
        self.uid = uid
        self.gid = gid


def parse(raw: str | None) -> list[ExportSpec]:
    """Parse the env value; raises ValueError naming the bad entry."""
    specs: list[ExportSpec] = []
    seen: set[str] = set()
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        principal, sep, rest = entry.partition("=")
        principal = principal.strip()
        parts = rest.strip().split(":")
        if not sep or not keys.valid_principal(principal) or not parts[0].startswith("/") \
                or len(parts) > 3 or principal in seen:
            raise ValueError(f"invalid {EXPORTS_ENV} entry: {entry!r}")
        try:
            ids = [int(p) if p else None for p in parts[1:]]
        except ValueError:
            raise ValueError(f"invalid {EXPORTS_ENV} entry: {entry!r}") from None
        ids += [None] * (2 - len(ids))
        seen.add(principal)
        specs.append(ExportSpec(principal, Path(parts[0]), ids[0], ids[1]))
    return specs


_lock = threading.Lock()


def export_all(specs: list[ExportSpec] | None = None, *, root=None) -> dict[str, str]:
    """Write every configured key; ``{principal: "written"|"kept"|"failed"}``.

    Never raises. Unset or invalid configuration exports nothing."""
    if specs is None:
        try:
            specs = parse(os.environ.get(EXPORTS_ENV))
        except ValueError as exc:
            log.error("api key export disabled: %s", exc)
            return {}
    from ..bawthub_auth.token_export import _write_atomically

    kwargs = {} if root is None else {"root": root}
    results: dict[str, str] = {}
    with _lock:
        for spec in specs:
            try:
                value = keys.header_value(spec.principal, **kwargs)
            except Exception as exc:  # noqa: BLE001 - e.g. no master key: keep last copy
                log.warning("api key export %s: source failed (%s)", spec.principal, type(exc).__name__)
                results[spec.principal] = "kept"
                continue
            try:
                _write_atomically(spec.directory / FILE_NAME, value, spec.uid, spec.gid)
                results[spec.principal] = "written"
            except OSError as exc:
                log.warning("api key export %s failed (%s)", spec.principal, type(exc).__name__)
                results[spec.principal] = "failed"
    return results
