"""The ops reconciler's secrets, handed over by the app (TASK-2386).

The reconciler used to mount the master key so it could derive its control
bearer and gate provenance key and decrypt three stored GitHub credentials.
Now only the app holds the key: it writes exactly those five values into a
volume that only the app (read-write) and the reconciler (read-only) mount,
and the reconciler reads them from there.

* App side: ``LLM_BAWT_OPS_SECRETS_EXPORT_DIR`` names the directory. The app
  exports at startup and again shortly after any write or delete of one of the
  exported credentials (:func:`credential_changed`), so a reconnect in the
  Providers UI reaches the reconciler without a restart.
* Reconciler side: ``LLM_BAWT_OPS_SECRETS_DIR`` names the same directory,
  mounted read-only. When it is set the readers use only the files and never
  touch the master key; a missing file means "unavailable", which each caller
  already treats as fail-closed. Files are read on every use.
* Each file is replaced atomically (0600, fsync, rename). A credential that is
  not connected removes its file. A value the app can't read right now (DB or
  decrypt trouble) leaves the last good file in place.
* Nothing here logs a value.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

EXPORT_DIR_ENV = "LLM_BAWT_OPS_SECRETS_EXPORT_DIR"
READ_DIR_ENV = "LLM_BAWT_OPS_SECRETS_DIR"

CONTROL_TOKEN = "control-token"
GATE_PROVENANCE_KEY = "gate-provenance-key"
GHCR_PULL = "ghcr-pull.json"
GITHUB_RELEASE = "github-release"
GITHUB_RELEASE_DISPATCH = "github-release-dispatch"
FILES = (CONTROL_TOKEN, GATE_PROVENANCE_KEY, GHCR_PULL, GITHUB_RELEASE, GITHUB_RELEASE_DISPATCH)

#: Stored provider credentials the reconciler uses; a write to one re-exports.
CREDENTIAL_PROVIDERS = frozenset({"ghcr-pull", "github-release", "github-release-dispatch"})
#: Re-export this long after a credential write, so a write made inside a
#: caller's transaction (``CredentialStore.save_in``) has committed first.
REEXPORT_DELAY_SECONDS = 1.0

_MISSING = object()


class Unreadable(Exception):
    """The app can't read this value right now; keep the last good file."""


# ── reconciler side ─────────────────────────────────────────────────────────
def read_dir() -> Path | None:
    raw = (os.environ.get(READ_DIR_ENV) or "").strip()
    return Path(raw) if raw else None


def read(name: str) -> str | None:
    """The exported value, or None when the file is absent or empty.

    Only call when :func:`read_dir` is set."""
    if name not in FILES:
        raise ValueError(f"unknown ops secret {name!r}")
    directory = read_dir()
    if directory is None:
        raise RuntimeError(f"{READ_DIR_ENV} is not set")
    try:
        value = (directory / name).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def read_pull_auth() -> dict | None:
    raw = read(GHCR_PULL)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if not (isinstance(value, dict) and set(value) == {"username", "password"}
            and all(isinstance(v, str) and v for v in value.values())):
        return None
    return value


# ── app side ────────────────────────────────────────────────────────────────
def _derived(label: str) -> str:
    from ..service.providers import crypto

    return crypto.derive_key(label).hex()


def _stored(provider: str) -> tuple[str | None, str] | None:
    """``(account, api_key)`` of a connected provider record; None only when
    the record is authoritatively absent or not connected.

    Strict on purpose (Codex): the generic adapter readers return None for an
    unavailable store, unreadable JSON or a secret that fails to decrypt, and
    None here deletes the reconciler's last good copy. Every one of those is
    :class:`Unreadable` instead, so the export keeps the previous file."""
    from sqlalchemy import select

    from ..runtime_settings import RuntimeSetting, RuntimeSettingsStore
    from ..service.providers import crypto
    from ..service.providers.base import _SETTING_PREFIX, STATUS_CONNECTED
    from ..utils.config import Config

    try:
        engine = RuntimeSettingsStore(Config()).engine
        if engine is None:
            raise Unreadable("store_unavailable")
        table = RuntimeSetting.__table__
        with engine.connect() as conn:
            raw = conn.execute(select(table.c.value_json).where(
                table.c.scope_type == "global", table.c.scope_id == "*",
                table.c.key == f"{_SETTING_PREFIX}{provider}")).scalar_one_or_none()
        if raw is None:
            return None
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise Unreadable("malformed")
        if row.get("status") != STATUS_CONNECTED:
            return None
        secret = json.loads(crypto.decrypt(row.get("secret_enc") or ""))
        key = secret.get("api_key") if isinstance(secret, dict) else None
        if not isinstance(key, str) or not key:
            raise Unreadable("malformed")
        account = row.get("account")
        return (account if isinstance(account, str) and account else None), key
    except Unreadable:
        raise
    except Exception as exc:  # noqa: BLE001 - never surface secret-bearing text
        raise Unreadable(type(exc).__name__) from None


def _credential(provider: str) -> str | None:
    found = _stored(provider)
    return found[1] if found else None


def _pull_auth() -> str | None:
    found = _stored("ghcr-pull")
    if found is None:
        return None
    account, token = found
    if not account:
        raise Unreadable("malformed")
    return json.dumps({"username": account, "password": token}, sort_keys=True)


def _sources():
    from . import gate_trust
    from .control_protocol import TOKEN_LABEL

    return {
        CONTROL_TOKEN: lambda: _derived(TOKEN_LABEL),
        GATE_PROVENANCE_KEY: lambda: _derived(gate_trust._KEY_LABEL),
        GHCR_PULL: _pull_auth,
        GITHUB_RELEASE: lambda: _credential("github-release"),
        GITHUB_RELEASE_DISPATCH: lambda: _credential("github-release-dispatch"),
    }


def export_dir() -> Path | None:
    raw = (os.environ.get(EXPORT_DIR_ENV) or "").strip()
    return Path(raw) if raw else None


_export_lock = threading.Lock()


def export_all(directory: Path | None = None, sources=None) -> dict[str, str]:
    """Write every value; return ``{file: "written"|"removed"|"kept"|"failed"}``.

    Never raises and never logs a value. Unset export dir = nothing to do."""
    directory = directory or export_dir()
    if directory is None:
        return {}
    from ..bawthub_auth.token_export import _write_atomically

    results: dict[str, str] = {}
    with _export_lock:
        for name, source in (sources or _sources()).items():
            path = directory / name
            try:
                value = source()
            except Unreadable:
                results[name] = "kept"
                continue
            except Exception as exc:  # noqa: BLE001 - e.g. no master key: keep the last good copy
                log.warning("ops secret export %s source failed (%s)", name, type(exc).__name__)
                results[name] = "kept"
                continue
            try:
                if value:
                    _write_atomically(path, value, None)
                    results[name] = "written"
                else:
                    path.unlink(missing_ok=True)
                    results[name] = "removed"
            except OSError as exc:
                log.warning("ops secret export %s failed (%s)", name, type(exc).__name__)
                results[name] = "failed"
    kept = sorted(n for n, r in results.items() if r in ("kept", "failed"))
    if kept:
        log.warning("ops secret export kept the previous copy of: %s", ", ".join(kept))
    return results


_pending: threading.Timer | None = None
_pending_lock = threading.Lock()


def credential_changed(provider: str) -> None:
    """Re-export shortly after a write/delete of an exported credential. Never raises."""
    global _pending
    if provider not in CREDENTIAL_PROVIDERS or export_dir() is None:
        return
    try:
        with _pending_lock:
            if _pending is not None:
                return
            timer = threading.Timer(REEXPORT_DELAY_SECONDS, _fire)
            timer.daemon = True
            _pending = timer
        timer.start()
    except Exception:  # noqa: BLE001 - a credential write must never fail on this
        log.debug("ops secret re-export scheduling failed", exc_info=True)


def _fire() -> None:
    global _pending
    with _pending_lock:
        _pending = None
    export_all()
