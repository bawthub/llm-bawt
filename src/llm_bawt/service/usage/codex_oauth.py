"""App-owned ChatGPT/codex OAuth bundle (TASK-636 Phase 2).

Mirrors the Claude credential model (TASK-635/636): the app is the SOLE owner
and refresher of the ChatGPT subscription OAuth bundle (the artifact ``codex
login`` produces). The bundle lives in the encrypted CredentialStore row
(``provider_connection:openai_chatgpt`` → ``secret_enc``, Fernet at rest).

Consumers:
* the proxy adapter ``OpenAIChatGPTAdapter`` in ``claude_code_bridge`` — reads
  tokens via the broker endpoint ``GET /v1/providers/openai_chatgpt/token``;
* the codex CLI in ``codex-bridge`` — reads a locally-materialized
  ``~/.codex/auth.json`` that the bridge writes from the broker response.

Neither consumer refreshes — the app owns the rotate-on-use refresh chain.
One-time cutover: on first load, if the DB row has no bundle but the legacy
file exists, the file is imported into the DB.

Refresh is serialized behind a process-wide lock (the refresh token is
single-use — two concurrent refreshes invalidate each other). A proactive
background loop refreshes at ``exp - buffer`` so the access token never
lapsies even when idle.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ── Constants (lifted from codex-rs/login + proxy/adapters/openai_chatgpt.py) ──

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token"
REFRESH_URL_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"

# Refresh when the access_token JWT's exp is within this window.
_REFRESH_BUFFER_S = 5 * 60
# Proactive loop interval + buffer.
_PROACTIVE_CHECK_S = 300.0
_PROACTIVE_BUFFER_S = 20 * 60

_PROVIDER_ID = "openai_chatgpt"
_SECRET_KEY = "codex_auth"

# Single-writer lock — the refresh token rotates on use.
_refresh_lock = threading.Lock()

# A rotated bundle whose DB save failed (TASK-2267). The provider has already
# consumed the refresh token still in the DB, so this bundle is the only valid
# one: it is used in place of the DB bundle and its save is retried, while the
# DB row still holds the refresh token it replaced (a reconnect or disconnect
# meanwhile drops it). Process memory only; not crash-proof.
_pending: tuple[str, dict] | None = None  # (DB refresh token, rotated bundle)


def _refresh_token_of(bundle: dict | None) -> str | None:
    tokens = (bundle or {}).get("tokens") or {}
    return tokens.get("refresh_token") if isinstance(tokens, dict) else None

# Health reporting (mirrors claude_oauth).
_last_refresh_ok_at: int | None = None
_last_refresh_error: str | None = None
_last_refresh_error_at: int | None = None

# Migration import runs at most once per process.
_migrated = False


def _record_refresh_outcome(error: str | None) -> None:
    global _last_refresh_ok_at, _last_refresh_error, _last_refresh_error_at
    was_failing = _last_refresh_error is not None
    now_ms = int(time.time() * 1000)
    if error is None:
        _last_refresh_ok_at = now_ms
        _last_refresh_error = None
        _last_refresh_error_at = None
    else:
        _last_refresh_error = error
        _last_refresh_error_at = now_ms
    if was_failing != (error is not None):
        # Provider health reads this without a credential write: publish the flip.
        from llm_bawt.service.health_changes import publish
        publish("refresh-outcome")


def refresh_health() -> dict:
    """Cleartext refresh-chain health for the status endpoint."""
    return {
        "last_refresh_at": _last_refresh_ok_at,
        "last_refresh_error": _last_refresh_error,
        "last_refresh_error_at": _last_refresh_error_at,
    }


def codex_auth_path() -> Path:
    """The LEGACY codex auth.json file (migration source).

    TASK-636: no longer the source of truth — the encrypted DB row is. This
    path is only ever read (one-time import on first load). ``CODEX_AUTH_PATH``
    env override honored for the import.
    """
    p = os.getenv("CODEX_AUTH_PATH")
    return Path(p) if p else Path.home() / ".codex" / "auth.json"


class CodexToken:
    """Result of resolving the ChatGPT access token."""

    token: str | None
    account_id: str | None
    expires_at: float | None  # epoch seconds (JWT exp claim)
    state: str  # ok | missing | stale

    def __init__(
        self,
        token: str | None,
        account_id: str | None,
        expires_at: float | None = None,
        state: str = "ok",
    ) -> None:
        self.token = token
        self.account_id = account_id
        self.expires_at = expires_at
        self.state = state


# ── DB-backed bundle storage ───────────────────────────────────────────


def _credential_store():
    """The app's CredentialStore, or None when the service isn't up yet."""
    try:
        from ..dependencies import get_service  # noqa: PLC0415

        from ..providers.base import CredentialStore  # noqa: PLC0415

        return CredentialStore(get_service().config)
    except Exception:  # noqa: BLE001 — service not initialized (CLI contexts)
        return None


def _read_file_bundle() -> dict | None:
    """Read the legacy ~/.codex/auth.json. Never writes."""
    path = codex_auth_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read legacy codex auth %s: %s", path, e)
        return None
    return data if isinstance(data, dict) else None


def _jwt_exp(token: str) -> float | None:
    """Best-effort decode of a JWT's ``exp`` claim (seconds since epoch)."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = claims.get("exp")
        return float(exp) if exp is not None else None
    except Exception:  # noqa: BLE001
        return None


def _bundle_expires_at(bundle: dict) -> float | None:
    """The access_token JWT exp (seconds), or None if undecodable."""
    token = (bundle.get("tokens") or {}).get("access_token") or ""
    return _jwt_exp(token)


def _expires_soon(bundle: dict, *, buffer_s: float = _REFRESH_BUFFER_S) -> bool:
    exp = _bundle_expires_at(bundle)
    if exp is None:
        # Fall back to last_refresh heuristic.
        last = bundle.get("last_refresh")
        if not last:
            return True
        try:
            issued = datetime.fromisoformat(
                last.replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            return True
        return (time.time() - issued) > (3600 - buffer_s)
    return (exp - time.time()) < buffer_s


def _load() -> tuple[dict, dict | None]:
    """Return (raw_record_json, auth_bundle), preferring an unsaved rotation.

    Never clears ``_pending``: a reader's DB snapshot may be stale, so only the
    locked save path (:func:`_persist_rotated`) decides a pending bundle is done.
    """
    pending = _pending
    raw, bundle = _load_db()
    if pending is not None and bundle and _refresh_token_of(bundle) == pending[0]:
        return raw, pending[1]
    return raw, bundle


def _persist_rotated(raw: dict, db_token: str | None, rotated: dict) -> bool:
    """Save a rotated bundle; on failure keep it pending and report degraded.

    The save is conditional on the DB row still holding ``db_token`` (checked
    under a row lock), so a reconnect or disconnect that landed meanwhile wins
    and the stale rotation is dropped. Returns True when nothing is left
    pending. Caller holds _refresh_lock.
    """
    global _pending
    try:
        saved = _save(raw, rotated, expect_refresh_token=db_token)
    except Exception as e:  # noqa: BLE001
        logger.warning("Refreshed ChatGPT token but could not persist to DB; keeping it in memory and retrying: %s", e)
        if db_token:
            _pending = (db_token, rotated)
        _record_refresh_outcome(f"rotated token not yet saved: {e}")
        return False
    _pending = None
    if not saved:
        logger.warning("ChatGPT credential changed during refresh (reconnect or disconnect); dropping rotated bundle")
    _record_refresh_outcome(None)
    return True


def _retry_pending_save() -> bool:
    """Retry saving an unsaved rotation. Returns True when nothing is pending."""
    with _refresh_lock:
        pending = _pending
        if pending is None:
            return True
        raw, _ = _load_db()
        return _persist_rotated(raw, pending[0], pending[1])


def _load_db() -> tuple[dict, dict | None]:
    """Return (raw_record_json, auth_bundle). bundle is None if absent.

    Source of truth is the DB row; the first load imports the legacy file
    into the DB when the row has no bundle (one-time cutover, TASK-636).
    """
    global _migrated
    store = _credential_store()
    if store is None or not store.available:
        # Service down / no DB — fall back to the legacy file read-only.
        return {}, _read_file_bundle()

    record = store.load(_PROVIDER_ID)
    raw = record.public() if record else {}
    bundle: dict | None = None
    if record:
        candidate = record.secret.get(_SECRET_KEY)
        if isinstance(candidate, dict):
            bundle = candidate

    if bundle is None and not _migrated:
        _migrated = True
        legacy = _read_file_bundle()
        if legacy:
            try:
                _save(raw, legacy)
                logger.info(
                    "Migrated legacy codex auth.json into the encrypted DB store"
                )
                bundle = legacy
            except Exception as e:  # noqa: BLE001
                logger.warning("Codex auth DB migration failed: %s", e)
    return raw, bundle


def _save(raw: dict, bundle: dict, *, expect_refresh_token: str | None = None) -> bool:
    """Persist the bundle to the DB row.

    With ``expect_refresh_token``, saves only if the row still holds that
    refresh token (atomic under a row lock); returns False when it does not.
    """
    store = _credential_store()
    if store is None or not store.available:
        raise RuntimeError("CredentialStore unavailable — cannot persist codex auth")
    from ..providers.base import (  # noqa: PLC0415
        AUTH_CLI_OAUTH,
        STATUS_CONNECTED,
        ConnectionRecord,
    )

    tokens = bundle.get("tokens") or {}
    auth_method = raw.get("auth_method") or AUTH_CLI_OAUTH
    record = ConnectionRecord(
        provider=_PROVIDER_ID,
        status=STATUS_CONNECTED,
        auth_method=auth_method,
        account="chatgpt-subscription",
        meta={
            "auth_mode": bundle.get("auth_mode", "chatgpt"),
            "account_id": tokens.get("account_id"),
            "last_refresh": bundle.get("last_refresh"),
        },
        secret={_SECRET_KEY: bundle},
        connected_at=raw.get("connected_at"),
    )
    if expect_refresh_token is None:
        store.save(record)
        return True

    def _holds_expected(current) -> bool:
        current_bundle = (current.secret or {}).get(_SECRET_KEY) if current else None
        return isinstance(current_bundle, dict) and _refresh_token_of(current_bundle) == expect_refresh_token

    return store.save_if(record, _holds_expected)


# ── Refresh ────────────────────────────────────────────────────────────


def _refresh_upstream(bundle: dict) -> dict:
    """Call auth.openai.com to rotate the token pair. Returns updated bundle."""
    refresh_token = (bundle.get("tokens") or {}).get("refresh_token")
    if not refresh_token:
        raise RuntimeError("codex auth bundle has no refresh_token — re-run `codex login`")
    url = os.getenv(REFRESH_URL_ENV) or DEFAULT_REFRESH_URL
    body = {
        "client_id": CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    # Sync httpx — runs inside asyncio.to_thread from the async callers.
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=body)
    if resp.status_code != 200:
        raise RuntimeError(
            f"ChatGPT OAuth refresh failed ({resp.status_code}): {resp.text[:300]}"
        )
    payload = resp.json()
    tokens = bundle.setdefault("tokens", {})
    for key in ("id_token", "access_token", "refresh_token"):
        val = payload.get(key)
        if val:
            tokens[key] = val
    bundle["last_refresh"] = datetime.now(timezone.utc).isoformat()
    return bundle


def _refresh_serialized(*, buffer_s: float, force: bool = False) -> dict | None:
    """Refresh + persist the bundle under the lock.

    Re-loads inside the lock and skips the upstream call if another caller
    already refreshed while we waited.
    """
    with _refresh_lock:
        raw, bundle = _load()
        if not bundle:
            return None
        is_pending = _pending is not None and bundle is _pending[1]
        if is_pending:
            # Still unsaved: retry the save before anything else.
            is_pending = not _persist_rotated(raw, _pending[0], bundle)
        if not force and not _expires_soon(bundle, buffer_s=buffer_s):
            return bundle
        # Captured first: _refresh_upstream rotates the bundle in place.
        db_token = _pending[0] if is_pending and _pending is not None else _refresh_token_of(bundle)
        try:
            refreshed = _refresh_upstream(dict(bundle, tokens=dict(bundle.get("tokens") or {})))
        except Exception as e:  # noqa: BLE001
            _record_refresh_outcome(str(e))
            raise
        # Health stays degraded until the rotated bundle is saved.
        _persist_rotated(raw, db_token, refreshed)
        logger.info("Refreshed ChatGPT/codex OAuth token")
        return refreshed


def get_access_token(*, force_refresh: bool = False) -> CodexToken:
    """Resolve the app-owned ChatGPT access token.

    Refreshes (serialized) when expired-or-near-expiry, or unconditionally
    with ``force_refresh`` (e.g. a reader got a 401).
    """
    raw, bundle = _load()
    if not bundle:
        return CodexToken(None, None, state="missing")

    needs = force_refresh or _expires_soon(bundle)
    if needs:
        try:
            bundle = _refresh_serialized(
                buffer_s=_REFRESH_BUFFER_S, force=force_refresh
            ) or bundle
        except Exception as e:  # noqa: BLE001 — fall back to existing token
            logger.warning("Failed to refresh codex token: %s", e)

    tokens = bundle.get("tokens") or {}
    token = tokens.get("access_token")
    if not token:
        return CodexToken(None, None, state="missing")

    exp = _bundle_expires_at(bundle)
    expired = exp is not None and exp <= time.time()
    return CodexToken(
        token=token,
        account_id=tokens.get("account_id"),
        expires_at=exp,
        state="stale" if expired else "ok",
    )


def bundle_status() -> dict:
    """Cleartext facts about the stored bundle (no tokens) — for UI status."""
    _, bundle = _load()
    if not bundle:
        return {"present": False}
    tokens = bundle.get("tokens") or {}
    exp = _bundle_expires_at(bundle)
    return {
        "present": True,
        "expired": exp is not None and exp <= time.time(),
        "expires_at": int(exp * 1000) if exp is not None else None,
        "account_id": tokens.get("account_id"),
        "last_refresh": bundle.get("last_refresh"),
        **refresh_health(),
    }


async def proactive_refresh_loop() -> None:
    """Keep the bundle fresh forever — refresh at exp - buffer.

    Started from the app lifespan. Sleeps only when no bundle exists yet.
    """
    logger.info("Codex credential proactive refresh loop started (DB store)")
    while True:
        try:
            await asyncio.to_thread(_retry_pending_save)
            _, bundle = _load()
            if bundle and _expires_soon(bundle, buffer_s=_PROACTIVE_BUFFER_S):
                await asyncio.to_thread(
                    _refresh_serialized, buffer_s=_PROACTIVE_BUFFER_S
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("Proactive codex token refresh failed (will retry): %s", e)
        await asyncio.sleep(_PROACTIVE_CHECK_S)
