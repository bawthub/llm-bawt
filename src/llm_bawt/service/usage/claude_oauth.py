"""The app-owned Claude OAuth credential — single source for inference + usage.

TASK-635: there is ONE Claude login for the whole deployment. The app owns the
full-scope ``claudeAiOauth`` bundle (``user:inference`` + ``user:profile`` + …,
minted by the ``claude`` provider adapter's wizard login) and is the SOLE
refresher of its rotate-on-use refresh-token chain.

TASK-636: the bundle now lives in the **encrypted CredentialStore** (the
``provider_connection:claude`` row's ``secret_enc`` — Fernet, key from
LLM_BAWT_SECRET_KEY) instead of a host file. The DB row is the single source
of truth; consumers:

* the ``/v1/usage`` Claude adapter (same process — calls :func:`load_usage_token`),
* the claude-code bridge (broker endpoint ``GET /v1/providers/claude/token``
  on each native inference request — no token cache or credential bind mounts).

One-time cutover: on first load, if the DB row has no bundle but the legacy
file exists (:func:`claude_credentials_path`), the file is imported into the
DB — so existing deployments switch over with zero re-login.

Refresh is serialized behind a process-wide lock and re-checks freshness after
acquiring it, so concurrent callers (usage fetch, broker endpoint, proactive
loop) can never race the single-use refresh token. A proactive background loop
(:func:`proactive_refresh_loop`, started from the app lifespan) refreshes at
``expiresAt - buffer`` so the access token never lapses even when idle.

``CLAUDE_USAGE_CREDENTIALS_MODE=shared`` is obsolete: the app owns the row and
is always the refresher. The legacy file, when it exists, is only ever READ
(once, for the migration import) — never written.

The bundle is the standard wrapper ``{"claudeAiOauth": {...}}`` on disk; in the
DB secret it is stored as the bare bundle (``accessToken``/``refreshToken``/
``expiresAt``/``scopes``/…).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Same OAuth client the Claude CLI / bridge use for subscription refresh.
_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_REFRESH_BUFFER_MS = 5 * 60 * 1000
# Proactive loop: check cadence + how early before expiry we refresh. The
# refresh window is deliberately wider than the on-demand buffer so the loop
# (not a request) is what normally performs the refresh.
_PROACTIVE_CHECK_S = 300.0
_PROACTIVE_BUFFER_MS = 20 * 60 * 1000

_PROVIDER_ID = "claude"
_SECRET_KEY = "claudeAiOauth"

# Serializes refresh across the usage fetch, the broker endpoint, and the
# proactive loop — the refresh token is single-use, so two concurrent
# refreshes would invalidate each other (the exact race TASK-635 exists to
# prevent).
_REFRESH_LOCK = threading.Lock()

# Refresh-outcome tracking (TASK-637) — the invisible-failure fix. The app is
# the SOLE refresher (TASK-635), so module-level state is authoritative for
# this process. When refresh starts failing while the access token is still
# valid, there is up to ~expiresAt of runway — these fields let the health
# layer surface that warning window instead of burying it in logs.
_last_refresh_ok_at: int | None = None  # epoch ms
_last_refresh_error: str | None = None
_last_refresh_error_at: int | None = None

# Migration import runs at most once per process.
_migrated = False


def _record_refresh_outcome(error: str | None) -> None:
    global _last_refresh_ok_at, _last_refresh_error, _last_refresh_error_at
    now = int(time.time() * 1000)
    if error is None:
        _last_refresh_ok_at = now
        _last_refresh_error = None
        _last_refresh_error_at = None
    else:
        _last_refresh_error = error[:300]
        _last_refresh_error_at = now


def refresh_health() -> dict:
    """Cleartext refresh-chain health for the provider health layer (no tokens)."""
    return {
        "last_refresh_at": _last_refresh_ok_at,
        "last_refresh_error": _last_refresh_error,
        "last_refresh_error_at": _last_refresh_error_at,
    }


def claude_credentials_path() -> Path:
    """The LEGACY app-owned Claude credential bundle file (migration source).

    TASK-636: no longer the source of truth — the encrypted DB row is. This
    path is only ever read (one-time import on first load) and kept so older
    docs/scripts still resolve. ``CLAUDE_CREDENTIALS_PATH`` /
    ``CLAUDE_USAGE_CREDENTIALS_PATH`` env overrides honored for the import.
    """
    override = os.getenv("CLAUDE_CREDENTIALS_PATH") or os.getenv(
        "CLAUDE_USAGE_CREDENTIALS_PATH"
    )
    if override:
        return Path(override)
    return Path.home() / ".config" / "llm-bawt" / "claude-usage-credentials.json"


# Backwards-compatible alias — existing callers/docs use the usage-era name.
usage_credentials_path = claude_credentials_path


def credentials_mode() -> str:
    """Legacy knob — always ``owned`` now (the app owns the DB row)."""
    return "owned"


# Backwards-compatible private alias (internal callers use _mode()).
_mode = credentials_mode


@dataclass
class UsageToken:
    """Result of resolving the app-owned credential.

    ``state`` is one of:
      * ``ok``      — usable access token (``token`` set)
      * ``missing`` — no credential configured
      * ``stale``   — credential present but its access token has expired and
                      we couldn't refresh it
    """

    token: str | None
    state: str
    expires_at: int | None = None


# ── DB-backed bundle storage (CredentialStore) ───────────────────────


def _credential_store():
    """The app's CredentialStore, or None when the service isn't up yet."""
    try:
        from ..dependencies import get_service  # noqa: PLC0415

        from ..providers.base import CredentialStore  # noqa: PLC0415

        return CredentialStore(get_service().config)
    except Exception:  # noqa: BLE001 — service not initialized (CLI contexts)
        return None


def _read_file_bundle() -> dict | None:
    """Read the legacy bundle file (wrapper or bare form). Never writes."""
    path = claude_credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read legacy Claude credential %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        return None
    bundle = data.get("claudeAiOauth")
    if bundle is None and data.get("accessToken"):
        bundle = data
    return bundle if isinstance(bundle, dict) else None


def _load() -> tuple[dict, dict | None]:
    """Return (raw_record_json, oauth_bundle). bundle is None if absent.

    Source of truth is the DB row; the first load imports the legacy file
    into the DB when the row has no bundle (one-time cutover, TASK-636).
    """
    global _migrated
    store = _credential_store()
    if store is None or not store.available:
        # Service down / no DB — fall back to the legacy file read-only so
        # CLI contexts keep working.
        return {}, _read_file_bundle()

    record = store.load(_PROVIDER_ID)
    raw = record.public() if record else {}
    bundle = None
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
                    "Migrated legacy Claude credential file into the encrypted DB store"
                )
                bundle = legacy
            except Exception as e:  # noqa: BLE001
                logger.warning("Claude credential DB migration failed: %s", e)
    return raw, bundle


def _save(raw: dict, bundle: dict) -> None:
    """Persist the bundle to the DB row (the only write path, TASK-636)."""
    store = _credential_store()
    if store is None or not store.available:
        raise RuntimeError("CredentialStore unavailable — cannot persist Claude bundle")
    from ..providers.base import (  # noqa: PLC0415
        AUTH_CLI_OAUTH,
        STATUS_CONNECTED,
        ConnectionRecord,
    )

    store.save(
        ConnectionRecord(
            provider=_PROVIDER_ID,
            status=STATUS_CONNECTED,
            auth_method=AUTH_CLI_OAUTH,
            account=bundle.get("subscriptionType") or "claude-subscription",
            meta={
                "scopes": bundle.get("scopes") or [],
                "expires_at": bundle.get("expiresAt"),
                # Preserve the wizard login timestamp across refreshes — the
                # ~30-day session-lifetime warning keys off it (providers/claude.py).
                **(
                    {"connected_at": raw["connected_at"]}
                    if raw.get("connected_at")
                    else {}
                ),
            },
            secret={_SECRET_KEY: bundle},
            connected_at=raw.get("connected_at"),
        )
    )


def _expired(expires_at: int | None, *, buffer_ms: int = 0) -> bool:
    if not expires_at:
        return False
    return (int(time.time() * 1000) + buffer_ms) >= int(expires_at)


def _refresh_upstream(bundle: dict) -> dict:
    """Exchange the refresh token upstream. Caller must hold _REFRESH_LOCK."""
    refresh_token = bundle.get("refreshToken")
    if not refresh_token:
        raise RuntimeError("Claude credential has no refreshToken")
    scopes = bundle.get("scopes") or []
    resp = httpx.post(
        _OAUTH_TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _OAUTH_CLIENT_ID,
            "scope": " ".join(scopes),
        },
        headers={"Content-Type": "application/json"},
        timeout=15.0,
    )
    if resp.is_error:
        detail = (resp.text or "").strip().replace("\n", " ")[:300]
        raise RuntimeError(f"Claude OAuth refresh failed ({resp.status_code}): {detail}")
    payload = resp.json()
    return {
        **bundle,
        "accessToken": payload["access_token"],
        "refreshToken": payload.get("refresh_token", refresh_token),
        "expiresAt": int(time.time() * 1000) + int(payload["expires_in"]) * 1000,
        "scopes": payload.get("scope", "").split() if payload.get("scope") else scopes,
    }


def _refresh_serialized(
    *, buffer_ms: int, force: bool = False, rejected_token_sha256: str | None = None,
) -> dict | None:
    """Refresh + persist the bundle under the lock.

    Re-loads inside the lock and skips the upstream call if another caller
    already refreshed while we waited. Returns the current bundle (refreshed
    or not), or None if no bundle exists. Raises on upstream refresh failure.
    """
    with _REFRESH_LOCK:
        raw, bundle = _load()
        if not bundle:
            return None
        # A 401 refers to the token used by ONE request, not necessarily the
        # current bundle. Compare under the same lock as rotation: concurrent
        # rejected-A callers must reuse B, not each rotate B -> C -> D.
        if force and rejected_token_sha256:
            current_hash = hashlib.sha256(
                str(bundle.get("accessToken") or "").encode()
            ).hexdigest()
            if current_hash != rejected_token_sha256:
                force = False
        if not force and not _expired(bundle.get("expiresAt"), buffer_ms=buffer_ms):
            return bundle  # someone else refreshed while we waited
        try:
            refreshed = _refresh_upstream(bundle)
        except Exception as e:  # noqa: BLE001 — record then re-raise for callers
            _record_refresh_outcome(str(e))
            raise
        _record_refresh_outcome(None)
        try:
            _save(raw, refreshed)
        except Exception as e:  # noqa: BLE001
            logger.warning("Refreshed Claude token but could not persist to DB: %s", e)
        logger.info("Refreshed Claude OAuth token")
        return refreshed


def get_access_token(
    *, force_refresh: bool = False, rejected_token_sha256: str | None = None,
) -> UsageToken:
    """Resolve the app-owned Claude access token.

    Refreshes when near expiry or forced. A reader recovering from a 401
    supplies the rejected token's SHA-256 so an already-rotated bundle is
    reused instead of revoking tokens held by other concurrent requests.
    """
    raw, bundle = _load()
    if not bundle:
        return UsageToken(None, "missing")

    expired = _expired(bundle.get("expiresAt"))
    needs = force_refresh or _expired(bundle.get("expiresAt"), buffer_ms=_REFRESH_BUFFER_MS)
    if needs:
        try:
            bundle = _refresh_serialized(
                buffer_ms=_REFRESH_BUFFER_MS, force=force_refresh,
                rejected_token_sha256=rejected_token_sha256,
            ) or bundle
            expired = _expired(bundle.get("expiresAt"))
        except Exception as e:  # noqa: BLE001 — fall back to existing token
            logger.warning("Failed to refresh Claude token: %s", e)
            expired = _expired(bundle.get("expiresAt"))

    token = bundle.get("accessToken")
    if not token:
        return UsageToken(None, "missing")
    return UsageToken(token, "stale" if expired else "ok", bundle.get("expiresAt"))


def load_usage_token() -> UsageToken:
    """Legacy name used by the usage adapter — same resolution."""
    return get_access_token()


def bundle_status() -> dict:
    """Cleartext facts about the stored bundle (no tokens) — for honest UI status."""
    _, bundle = _load()
    if not bundle:
        return {"present": False}
    return {
        "present": True,
        "expired": _expired(bundle.get("expiresAt")),
        "expires_at": bundle.get("expiresAt"),
        "scopes": bundle.get("scopes") or [],
        "subscription": bundle.get("subscriptionType"),
        "mode": _mode(),
        **refresh_health(),
    }


async def proactive_refresh_loop() -> None:
    """Keep the bundle fresh forever — refresh at expiresAt - buffer.

    Started from the app lifespan. Sleeps only when no bundle exists yet.
    Failures are logged and retried next tick; the on-demand path in
    :func:`get_access_token` remains the backstop.
    """
    logger.info("Claude credential proactive refresh loop started (DB store)")
    while True:
        try:
            _, bundle = _load()
            if bundle and _expired(bundle.get("expiresAt"), buffer_ms=_PROACTIVE_BUFFER_MS):
                await asyncio.to_thread(
                    _refresh_serialized, buffer_ms=_PROACTIVE_BUFFER_MS
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("Proactive Claude token refresh failed (will retry): %s", e)
        await asyncio.sleep(_PROACTIVE_CHECK_S)
