"""Credential-bound, secret-free upstream observations (no probes or polling)."""
from __future__ import annotations

import hmac
import json
import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping

from .base import (
    AUTH_API_KEY, HEALTH_BROKEN, HEALTH_OK, HEALTH_UNCONFIGURED, HEALTH_WARNING,
    STATUS_CONNECTED, ConnectionRecord, CredentialStore, health_block,
)

logger = logging.getLogger(__name__)
WARNING_MS = 7 * 24 * 60 * 60 * 1000


def parse_expiry(value: str | None) -> int | None:
    """GitHub uses an HTTP date; tolerate ISO dates too. Unknown stays unknown."""
    if not isinstance(value, str) or not value:
        return None
    try:
        try:
            parsed = parsedate_to_datetime(value)
        except (ValueError, TypeError):
            parsed = datetime.fromisoformat(value.strip().replace(" UTC", "+00:00").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        stamp = parsed.timestamp() * 1000
        return int(stamp) if math.isfinite(stamp) and stamp > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def github_observation(status: int, headers: Mapping[str, str], body: bytes = b"") -> dict[str, Any]:
    headers = {k.lower(): v for k, v in headers.items()}
    meta: dict[str, Any] = {}
    expiry = parse_expiry(headers.get("github-authentication-token-expiration"))
    if expiry is not None:
        meta["expires_at"] = expiry
    # GitHub also returns 403 for rate limits. Those are not bad credentials.
    limited = (
        headers.get("x-ratelimit-remaining") == "0" or "retry-after" in headers
        or b"rate limit" in body[:4096].lower()
    )
    if status == 401 or (status == 403 and not limited):
        meta["auth_error_status"] = status
    elif 200 <= status < 400:
        meta["auth_error_status"] = None
    return meta


def api_key_health(record: ConnectionRecord | None, *, detail: str | None = None) -> dict:
    if record is None or record.status != STATUS_CONNECTED:
        return health_block(HEALTH_UNCONFIGURED, detail="not connected")
    expiry = record.meta.get("expires_at")
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or not math.isfinite(expiry) or expiry <= 0:
        expiry = None
    else:
        expiry = int(expiry)
    code = record.meta.get("auth_error_status")
    if code in (401, 403):
        reason = "Upstream rejected the API key (HTTP 401); reconnect."
        if code == 403:
            reason = "Upstream denied token access (HTTP 403); check token permissions and reconnect."
        return health_block(HEALTH_BROKEN, detail=reason, expires_at=expiry, fix="reconnect")
    remaining = expiry - datetime.now(timezone.utc).timestamp() * 1000 if expiry else None
    if remaining is not None and remaining <= 0:
        return health_block(HEALTH_BROKEN, detail="API key expired; reconnect.", expires_at=expiry, fix="reconnect")
    if remaining is not None and remaining <= WARNING_MS:
        return health_block(HEALTH_WARNING, detail="API key expires within 7 days; replace it before expiry.",
                            expires_at=expiry, fix="reconnect")
    return health_block(HEALTH_OK, detail=detail, expires_at=expiry)


def record_observation(store: CredentialStore, provider: str, key: str, meta: dict[str, Any]) -> bool:
    """CAS the metadata only; never overwrite a rotation/disconnect or leak errors."""
    if not meta or not key:
        return False
    try:
        return _record_observation(store, provider, key, meta)
    except Exception as exc:  # telemetry must not change the caller's operation
        logger.debug("Provider observation skipped (%s)", type(exc).__name__)
        return False


def _record_observation(store: CredentialStore, provider: str, key: str, meta: dict[str, Any]) -> bool:
    from sqlalchemy import select, update

    from ...runtime_settings import RuntimeSetting

    if not store.available:
        return False
    table = RuntimeSetting.__table__
    where = (table.c.scope_type == "global", table.c.scope_id == "*", table.c.key == store._key(provider))
    changed = False
    with store._store.engine.begin() as conn:
        raw = conn.execute(select(table.c.value_json).where(*where)).scalar_one_or_none()
        if raw is None:
            return False
        value = json.loads(raw)
        current = store._record(provider, value)
        current_key = current.secret.get("api_key")
        if (current.status != STATUS_CONNECTED or current.auth_method != AUTH_API_KEY
                or not isinstance(current_key, str) or not hmac.compare_digest(current_key, key)):
            return False
        observed = {k: v for k, v in meta.items() if k in ("expires_at", "auth_error_status")}
        # Permission failures may be endpoint-specific: an unrelated successful
        # request must not erase the warning and make the card flap.
        if current.meta.get("auth_error_status") == 403 and observed.get("auth_error_status") is None:
            observed.pop("auth_error_status", None)
        updated = {**current.meta, **observed}
        if updated == current.meta:
            return False
        now = datetime.now(timezone.utc)
        value.update(meta=updated, updated_at=now.isoformat())
        changed = conn.execute(update(table).where(*where, table.c.value_json == raw).values(
            value_json=json.dumps(value, ensure_ascii=False), updated_at=now,
        )).rowcount == 1
    if changed:
        from ..health_changes import publish

        publish(f"credential observation: {provider}")
    return changed


def observe_github_response(provider: str, key: str, status: int, headers: Mapping[str, str],
                            body: bytes = b"") -> None:
    """Only existing stored API keys can be updated, even for injected callers."""
    meta = github_observation(status, headers, body)
    if not meta:
        return
    try:
        from ...utils.config import Config

        record_observation(CredentialStore(Config()), provider, key, meta)
    except Exception as exc:
        logger.debug("Provider observation unavailable (%s)", type(exc).__name__)
