"""The ONE Claude login adapter (TASK-635) — mints the app-owned bundle.

Collapses the old ``claude-sub`` (bridge inference) + ``claude-usage`` (usage
panel) dual auth into a single wizard flow. It drives the FULL login —
``claude auth login --claudeai`` — whose scope set covers everything the
deployment needs (``user:inference`` for the claude-code bridge,
``user:profile`` for ``/v1/usage``), in an ISOLATED throwaway config dir so it
can never rotate or clobber an interactive TUI login.

TASK-636: the minted ``claudeAiOauth`` bundle is stored in the encrypted
CredentialStore row (``secret["claudeAiOauth"]`` — Fernet at rest) instead of
a host file. The DB row is the single source of truth:

* the app's owned-mode refresher (``usage/claude_oauth.py``) reads/writes the
  row and keeps it fresh forever (proactive loop + on-demand), and
* the claude-code bridge consumes it via the
  ``GET /v1/providers/claude/token`` broker endpoint — readers never refresh,
  and no credential bind mounts exist.

Reuses the PTY engine from :class:`ClaudeSubAdapter` (which remains as the
unregistered base class).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..usage.claude_oauth import bundle_status, claude_credentials_path
from .base import (
    AUTH_CLI_OAUTH,
    HEALTH_BROKEN,
    HEALTH_OK,
    HEALTH_UNCONFIGURED,
    HEALTH_WARNING,
    STATUS_CONNECTED,
    ConnectionRecord,
    health_block,
)
from .claude_sub import ClaudeSubAdapter

logger = logging.getLogger(__name__)

_TMP_PREFIX = "claude-login-"
# Observed hard max session lifetime (TASK-637): the OAuth grant dies ~30 days
# after login REGARDLESS of refresh rotation — empirical: the 2026-06-21 usage
# login died 2026-07-21 with `invalid_grant — Refresh token expired`, on its
# 30-day anniversary, despite fresh rotations. Rotation keeps the token pair
# alive WITHIN the session; it does not extend the session itself. Warn a few
# days ahead so re-login is a planned 15-second click, not an outage.
_SESSION_MAX_AGE_DAYS = 30
_SESSION_WARN_AGE_DAYS = 27
# Scopes the single credential must carry to serve every consumer.
_REQUIRED_SCOPES = ("user:inference", "user:profile")
# Legacy adapter ids whose stored connection records this adapter supersedes.
_LEGACY_IDS = ("claude-sub", "claude-usage")


class ClaudeAdapter(ClaudeSubAdapter):
    id = "claude"
    label = "Claude"
    auth_methods = (AUTH_CLI_OAUTH,)
    access_path_keys = ("anthropic-oauth",)

    # --- hooks --------------------------------------------------------------
    def _cli_args(self) -> list[str]:
        return ["auth", "login", "--claudeai"]

    def _login_config_dir(self) -> Path:
        # Isolated per-session dir — NEVER a live config dir. Reaped sessions
        # may strand a dir under /tmp; that's container-ephemeral.
        return Path(tempfile.mkdtemp(prefix=_TMP_PREFIX))

    def _cleanup_login_dir(self, config_dir: Path) -> None:
        if _TMP_PREFIX in config_dir.name:
            shutil.rmtree(config_dir, ignore_errors=True)

    def _build_record(
        self, bundle: dict | None, token: str | None, creds_path: Path
    ) -> ConnectionRecord:
        if not bundle:
            raise RuntimeError(
                "login finished but no claudeAiOauth bundle was written — "
                "cannot install the Claude credential"
            )
        scopes = bundle.get("scopes") or []
        missing = [s for s in _REQUIRED_SCOPES if scopes and s not in scopes]
        if missing:
            raise RuntimeError(
                f"minted bundle lacks required scope(s) {missing} (got {scopes}) — "
                "it could not serve both inference and usage; not installing"
            )

        self._cleanup_login_dir(creds_path.parent)
        # TASK-636: the bundle goes into the encrypted CredentialStore row —
        # the DB is the single source of truth. The caller's save() persists
        # both the record and the secret (Fernet at rest).
        logger.info("Installed fresh Claude credential into the encrypted DB store")

        return ConnectionRecord(
            provider=self.id,
            status=STATUS_CONNECTED,
            auth_method=AUTH_CLI_OAUTH,
            account=bundle.get("subscriptionType") or "claude-subscription",
            meta={
                "scopes": scopes,
                "expires_at": bundle.get("expiresAt"),
            },
            secret={"claudeAiOauth": bundle},
        )

    # --- honest status -------------------------------------------------------
    def descriptor(self) -> dict[str, Any]:
        """Report truth from the DB-backed bundle, not just wizard bookkeeping.

        The bundle may predate this adapter (minted by the legacy flow or
        imported from the legacy file on first load) — if a usable bundle
        exists, the deployment IS connected regardless of stored bookkeeping.
        """
        desc = super().descriptor()
        status = bundle_status()
        if status.get("present"):
            conn = desc["connection"]
            conn["connected"] = True
            conn["status"] = STATUS_CONNECTED
            if not conn.get("auth_method"):
                conn["auth_method"] = AUTH_CLI_OAUTH
            if not conn.get("account"):
                conn["account"] = status.get("subscription") or "claude-subscription"
            conn["meta"] = {
                **(conn.get("meta") or {}),
                "scopes": status.get("scopes"),
                "expires_at": status.get("expires_at"),
                "expired": status.get("expired"),
            }
        return desc

    def health(self) -> dict[str, Any]:
        """Real credential health from the bundle + refresh-chain outcomes.

        The warning state is the whole point (TASK-637): the app refreshes at
        expiry−20min, so when refresh starts FAILING there is up to ~8h of
        runway while the access token is still valid — surface that window
        instead of waiting for the hard break.
        """
        status = bundle_status()
        if not status.get("present"):
            return health_block(
                HEALTH_UNCONFIGURED,
                detail="No Claude credential — connect to enable claude-code bots and usage.",
                fix="reconnect",
            )
        expires_at = status.get("expires_at")
        last_ok = status.get("last_refresh_at")
        err = status.get("last_refresh_error")
        if status.get("expired"):
            detail = "Access token expired — claude-code bots and usage are down."
            if err:
                detail += f" Last refresh error: {err}"
            return health_block(
                HEALTH_BROKEN,
                detail=detail,
                expires_at=expires_at,
                last_refresh_at=last_ok,
                fix="reconnect",
            )
        if err:
            return health_block(
                HEALTH_WARNING,
                detail=(
                    "Auto-refresh is failing — the access token still works but "
                    f"will not renew. Reconnect before it expires. ({err})"
                ),
                expires_at=expires_at,
                last_refresh_at=last_ok,
                fix="reconnect",
            )
        # Session-anniversary warning: the grant has a hard ~30-day lifetime
        # from login (see _SESSION_MAX_AGE_DAYS) that refresh rotation cannot
        # extend. connected_at is stamped by each wizard login, so it IS the
        # session birth time.
        session_death = self._estimated_session_death_ms()
        if session_death is not None:
            now_ms = int(time.time() * 1000)
            warn_from = session_death - (
                (_SESSION_MAX_AGE_DAYS - _SESSION_WARN_AGE_DAYS) * 86_400_000
            )
            if now_ms >= warn_from:
                days_left = max(0, (session_death - now_ms) // 86_400_000)
                return health_block(
                    HEALTH_WARNING,
                    detail=(
                        f"This Claude login session is ~{_SESSION_MAX_AGE_DAYS - days_left} "
                        f"days old — sessions hard-expire at ~{_SESSION_MAX_AGE_DAYS} days "
                        "regardless of refresh. Reconnect at your convenience to avoid "
                        "a hard stop."
                    ),
                    expires_at=session_death,
                    last_refresh_at=last_ok,
                    fix="reconnect",
                )
        if status.get("mode") == "shared":
            return health_block(
                HEALTH_OK,
                detail="Shared mode — an external owner refreshes this credential.",
                expires_at=expires_at,
                last_refresh_at=last_ok,
            )
        return health_block(
            HEALTH_OK, expires_at=expires_at, last_refresh_at=last_ok
        )

    def _estimated_session_death_ms(self) -> int | None:
        """connected_at + observed max session lifetime, or None if unknown.

        Falls back to the legacy claude-usage/claude-sub records — a login
        minted through the pre-TASK-635 flow is the same session, just filed
        under the old id.
        """
        for pid in (self.id, *_LEGACY_IDS):
            rec = self.store.load(pid)
            if not rec or not rec.connected_at:
                continue
            try:
                born = datetime.fromisoformat(rec.connected_at)
            except ValueError:
                continue
            return int(born.timestamp() * 1000) + _SESSION_MAX_AGE_DAYS * 86_400_000
        return None

    def disconnect(self) -> bool:
        ok = self.store.delete(self.id)
        for legacy in _LEGACY_IDS:  # supersede old dual-flow records
            self.store.delete(legacy)
        target = claude_credentials_path()
        try:
            if target.exists():
                target.write_text("{}")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to clear Claude credential: %s", e)
        return ok
