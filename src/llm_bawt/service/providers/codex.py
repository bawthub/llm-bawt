"""Codex (ChatGPT subscription) — provider adapter with device-code login (TASK-773).

Implements the OpenAI device-code OAuth flow so users can connect their ChatGPT
subscription from the BawtHub UI without SSH + ``codex login``.

Flow (non-standard — custom OpenAI endpoints, not RFC 8628):
  1. ``POST /api/accounts/deviceauth/usercode`` → ``device_auth_id``, ``user_code``
  2. User visits ``https://auth.openai.com/codex/device``, enters the user code
  3. ``POST /api/accounts/deviceauth/token`` → polls until the user authorizes →
     returns ``authorization_code`` + ``code_verifier``
  4. ``POST /oauth/token`` exchanges the auth code → ``access_token``,
     ``refresh_token``, ``id_token``
  5. Token bundle stored encrypted in the DB; the codex bridge materializes it
     on next turn via the broker endpoint.

Health: the adapter also monitors the stored token's JWT ``exp`` claim.
Expired → ``broken`` (immediate UI action item), missing → ``unconfigured``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..usage.codex_oauth import refresh_health
from .base import (
    AUTH_DEVICE_OAUTH,
    HEALTH_BROKEN,
    HEALTH_OK,
    HEALTH_UNCONFIGURED,
    HEALTH_WARNING,
    STATUS_CONNECTED,
    ConnectionRecord,
    DeviceFlowStart,
    DevicePollResult,
    ProviderAdapter,
    health_block,
)

logger = logging.getLogger(__name__)

# ── Device-code OAuth constants (codex-rs/login/src/auth/manager.rs) ──

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_CLIENT_ID_ENV = "CODEX_APP_SERVER_LOGIN_CLIENT_ID"
_ISSUER = "https://auth.openai.com"
_DEVICE_CODE_URL = f"{_ISSUER}/api/accounts/deviceauth/usercode"
_DEVICE_POLL_URL = f"{_ISSUER}/api/accounts/deviceauth/token"
_TOKEN_EXCHANGE_URL = f"{_ISSUER}/oauth/token"
_DEVICE_REDIRECT_URI = f"{_ISSUER}/deviceauth/callback"
_VERIFICATION_URI = f"{_ISSUER}/codex/device"

_TIMEOUT = 15.0
_POLL_EXPIRES_IN = 900  # 15 minutes

# DB provider key and secret envelope key (matches codex_oauth.py).
_DB_PROVIDER_ID = "openai_chatgpt"
_DB_SECRET_KEY = "codex_auth"


def _client_id() -> str:
    return os.getenv(_CLIENT_ID_ENV) or _CLIENT_ID


def codex_auth_path() -> Path:
    return Path(os.getenv("CODEX_AUTH_PATH") or (Path.home() / ".codex" / "auth.json"))


def _jwt_claims(token: str) -> dict[str, Any]:
    """Best-effort unverified decode of a JWT payload — scheduling facts only,
    never a trust decision (same rationale as the bridge proxy's ``_jwt_exp``)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def codex_auth_status() -> dict[str, Any]:
    """Cleartext facts about the codex OAuth bundle (no tokens)."""
    path = codex_auth_path()
    if not path.exists():
        return {"present": False}
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read codex auth bundle %s: %s", path, e)
        return {"present": False, "error": str(e)}
    tokens = data.get("tokens") or {}
    access = tokens.get("access_token")
    if not access:
        return {"present": False}
    claims = _jwt_claims(access)
    id_claims = _jwt_claims(tokens.get("id_token") or "")
    auth_claims = id_claims.get("https://api.openai.com/auth") or {}
    exp = claims.get("exp")
    return {
        "present": True,
        # epoch ms, matching the health-block convention.
        "expires_at": int(exp) * 1000 if exp else None,
        "account": id_claims.get("email"),
        "plan": auth_claims.get("chatgpt_plan_type"),
        "last_refresh": data.get("last_refresh"),
    }


class CodexAdapter(ProviderAdapter):
    id = "codex"
    label = "Codex (ChatGPT)"
    auth_methods = (AUTH_DEVICE_OAUTH,)
    access_path_keys = ("openai-oauth",)

    def descriptor(self) -> dict[str, Any]:
        desc = super().descriptor()
        db_record = self.store.load(_DB_PROVIDER_ID)
        status = self._status_from_record(db_record) or codex_auth_status()
        if status.get("present"):
            conn = desc["connection"]
            conn["connected"] = True
            conn["status"] = STATUS_CONNECTED
            conn["account"] = (
                (db_record.account if db_record else None)
                or status.get("account")
                or "chatgpt-subscription"
            )
            conn["meta"] = {
                **(db_record.meta if db_record else {}),
                **(conn.get("meta") or {}),
                "credentials_path": str(codex_auth_path()),
                "plan": status.get("plan"),
                "expires_at": status.get("expires_at"),
                "last_refresh": status.get("last_refresh"),
            }
        return desc

    @staticmethod
    def _status_from_record(record: ConnectionRecord | None) -> dict[str, Any] | None:
        if not record or not record.secret:
            return None
        bundle = record.secret.get(_DB_SECRET_KEY) or record.secret
        tokens = bundle.get("tokens") or bundle
        access = tokens.get("access_token")
        if not access:
            return None
        claims = _jwt_claims(access)
        exp = claims.get("exp")
        id_claims = _jwt_claims(tokens.get("id_token") or "")
        auth_claims = id_claims.get("https://api.openai.com/auth") or {}
        return {
            "present": True,
            "expires_at": int(exp) * 1000 if exp else None,
            "account": id_claims.get("email"),
            "plan": auth_claims.get("chatgpt_plan_type"),
            "last_refresh": (record.meta or {}).get("last_refresh"),
        }

    def _status_from_db(self) -> dict[str, Any] | None:
        """Try to derive codex auth status from the DB credential store.

        The codex bridge syncs the OAuth bundle into
        ``provider_connection:openai_chatgpt`` with the tokens encrypted in
        ``secret_enc``.  The decrypted secret nests:
        ``{"codex_auth": {"tokens": {"access_token": "…", …}, …}}``.
        We extract the JWT ``exp`` claim — same facts ``codex_auth_status()``
        derives from the filesystem file.
        """
        for provider_key in (_DB_PROVIDER_ID, self.id):
            status = self._status_from_record(self.store.load(provider_key))
            if status is not None:
                return status
        return None

    def health(self) -> dict[str, Any]:
        # Prefer DB credential store (available in every container), fall back
        # to filesystem auth.json (only in the codex bridge container).
        status = self._status_from_db() or codex_auth_status()
        if not status.get("present"):
            return health_block(
                HEALTH_UNCONFIGURED,
                detail=(
                    "No codex credential — connect your ChatGPT subscription "
                    "to enable codex-routed bots."
                ),
                fix="reconnect",
            )
        expires_at = status.get("expires_at")
        now_ms = int(time.time() * 1000)
        token_expired = expires_at is not None and now_ms >= expires_at

        if token_expired:
            return health_block(
                HEALTH_BROKEN,
                detail=(
                    "Access token expired — codex bots will fail with auth "
                    "errors. Reconnect your ChatGPT subscription."
                ),
                expires_at=expires_at,
                fix="reconnect",
            )

        # The JWT exp claim only proves the token *should* be valid. If the
        # in-app refresh chain is failing (e.g. a 401-triggered force refresh
        # couldn't rotate the pair), the credential is effectively dead even
        # with a future exp — surface it as an action item instead of "ok".
        chain = refresh_health()
        err_at = chain.get("last_refresh_error_at")
        ok_at = chain.get("last_refresh_at")
        if err_at is not None and (ok_at is None or err_at > ok_at):
            return health_block(
                HEALTH_WARNING,
                detail=(
                    "Token refresh is failing — codex may reject requests "
                    f"despite an unexpired token: {chain.get('last_refresh_error')}"
                ),
                expires_at=expires_at,
                last_refresh_at=ok_at,
                fix="reconnect",
            )
        return health_block(HEALTH_OK, expires_at=expires_at, last_refresh_at=ok_at)

    # --- device-code OAuth (TASK-773) -----------------------------------------

    def start_device_flow(self) -> DeviceFlowStart:
        """POST to OpenAI's device-code endpoint, return user code + URL."""
        client_id = _client_id()
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(_DEVICE_CODE_URL, json={"client_id": client_id})
        if resp.status_code == 404:
            raise RuntimeError("Device-code login not supported by this OpenAI server")
        resp.raise_for_status()
        data = resp.json()
        device_auth_id = data.get("device_auth_id") or data.get("device_code")
        user_code = data.get("user_code") or data.get("usercode") or data.get("userCode")
        if not device_auth_id or not user_code:
            raise RuntimeError(f"unexpected device-code response: {data!r}")
        interval = 5
        raw_interval = data.get("interval")
        if raw_interval is not None:
            try:
                interval = max(3, int(str(raw_interval).strip()))
            except (ValueError, TypeError):
                pass
        return DeviceFlowStart(
            user_code=user_code,
            verification_uri=_VERIFICATION_URI,
            device_code=device_auth_id,
            interval=interval,
            expires_in=_POLL_EXPIRES_IN,
        )

    def poll_device_flow(self, device_code: str, user_code: str = "") -> DevicePollResult:
        """Poll the device-code token endpoint.

        ``device_code`` is the ``device_auth_id`` from ``start_device_flow``.
        ``user_code`` is the human-visible code the user entered at the provider.
        On success, exchanges the returned authorization_code for an OAuth
        token bundle and persists it to the DB.
        """
        device_auth_id = device_code

        client_id = _client_id()
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                _DEVICE_POLL_URL,
                json={"device_auth_id": device_auth_id, "user_code": user_code},
            )

        # 403/404 = user hasn't approved yet
        if resp.status_code in (403, 404):
            return DevicePollResult(status="pending")

        if not resp.is_success:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            # OpenAI returns errors in two formats:
            #   Standard OAuth: {"error": "string", "error_description": "string"}
            #   OpenAI-style:   {"error": {"message": "...", "code": "..."}}
            raw_error = data.get("error", "")
            if isinstance(raw_error, dict):
                error = raw_error.get("code", "")
                desc = raw_error.get("message", "")
            else:
                error = str(raw_error)
                desc = data.get("error_description", "")
            if error in ("authorization_declined", "access_denied"):
                return DevicePollResult(status="denied", detail=desc or "authorization declined")
            if error == "slow_down":
                return DevicePollResult(status="slow_down")
            return DevicePollResult(
                status="error",
                detail=desc or f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        # Success — extract authorization_code + code_verifier, exchange for tokens.
        poll_data = resp.json()
        auth_code = poll_data.get("authorization_code")
        code_verifier = poll_data.get("code_verifier")
        if not auth_code:
            return DevicePollResult(status="error", detail="no authorization_code in poll response")

        # Exchange the authorization code for real tokens.
        exchange_body = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": _DEVICE_REDIRECT_URI,
            "client_id": client_id,
        }
        if code_verifier:
            exchange_body["code_verifier"] = code_verifier

        with httpx.Client(timeout=_TIMEOUT) as client:
            token_resp = client.post(
                _TOKEN_EXCHANGE_URL,
                data=exchange_body,  # form-encoded
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if not token_resp.is_success:
            return DevicePollResult(
                status="error",
                detail=f"token exchange failed ({token_resp.status_code}): {token_resp.text[:200]}",
            )

        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        id_token = token_data.get("id_token")

        if not access_token or not refresh_token:
            return DevicePollResult(
                status="error",
                detail="token exchange returned no access_token/refresh_token",
            )

        # Derive account_id from the id_token JWT.
        account_id = None
        if id_token:
            id_claims = _jwt_claims(id_token)
            auth_info = id_claims.get("https://api.openai.com/auth") or {}
            # account_id is in the proxy adapter's "account_id" field
            # or under the auth info's user_id / sub claim.
            account_id = (
                auth_info.get("chatgpt_account_id")
                or auth_info.get("account_id")
                or auth_info.get("user_id")
                or id_claims.get("chatgpt_account_id")
                or id_claims.get("account_id")
                or id_claims.get("sub")
            )

        # Build the auth.json-shaped bundle (same structure the bridge expects).
        bundle: dict[str, Any] = {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "last_refresh": datetime.now(timezone.utc).isoformat(),
            "tokens": {
                "id_token": id_token or "",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "account_id": account_id or "",
            },
        }

        # Persist to the DB so the bridge can materialize it.
        try:
            self._save_bundle(bundle, account_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Codex login succeeded but failed to persist: %s", e)
            return DevicePollResult(
                status="error",
                detail=f"login succeeded but could not persist credential: {e}",
            )

        record = ConnectionRecord(
            provider=_DB_PROVIDER_ID,
            status=STATUS_CONNECTED,
            auth_method=AUTH_DEVICE_OAUTH,
            account="chatgpt-subscription",
            meta={
                "auth_mode": "chatgpt",
                "account_id": account_id,
                "last_refresh": bundle["last_refresh"],
            },
        )
        return DevicePollResult(status="authorized", record=record)

    def _save_bundle(self, bundle: dict[str, Any], account_id: str | None) -> None:
        """Persist the full auth bundle to the DB credential store."""
        record = ConnectionRecord(
            provider=_DB_PROVIDER_ID,
            status=STATUS_CONNECTED,
            auth_method=AUTH_DEVICE_OAUTH,
            account="chatgpt-subscription",
            meta={
                "auth_mode": bundle.get("auth_mode", "chatgpt"),
                "account_id": account_id,
                "last_refresh": bundle.get("last_refresh"),
            },
            secret={_DB_SECRET_KEY: bundle},
        )
        self.store.save(record)
        logger.info("Saved codex OAuth bundle to DB (account=%s)", account_id)

    def disconnect(self) -> bool:
        """Clear both the adapter's own key and the shared provider key."""
        removed_adapter = self.store.delete(self.id)
        removed_bundle = self.store.delete(_DB_PROVIDER_ID)
        return removed_adapter or removed_bundle
