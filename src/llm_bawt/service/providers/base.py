"""Provider adapter interface + encrypted credential store.

A ``ProviderAdapter`` describes one connectable provider and the auth method(s)
it supports. The wizard/settings UI drives adapters through the ``/v1/providers``
API. Credentials persist (encrypted) via :class:`CredentialStore`, which layers
on top of ``RuntimeSettingsStore`` (global scope) so we reuse the existing
``runtime_settings`` table instead of adding a new one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ...runtime_settings import RuntimeSettingsStore
from ...utils.config import Config
from . import crypto

logger = logging.getLogger(__name__)

# Auth methods a provider may advertise.
AUTH_DEVICE_OAUTH = "device_oauth"
AUTH_API_KEY = "api_key"
# CLI-driven OAuth: drive a first-party CLI (e.g. `claude setup-token`) under a
# PTY. Two-step + stateful — the authorize URL carries a per-session PKCE
# challenge, so the same process must live between start (URL) and complete (code).
AUTH_CLI_OAUTH = "cli_oauth"

# Connection status values.
STATUS_CONNECTED = "connected"
STATUS_PENDING = "pending"
STATUS_DISCONNECTED = "disconnected"
STATUS_ERROR = "error"

# Credential health states (TASK-637). ``warning`` is the load-bearing one: it
# means "still working, but will break — act now" (e.g. refresh chain failing
# while the access token is still valid).
HEALTH_OK = "ok"
HEALTH_WARNING = "warning"
HEALTH_BROKEN = "broken"
HEALTH_UNCONFIGURED = "unconfigured"
HEALTH_UNKNOWN = "unknown"

#: Health states under which a credential is usable right now.
CONNECTED_HEALTH_STATES = frozenset({HEALTH_OK, HEALTH_WARNING})


def health_block(
    state: str,
    *,
    detail: str | None = None,
    expires_at: int | None = None,
    last_refresh_at: int | None = None,
    fix: str | None = None,
) -> dict[str, Any]:
    """Normalized credential-health dict every adapter reports.

    ``expires_at``/``last_refresh_at`` are epoch **milliseconds** (matching the
    claudeAiOauth bundle convention). ``fix`` names the recovery action the UI
    should offer (``"reconnect"`` → the adapter's own auth flow, ``"cli"`` →
    an operator command described in ``detail``).
    """
    return {
        "state": state,
        "detail": detail,
        "expires_at": expires_at,
        "last_refresh_at": last_refresh_at,
        "fix": fix,
    }

_SETTING_PREFIX = "provider_connection:"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConnectionRecord:
    """Persisted state of a provider connection.

    ``secret`` holds sensitive material (tokens/keys) and is encrypted at rest;
    everything else is stored in cleartext so ``GET /v1/providers`` can report
    status without decrypting.
    """

    provider: str
    status: str = STATUS_DISCONNECTED
    auth_method: str | None = None
    account: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    secret: dict[str, Any] = field(default_factory=dict)
    connected_at: str | None = None
    updated_at: str | None = None

    def public(self) -> dict[str, Any]:
        """Serializable view WITHOUT secrets — safe for API responses."""
        return {
            "provider": self.provider,
            "status": self.status,
            "auth_method": self.auth_method,
            "account": self.account,
            "meta": self.meta,
            "connected": self.status == STATUS_CONNECTED,
            "connected_at": self.connected_at,
            "updated_at": self.updated_at,
        }


@dataclass
class DeviceFlowStart:
    user_code: str
    verification_uri: str
    device_code: str
    interval: int = 5
    expires_in: int = 900


@dataclass
class DevicePollResult:
    # status: pending | authorized | denied | slow_down | expired | error
    status: str
    record: ConnectionRecord | None = None
    detail: str | None = None


@dataclass
class ValidateResult:
    ok: bool
    account: str | None = None
    detail: str | None = None


@dataclass
class CliLoginStart:
    """Result of starting a CLI-driven OAuth login.

    ``verification_uri`` is the authorize URL the user opens in a browser;
    ``session_id`` ties the follow-up code submission back to the live process.
    """

    session_id: str
    verification_uri: str
    instructions: str | None = None


@dataclass
class CliLoginResult:
    # status: connected | error
    status: str
    record: ConnectionRecord | None = None
    detail: str | None = None


class CredentialStore:
    """Encrypted persistence for provider connections (global scope)."""

    def __init__(self, config: Config):
        self._store = RuntimeSettingsStore(config)

    @property
    def available(self) -> bool:
        return self._store.engine is not None

    def _key(self, provider: str) -> str:
        return f"{_SETTING_PREFIX}{provider}"

    def load(self, provider: str) -> ConnectionRecord | None:
        if not self.available:
            return None
        settings = self._store.get_scope_settings("global", "*")
        raw = settings.get(self._key(provider))
        if not isinstance(raw, dict):
            return None
        secret: dict[str, Any] = {}
        enc = raw.get("secret_enc")
        if enc:
            try:
                secret = json.loads(crypto.decrypt(enc))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to decrypt secret for provider %s: %s", provider, e)
        return ConnectionRecord(
            provider=provider,
            status=raw.get("status", STATUS_DISCONNECTED),
            auth_method=raw.get("auth_method"),
            account=raw.get("account"),
            meta=raw.get("meta") or {},
            secret=secret,
            connected_at=raw.get("connected_at"),
            updated_at=raw.get("updated_at"),
        )

    def save(self, record: ConnectionRecord) -> None:
        if not self.available:
            raise RuntimeError("Runtime settings DB unavailable — cannot persist provider credential")
        record.updated_at = _now()
        if record.status == STATUS_CONNECTED and not record.connected_at:
            record.connected_at = record.updated_at
        payload: dict[str, Any] = {
            "status": record.status,
            "auth_method": record.auth_method,
            "account": record.account,
            "meta": record.meta or {},
            "connected_at": record.connected_at,
            "updated_at": record.updated_at,
        }
        if record.secret:
            payload["secret_enc"] = crypto.encrypt(json.dumps(record.secret))
        self._store.set_value("global", "*", self._key(record.provider), payload)

    def delete(self, provider: str) -> bool:
        if not self.available:
            return False
        return self._store.delete_value("global", "*", self._key(provider))


class ProviderAdapter:
    """Base class for a connectable provider.

    Subclasses set ``id``, ``label``, ``auth_methods`` and implement whichever of
    the auth flows they advertise. Default implementations raise so an
    unsupported call surfaces clearly rather than silently no-op'ing.
    """

    id: str = ""
    label: str = ""
    auth_methods: tuple[str, ...] = ()
    #: ``access_paths.key`` values whose requests authenticate with this
    #: adapter's credential. The registry is the only place this mapping is
    #: read, and a key may have at most one owner.
    access_path_keys: tuple[str, ...] = ()

    def __init__(self, config: Config):
        self.config = config
        self.store = CredentialStore(config)

    # --- capability introspection -------------------------------------------
    def supports(self, method: str) -> bool:
        return method in self.auth_methods

    def descriptor(self) -> dict[str, Any]:
        rec = self.store.load(self.id)
        return {
            "id": self.id,
            "label": self.label,
            "auth_methods": list(self.auth_methods),
            "connection": (rec or ConnectionRecord(provider=self.id)).public(),
            "health": self.health(),
        }

    def health(self) -> dict[str, Any]:
        """Normalized credential health — override where real expiry is known.

        Default derives from the stored connection record only (fine for
        adapters whose credentials don't age, e.g. GitHub App mints
        installation tokens on demand).
        """
        rec = self.store.load(self.id)
        if not rec or rec.status == STATUS_DISCONNECTED:
            return health_block(HEALTH_UNCONFIGURED, detail="not connected")
        if rec.status == STATUS_ERROR:
            return health_block(
                HEALTH_BROKEN,
                detail=str(rec.meta.get("error") or "connection in error state"),
                fix="reconnect",
            )
        return health_block(HEALTH_OK)

    # --- device oauth --------------------------------------------------------
    def start_device_flow(self) -> DeviceFlowStart:
        raise NotImplementedError(f"{self.id} does not support device oauth")

    def poll_device_flow(self, device_code: str, user_code: str = "") -> DevicePollResult:
        raise NotImplementedError(f"{self.id} does not support device oauth")

    # --- cli oauth -----------------------------------------------------------
    def start_cli_login(self) -> "CliLoginStart":
        raise NotImplementedError(f"{self.id} does not support cli oauth")

    def complete_cli_login(self, session_id: str, code: str) -> "CliLoginResult":
        raise NotImplementedError(f"{self.id} does not support cli oauth")

    # --- api key -------------------------------------------------------------
    def set_api_key(self, api_key: str) -> ValidateResult:
        raise NotImplementedError(f"{self.id} does not support api key auth")

    # --- shared --------------------------------------------------------------
    def validate(self, record: ConnectionRecord) -> ValidateResult:
        raise NotImplementedError

    def disconnect(self) -> bool:
        return self.store.delete(self.id)
