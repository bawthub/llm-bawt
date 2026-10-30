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

# Every runtime_settings key prefix a CredentialStore writes (this class and
# integrations.connections.IntegrationStore). The generic /v1/settings routes
# have no service-token auth, so they neither list, write nor delete these rows
# (TASK-1520); the provider and integration APIs are the only way in.
CREDENTIAL_SETTING_PREFIXES: tuple[str, ...] = (_SETTING_PREFIX, "integration_connection:")


def is_credential_setting_key(key: str) -> bool:
    """True when ``key`` names (or imitates) a CredentialStore row.

    Case and surrounding whitespace are ignored, so a look-alike key can't be
    planted through the generic settings routes either.
    """
    return key.strip().lower().startswith(CREDENTIAL_SETTING_PREFIXES)


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
        return self._record(provider, raw)

    @staticmethod
    def _record(provider: str, raw: dict[str, Any]) -> ConnectionRecord:
        """Decode a stored row's JSON into a record (secret decrypted)."""
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
        self._store.set_value("global", "*", self._key(record.provider), self._payload(record))

    @staticmethod
    def _payload(record: ConnectionRecord) -> dict[str, Any]:
        """Stamp ``record`` and build the stored row JSON (secret encrypted)."""
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
        return payload

    @staticmethod
    def durable_key_problem() -> str | None:
        """``None`` when records are encrypted under the persisted root, else a code.

        ``ephemeral_root`` means the active key exists only in this process, so
        what this process writes the next one can't read. Any other failure (an
        unusable key) is reported by exception type only.
        """
        try:
            crypto.require_persistent_root()
        except crypto.EphemeralKeyError:
            return "ephemeral_root"
        except Exception as error:  # noqa: BLE001 - never echo: may include key material
            return f"error:{type(error).__name__}"
        return None

    def create_if_absent(self, record: ConnectionRecord, *, conn: Any = None) -> bool:
        """Store ``record`` only if no row exists for it; True when this call inserted it.

        A single ``INSERT … ON CONFLICT DO NOTHING`` on the scope/key unique
        index, so a row in ANY status (including an operator's disconnected
        one) is never overwritten, even by a concurrent writer. Pass ``conn``
        to insert inside the caller's transaction.
        """
        if not self.available:
            raise RuntimeError("Runtime settings DB unavailable — cannot persist provider credential")
        if conn is not None:
            return self._insert_absent(conn, record)
        with self._store.engine.begin() as own:
            return self._insert_absent(own, record)

    def _insert_absent(self, conn: Any, record: ConnectionRecord) -> bool:
        from sqlalchemy.dialects import postgresql, sqlite

        from ...runtime_settings import RuntimeSetting

        dialect = {"postgresql": postgresql, "sqlite": sqlite}[conn.dialect.name]
        statement = dialect.insert(RuntimeSetting.__table__).values(
            scope_type="global", scope_id="*", key=self._key(record.provider),
            value_json=json.dumps(self._payload(record), ensure_ascii=False),
            updated_at=datetime.now(timezone.utc),
        ).on_conflict_do_nothing(index_elements=["scope_type", "scope_id", "key"])
        return conn.execute(statement).rowcount == 1

    def load_in(self, conn: Any, provider: str) -> ConnectionRecord | None:
        """:meth:`load` through ``conn``, so it sees that transaction's own writes."""
        from sqlalchemy import select

        from ...runtime_settings import RuntimeSetting

        table = RuntimeSetting.__table__
        raw = conn.execute(select(table.c.value_json).where(
            table.c.scope_type == "global", table.c.scope_id == "*",
            table.c.key == self._key(provider),
        )).scalar_one_or_none()
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except ValueError:
            return None
        return self._record(provider, value) if isinstance(value, dict) else None

    def exists_in(self, conn: Any, provider: str) -> bool:
        """True when ANY row (any status, even undecodable) exists for ``provider``."""
        from sqlalchemy import select

        from ...runtime_settings import RuntimeSetting

        table = RuntimeSetting.__table__
        return conn.execute(select(table.c.id).where(
            table.c.scope_type == "global", table.c.scope_id == "*",
            table.c.key == self._key(provider),
        )).first() is not None

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
