"""Shared API-key provider machinery (TASK-825).

Two pieces live here:

- :class:`ApiKeyAdapter` — base class for straight API-key providers (xAI,
  OpenAI platform, Anthropic API). Subclasses supply ``id``/``label`` and a
  ``_probe`` that validates the key upstream and returns an account label.
  OpenRouter (TASK-822) predates this base and keeps its own copy of the same
  shape; consolidating it here is a follow-up, not a requirement.

- :func:`resolve_api_key` — the DB-first key resolution used by the LLM
  clients: explicit key -> encrypted CredentialStore -> env fallback (kept for
  dev/backward-compat, logs a deprecation note when used). This is what
  retires env vars as the source of truth for LLM credentials.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from ...utils.config import Config
from .base import (
    AUTH_API_KEY,
    HEALTH_BROKEN,
    HEALTH_OK,
    HEALTH_UNCONFIGURED,
    HEALTH_UNKNOWN,
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    ConnectionRecord,
    ProviderAdapter,
    ValidateResult,
    health_block,
)

logger = logging.getLogger(__name__)

_SECRET_FIELD = "api_key"


class ApiKeyAdapter(ProviderAdapter):
    """Base class for providers whose only auth method is a static API key.

    Subclasses set ``id``, ``label`` and implement ``_probe(key)`` returning
    ``(account_label, error_detail)`` — exactly one of the two is non-None.
    """

    auth_methods = (AUTH_API_KEY,)
    #: Disconnect leaves a key-free ``disconnected`` record instead of deleting
    #: the row. Set when a reader falls back to legacy env while no record
    #: exists (:func:`resolve_stored_api_key`, the bridge broker's 503), so a
    #: disconnect can't bring the env key back.
    keeps_disconnected_record: bool = False

    def disconnect(self) -> bool:
        if not self.keeps_disconnected_record:
            return super().disconnect()
        record = self.store.load(self.id)
        if record is None or record.status == STATUS_DISCONNECTED:
            return False
        self.store.save(ConnectionRecord(
            provider=self.id, status=STATUS_DISCONNECTED, auth_method=AUTH_API_KEY,
        ))
        return True

    def health(self) -> dict:
        if not self.keeps_disconnected_record:
            return super().health()
        # Same strict read as the broker and resolve_stored_api_key, so a
        # route is assignable only when the key its readers get is usable.
        from ..connection_secrets import (  # noqa: PLC0415 — import cycle
            ConnectionUnavailable,
            FieldSpec,
            read_connection,
        )

        try:
            found = read_connection(
                self.config, "provider", self.id, FieldSpec(secret=(_SECRET_FIELD,)), store=self.store,
            )
        except ConnectionUnavailable as exc:
            if exc.code == "not_connected":
                return health_block(HEALTH_UNCONFIGURED, detail="not connected")
            if exc.code == "unreadable":
                return health_block(
                    HEALTH_BROKEN, detail=f"{self.label} API key is unreadable; reconnect.", fix="reconnect",
                )
            return health_block(HEALTH_UNKNOWN, detail="credential store unavailable")
        if found is None:
            return health_block(HEALTH_UNCONFIGURED, detail="not connected")
        return health_block(HEALTH_OK)

    # --- api key -------------------------------------------------------------
    def set_api_key(self, api_key: str) -> ValidateResult:
        key = (api_key or "").strip()
        if not key:
            return ValidateResult(ok=False, detail="API key is empty")
        account, detail = self._probe(key)
        if account is None:
            return ValidateResult(ok=False, detail=detail)
        record = ConnectionRecord(
            provider=self.id,
            status=STATUS_CONNECTED,
            auth_method=AUTH_API_KEY,
            account=account,
            secret={_SECRET_FIELD: key},
        )
        self.store.save(record)
        logger.info("%s API key connected (account=%s)", self.label, account)
        return ValidateResult(ok=True, account=account)

    def validate(self, record: ConnectionRecord) -> ValidateResult:
        key = (record.secret or {}).get(_SECRET_FIELD)
        if not key:
            return ValidateResult(ok=False, detail="no API key stored")
        account, detail = self._probe(key)
        if account is None:
            return ValidateResult(ok=False, detail=detail)
        return ValidateResult(ok=True, account=account)

    # --- broker / in-process access ------------------------------------------
    def api_key(self) -> str | None:
        """Decrypted key for internal consumers; None if not connected."""
        record = self.store.load(self.id)
        if not record or record.status != STATUS_CONNECTED:
            return None
        return (record.secret or {}).get(_SECRET_FIELD) or None

    # --- subclass hook --------------------------------------------------------
    def _probe(self, key: str) -> tuple[str | None, str | None]:
        raise NotImplementedError


def resolve_api_key(
    config: Config,
    provider_id: str,
    env_vars: Sequence[str] = (),
    explicit: str | None = None,
    config_attr: str | None = None,
) -> str | None:
    """Resolve an LLM provider API key, DB-first.

    Order (first hit wins):
    1. ``explicit`` — a key handed in by the caller (model_definition.api_key).
    2. Encrypted :class:`CredentialStore` via the provider adapter (the
       canonical source since TASK-636/825).
    3. ``env_vars`` / ``config_attr`` — legacy env fallback, kept so dev
       setups keep working. Logs a deprecation note when hit.

    Returns None when no key is found anywhere.
    """
    if explicit:
        return explicit

    try:
        from .registry import get_adapter

        adapter = get_adapter(config, provider_id)
        key = adapter.api_key() if isinstance(adapter, ApiKeyAdapter) else None
        if key:
            return key
    except Exception as e:  # noqa: BLE001 — DB down must not kill the client path
        logger.debug("CredentialStore lookup failed for %s: %s", provider_id, e)

    for var in env_vars:
        val = os.getenv(var)
        if val:
            logger.warning(
                "Using %s from environment for provider '%s' — deprecated; "
                "connect the key via the providers UI (CredentialStore) instead.",
                var,
                provider_id,
            )
            return val

    if config_attr:
        val = getattr(config, config_attr, None)
        if val:
            logger.warning(
                "Using Config.%s for provider '%s' — deprecated; connect the "
                "key via the providers UI (CredentialStore) instead.",
                config_attr,
                provider_id,
            )
            return val

    return None


# Providers whose env fallback has already logged its deprecation warning.
_env_fallback_warned: set[str] = set()


def resolve_stored_api_key(config: Config, provider_id: str, env_vars: Sequence[str] = ()) -> str | None:
    """A provider's API key, record-first and fail-closed (TASK-1476).

    The ``provider_connection:<id>`` record wins. ``env_vars`` are read only
    when no record exists, with one warning per provider that names the vars
    and never a value. A record that exists but can't be used (disconnected,
    undecryptable, store down) raises
    :class:`~llm_bawt.service.connection_secrets.ConnectionUnavailable`; it
    never falls back to env. Returns None when there is no record and no env.
    """
    from ..connection_secrets import FieldSpec, read_connection  # noqa: PLC0415 — import cycle

    found = read_connection(config, "provider", provider_id, FieldSpec(secret=(_SECRET_FIELD,)))
    if found is not None:
        return found.secret[_SECRET_FIELD]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if provider_id not in _env_fallback_warned:
                _env_fallback_warned.add(provider_id)
                logger.warning(
                    "Using %s from the environment for provider '%s' because no "
                    "connection is stored (deprecated; connect the key in "
                    "Settings → Providers). Checked: %s.",
                    var, provider_id, ", ".join(env_vars),
                )
            return value
    return None
