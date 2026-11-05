"""Ollama server connection (TASK-2047).

Ollama is a self-hosted server, so "connecting" means pointing llm-bawt at a
server URL (plus an optional bearer key for a proxied server) rather than
handing over an account credential.

Where things live:

- The server URL is the ``ollama`` access path's ``base_url`` (DB), the same
  value the chat and background clients already read through the model
  catalog. Connecting writes it; disconnecting clears it, which falls back to
  the ``OLLAMA_URL`` config value so installs that never configured anything
  here keep working.
- The optional key is encrypted in the :class:`CredentialStore`.

The adapter deliberately owns no access path for route eligibility: Ollama
routes need no credential, so existing Ollama models stay assignable whether
or not this connection was ever saved.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import (
    HEALTH_BROKEN,
    HEALTH_OK,
    HEALTH_UNCONFIGURED,
    STATUS_CONNECTED,
    ConnectionRecord,
    ProviderAdapter,
    ValidateResult,
    health_block,
)

logger = logging.getLogger(__name__)

#: Auth method for providers configured by a server URL (+ optional key).
AUTH_ENDPOINT = "endpoint"

ACCESS_PATH_KEY = "ollama"
_SECRET_FIELD = "api_key"
_PROBE_TIMEOUT = 5.0
_HEALTH_TIMEOUT = 2.0


class OllamaUrlError(ValueError):
    """The submitted server URL is unusable."""


def normalize_server_url(raw: str) -> str:
    """Server root (``scheme://host[:port][/prefix]``) without ``/v1`` or ``/api``."""
    url = (raw or "").strip().rstrip("/")
    if not url:
        raise OllamaUrlError("Server URL is empty")
    if "://" not in url:
        url = f"http://{url}"
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise OllamaUrlError("Server URL must be an http(s) address, e.g. http://host:11434")
    if parts.query or parts.fragment:
        raise OllamaUrlError("Server URL must not contain a query or fragment")
    for suffix in ("/v1", "/api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def _catalog_engine(config: Any):
    from ..dependencies import get_model_catalog_engine  # noqa: PLC0415 — import cycle

    return get_model_catalog_engine(config)


def _stored_path_url(config: Any) -> str | None:
    """The ``ollama`` access path's base_url as stored, or None."""
    from sqlalchemy import text  # noqa: PLC0415

    engine = _catalog_engine(config)
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            value = conn.execute(
                text("SELECT base_url FROM access_paths WHERE key = :key"), {"key": ACCESS_PATH_KEY},
            ).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001 — a missing catalog falls back to config
        logger.debug("ollama access path lookup failed: %s", exc)
        return None
    return str(value).strip() or None if value else None


def _write_path_url(config: Any, base_url: str | None) -> None:
    from sqlalchemy import text  # noqa: PLC0415

    engine = _catalog_engine(config)
    if engine is None:
        raise RuntimeError("Model catalog database unavailable")
    with engine.begin() as conn:
        updated = conn.execute(
            text("UPDATE access_paths SET base_url = :url, updated_at = now() WHERE key = :key"),
            {"url": base_url, "key": ACCESS_PATH_KEY},
        ).rowcount
    if updated != 1:
        raise RuntimeError("The ollama access path is missing from the model catalog")


def server_url(config: Any) -> str:
    """Effective server root: the configured access path, else ``OLLAMA_URL``."""
    stored = _stored_path_url(config)
    if stored:
        try:
            return normalize_server_url(stored)
        except OllamaUrlError:
            logger.warning("ignoring unusable ollama access-path base_url")
    return normalize_server_url(getattr(config, "OLLAMA_URL", "") or "http://localhost:11434")


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def probe(url: str, api_key: str | None, *, timeout: float = _PROBE_TIMEOUT) -> tuple[str | None, str | None]:
    """``(version, error)`` from ``GET {url}/api/version`` — exactly one is set."""
    try:
        response = httpx.get(f"{url}/api/version", headers=_headers(api_key), timeout=timeout)
    except httpx.TimeoutException:
        return None, f"Ollama at {url} did not answer within {timeout:g}s"
    except httpx.HTTPError as exc:
        return None, f"Couldn't reach Ollama at {url} ({type(exc).__name__})"
    if response.status_code in (401, 403):
        return None, f"Ollama at {url} rejected the request (HTTP {response.status_code}); check the API key"
    if response.is_error:
        return None, f"Ollama at {url} returned HTTP {response.status_code}"
    try:
        version = str(response.json().get("version") or "").strip()
    except (ValueError, AttributeError):
        version = ""
    if not version:
        return None, f"{url} answered, but not like an Ollama server (no /api/version)"
    return version, None


class OllamaAdapter(ProviderAdapter):
    """Server-URL connection for a self-hosted Ollama."""

    id = "ollama"
    label = "Ollama"
    auth_methods = (AUTH_ENDPOINT,)
    access_path_keys = ()

    def descriptor(self) -> dict[str, Any]:
        base = super().descriptor()
        base.update({
            "endpoint_label": "Server URL",
            "endpoint_default": server_url(self.config),
            "credential_label": "API key (optional)",
            "credential_help": "Only needed when the server sits behind a proxy that requires a bearer token.",
        })
        return base

    def api_key(self) -> str | None:
        record = self.store.load(self.id)
        if not record or record.status != STATUS_CONNECTED:
            return None
        return (record.secret or {}).get(_SECRET_FIELD) or None

    def connect_endpoint(self, url: str, api_key: str | None = None) -> ValidateResult:
        try:
            root = normalize_server_url(url)
        except OllamaUrlError as exc:
            return ValidateResult(ok=False, detail=str(exc))
        key = (api_key or "").strip() or None
        version, detail = probe(root, key)
        if version is None:
            return ValidateResult(ok=False, detail=detail)
        _write_path_url(self.config, f"{root}/v1")
        account = f"Ollama {version} · {root}"
        self.store.save(ConnectionRecord(
            provider=self.id,
            status=STATUS_CONNECTED,
            auth_method=AUTH_ENDPOINT,
            account=account,
            meta={"base_url": root, "version": version},
            secret={_SECRET_FIELD: key} if key else {},
        ))
        logger.info("Ollama server configured (%s)", account)
        return ValidateResult(ok=True, account=account)

    def disconnect(self) -> bool:
        removed = super().disconnect()
        _write_path_url(self.config, None)
        return removed

    def validate(self, record: ConnectionRecord) -> ValidateResult:
        root = str((record.meta or {}).get("base_url") or server_url(self.config))
        version, detail = probe(root, (record.secret or {}).get(_SECRET_FIELD))
        if version is None:
            return ValidateResult(ok=False, detail=detail)
        return ValidateResult(ok=True, account=f"Ollama {version} · {root}")

    def health(self) -> dict[str, Any]:
        record = self.store.load(self.id)
        if not record or record.status != STATUS_CONNECTED:
            return health_block(HEALTH_UNCONFIGURED, detail=f"not configured here; using {server_url(self.config)}")
        root = str((record.meta or {}).get("base_url") or server_url(self.config))
        version, detail = probe(root, (record.secret or {}).get(_SECRET_FIELD), timeout=_HEALTH_TIMEOUT)
        if version is None:
            return health_block(HEALTH_BROKEN, detail=detail, fix="reconnect")
        return health_block(HEALTH_OK, detail=f"Ollama {version}")


def resolve_api_key(config: Any) -> str | None:
    """Stored Ollama key for the request clients; None when none is configured."""
    try:
        return OllamaAdapter(config).api_key()
    except Exception as exc:  # noqa: BLE001 — a credential-store outage must not break keyless servers
        logger.debug("ollama key lookup failed: %s", exc)
        return None
