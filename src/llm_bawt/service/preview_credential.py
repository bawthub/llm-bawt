"""Read-only, per-run access-token broker for an isolated Preview bridge.

This router is registered only by the production app; the Preview app does not
mount it. The gateway, not the editable Preview
bridge, holds the signed capability. A missing grant or token refuses service.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from binascii import Error as Base64Error
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..model_catalog import ModelCatalog
from . import bridge_gate
from .providers.base import CredentialStore, STATUS_CONNECTED
from .providers import crypto

router = APIRouter()
GRANT_KEY = "preview_broker:active_grant"
CAPABILITY_HEADER = "X-Preview-Broker-Token"
_SIGNING_LABEL = "preview-broker-capability/v1"
_CAPABILITY_SECONDS = 600
_MAX_RUN_SECONDS = 8 * 60 * 60
_MIN_TOKEN_SECONDS = 60
_SUPPORTED_PROVIDERS = frozenset({"claude", "openai_chatgpt"})


@dataclass(frozen=True)
class PreviewGrant:
    run_id: str
    provider: str
    endpoint_id: int
    expires_at: int

    @classmethod
    def parse(cls, raw: Any, now: int) -> PreviewGrant:
        if not isinstance(raw, dict) or set(raw) != {"run_id", "provider", "endpoint_id", "expires_at"}:
            raise HTTPException(403, "Preview broker grant unavailable")
        run_id, provider = raw["run_id"], raw["provider"]
        endpoint_id, expires_at = raw["endpoint_id"], raw["expires_at"]
        try:
            if not isinstance(run_id, str) or str(UUID(run_id)) != run_id:
                raise ValueError("invalid run")
        except (ValueError, AttributeError) as exc:
            raise HTTPException(403, "Preview broker grant unavailable") from exc
        if (not isinstance(provider, str) or provider not in _SUPPORTED_PROVIDERS
                or type(endpoint_id) is not int
                or endpoint_id <= 0 or type(expires_at) is not int
                or not now < expires_at <= now + _MAX_RUN_SECONDS):
            raise HTTPException(403, "Preview broker grant unavailable")
        return cls(run_id, provider, endpoint_id, expires_at)

    def payload(self, expires_at: int) -> bytes:
        return json.dumps({"version": 1, "run_id": self.run_id,
                           "provider": self.provider, "endpoint_id": self.endpoint_id,
                           "expires_at": expires_at},
                          sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _b64_encode(raw) != value:
        raise ValueError("noncanonical capability")
    return raw


def issue_capability(grant: PreviewGrant, *, now: int | None = None) -> str:
    """Called only by the owner-side grant publisher; never exposed as a route."""
    now = int(time.time()) if now is None else now
    if not now < grant.expires_at <= now + _MAX_RUN_SECONDS:
        raise ValueError("inactive Preview grant")
    payload = grant.payload(min(grant.expires_at, now + _CAPABILITY_SECONDS))
    signature = hmac.new(crypto.derive_key(_SIGNING_LABEL), payload, hashlib.sha256).digest()
    return f"{_b64_encode(payload)}.{_b64_encode(signature)}"


def verify_capability(token: str | None, grant: PreviewGrant, *, now: int) -> None:
    if not token or len(token) > 1024 or token.count(".") != 1:
        raise HTTPException(403, "Invalid Preview broker capability")
    try:
        part, signature = token.split(".", 1)
        payload = _b64_decode(part)
        expected = hmac.new(crypto.derive_key(_SIGNING_LABEL), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64_decode(signature), expected):
            raise ValueError("bad signature")
        claims = json.loads(payload)
        cap_exp = claims.get("expires_at")
        if (set(claims) != {"version", "run_id", "provider", "endpoint_id", "expires_at"}
                or type(claims["version"]) is not int or claims["version"] != 1
                or type(cap_exp) is not int or not now < cap_exp <= min(
                    grant.expires_at, now + _CAPABILITY_SECONDS
                ) or payload != grant.payload(cap_exp)):
            raise ValueError("wrong Preview grant or expiration")
    except (RuntimeError, OSError) as exc:
        raise HTTPException(503, "Preview broker signing key unavailable") from exc
    except (ValueError, TypeError, KeyError, UnicodeError, Base64Error) as exc:
        raise HTTPException(403, "Invalid Preview broker capability") from exc


def _read_setting(engine, key: str) -> Any:
    """Read one value without RuntimeSettingsStore's schema/bootstrap writes."""
    with engine.connect() as conn:
        raw = conn.execute(text("SELECT value_json FROM runtime_settings "
                                "WHERE scope_type='global' AND scope_id='*' AND key=:key"),
                           {"key": key}).scalar_one_or_none()
    return json.loads(raw) if raw is not None else None


def _provider_for_endpoint(catalog: ModelCatalog | None, grant: PreviewGrant) -> None:
    if catalog is None:
        raise HTTPException(503, "Preview broker model catalog unavailable")
    harness = "claude-code" if grant.provider == "claude" else "claude-proxy"
    try:
        endpoint = catalog.resolve_endpoint(grant.endpoint_id, harness=harness)
    except (LookupError, ValueError) as exc:
        raise HTTPException(403, "Preview broker endpoint is not authorized") from exc
    path = endpoint.access_path
    expected = (path.vendor == "anthropic" if grant.provider == "claude"
                else path.key == "openai-oauth" and path.vendor == "openai")
    if not expected or path.auth_mechanism != "oauth":
        raise HTTPException(403, "Preview broker endpoint is not authorized")


def _jwt_exp(token: str) -> float | None:
    try:
        payload = token.split(".", 2)[1]
        claims = json.loads(_b64_decode(payload))
        value = claims.get("exp")
        expiration = float(value) if type(value) in (int, float) else None
        return expiration if expiration is not None and math.isfinite(expiration) else None
    except (ValueError, IndexError, TypeError, UnicodeError, Base64Error):
        return None


def _access_token(engine, grant: PreviewGrant, now: int) -> dict[str, Any]:
    """Only DB reads/decryption. Never invoke the OAuth _load/get_access_token APIs."""
    raw = _read_setting(engine, f"provider_connection:{grant.provider}")
    if not isinstance(raw, dict):
        raise HTTPException(503, "Preview provider credential unavailable")
    record = CredentialStore._record(grant.provider, raw)
    if record.status != STATUS_CONNECTED or not isinstance(record.secret, dict):
        raise HTTPException(503, "Preview provider credential unavailable")
    if grant.provider == "claude":
        bundle = record.secret.get("claudeAiOauth")
        token = bundle.get("accessToken") if isinstance(bundle, dict) else None
        expires_ms = bundle.get("expiresAt") if isinstance(bundle, dict) else None
        if type(expires_ms) is not int:
            raise HTTPException(503, "Preview provider token expiration unavailable")
        expires_at = expires_ms / 1000
        response = {"access_token": token, "expires_at": expires_ms, "state": "ok"}
    else:
        bundle = record.secret.get("codex_auth")
        tokens = bundle.get("tokens") if isinstance(bundle, dict) else None
        token = tokens.get("access_token") if isinstance(tokens, dict) else None
        account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
        expires_at = _jwt_exp(token) if isinstance(token, str) else None
        if not isinstance(account_id, str) or not account_id:
            raise HTTPException(503, "Preview provider account unavailable")
        response = {"access_token": token, "account_id": account_id,
                    "expires_at": expires_at, "state": "ok"}
    if (not isinstance(token, str) or not token or expires_at is None
            or expires_at <= now + _MIN_TOKEN_SECONDS):
        raise HTTPException(503, "Preview provider token unavailable or expiring")
    return response


@router.get("/v1/preview/credential")
async def preview_credential(request: Request) -> JSONResponse:
    """Gateway-only route: no query, broad bridge token or refresh control."""
    bridge_gate.reject_bridge_token_for_preview(request)
    if request.query_params:
        raise HTTPException(403, "Preview broker does not accept options")
    presented = request.headers.get(CAPABILITY_HEADER)
    if not presented:
        raise HTTPException(403, "Invalid Preview broker capability")
    from ..utils.db import get_shared_engine
    from .dependencies import get_service

    config = get_service().config
    engine = get_shared_engine(config)
    if engine is None:
        raise HTTPException(503, "Preview broker database unavailable")
    now = int(time.time())
    try:
        grant = PreviewGrant.parse(_read_setting(engine, GRANT_KEY), now)
    except (ValueError, OSError, SQLAlchemyError) as exc:
        raise HTTPException(503, "Preview broker grant unavailable") from exc
    verify_capability(presented, grant, now=now)
    _provider_for_endpoint(config.model_catalog, grant)
    try:
        result = _access_token(engine, grant, now)
    except (ValueError, OSError, SQLAlchemyError) as exc:
        raise HTTPException(503, "Preview provider credential unavailable") from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
