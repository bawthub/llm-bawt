"""Preview-role credential client for agent bridges (TASK-1802 / TASK-1789).

A preview bridge runs untrusted preview code, so it must never reach the
production broker, a mounted bundle, a token file or a credential env var.
When ``LLM_BAWT_PREVIEW_ENABLED=1`` (the same flag the preview app uses), every
bridge credential lookup routes here and nowhere else:

    GET {LLM_BAWT_PREVIEW_CREDENTIAL_URL}/v1/providers/{provider}/token

No headers, no query string, no redirects. The credential gateway (Al's
component) injects the per-run capability; the bridge holds none. Anything
other than HTTP 200 with a valid, unexpired ``{access_token, expires_at,
state: "ok"}`` (plus ``account_id`` for ChatGPT) raises
``PreviewCredentialError``. Callers must let it propagate: there is no fallback.

Wire units match the existing broker routes: Claude ``expires_at`` is epoch
milliseconds, ChatGPT ``expires_at`` is epoch seconds. A value in the wrong
unit is refused, never normalized, so a producer bug fails loudly.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import httpx

ENABLED_ENV = "LLM_BAWT_PREVIEW_ENABLED"
URL_ENV = "LLM_BAWT_PREVIEW_CREDENTIAL_URL"

# The gateway answers 503 under this floor; the client enforces it too.
MIN_REMAINING_S = 60.0
TIMEOUT_S = 15.0

# Epoch seconds below this are before 2001 or are milliseconds read as
# seconds (and vice versa): a 1e11 boundary separates the two for any
# plausible expiry (1e11 s is year 5138; 1e11 ms is 1973).
_UNIT_BOUNDARY = 100_000_000_000


class PreviewCredentialError(RuntimeError):
    """The preview gateway did not supply a usable credential. Never fall back."""


@dataclass(frozen=True)
class _ProviderShape:
    expires_unit_s: float  # seconds per wire unit
    requires_account_id: bool


PROVIDERS: Mapping[str, _ProviderShape] = {
    "claude": _ProviderShape(expires_unit_s=0.001, requires_account_id=False),
    "openai_chatgpt": _ProviderShape(expires_unit_s=1.0, requires_account_id=True),
}


@dataclass(frozen=True)
class PreviewCredential:
    access_token: str
    expires_at_s: float
    account_id: str | None


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENABLED_ENV) == "1"


def _gateway_url(env: Mapping[str, str]) -> str:
    url = (env.get(URL_ENV) or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise PreviewCredentialError(f"{URL_ENV} is unset or not an http(s) URL")
    return url


def _expires_at_s(raw: object, shape: _ProviderShape, provider: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise PreviewCredentialError(f"{provider} credential has no numeric expires_at")
    is_ms = raw >= _UNIT_BOUNDARY
    if is_ms != (shape.expires_unit_s < 1.0):
        raise PreviewCredentialError(f"{provider} credential expires_at is in the wrong unit")
    return float(raw) * shape.expires_unit_s


def fetch(
    provider: str,
    *,
    environ: Mapping[str, str] | None = None,
    get: Callable[..., httpx.Response] | None = None,
    now: Callable[[], float] | None = None,
) -> PreviewCredential:
    """Return the gateway's credential for ``provider`` or raise. No fallback."""
    get = get or httpx.get
    now = now or time.time
    env = os.environ if environ is None else environ
    if not enabled(env):
        raise PreviewCredentialError("preview credential client used outside preview role")
    shape = PROVIDERS.get(provider)
    if shape is None:
        raise PreviewCredentialError(f"provider {provider!r} is not served in preview")
    url = f"{_gateway_url(env)}/v1/providers/{provider}/token"
    try:
        resp = get(url, headers={}, follow_redirects=False, timeout=TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise PreviewCredentialError(
            f"preview credential gateway unreachable: {type(exc).__name__}"
        ) from exc
    if resp.status_code != 200:
        raise PreviewCredentialError(f"preview credential gateway returned {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise PreviewCredentialError("preview credential gateway returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("state") != "ok":
        raise PreviewCredentialError("preview credential state is not ok")
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise PreviewCredentialError(f"{provider} credential has no access_token")
    account_id = payload.get("account_id")
    if shape.requires_account_id and (not isinstance(account_id, str) or not account_id):
        raise PreviewCredentialError(f"{provider} credential has no account_id")
    expires_at_s = _expires_at_s(payload.get("expires_at"), shape, provider)
    if expires_at_s - now() < MIN_REMAINING_S:
        raise PreviewCredentialError(f"{provider} credential expires in under {MIN_REMAINING_S:.0f}s")
    return PreviewCredential(
        access_token=token,
        expires_at_s=expires_at_s,
        account_id=account_id if isinstance(account_id, str) and account_id else None,
    )
