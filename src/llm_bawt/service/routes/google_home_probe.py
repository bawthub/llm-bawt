"""Narrow, disabled-by-default Google Home OAuth + capture-only fulfillment."""
from __future__ import annotations

import hmac
import json
import logging
from html import escape
from urllib.parse import parse_qsl, urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from llm_bawt.integrations.google_home_oauth import (
    OAuthError, ProbeAuthenticator, ProbeOAuthStore, ProbeSettings,
    ProbeUnauthorized, ProbeUnavailable,
)
from llm_bawt.integrations.google_home_probe import GoogleHomeProbe, ProbeRequest
from llm_bawt.integrations.google_home_probe_store import ProbeCaptureStore
from llm_bawt.service.dependencies import get_runtime_settings_store, get_service
from llm_bawt.service.providers.base import CredentialStore
from llm_bawt.utils.db import get_shared_engine

router = APIRouter()
logger = logging.getLogger(__name__)
BASE = "/integrations/google-home"
MAX_BODY = 32768
CONSENT_COOKIE = "__Secure-google-home-consent"
SAFE_HEADERS = {
    "Cache-Control": "no-store", "Pragma": "no-cache", "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
}


def consent_headers(settings: ProbeSettings) -> dict[str, str]:
    # Browsers enforce form-action against the POST's redirect destination too.
    # Permit only this project's already allowlisted Google callback paths.
    destinations = " ".join(settings.redirects)
    return SAFE_HEADERS | {
        "Referrer-Policy": "same-origin",
        "Content-Security-Policy": (
            f"default-src 'none'; form-action 'self' {destinations}; "
            "frame-ancestors 'none'; base-uri 'none'"
        ),
    }


def get_probe_auth() -> ProbeAuthenticator:
    config = get_service().config
    raw = get_runtime_settings_store(config).get_scope_settings("global", "*").get("google_home_probe", {})
    try:
        settings = ProbeSettings(**raw)
        settings.validate()
    except (TypeError, ValueError, ProbeUnavailable) as exc:
        raise HTTPException(503, "Google Home probe is disabled or unconfigured", headers=SAFE_HEADERS) from exc
    record = CredentialStore(config).load("google-home-probe")
    secret = record.secret.get("client_secret", "") if record and record.status == "connected" else ""
    if not isinstance(secret, str) or len(secret) < 32:
        raise HTTPException(503, "Google Home probe authentication is unconfigured", headers=SAFE_HEADERS)
    return ProbeAuthenticator(settings, secret, ProbeOAuthStore(get_shared_engine(config), settings))


def get_probe() -> GoogleHomeProbe:
    return GoogleHomeProbe(ProbeCaptureStore(get_shared_engine(get_service().config)))


def error(code: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status, headers=SAFE_HEADERS)


def pairs(raw: str) -> dict[str, str]:
    try:
        entries = parse_qsl(raw, keep_blank_values=True, max_num_fields=20, encoding="utf-8", errors="strict")
    except (ValueError, UnicodeError) as exc:
        raise OAuthError("invalid_request") from exc
    if len(raw) > 8192 or len(dict(entries)) != len(entries):
        raise OAuthError("invalid_request")
    return dict(entries)


async def read_body(request: Request, limit: int = MAX_BODY) -> bytes:
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > limit:
            raise HTTPException(413, "Request too large", headers=SAFE_HEADERS)
        raw.extend(chunk)
    return bytes(raw)


async def form(request: Request) -> dict[str, str]:
    if request.headers.get("content-type", "").split(";")[0].lower() != "application/x-www-form-urlencoded":
        raise OAuthError("invalid_request")
    try:
        return pairs((await read_body(request, 8192)).decode("utf-8"))
    except UnicodeError as exc:
        raise OAuthError("invalid_request") from exc


async def browser_subject(request: Request, auth: ProbeAuthenticator) -> str:
    async with httpx.AsyncClient() as http:
        return await auth.browser_identity(request.headers.get("cookie", ""), http)


@router.get(BASE + "/authorize", tags=["Webhooks"])
async def google_home_authorize(request: Request, auth: ProbeAuthenticator = Depends(get_probe_auth)):
    try:
        params = pairs(request.url.query)
        auth.settings.authorization(params)  # Validate before login or any redirect.
        try:
            await browser_subject(request, auth)
        except ProbeUnauthorized:
            return_url = auth.settings.public_origin + BASE + "/authorize?" + urlencode(params)
            signin = auth.settings.sso_signin_url + "?" + urlencode({"rd": return_url})
            return HTMLResponse(
                '<!doctype html><title>BawtHub Google Home probe — sign in</title><h1>BawtHub Google Home probe</h1>'
                '<p>Sign in with your BawtHub owner account to review linking with Google.</p>'
                f'<a href="{escape(signin, quote=True)}">Sign in to BawtHub</a>', headers=SAFE_HEADERS,
            )
        consent = await run_in_threadpool(auth.store.consent, params)
        response = HTMLResponse(
            '<!doctype html><title>Link BawtHub Google Home probe</title><h1>Link BawtHub Google Home probe with Google</h1>'
            '<p>By linking, you authorize Google to control your BawtHub TV test device. '
            'This experiment only records commands sent by Google; it never executes bot actions. '
            'Command payloads may contain spoken text and are retained for up to 24 hours of readable history.</p>'
            '<p><a href="https://policies.google.com/privacy">Google Privacy Policy</a> · '
            '<a href="https://myaccount.google.com/connections">Manage linked accounts</a></p>'
            f'<form method="post" action="{BASE}/authorize">'
            f'<input type="hidden" name="consent" value="{consent}">'
            '<button name="decision" value="allow">Agree and link</button> '
            '<button name="decision" value="deny">Cancel</button></form>',
            # no-referrer makes native form POSTs send Origin: null. Preserve
            # same-origin form provenance without leaking queries cross-origin.
            headers=consent_headers(auth.settings),
        )
        response.set_cookie(CONSENT_COOKIE, consent, max_age=600, secure=True, httponly=True,
                            samesite="lax", path=BASE + "/authorize")
        return response
    except OAuthError as exc:
        return error(exc.error)
    except ProbeUnavailable:
        return error("temporarily_unavailable", 503)


@router.post(BASE + "/authorize", tags=["Webhooks"])
async def google_home_consent(request: Request, auth: ProbeAuthenticator = Depends(get_probe_auth)):
    try:
        params = await form(request)
        consent = params.get("consent", "")
        cookie = request.cookies.get(CONSENT_COOKIE, "")
        if (request.headers.get("origin") != auth.settings.public_origin
                or not consent or not cookie or not hmac.compare_digest(consent.encode(), cookie.encode())
                or params.get("decision") not in ("allow", "deny")):
            return error("invalid_request", 403)
        await browser_subject(request, auth)
        result = await run_in_threadpool(auth.store.approve, consent, params["decision"] == "allow")
        redirect = result.pop("redirect_uri")
        response = RedirectResponse(redirect + "?" + urlencode(result), status_code=303,
                                    headers=consent_headers(auth.settings) | {"Referrer-Policy": "no-referrer"})
        response.delete_cookie(CONSENT_COOKIE, path=BASE + "/authorize", secure=True, httponly=True, samesite="lax")
        return response
    except OAuthError as exc:
        return error(exc.error)
    except ProbeUnauthorized:
        return error("access_denied", 403)
    except ProbeUnavailable:
        return error("temporarily_unavailable", 503)


@router.post(BASE + "/token", tags=["Webhooks"])
async def google_home_token(request: Request, auth: ProbeAuthenticator = Depends(get_probe_auth)):
    try:
        params = await form(request)
        # Google defaults to credentials in the form body; leave its Basic auth
        # checkbox OFF. Reject ambiguous mixed authentication rather than guess.
        if request.headers.get("authorization"):
            return error("invalid_client", 401)
        auth.authenticate_client(params)
        result = await run_in_threadpool(auth.store.exchange, params)
        return JSONResponse(result, headers=SAFE_HEADERS)
    except OAuthError as exc:
        return error(exc.error, 401 if exc.error == "invalid_client" else 400)


@router.post(BASE + "/fulfillment", tags=["Webhooks"])
async def google_home_fulfillment(request: Request, auth: ProbeAuthenticator = Depends(get_probe_auth)):
    authorization = request.headers.get("authorization", "")
    try:
        subject = await auth.authenticate(authorization)
    except ProbeUnauthorized as exc:
        raise HTTPException(401, "Invalid access token", headers=SAFE_HEADERS | {"WWW-Authenticate": "Bearer"}) from exc
    except ProbeUnavailable as exc:
        raise HTTPException(503, "Probe authentication unavailable", headers=SAFE_HEADERS) from exc
    if request.headers.get("content-type", "").split(";")[0].lower() != "application/json":
        raise HTTPException(415, "Expected application/json", headers=SAFE_HEADERS)
    try:
        message = ProbeRequest.model_validate(json.loads(await read_body(request)))
    except ValidationError as exc:
        # Never log input values, request bodies, tokens, or attacker-chosen keys.
        known = {"requestId", "inputs", "intent", "payload", "devices", "commands",
                 "execution", "id", "customData", "command", "params", "newApplication",
                 "newApplicationName", "on", "context", "locale", "challenge", "challengeId"}
        issues = [{"type": e["type"], "path": [
            part if isinstance(part, int) or part in known else "<extra>"
            for part in e["loc"]
        ]} for e in exc.errors(include_input=False, include_context=False, include_url=False)[:12]]
        logger.warning("Google Home probe rejected authenticated request: %s", json.dumps(issues))
        raise HTTPException(400, "Invalid Google Home probe request", headers=SAFE_HEADERS) from exc
    except (ValueError, RecursionError) as exc:
        logger.warning("Google Home probe rejected malformed JSON")
        raise HTTPException(400, "Invalid Google Home probe request", headers=SAFE_HEADERS) from exc
    if message.inputs[0].intent == "action.devices.DISCONNECT":
        try:
            await run_in_threadpool(auth.store.revoke, authorization)
        except ProbeUnauthorized as exc:
            raise HTTPException(401, "Invalid access token", headers=SAFE_HEADERS) from exc
        # Revocation must not depend on capture storage succeeding.
        return Response(status_code=200, headers=SAFE_HEADERS)
    def capture():
        return get_probe().handle(subject, message)
    result = await run_in_threadpool(capture)
    return JSONResponse(result, headers=SAFE_HEADERS)
