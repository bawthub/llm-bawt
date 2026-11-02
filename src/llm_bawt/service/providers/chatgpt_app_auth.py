"""Own-app SIWC OAuth/OIDC contract. No Codex credentials or OAuth client identity.

Protocol: developers.openai.com/siwc/token-sharing-open-source/sign-in
Authlib owns authorization response/state, PKCE, token grants and OIDC claim
validation. All network targets are fixed; upstream error text is never public.
"""
from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

import httpx
from authlib.integrations.httpx_client import OAuth2Client
from authlib.jose import JsonWebToken
from authlib.oauth2.rfc6749.parameters import parse_authorization_code_response
from authlib.oidc.core import CodeIDToken

ISSUER = "https://auth.openai.com"
AUTHORIZE = f"{ISSUER}/api/accounts/authorize"
TOKEN = f"{ISSUER}/api/accounts/oauth/token"
JWKS = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
# A constant product identity, not a deployment/bot name.
APP_NAME = "BawtHub"
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DIRECT_SCOPES = frozenset(SCOPE.split())
REDIRECT_URI = "http://127.0.0.1:1455/auth/callback"
DYNAMIC_CLIENT = "dynamic_agent_client"


class SignInError(RuntimeError):
    """Only fixed, nonsecret diagnostics may cross the API boundary."""


@dataclass(repr=False)
class PendingSignIn:
    state: str
    nonce: str
    verifier: str
    client_id: str
    subject: str | None
    generation: str | None
    deadline: float = field(default_factory=lambda: time.monotonic() + 600)


def client(client_id: str, **kwargs) -> OAuth2Client:
    return OAuth2Client(
        client_id=client_id, token_endpoint_auth_method="none",
        code_challenge_method="S256", timeout=25.0, follow_redirects=False,
        trust_env=False, **kwargs,
    )


def begin(host_id: str, *, registration: dict, generation: str | None,
          reconsent: bool = False) -> tuple[PendingSignIn, str]:
    pending = PendingSignIn(
        state=secrets.token_urlsafe(32), nonce=secrets.token_urlsafe(32),
        verifier=secrets.token_urlsafe(48),
        client_id=registration.get("client_id") or DYNAMIC_CLIENT,
        subject=registration.get("subject"), generation=generation,
    )
    parameters = {"nonce": pending.nonce, "ext_agent_host_id": host_id, "resource": RESOURCE}
    if pending.client_id == DYNAMIC_CLIENT:
        parameters["agent_name_hint"] = APP_NAME
    if reconsent:
        parameters["prompt"] = "consent"  # Explicit enable action after a declined grant only.
    with client(pending.client_id, scope=SCOPE, redirect_uri=REDIRECT_URI) as oauth:
        url, _ = oauth.create_authorization_url(
            AUTHORIZE, state=pending.state, code_verifier=pending.verifier, **parameters,
        )
    return pending, url


def callback_parameters(callback_url: str, pending: PendingSignIn) -> tuple[str, str]:
    if time.monotonic() >= pending.deadline:
        raise SignInError("Sign-in expired. Start again.")
    if len(callback_url) > 8192:
        raise SignInError("Invalid callback URL.")
    parts = urlsplit(callback_url)
    target = urlsplit(REDIRECT_URI)
    if (parts.scheme, parts.netloc, parts.path) != (target.scheme, target.netloc, target.path) or parts.fragment:
        raise SignInError("Paste the full original 127.0.0.1 callback URL.")
    params = parse_qs(parts.query, keep_blank_values=True)
    if any(len(params.get(key, [])) > 1 for key in ("state", "code", "client_id", "error", "scope")):
        raise SignInError("Invalid duplicate callback parameters.")
    # The library verifies state before accepting authorization or denial.
    try:
        response = parse_authorization_code_response(callback_url, state=pending.state)
    except Exception:
        raise SignInError("Authorization was denied or callback validation failed.") from None
    issued = response.get("client_id") or (pending.client_id if pending.client_id != DYNAMIC_CLIENT else "")
    if (not issued or issued == DYNAMIC_CLIENT or len(issued) > 200
            or not all(c.isascii() and (c.isalnum() or c in "_-") for c in issued)):
        raise SignInError("Registration did not return an issued application ID.")
    if pending.client_id != DYNAMIC_CLIENT and issued != pending.client_id:
        raise SignInError("Application registration changed. Start again.")
    return response["code"], issued


def metadata_client() -> httpx.Client:
    return httpx.Client(timeout=20.0, follow_redirects=False, trust_env=False)


def validate_identity(token: dict, client_id: str, *, nonce: str | None, subject: str | None) -> dict:
    with metadata_client() as http:
        response = http.get(JWKS)
        response.raise_for_status()
        keys = response.json()
    claims = JsonWebToken(["RS256"]).decode(
        token["id_token"], keys, claims_cls=CodeIDToken,
        claims_options={
            "iss": {"essential": True, "value": ISSUER},
            "aud": {"essential": True, "value": client_id},
            "exp": {"essential": True}, "iat": {"essential": True},
            "sub": {"essential": True},
        },
        claims_params={"nonce": nonce, "client_id": client_id, "access_token": token["access_token"]},
    )
    claims.validate(leeway=30)
    if not isinstance(claims.get("sub"), str) or not claims["sub"] or (subject and claims["sub"] != subject):
        raise SignInError("Signed-in account does not match this registration.")
    return dict(claims)


def token_bundle(token: dict, *, previous: dict | None = None, require_plan: bool = True) -> dict:
    previous = previous or {}
    scopes = token.get("scope", " ".join(previous.get("scopes", [])))
    scopes = scopes.split() if isinstance(scopes, str) else []
    expires = token.get("expires_in")
    required = DIRECT_SCOPES if require_plan else {"openid"}
    if (not required.issubset(scopes) or str(token.get("token_type", "")).lower() != "bearer"
            or not isinstance(expires, (int, float)) or isinstance(expires, bool)
            or not math.isfinite(expires) or expires <= 0
            or not isinstance(token.get("access_token"), str) or not token["access_token"]):
        raise SignInError("ChatGPT plan usage permission or token expiry is missing.")
    result = {
        **previous, "access_token": token["access_token"],
        "refresh_token": token.get("refresh_token") or previous.get("refresh_token"),
        "id_token": token.get("id_token") or previous.get("id_token"),
        "expires_at": time.time() + expires, "scopes": scopes,
    }
    if require_plan and (not isinstance(result["refresh_token"], str) or not result["refresh_token"]):
        raise SignInError("No renewable session was returned.")
    return result


def exchange(callback_url: str, pending: PendingSignIn) -> dict:
    code, issued = callback_parameters(callback_url, pending)
    with client(issued, redirect_uri=REDIRECT_URI) as oauth:
        token = oauth.fetch_token(TOKEN, grant_type="authorization_code", code=code,
                                  code_verifier=pending.verifier, resource=RESOURCE)
    try:
        identity = validate_identity(token, issued, nonce=pending.nonce, subject=pending.subject)
        # A valid sign-in without direct-plan permission remains a sign-in,
        # not an inference credential. Never revoke it merely for declining use.
        plan_enabled = "chatgpt.tokens.use.direct" in str(token.get("scope", "")).split()
        bundle = token_bundle(token, require_plan=plan_enabled)
    except Exception:
        # A code exchange may have minted a fresh renewable session even when
        # identity/permission validation fails. Best-effort revoke only that
        # new session; never the existing connection or account-wide grants.
        try:
            revoke({"client_id": issued, "refresh_token": token.get("refresh_token")})
        except Exception:
            pass
        raise
    return {**bundle, "client_id": issued, "subject": identity["sub"], "issuer": ISSUER,
            "email": identity.get("email"), "plan_enabled": plan_enabled}


def renew(bundle: dict) -> dict:
    with client(bundle["client_id"]) as oauth:
        token = oauth.refresh_token(TOKEN, refresh_token=bundle["refresh_token"], resource=RESOURCE)
    if token.get("id_token"):
        validate_identity(token, bundle["client_id"], nonce=None, subject=bundle["subject"])
    return token_bundle(token, previous=bundle)


def revoke(bundle: dict) -> bool:
    if not bundle.get("refresh_token"):
        return True
    # Discover the published revocation target. Never follow arbitrary metadata.
    with metadata_client() as http:
        metadata = http.get(f"{ISSUER}/.well-known/openid-configuration")
        metadata.raise_for_status()
        url = metadata.json().get("revocation_endpoint")
        parts = urlsplit(url or "")
        if parts.scheme != "https" or parts.netloc != "auth.openai.com" or parts.fragment or parts.query:
            return False
    with client(bundle["client_id"]) as oauth:
        response = oauth.revoke_token(url, token=bundle["refresh_token"], token_type_hint="refresh_token")
        return response.status_code == 200
