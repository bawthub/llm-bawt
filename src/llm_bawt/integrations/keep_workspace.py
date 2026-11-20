"""Google Keep Workspace delegation. Fixed endpoints; no browser refresh tokens.

The configured subject pins application behavior, not the domain-wide authority
of the service-account credential. Never accept a per-call impersonation target.
"""
from __future__ import annotations

import json
import re
import time

import httpx
import jwt
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

TOKEN_URL = "https://oauth2.googleapis.com/token"
KEEP_URL = "https://keep.googleapis.com/v1/notes"
KEEP_SCOPE = "https://www.googleapis.com/auth/keep"


class WorkspaceAuthError(Exception):
    """Bounded public code, never an upstream response or private key."""


def validate_service_account(raw: str, subject: str) -> tuple[dict, dict]:
    subject = subject.strip().lower()
    if not re.fullmatch(r"[^\s@]+@(?:[a-z0-9-]+\.)+[a-z]{2,}", subject):
        raise WorkspaceAuthError("invalid_workspace_user")
    if subject.rsplit("@", 1)[1] in {"gmail.com", "googlemail.com"}:
        raise WorkspaceAuthError("workspace_required")
    try:
        key = json.loads(raw)
        if not isinstance(key, dict) or key.get("type") != "service_account":
            raise ValueError("type")
        required = ("client_email", "client_id", "private_key", "private_key_id", "project_id")
        if any(not isinstance(key.get(k), str) or not key[k].strip() for k in required):
            raise ValueError("fields")
        if not re.fullmatch(r"[a-z0-9-]+@[a-z0-9-]+\.iam\.gserviceaccount\.com", key["client_email"]):
            raise ValueError("email")
        if not key["client_id"].isascii() or not key["client_id"].isdigit():
            raise ValueError("client id")
        if key.get("token_uri") != TOKEN_URL:
            raise ValueError("token URI")
        private = load_pem_private_key(key["private_key"].encode(), password=None)
        if not isinstance(private, RSAPrivateKey) or private.key_size < 2048:
            raise ValueError("RSA key")
    except (ValueError, TypeError, KeyError) as exc:
        raise WorkspaceAuthError("invalid_service_account_key") from exc
    return (
        {"client_email": key["client_email"], "client_id": key["client_id"],
         "private_key_id": key["private_key_id"], "project_id": key["project_id"],
         "subject": subject, "workspace_domain": subject.rsplit("@", 1)[1]},
        {"private_key": key["private_key"]},
    )


def delegated_token(http: httpx.Client, meta: dict, secret: dict, scope: str = KEEP_SCOPE) -> tuple[str, int]:
    """A delegated token for the configured subject and one fixed product scope."""
    now = int(time.time())
    assertion = jwt.encode(
        {"iss": meta["client_email"], "sub": meta["subject"], "scope": scope,
         "aud": TOKEN_URL, "iat": now, "exp": now + 3600},
        secret["private_key"], algorithm="RS256", headers={"kid": meta["private_key_id"]},
    )
    response = http.post(TOKEN_URL, data={
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion,
    })
    if response.status_code != 200:
        try:
            body = response.json()
            reason = body.get("error") if isinstance(body, dict) else None
        except ValueError:
            reason = None
        code = {"unauthorized_client": "workspace_delegation_denied",
                "invalid_grant": "workspace_assertion_rejected",
                "invalid_scope": "workspace_scope_denied"}.get(reason if isinstance(reason, str) else "")
        raise WorkspaceAuthError(code or "workspace_token_failed")
    try:
        token = response.json()
        access = token["access_token"]
        lifetime = int(token["expires_in"])
        if not isinstance(access, str) or not access or not 0 < lifetime <= 3600:
            raise ValueError("token")
        if "scope" in token and scope not in token["scope"].split():
            raise WorkspaceAuthError("workspace_scope_denied")
        return access, now + lifetime
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise WorkspaceAuthError("workspace_token_failed") from exc
