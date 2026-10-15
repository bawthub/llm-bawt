"""External integration registry and encrypted DB connections (not model providers).

TASK-906: connection management only. No note ingestion or bot dispatch.
One account per integration per deployment, matching provider-account ownership.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import secrets
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit

import httpx
from sqlalchemy import delete, update
from sqlmodel import Session, select

from llm_bawt.integrations.keep_workspace import (
    KEEP_URL, WorkspaceAuthError, delegated_token, validate_service_account,
)
from llm_bawt.runtime_settings import RuntimeSetting
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import ConnectionRecord, CredentialStore

KEEP_SCOPE = "https://www.googleapis.com/auth/keep"
CALLBACK_PATH = "/api/chat/proxy/v1/integrations/google-keep/callback"
RETURN_PATHS = {"/tools/settings/integrations", "/setup"}
REGISTRY = {
    "nextcloud-talk": {
        "id": "nextcloud-talk", "label": "Nextcloud Talk", "auth_method": "nextcloud_login_v2",
        "description": "Authorize your Nextcloud account and discover existing Talk rooms and installed bots.",
        "supported_account_types": ["nextcloud"], "scopes": [],
        "capabilities": {"discover_rooms": True, "discover_bots": True, "create_rooms": False},
        "requirements": "The instance must match the server's HTTPS NEXTCLOUD_URL. Bot discovery requires a Nextcloud administrator.",
        "limitations": "Account authorization does not grant signing-secret recovery or webhook bot installation. Delivery is not verified by account login.",
        "setup_url": "https://nextcloud-talk.readthedocs.io/en/latest/bots/",
    },
    "google-keep": {
        "id": "google-keep", "label": "Google Keep · Workspace", "auth_method": "service_account_delegation",
        "supported_account_types": ["google_workspace"],
        "description": "Connect Google Workspace Keep with read/write permission for supported note operations.",
        "scopes": [KEEP_SCOPE],
        "capabilities": {"read_notes": True, "create_notes": True, "delete_notes": True, "edit_existing_notes": False},
        "limitations": "The official Keep API cannot edit existing notes or add, remove, or check items in an existing checklist. Connection setup does not import or modify notes.",
        "requirements": "Workspace domains only. Enable Keep API in the service account's Cloud project and authorize its numeric client ID for the Keep scope through Workspace domain-wide delegation. The configured Workspace user must have Keep enabled and API entitlement. This credential has domain-wide authority within its granted scopes; BawtHub pins use to the configured user.",
        "admin_scopes": [KEEP_SCOPE, "openid", "https://www.googleapis.com/auth/userinfo.email"],
        "setup_url": "https://developers.google.com/workspace/keep/api/guides",
    },
}


class IntegrationError(Exception):
    """A bounded, public error code; never includes an upstream body or token."""


def keep_failure_code(response: httpx.Response) -> str:
    """Classify Google's structured errors without exposing upstream text or secrets."""
    reasons: set[str] = set()
    message = ""
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if isinstance(error, dict):
            message = str(error.get("message", "")).lower()
            for field in ("details", "errors"):
                entries = error.get(field, [])
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and isinstance(entry.get("reason"), str):
                            reasons.add(entry["reason"])
    except ValueError:
        pass
    if reasons & {"SERVICE_DISABLED", "accessNotConfigured"}:
        code = "keep_api_disabled"
    elif reasons & {"ACCESS_TOKEN_SCOPE_INSUFFICIENT", "insufficientPermissions"}:
        code = "keep_scope_insufficient"
    elif reasons & {"ORG_RESTRICTION_VIOLATION", "SECURITY_POLICY_VIOLATED", "ORG_POLICY_VIOLATION"}:
        code = "keep_organization_policy_denied"
    elif response.status_code == 401:
        code = "keep_token_rejected"
    elif response.status_code == 403 and (
        "not available for your account" in message or "not enabled for this user" in message
    ):
        code = "keep_account_unavailable"
    elif response.status_code == 429:
        code = "keep_rate_limited"
    else:
        code = "keep_access_denied" if response.status_code == 403 else "keep_verification_failed"
    # Only our bounded code and HTTP status are logged, never Google's response body.
    logging.getLogger(__name__).warning("Keep verification failed: http_status=%s code=%s", response.status_code, code)
    return code


class IntegrationStore(CredentialStore):
    def _key(self, provider: str) -> str:
        return f"integration_connection:{provider}"

    def consume_state(self, integration_id: str, state: str, browser: str) -> dict:
        """Atomically consume a matching, unexpired, browser-bound pending flow.

        A single pending flow per integration keeps abandoned state bounded.
        Compare-and-delete also works when multiple app workers race a callback.
        """
        if not self.available:
            raise IntegrationError("storage_unavailable")
        key = self._key(f"pending:{integration_id}")
        with Session(self._store.engine) as session:
            row = session.exec(select(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == key,
            )).first()
            if row is None:
                raise IntegrationError("invalid_state")
            raw = row.value_json
            payload = json.loads(raw)
            pending = json.loads(crypto.decrypt(payload["secret_enc"]))
            if (payload.get("status") != "pending" or not state or not browser or pending["expires_at"] < time.time()
                    or not secrets.compare_digest(pending["state_hash"], digest(state))
                    or not secrets.compare_digest(pending["browser_hash"], digest(browser))):
                raise IntegrationError("invalid_state")
            payload["status"] = "exchanging"
            claimed = json.dumps(payload)
            result = session.execute(update(RuntimeSetting).where(
                RuntimeSetting.id == row.id, RuntimeSetting.value_json == raw,
            ).values(value_json=claimed))
            if result.rowcount != 1:
                raise IntegrationError("invalid_state")
            session.commit()
            pending["claim"] = claimed
            return pending

    def commit_connection(self, integration_id: str, claim: str, record: ConnectionRecord, *, provisional: bool = False) -> None:
        """Commit only while this callback still owns the pending flow.

        Disconnect, configuration changes, or a newer login invalidate the claim
        even if the token exchange is already in flight.
        """
        key = self._key(f"pending:{integration_id}")
        now = datetime.now(timezone.utc).isoformat()
        record.connected_at = (record.connected_at or now) if not provisional else None
        record.updated_at = now
        target = f"unverified:{integration_id}" if provisional else integration_id
        payload = record.public()
        payload.pop("provider")
        payload.pop("connected")
        payload["secret_enc"] = crypto.encrypt(json.dumps(record.secret))
        with Session(self._store.engine) as session:
            result = session.execute(delete(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == key, RuntimeSetting.value_json == claim,
            ))
            if result.rowcount != 1:
                raise IntegrationError("invalid_state")
            row = session.exec(select(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(target),
            )).first()
            if row is None:
                row = RuntimeSetting(scope_type="global", scope_id="*", key=self._key(target), value_json="{}")
            row.value_json = json.dumps(payload)
            session.add(row)
            if not provisional:
                session.execute(delete(RuntimeSetting).where(
                    RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                    RuntimeSetting.key == self._key(f"unverified:{integration_id}"),
                ))
            session.commit()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def validate_redirect(uri: str) -> str:
    parsed = urlsplit(uri)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path != CALLBACK_PATH):
        raise IntegrationError("invalid_redirect_uri")
    return uri


class IntegrationConnections:
    def __init__(self, config, *, store=None, client_factory=httpx.Client):
        self.store = store if store is not None else IntegrationStore(config)
        self.client_factory = client_factory

    def descriptor(self, integration_id: str) -> dict:
        definition = REGISTRY[integration_id]
        client = self.store.load(f"client:{integration_id}")
        connection = self.store.load(integration_id)
        configured = bool(client and client.secret.get("client_secret"))
        workspace = self.store.load(f"workspace:{integration_id}")
        workspace_public = {
            "configured": bool(workspace and workspace.secret.get("private_key")),
            **(workspace.meta if workspace else {}),
        }
        return {
            **definition,
            "configured": configured or workspace_public["configured"],
            "workspace": workspace_public,
            "workspace_admin_scopes": [KEEP_SCOPE],
            "has_unverified_credentials": self.store.load(f"unverified:{integration_id}") is not None,
            "reauthorization_required": bool(connection and connection.status == "connected"
                and KEEP_SCOPE not in connection.meta.get("scopes", [])),
            "client": {
                "client_id": client.meta.get("client_id", "") if client else "",
                "redirect_uri": client.meta.get("redirect_uri", "") if client else "",
                "has_client_secret": configured,
            },
            "connection": (connection or ConnectionRecord(provider=integration_id)).public(),
        }

    def list_keep_notes(self, integration_id: str) -> dict:
        """Read-only discovery. Return note titles/shape, never checklist contents."""
        connection = self.store.load(integration_id)
        config = self.store.load(f"workspace:{integration_id}")
        if not connection or connection.status != "connected" or connection.auth_method != "service_account_delegation":
            raise IntegrationError("keep_not_connected")
        if not config or not config.secret.get("private_key") or config.meta["subject"] != connection.account:
            raise IntegrationError("workspace_not_configured")
        results = []
        page_token = None
        seen = set()
        try:
            with self.client_factory(timeout=20, follow_redirects=False) as http:
                token, _ = delegated_token(http, config.meta, config.secret)
                for _ in range(20):
                    params = {"pageSize": 100}
                    if page_token:
                        params["pageToken"] = page_token
                    response = http.get(KEEP_URL, params=params, headers={"Authorization": f"Bearer {token}"})
                    if response.status_code != 200:
                        raise IntegrationError(keep_failure_code(response))
                    page = response.json()
                    if not isinstance(page, dict) or not isinstance(page.get("notes", []), list):
                        raise IntegrationError("keep_invalid_response")
                    for note in page.get("notes", []):
                        if not isinstance(note, dict) or note.get("trashed"):
                            continue
                        name = note.get("name")
                        if not isinstance(name, str) or not re.fullmatch(r"notes/[A-Za-z0-9_-]+", name):
                            continue
                        body = note.get("body") or {}
                        if not isinstance(body, dict):
                            body = {}
                        items = body.get("list", {}).get("listItems", []) if isinstance(body.get("list"), dict) else []
                        if not isinstance(items, list):
                            items = []
                        title = note.get("title")
                        results.append({"name": name, "title": title[:1000] if isinstance(title, str) else "",
                                        "type": "list" if "list" in body else "text",
                                        "item_count": len(items),
                                        "unchecked_count": sum(isinstance(item, dict) and not item.get("checked", False) for item in items)})
                    next_token = page.get("nextPageToken")
                    if not next_token:
                        return {"notes": results, "total": len(results)}
                    if not isinstance(next_token, str) or next_token in seen:
                        raise IntegrationError("keep_invalid_response")
                    seen.add(next_token)
                    page_token = next_token
        except WorkspaceAuthError as exc:
            raise IntegrationError(str(exc)) from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise IntegrationError("google_unavailable") from exc
        raise IntegrationError("keep_too_many_pages")

    def read_keep_list(self, integration_id: str, name: str) -> dict:
        """Return one explicitly named list's unchecked texts; never mutate Keep."""
        if not re.fullmatch(r"notes/[A-Za-z0-9_-]+", name):
            raise IntegrationError("invalid_keep_note")
        connection = self.store.load(integration_id)
        config = self.store.load(f"workspace:{integration_id}")
        if (not connection or connection.status != "connected"
                or connection.auth_method != "service_account_delegation"
                or not config or config.meta.get("subject") != connection.account
                or not config.secret.get("private_key")):
            raise IntegrationError("keep_not_connected")
        try:
            with self.client_factory(timeout=20, follow_redirects=False) as http:
                token, _ = delegated_token(http, config.meta, config.secret)
                response = http.get(f"{KEEP_URL}/{name.removeprefix('notes/')}",
                                    headers={"Authorization": f"Bearer {token}"})
                if response.status_code != 200:
                    raise IntegrationError(keep_failure_code(response))
                note = response.json()
        except WorkspaceAuthError as exc:
            raise IntegrationError(str(exc)) from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise IntegrationError("google_unavailable") from exc
        if not isinstance(note, dict) or note.get("name") != name or note.get("trashed"):
            raise IntegrationError("keep_invalid_response")
        body = note.get("body")
        listing = body.get("list") if isinstance(body, dict) else None
        if not isinstance(listing, dict) or not isinstance(listing.get("listItems"), list):
            raise IntegrationError("keep_not_list")
        items = []
        for item in listing["listItems"]:
            if not isinstance(item, dict) or item.get("checked") or item.get("childListItems"):
                continue  # Nested items need explicit semantics before importing.
            text = item.get("text")
            value = text.get("text") if isinstance(text, dict) else None
            if isinstance(value, str) and value.strip():
                items.append(value.strip())
        return {"name": name, "title": note.get("title", ""), "unchecked_items": items,
                "account": connection.account, "updated_at": note.get("updateTime")}

    def configure_workspace(self, integration_id: str, raw: str, subject: str):
        try:
            meta, secret = validate_service_account(raw, subject)
        except WorkspaceAuthError as exc:
            raise IntegrationError(str(exc)) from exc
        previous = self.store.load(f"workspace:{integration_id}")
        self.store.save(ConnectionRecord(provider=f"workspace:{integration_id}", status="configured",
                                         auth_method="service_account_delegation", meta=meta, secret=secret))
        connection = self.store.load(integration_id)
        if (connection and connection.auth_method == "service_account_delegation"
                and (not previous or previous.meta != meta or previous.secret != secret)):
            self.store.delete(integration_id)
        self.store.delete(f"pending:{integration_id}")
        self.store.delete(f"unverified:{integration_id}")
        return self.descriptor(integration_id)

    def verify_workspace(self, integration_id: str):
        config = self.store.load(f"workspace:{integration_id}")
        if not config or not config.secret.get("private_key"):
            raise IntegrationError("workspace_not_configured")
        # Reuse the atomic claim so disconnect/reconfiguration invalidates in-flight verification.
        state, browser = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self.store.save(ConnectionRecord(provider=f"pending:{integration_id}", status="pending", secret={
            "state_hash": digest(state), "browser_hash": digest(browser), "expires_at": time.time() + 120,
        }))
        pending = self.store.consume_state(integration_id, state, browser)
        current = self.store.load(f"workspace:{integration_id}")
        if not current or current.meta != config.meta or current.secret != config.secret:
            raise IntegrationError("invalid_state")
        try:
            with self.client_factory(timeout=20, follow_redirects=False) as http:
                token, expires_at = delegated_token(http, config.meta, config.secret)
                probe = http.get(KEEP_URL, params={"pageSize": 1}, headers={"Authorization": f"Bearer {token}"})
                if probe.status_code != 200:
                    raise IntegrationError(keep_failure_code(probe))
        except WorkspaceAuthError as exc:
            raise IntegrationError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError("google_unavailable") from exc
        self.store.commit_connection(integration_id, pending["claim"], ConnectionRecord(
            provider=integration_id, status="connected", auth_method="service_account_delegation",
            account=config.meta["subject"],
            meta={"client_id": config.meta["client_id"], "scopes": [KEEP_SCOPE],
                  "service_account_email": config.meta["client_email"], "workspace_domain": config.meta["workspace_domain"],
                  "verified_at": time.time(), "expires_at": expires_at, "read_only": False},
            secret={"access_token": token},
        ))
        return self.descriptor(integration_id)

    def configure(self, integration_id: str, client_id: str, client_secret: str | None, redirect_uri: str):
        redirect_uri = validate_redirect(redirect_uri)
        if not client_id.endswith(".apps.googleusercontent.com"):
            raise IntegrationError("invalid_client_id")
        previous = self.store.load(f"client:{integration_id}")
        same_client = bool(previous and previous.meta.get("client_id") == client_id)
        secret = client_secret or (previous.secret.get("client_secret") if same_client else None)
        if not secret:
            raise IntegrationError("missing_client_secret")
        self.store.save(ConnectionRecord(
            provider=f"client:{integration_id}", status="configured",
            meta={"client_id": client_id, "redirect_uri": redirect_uri},
            secret={"client_secret": secret},
        ))
        # A flow created with different client settings must not finish afterward.
        self.store.delete(f"pending:{integration_id}")
        self.store.delete(f"unverified:{integration_id}")
        return self.descriptor(integration_id)

    def start(self, integration_id: str, return_to: str, origin: str) -> dict:
        if return_to not in RETURN_PATHS:
            raise IntegrationError("invalid_return_path")
        client = self.store.load(f"client:{integration_id}")
        if not client or not client.secret.get("client_secret"):
            raise IntegrationError("client_not_configured")
        uri = validate_redirect(client.meta["redirect_uri"])
        parsed = urlsplit(uri)
        if origin != f"{parsed.scheme}://{parsed.netloc}":
            raise IntegrationError("callback_origin_mismatch")
        state, browser, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        self.store.save(ConnectionRecord(
            provider=f"pending:{integration_id}", status="pending", secret={
                "state_hash": digest(state), "browser_hash": digest(browser),
                "verifier": verifier, "expires_at": time.time() + 600,
                "redirect_uri": uri, "return_to": return_to,
                "client_id": client.meta["client_id"],
            },
        ))
        # Keep scopes cannot be shown on Google's user-consent screen. Never
        # force consent, including when recovering a missing offline token.
        # Live acceptance: forcing consent returned invalid_scope (TASK-906).
        params = {
            "client_id": client.meta["client_id"], "redirect_uri": uri,
            "response_type": "code", "scope": " ".join(REGISTRY[integration_id]["admin_scopes"]),
            "access_type": "offline", "prompt": "select_account", "state": state,
            "code_challenge": challenge, "code_challenge_method": "S256",
        }
        return {"authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params), "browser": browser}

    def finish(self, integration_id: str, state: str, browser: str, code: str, error: str) -> str:
        pending = self.store.consume_state(integration_id, state, browser)
        if error:
            raise IntegrationError("google_scope_not_authorized" if error == "invalid_scope" else "authorization_denied")
        if not code:
            raise IntegrationError("missing_code")
        client = self.store.load(f"client:{integration_id}")
        if not client or client.meta["client_id"] != pending["client_id"]:
            raise IntegrationError("client_not_configured")
        try:
            with self.client_factory(timeout=20, follow_redirects=False) as http:
                response = http.post("https://oauth2.googleapis.com/token", data={
                    "client_id": pending["client_id"], "client_secret": client.secret["client_secret"],
                    "code": code, "code_verifier": pending["verifier"],
                    "redirect_uri": pending["redirect_uri"], "grant_type": "authorization_code",
                })
                if response.status_code != 200:
                    raise IntegrationError("token_exchange_failed")
                token = response.json()
                expires_in = int(token.get("expires_in", 3600))
                if expires_in <= 0:
                    raise IntegrationError("token_exchange_failed")
                if KEEP_SCOPE not in token.get("scope", "").split():
                    raise IntegrationError("missing_keep_scope")
                if not token.get("access_token"):
                    raise IntegrationError("token_exchange_failed")
                headers = {"Authorization": f"Bearer {token['access_token']}"}
                identity = http.get("https://openidconnect.googleapis.com/v1/userinfo", headers=headers)
                if identity.status_code != 200:
                    raise IntegrationError("account_verification_failed")
                user = identity.json()
                if not user.get("sub") or not user.get("email") or user.get("email_verified") is not True:
                    raise IntegrationError("account_verification_failed")
                # Google may omit a refresh token on subsequent authorizations. Reuse
                # only a token belonging to the same client, subject, and required grant.
                refresh_token = token.get("refresh_token")
                for key in (f"unverified:{integration_id}", integration_id):
                    previous = self.store.load(key)
                    if (not refresh_token and previous
                            and previous.meta.get("client_id") == pending["client_id"]
                            and previous.meta.get("google_subject") == user["sub"]
                            and KEEP_SCOPE in previous.meta.get("scopes", [])):
                        refresh_token = previous.secret.get("refresh_token")
                # Probe even without offline access: don't hide the original Keep denial.
                # Retain OAuth credentials separately if verification fails, never as connected.
                failure = None
                try:
                    probe = http.get("https://keep.googleapis.com/v1/notes", params={"pageSize": 1}, headers=headers)
                    if probe.status_code != 200:
                        failure = keep_failure_code(probe)
                except httpx.HTTPError:
                    failure = "google_unavailable"
                if failure or not refresh_token:
                    self.store.commit_connection(integration_id, pending["claim"], ConnectionRecord(
                        provider=integration_id, status="unverified", auth_method="oauth_redirect",
                        account=user["email"],
                        meta={"google_subject": user["sub"], "scopes": token["scope"].split(),
                              "client_id": pending["client_id"], "expires_at": time.time() + expires_in,
                              "verification_error": failure or "offline_access_required"},
                        secret={"access_token": token["access_token"],
                                **({"refresh_token": refresh_token} if refresh_token else {})},
                    ), provisional=True)
                    raise IntegrationError(failure or "offline_access_required")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise IntegrationError("google_unavailable") from exc
        self.store.commit_connection(integration_id, pending["claim"], ConnectionRecord(
            provider=integration_id, status="connected", auth_method="oauth_redirect",
            account=user["email"],
            meta={"google_subject": user["sub"], "scopes": token["scope"].split(),
                  "client_id": pending["client_id"], "verified_at": time.time(),
                  "expires_at": time.time() + expires_in,
                  "read_only": False},
            secret={"access_token": token["access_token"], "refresh_token": refresh_token},
        ))
        return pending["return_to"]

    def disconnect(self, integration_id: str) -> bool:
        self.store.delete(f"pending:{integration_id}")
        removed_unverified = self.store.delete(f"unverified:{integration_id}")
        removed_workspace = self.store.delete(f"workspace:{integration_id}")
        return self.store.delete(integration_id) or removed_unverified or removed_workspace
