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
import threading
import time
from contextlib import contextmanager, nullcontext
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit

import httpx
from sqlalchemy import delete, text, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from llm_bawt.integrations.keep_workspace import (
    KEEP_URL, TOKEN_URL, WorkspaceAuthError, delegated_token, validate_service_account,
)
from llm_bawt.runtime_settings import RuntimeSetting
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import ConnectionRecord, CredentialStore

KEEP_SCOPE = "https://www.googleapis.com/auth/keep"
TASKS_SCOPE = "https://www.googleapis.com/auth/tasks.readonly"
TASKS_LISTS_URL = "https://tasks.googleapis.com/tasks/v1/users/@me/lists"
IDENTITY_SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email"]


def callback_path(integration_id: str) -> str:
    return f"/api/chat/proxy/v1/integrations/{integration_id}/callback"


CALLBACK_PATH = callback_path("google-keep")
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
        "admin_scopes": [KEEP_SCOPE, *IDENTITY_SCOPES],
        "setup_url": "https://developers.google.com/workspace/keep/api/guides",
    },
    "google-tasks": {
        "id": "google-tasks", "label": "Google Tasks", "auth_method": "oauth_web",
        # Browser sign-in is the general path; a Workspace domain can instead
        # delegate a service account it already trusts (for example Keep's).
        "auth_methods": ["oauth_web", "service_account_delegation"],
        "supported_account_types": ["google", "google_workspace"],
        "description": "Import tasks from selected Google Tasks lists (including ones Google Assistant creates) as BawtHub tasks.",
        "scopes": [TASKS_SCOPE],
        "capabilities": {"read_tasks": True, "write_tasks": False},
        "limitations": "Read-only. Google Tasks has no push notifications, so BawtHub checks enabled lists every 10 minutes or on Sync now.",
        "requirements": "Enable the Google Tasks API in the OAuth client's Cloud project, then approve read-only Tasks access with the account whose lists you want to import.",
        "admin_scopes": [TASKS_SCOPE, *IDENTITY_SCOPES],
        "setup_url": "https://developers.google.com/workspace/tasks/overview",
    },
}
# Per-integration browser OAuth contract: the scope a grant must include, the
# read that proves it works, and the prefix for bounded probe error codes.
OAUTH = {
    "google-keep": {"required_scope": KEEP_SCOPE, "probe_url": "https://keep.googleapis.com/v1/notes",
                    "probe_params": {"pageSize": 1}, "error_prefix": "keep", "read_only": False},
    "google-tasks": {"required_scope": TASKS_SCOPE, "probe_url": TASKS_LISTS_URL,
                     "probe_params": {"maxResults": 1}, "error_prefix": "tasks", "read_only": True,
                     # tasks.readonly is user-consentable, so forcing consent always
                     # returns a refresh token, including after disconnect or token loss.
                     "prompt": "consent select_account", "bawthub_connection": True},
}


FINALIZE_SECONDS = 120


class IntegrationError(Exception):
    """A bounded, public error code; never includes an upstream body or token."""


def _connection_subject(record) -> str | None:
    """The account BawtHub fences a connected integration's runs by, else None."""
    return record.meta.get("google_subject") if record and record.status == "connected" else None


def keep_failure_code(response: httpx.Response, prefix: str = "keep") -> str:
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
        code = f"{prefix}_api_disabled"
    elif reasons & {"ACCESS_TOKEN_SCOPE_INSUFFICIENT", "insufficientPermissions"}:
        code = f"{prefix}_scope_insufficient"
    elif reasons & {"ORG_RESTRICTION_VIOLATION", "SECURITY_POLICY_VIOLATED", "ORG_POLICY_VIOLATION"}:
        code = f"{prefix}_organization_policy_denied"
    elif response.status_code == 401:
        code = f"{prefix}_token_rejected"
    elif response.status_code == 403 and (
        "not available for your account" in message or "not enabled for this user" in message
    ):
        code = f"{prefix}_account_unavailable"
    elif response.status_code == 429:
        code = f"{prefix}_rate_limited"
    else:
        code = f"{prefix}_access_denied" if response.status_code == 403 else f"{prefix}_verification_failed"
    # Only our bounded code and HTTP status are logged, never Google's response body.
    logging.getLogger(__name__).warning("%s verification failed: http_status=%s code=%s", prefix, response.status_code, code)
    return code


@dataclass(frozen=True)
class RowSnapshot:
    """One integration row exactly as read, for compare-and-set writes.

    ``record`` is None when there is no row or its JSON isn't an object.
    """

    row_id: int | None = None
    raw: str | None = None
    record: ConnectionRecord | None = None

    @property
    def exists(self) -> bool:
        return self.row_id is not None


_LOCAL_LIFECYCLE_LOCKS: dict[str, threading.Lock] = {}


class IntegrationStore(CredentialStore):
    def _key(self, provider: str) -> str:
        return f"integration_connection:{provider}"

    @contextmanager
    def lifecycle(self, integration_id: str) -> Iterator[None]:
        """Serialize one integration's lifecycle across workers.

        start, configure, disconnect and a callback's finalize/notify/commit all
        run under this lock, so the finalizing check and the write it guards are
        atomic with respect to each other. PostgreSQL uses a transaction-scoped
        advisory lock on a dedicated connection; other engines (tests) fall back
        to a process lock.
        """
        if not self.available:
            yield
            return
        engine = self._store.engine
        if engine.dialect.name == "postgresql":
            with engine.connect() as conn, conn.begin():
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
                             {"k": f"integration-lifecycle:{integration_id}"})
                yield
        else:
            with _LOCAL_LIFECYCLE_LOCKS.setdefault(integration_id, threading.Lock()):
                yield

    def _where(self, provider: str):
        return (RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(provider))

    def snapshot(self, provider: str) -> RowSnapshot:
        if not self.available:
            return RowSnapshot()
        with Session(self._store.engine) as session:
            row = session.exec(select(RuntimeSetting).where(*self._where(provider))).first()
            if row is None:
                return RowSnapshot()
            row_id, raw = row.id, row.value_json
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        record = self._record(provider, data) if isinstance(data, dict) else None
        return RowSnapshot(row_id, raw, record)

    def replace(self, provider: str, expected: RowSnapshot, record: ConnectionRecord) -> bool:
        """Write ``record`` only if the row is still exactly ``expected``.

        Returns False when another writer got there first (changed or deleted
        the row, or inserted one where none was), so the caller can re-read.
        """
        value = json.dumps(self._payload(record), ensure_ascii=False)
        with Session(self._store.engine) as session:
            if not expected.exists:
                session.add(RuntimeSetting(scope_type="global", scope_id="*",
                                           key=self._key(provider), value_json=value))
                try:
                    session.commit()
                except IntegrityError:  # unique (scope_type, scope_id, key)
                    session.rollback()
                    return False
                return True
            result = session.execute(update(RuntimeSetting).where(
                RuntimeSetting.id == expected.row_id, RuntimeSetting.value_json == expected.raw,
            ).values(value_json=value, updated_at=datetime.now(timezone.utc)))
            session.commit()
            return result.rowcount == 1

    def remove(self, provider: str, expected: RowSnapshot) -> bool:
        """Delete the row only if it is still exactly ``expected``."""
        with Session(self._store.engine) as session:
            result = session.execute(delete(RuntimeSetting).where(
                RuntimeSetting.id == expected.row_id, RuntimeSetting.value_json == expected.raw,
            ))
            session.commit()
            return result.rowcount == 1

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

    def finalize_claim(self, integration_id: str, claim: str) -> str:
        """Promote an exchanging claim to finalizing before an external side effect.

        While finalizing, start, configure and disconnect refuse (``finalizing``),
        so nothing can replace this flow between the BawtHub notification and
        the local commit.
        """
        payload = json.loads(claim)
        payload["status"] = "finalizing"
        payload["finalizing_at"] = time.time()
        promoted = json.dumps(payload)
        with Session(self._store.engine) as session:
            result = session.execute(update(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(f"pending:{integration_id}"), RuntimeSetting.value_json == claim,
            ).values(value_json=promoted))
            if result.rowcount != 1:
                raise IntegrationError("invalid_state")
            session.commit()
        return promoted

    def release_claim(self, integration_id: str, claim: str) -> None:
        """Drop this flow's own claim (a failed finalization); a newer flow is untouched."""
        with Session(self._store.engine) as session:
            session.execute(delete(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(f"pending:{integration_id}"), RuntimeSetting.value_json == claim,
            ))
            session.commit()

    def finalizing(self, integration_id: str) -> bool:
        """True while a callback owns the final step (bounded: a crashed one expires)."""
        if not self.available:
            return False
        with Session(self._store.engine) as session:
            row = session.exec(select(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(f"pending:{integration_id}"),
            )).first()
        if row is None:
            return False
        payload = json.loads(row.value_json)
        return payload.get("status") == "finalizing" and payload.get("finalizing_at", 0) > time.time() - FINALIZE_SECONDS

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


def validate_redirect(uri: str, integration_id: str = "google-keep") -> str:
    parsed = urlsplit(uri)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path != callback_path(integration_id)):
        raise IntegrationError("invalid_redirect_uri")
    return uri


class IntegrationConnections:
    def __init__(self, config, *, store=None, client_factory=httpx.Client, record_connection=None):
        self.store = store if store is not None else IntegrationStore(config)
        self.client_factory = client_factory
        # Google Tasks: BawtHub fences import runs by connection generation, so
        # it hears about connect, disconnect and account change first.
        self.record_connection = record_connection or record_bawthub_connection

    def _lifecycle(self, integration_id: str):
        """The cross-worker lifecycle lock for BawtHub-fenced integrations (Keep is unchanged)."""
        if OAUTH.get(integration_id, {}).get("bawthub_connection") and hasattr(self.store, "lifecycle"):
            return self.store.lifecycle(integration_id)
        return nullcontext()

    def _fenced(self, integration_id: str) -> None:
        """Lifecycle changes wait while a callback finalizes a BawtHub-fenced connection."""
        if OAUTH.get(integration_id, {}).get("bawthub_connection") and self.store.finalizing(integration_id):
            raise IntegrationError("connection_in_progress")

    def _notify_bawthub(self, integration_id: str, subject: str | None) -> None:
        if OAUTH.get(integration_id, {}).get("bawthub_connection"):
            self.record_connection(subject)

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
            "workspace_admin_scopes": [OAUTH.get(integration_id, {"required_scope": KEEP_SCOPE})["required_scope"]],
            # Other integrations' saved Workspace keys this one may reuse by reference.
            "workspace_sources": [
                other for other in OAUTH if other != integration_id
                and "service_account_delegation" in REGISTRY[other].get("auth_methods", [REGISTRY[other]["auth_method"]])
                and (source := self.store.load(f"workspace:{other}")) is not None and source.secret.get("private_key")
            ] if "service_account_delegation" in definition.get("auth_methods", []) else [],
            "has_unverified_credentials": self.store.load(f"unverified:{integration_id}") is not None,
            "reauthorization_required": bool(connection and connection.status == "connected"
                and OAUTH.get(integration_id, {"required_scope": KEEP_SCOPE})["required_scope"]
                not in connection.meta.get("scopes", [])),
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

    def _delegation_scope(self, integration_id: str) -> str:
        definition = REGISTRY.get(integration_id, {})
        if "service_account_delegation" not in definition.get("auth_methods", [definition.get("auth_method")]):
            raise IntegrationError("workspace_not_supported")
        return OAUTH[integration_id]["required_scope"]

    def configure_workspace(self, integration_id: str, raw: str | None, subject: str, copy_from: str | None = None):
        """Save this integration's Workspace key, pasted or copied from another integration's.

        A copy reuses the stored key server-side, so it never crosses the
        browser again; Google still authorizes each scope separately.
        """
        self._delegation_scope(integration_id)
        if copy_from is not None:
            if copy_from == integration_id or copy_from not in self.descriptor(integration_id)["workspace_sources"]:
                raise IntegrationError("workspace_source_unavailable")
            source = self.store.load(f"workspace:{copy_from}")
            if not source or not source.secret.get("private_key"):
                raise IntegrationError("workspace_source_unavailable")  # removed since the allowlist read
            try:
                meta, _ = validate_service_account(json.dumps({
                    "type": "service_account", "token_uri": TOKEN_URL,
                    "client_email": source.meta["client_email"], "client_id": source.meta["client_id"],
                    "private_key_id": source.meta["private_key_id"], "project_id": source.meta["project_id"],
                    "private_key": source.secret["private_key"],
                }), subject)
            except (WorkspaceAuthError, KeyError) as exc:
                raise IntegrationError("invalid_service_account_key") from exc
            secret = {"private_key": source.secret["private_key"]}
        else:
            try:
                meta, secret = validate_service_account(raw or "", subject)
            except WorkspaceAuthError as exc:
                raise IntegrationError(str(exc)) from exc
        with self._lifecycle(integration_id):
            self._fenced(integration_id)
            previous = self.store.load(f"workspace:{integration_id}")
            connection = self.store.load(integration_id)
            replaced = (connection and connection.auth_method == "service_account_delegation"
                        and (not previous or previous.meta != meta or previous.secret != secret))
            if replaced:
                # Fence imports, then drop the connection before the new key lands so
                # no refresh can pair the old verified record with the new key.
                self._notify_bawthub(integration_id, None)
                self.store.delete(integration_id)
            self.store.save(ConnectionRecord(provider=f"workspace:{integration_id}", status="configured",
                                             auth_method="service_account_delegation", meta=meta, secret=secret))
            self.store.delete(f"pending:{integration_id}")
            self.store.delete(f"unverified:{integration_id}")
        return self.descriptor(integration_id)

    def verify_workspace(self, integration_id: str):
        with self._lifecycle(integration_id):
            self._fenced(integration_id)
            return self._verify_workspace(integration_id)

    def _verify_workspace(self, integration_id: str):
        scope = self._delegation_scope(integration_id)
        oauth = OAUTH[integration_id]
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
                token, expires_at = delegated_token(http, config.meta, config.secret, scope)
                probe = http.get(oauth["probe_url"], params=oauth["probe_params"],
                                 headers={"Authorization": f"Bearer {token}"})
                if probe.status_code != 200:
                    raise IntegrationError(keep_failure_code(probe, oauth["error_prefix"]))
        except WorkspaceAuthError as exc:
            raise IntegrationError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError("google_unavailable") from exc
        claim = pending["claim"]
        subject = config.meta["subject"]
        if oauth.get("bawthub_connection"):
            # Same order as browser OAuth: own the final step, tell BawtHub the
            # account (the delegated user's address), then commit.
            claim = self.store.finalize_claim(integration_id, claim)
            try:
                self._notify_bawthub(integration_id, subject)
            except Exception:
                current = self.store.load(integration_id)
                try:
                    self.record_connection(_connection_subject(current))
                except Exception:
                    pass
                self.store.release_claim(integration_id, claim)
                raise
        self._commit_after_notify(integration_id, claim, bool(oauth.get("bawthub_connection")), ConnectionRecord(
            provider=integration_id, status="connected", auth_method="service_account_delegation",
            account=subject,
            meta={"client_id": config.meta["client_id"], "scopes": [scope], "google_subject": subject,
                  "service_account_email": config.meta["client_email"], "workspace_domain": config.meta["workspace_domain"],
                  "key_digest": digest(config.secret["private_key"]),
                  "verified_at": time.time(), "expires_at": expires_at, "read_only": oauth["read_only"]},
            secret={"access_token": token},
        ))
        return self.descriptor(integration_id)

    def configure(self, integration_id: str, client_id: str, client_secret: str | None, redirect_uri: str):
        if integration_id not in OAUTH:
            raise IntegrationError("unsupported_integration")
        with self._lifecycle(integration_id):
            return self._configure(integration_id, client_id, client_secret, redirect_uri)

    def _configure(self, integration_id: str, client_id: str, client_secret: str | None, redirect_uri: str):
        self._fenced(integration_id)
        redirect_uri = validate_redirect(redirect_uri, integration_id)
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
        with self._lifecycle(integration_id):
            return self._start(integration_id, return_to, origin)

    def _start(self, integration_id: str, return_to: str, origin: str) -> dict:
        self._fenced(integration_id)
        client = self.store.load(f"client:{integration_id}")
        if not client or not client.secret.get("client_secret"):
            raise IntegrationError("client_not_configured")
        uri = validate_redirect(client.meta["redirect_uri"], integration_id)
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
        # force consent (kept for every integration; Google still issues a
        # refresh token on first grant and refresh reuse covers later ones), including when recovering a missing offline token.
        # Live acceptance: forcing consent returned invalid_scope (TASK-906).
        params = {
            "client_id": client.meta["client_id"], "redirect_uri": uri,
            "response_type": "code", "scope": " ".join(REGISTRY[integration_id]["admin_scopes"]),
            "access_type": "offline", "prompt": OAUTH[integration_id].get("prompt", "select_account"), "state": state,
            "code_challenge": challenge, "code_challenge_method": "S256",
        }
        return {"authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params), "browser": browser}

    def finish(self, integration_id: str, state: str, browser: str, code: str, error: str) -> str:
        oauth = OAUTH[integration_id]
        required = oauth["required_scope"]
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
                if required not in token.get("scope", "").split():
                    raise IntegrationError(f"missing_{oauth['error_prefix']}_scope")
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
                            and required in previous.meta.get("scopes", [])):
                        refresh_token = previous.secret.get("refresh_token")
                # Probe even without offline access: don't hide the original denial.
                # Retain OAuth credentials separately if verification fails, never as connected.
                failure = None
                try:
                    probe = http.get(oauth["probe_url"], params=oauth["probe_params"], headers=headers)
                    if probe.status_code != 200:
                        failure = keep_failure_code(probe, oauth["error_prefix"])
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
        # Finalize, notify and commit under the lifecycle lock: a start, configure or
        # disconnect either completed before (and replaced the claim, so the
        # finalize CAS fails here) or waits until this commit lands.
        with self._lifecycle(integration_id):
            self._finalize(integration_id, pending, oauth, user, token, expires_in, refresh_token)
        return pending["return_to"]

    def _finalize(self, integration_id, pending, oauth, user, token, expires_in, refresh_token) -> None:
        claim = pending["claim"]
        if oauth.get("bawthub_connection"):
            # Own the final step before telling BawtHub: a newer start, configure or
            # disconnect now refuses until this commit lands or the claim is released.
            claim = self.store.finalize_claim(integration_id, claim)
            try:
                self._notify_bawthub(integration_id, user["sub"])
            except Exception:
                # An ambiguous bump may have landed: restore BawtHub to the local truth.
                current = self.store.load(integration_id)
                try:
                    self.record_connection(_connection_subject(current))
                except Exception:
                    pass  # runs still refuse a subject mismatch; the next connect heals it
                self.store.release_claim(integration_id, claim)
                raise
        self._commit_after_notify(integration_id, claim, bool(oauth.get("bawthub_connection")), ConnectionRecord(
            provider=integration_id, status="connected", auth_method="oauth_redirect",
            account=user["email"],
            meta={"google_subject": user["sub"], "scopes": token["scope"].split(),
                  "client_id": pending["client_id"], "verified_at": time.time(),
                  "expires_at": time.time() + expires_in,
                  "read_only": oauth["read_only"]},
            secret={"access_token": token["access_token"], "refresh_token": refresh_token},
        ))

    def _commit_after_notify(self, integration_id: str, claim: str, notified: bool, record: ConnectionRecord) -> None:
        """Commit a connection BawtHub was already told about; on failure, re-align BawtHub.

        If the write may have failed, BawtHub's subject is set back to whatever the
        local record actually holds and the finalizing claim is released for retry.
        """
        try:
            self.store.commit_connection(integration_id, claim, record)
        except Exception:
            if not notified:
                raise
            current = self.store.load(integration_id)
            landed = bool(current and current.status == "connected"
                          and current.meta.get("verified_at") == record.meta.get("verified_at"))
            if landed:
                return  # ambiguous error, but the write is in place and matches BawtHub
            try:
                self.record_connection(_connection_subject(current))
            except Exception:
                pass  # runs still refuse a subject mismatch; the next connect heals it
            try:
                self.store.release_claim(integration_id, claim)
            except Exception:
                pass
            raise

    def disconnect(self, integration_id: str) -> bool:
        with self._lifecycle(integration_id):
            return self._disconnect(integration_id)

    def _disconnect(self, integration_id: str) -> bool:
        self._fenced(integration_id)
        self._notify_bawthub(integration_id, None)  # fence import runs first
        self.store.delete(f"pending:{integration_id}")
        removed_unverified = self.store.delete(f"unverified:{integration_id}")
        removed_workspace = self.store.delete(f"workspace:{integration_id}")
        return self.store.delete(integration_id) or removed_unverified or removed_workspace


GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def access_token(manager: IntegrationConnections, integration_id: str, http: httpx.Client) -> tuple[str, str]:
    """A live access token and the Google subject for a connected browser-OAuth integration.

    Refreshes with the stored refresh token near expiry. The write is
    compare-and-set; a concurrent refresher's token is equally valid, so a lost
    race simply reloads.
    """
    snap = manager.store.snapshot(integration_id)
    record = snap.record
    if (not record or record.status != "connected"
            or record.auth_method not in ("oauth_redirect", "service_account_delegation")):
        raise IntegrationError("not_connected")
    subject = record.meta.get("google_subject")
    if record.secret.get("access_token") and record.meta.get("expires_at", 0) > time.time() + 60:
        return record.secret["access_token"], subject
    if record.auth_method == "service_account_delegation":
        # Re-mint for the verified subject only; a changed key needs re-verification.
        config = manager.store.load(f"workspace:{integration_id}")
        if (not config or not config.secret.get("private_key") or config.meta.get("subject") != record.account
                or config.meta.get("client_id") != record.meta.get("client_id")
                or digest(config.secret["private_key"]) != record.meta.get("key_digest")):
            raise IntegrationError("reauthorization_required")
        try:
            token, expires_at = delegated_token(http, config.meta, config.secret, OAUTH[integration_id]["required_scope"])
        except WorkspaceAuthError as exc:
            raise IntegrationError(str(exc)) from exc
        record.secret = {**record.secret, "access_token": token}
        record.meta = {**record.meta, "expires_at": expires_at}
        if not manager.store.replace(integration_id, snap, record):
            # Lost the write: use only a winner that is still this verified connection.
            winner = manager.store.load(integration_id)
            if (winner and winner.status == "connected" and winner.auth_method == "service_account_delegation"
                    and winner.account == record.account and winner.meta.get("key_digest") == record.meta.get("key_digest")
                    and winner.secret.get("access_token") and winner.meta.get("expires_at", 0) > time.time() + 60):
                return winner.secret["access_token"], winner.meta.get("google_subject")
            raise IntegrationError("reauthorization_required")
        return token, subject
    client = manager.store.load(f"client:{integration_id}")
    refresh = record.secret.get("refresh_token")
    if not client or not refresh or client.meta.get("client_id") != record.meta.get("client_id"):
        raise IntegrationError("reauthorization_required")
    response = http.post(GOOGLE_TOKEN_URL, data={
        "client_id": client.meta["client_id"], "client_secret": client.secret["client_secret"],
        "refresh_token": refresh, "grant_type": "refresh_token",
    })
    if response.status_code in (400, 401):
        raise IntegrationError("reauthorization_required")  # revoked or expired grant
    if response.status_code != 200:
        raise IntegrationError("google_unavailable")
    token = response.json()
    if not isinstance(token, dict) or not token.get("access_token"):
        raise IntegrationError("google_unavailable")
    record.secret = {**record.secret, "access_token": token["access_token"]}
    record.meta = {**record.meta, "expires_at": time.time() + int(token.get("expires_in", 3600))}
    if not manager.store.replace(integration_id, snap, record):
        logging.getLogger(__name__).info("%s token refresh raced; keeping the fresh token in memory", integration_id)
    return token["access_token"], subject


GOOGLE_TASKS_CONNECTION_PATH = "/internal/integrations/google-tasks/connection"


def record_bawthub_connection(subject: str | None) -> None:
    """Tell BawtHub the Google Tasks account changed; bounded error if it can't hear."""
    from llm_bawt.bawthub_auth import service_auth
    from llm_bawt.integrations.notify_bell import bell_base_url
    try:
        with httpx.Client(base_url=bell_base_url(), timeout=10, auth=service_auth()) as http:
            response = http.post(GOOGLE_TASKS_CONNECTION_PATH, json={"subject": subject})
    except httpx.HTTPError as exc:
        raise IntegrationError("bawthub_unavailable") from exc
    if not response.is_success:
        logging.getLogger(__name__).warning("BawtHub connection record failed: http_status=%s", response.status_code)
        raise IntegrationError("bawthub_unavailable")
