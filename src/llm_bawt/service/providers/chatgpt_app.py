"""Separate app-authorized ChatGPT connection. The app is its sole refresher.

The existing Codex provider/file/broker are deliberately not consulted. A DB
transaction lock serializes rotating refresh, sign-in replacement and disconnect
across app workers. A pending browser attempt never replaces an active account.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from contextlib import contextmanager
from uuid import uuid4

from authlib.integrations.base_client.errors import OAuthError
from sqlalchemy import text

from . import chatgpt_app_auth as auth
from .base import (
    HEALTH_BROKEN, HEALTH_OK, HEALTH_UNCONFIGURED, STATUS_CONNECTED,
    STATUS_DISCONNECTED, STATUS_ERROR, ConnectionRecord, ProviderAdapter, health_block,
)

AUTH_BROWSER_OAUTH = "browser_oauth"
STATUS_PLAN_DISABLED = "plan_disabled"
_lock = threading.RLock()
_pending: dict[str, auth.PendingSignIn] = {}


class ChatGPTAppAdapter(ProviderAdapter):
    id = "chatgpt-app"
    label = "ChatGPT (app-authorized)"
    auth_methods = (AUTH_BROWSER_OAUTH,)
    access_path_keys = ("openai-chatgpt-app",)
    billing_mode = "subscription"

    @contextmanager
    def _transaction(self):
        if not self.store.available or self.store.durable_key_problem():
            raise auth.SignInError("Protected credential storage is unavailable.")
        with _lock, self.store._store.engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(1944, 1)"))
            yield conn

    def descriptor(self):
        result = super().descriptor()
        result["description"] = (
            "Own-app ChatGPT plan connection to the public Responses API. "
            "Separate from Codex; never switches existing bots automatically. "
            "Subscription quota telemetry is not currently available."
        )
        return result

    def health(self):
        record = self.store.load(self.id)
        if not record or record.status == STATUS_DISCONNECTED:
            return health_block(HEALTH_UNCONFIGURED, detail="not connected")
        if record.status == STATUS_PLAN_DISABLED:
            return health_block(HEALTH_BROKEN, detail="Signed in; ChatGPT plan usage is not enabled.", fix="reconnect")
        if record.status == STATUS_ERROR or not record.secret.get("access_token"):
            return health_block(HEALTH_BROKEN, detail="Reconnect this app-authorized account.", fix="reconnect")
        return health_block(
            HEALTH_OK,
            expires_at=int(record.secret.get("expires_at", 0) * 1000),
            last_refresh_at=record.meta.get("last_refresh_at"),
        )

    def start_browser_login(self):
        with self._transaction() as conn:
            record = self.store.load_in(conn, self.id)
            host = self.store.load_in(conn, f"{self.id}-host")
            if host is None:
                host = ConnectionRecord(provider=f"{self.id}-host", meta={"host_id": f"urn:uuid:{uuid4()}"})
                self.store.save_in(conn, host)
            if record and record.status != STATUS_DISCONNECTED and not record.secret.get("client_id"):
                raise auth.SignInError("Stored registration is unreadable; restore protected storage first.")
            pending, url = auth.begin(
                host.meta["host_id"], registration=record.secret if record else {},
                generation=record.updated_at if record else None,
                reconsent=bool(record and record.status == STATUS_PLAN_DISABLED),
            )
            for key in list(_pending):
                if _pending[key].deadline <= time.monotonic():
                    _pending.pop(key)
            if len(_pending) >= 8:
                raise auth.SignInError("Too many pending sign-ins. Cancel an attempt or wait ten minutes.")
            session = secrets.token_urlsafe(32)
            _pending[session] = pending
        return {"session_id": session, "verification_uri": url, "instructions": (
            "Continue in your browser. If the final 127.0.0.1 page cannot load, "
            "copy its full address into this connection form only, never chat."
        )}

    def cancel_browser_login(self, session_id: str):
        with _lock:
            _pending.pop(session_id, None)
        return {"cancelled": True}

    def complete_browser_login(self, session_id: str, callback_url: str):
        # Validate the callback before consuming the attempt. All valid attempts
        # are single-use, including exchanges which fail upstream.
        with self._transaction() as conn:
            pending = _pending.get(session_id)
            if pending is None:
                raise auth.SignInError("Sign-in is missing or expired. Start again.")
            _, issued = auth.callback_parameters(callback_url, pending)
            _pending.pop(session_id)
            current = self.store.load_in(conn, self.id)
            if (current.updated_at if current else None) != pending.generation:
                raise auth.SignInError("Connection changed during sign-in. Start again.")
            try:
                bundle = auth.exchange(callback_url, pending)
            except Exception:
                # A validated callback's issued registration must survive a
                # failed/expired exchange. It is not yet a verified identity or
                # connected account. Never replace an existing active record.
                if current is None or not current.secret.get("client_id"):
                    self.store.save_in(conn, ConnectionRecord(
                        provider=self.id, secret={"client_id": issued},
                    ))
                failure = True
            else:
                record = ConnectionRecord(
                    provider=self.id, status=STATUS_CONNECTED if bundle["plan_enabled"] else STATUS_PLAN_DISABLED,
                    auth_method=AUTH_BROWSER_OAUTH,
                    account=bundle.get("email") or "ChatGPT account", secret=bundle,
                    meta={"last_refresh_at": int(time.time() * 1000)},
                )
                self.store.save_in(conn, record)
                failure = False
        if failure:
            raise auth.SignInError("Sign-in validation or exchange failed. Start again; if needed disconnect BawtHub in ChatGPT Settings.")
        return {"status": record.status, "connection": record.public()}

    def access_token(self, *, rejected_token_sha256: str | None = None):
        with self._transaction() as conn:
            record = self.store.load_in(conn, self.id)
            if not record or record.status != STATUS_CONNECTED or not record.secret.get("access_token"):
                raise auth.SignInError("Connect ChatGPT (app-authorized) in Providers.")
            bundle = record.secret
            rejected = rejected_token_sha256 == hashlib.sha256(bundle["access_token"].encode()).hexdigest()
            if bundle.get("expires_at", 0) <= time.time() + 300 or rejected:
                try:
                    bundle = auth.renew(bundle)
                except OAuthError as error:
                    # Only terminal credential errors clear tokens. Transient
                    # network/5xx leaves the saved pair intact, never falls back.
                    if error.error in {"invalid_grant", "invalid_refresh_token", "refresh_token_expired",
                                       "refresh_token_invalidated", "refresh_token_reused", "token_expired"}:
                        record.secret = self._registration(bundle)
                        record.status = STATUS_ERROR
                        self.store.save_in(conn, record)
                        # Commit the terminal state instead of raising inside the transaction.
                        return {"state": "reconnect", "access_token": None}
                    raise auth.SignInError("ChatGPT token renewal is unavailable.") from None
                except Exception:
                    raise auth.SignInError("ChatGPT token renewal is unavailable.") from None
                record.secret = bundle
                record.meta["last_refresh_at"] = int(time.time() * 1000)
                self.store.save_in(conn, record)
            # An app registration is workspace-scoped. Include both registration
            # and verified subject in replay/cache isolation, never the email.
            identity = hashlib.sha256(f"{bundle['client_id']}\0{bundle['subject']}".encode()).hexdigest()
            return {"access_token": bundle["access_token"], "account_id": identity,
                    "expires_at": bundle["expires_at"], "state": "ok"}

    @staticmethod
    def _registration(bundle):
        return {key: bundle[key] for key in ("client_id", "subject", "issuer", "email") if key in bundle}

    def disconnect(self):
        with self._transaction() as conn:
            _pending.clear()
            record = self.store.load_in(conn, self.id)
            remote = True
            if record is None:
                # A tombstone fences a late initial exchange in another worker.
                self.store.save_in(conn, ConnectionRecord(provider=self.id))
            if record:
                try:
                    remote = auth.revoke(record.secret)
                except Exception:
                    remote = False
                record.secret = self._registration(record.secret)
                record.status = STATUS_DISCONNECTED
                record.meta = {"remote_revocation": "confirmed" if remote else "unconfirmed"}
                self.store.save_in(conn, record)
            return {"disconnected": True, "remote_revocation": "confirmed" if remote else "unconfirmed",
                    "detail": None if remote else "Local tokens removed. Disconnect BawtHub in ChatGPT Settings to confirm remote revocation."}
