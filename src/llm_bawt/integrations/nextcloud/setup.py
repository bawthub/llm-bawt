"""Nextcloud Login Flow v2 and read-only Talk discovery.

One account/instance per deployment, matching the existing Talk transport. No
provisioner, shell access or second message queue is introduced here.
"""
from __future__ import annotations

import json
import secrets
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy import update
from sqlmodel import Session

from llm_bawt.integrations.connections import IntegrationError, IntegrationStore, REGISTRY, digest
from llm_bawt.runtime_settings import RuntimeSetting
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import ConnectionRecord

INTEGRATION_ID = "nextcloud-talk"
LOGIN_TTL = 600
PROVISIONING_REASON = (
    "Account authorization permits room discovery, not webhook bot installation or signing-secret recovery. "
    "A restricted server-side Nextcloud administration operation must securely import the existing bot's "
    "signing secret and verify its callback before reconnecting. No external provisioner is used."
)
DEFINITION = REGISTRY[INTEGRATION_ID]


class SetupError(IntegrationError):
    """Public, bounded diagnostics; upstream bodies and exceptions are never echoed."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(code)
        self.code, self.message, self.status = code, message, status


def canonical_instance(value: str) -> str:
    try:
        p = urlsplit(value)
        if (p.scheme != "https" or not p.hostname or p.username is not None or p.password is not None
                or p.query or p.fragment or p.port not in (None, 443)
                or any(c.isspace() or ord(c) < 32 for c in value)
                or any(c in value for c in ("\\", "%"))
                or any(part in (".", "..") for part in p.path.split("/"))
                or "//" in p.path):
            raise ValueError
        host = p.hostname.lower()
        # IP literals, loopback and credential-bearing URLs are never setup targets.
        import ipaddress
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or host.endswith((".localhost", ".local")):
                raise ValueError
        else:
            raise ValueError
        return urlunsplit(("https", host, p.path.rstrip("/"), "", ""))
    except (ValueError, TypeError):
        raise SetupError("invalid_instance_url", "Use the configured HTTPS Nextcloud base URL, without credentials, query or fragment.") from None


def pinned_url(value: str, instance: str, *, poll: bool = False) -> str:
    """Validate returned URLs before returning a login link or sending a token."""
    if not isinstance(value, str) or len(value) > 4096:
        raise SetupError("invalid_login_response", "Nextcloud returned an invalid login endpoint.", 502)
    p, base = urlsplit(value), urlsplit(instance)
    if (p.scheme != base.scheme or p.netloc != base.netloc or p.username is not None
            or p.password is not None or p.fragment or "\\" in value or "%" in p.path
            or any(c.isspace() or ord(c) < 32 for c in value)
            or any(part in (".", "..") for part in p.path.split("/"))
            or not p.path.startswith(base.path + "/")):
        raise SetupError("untrusted_login_url", "Nextcloud returned a login URL outside the configured instance.", 502)
    paths = (base.path + "/login/v2/", base.path + "/index.php/login/v2/")
    if poll and (p.path not in tuple(path + "poll" for path in paths) or p.query):
        raise SetupError("untrusted_login_url", "Nextcloud returned an unexpected polling endpoint.", 502)
    if not poll and not any(p.path.startswith(path + "flow/") for path in paths):
        raise SetupError("untrusted_login_url", "Nextcloud returned an unexpected browser login endpoint.", 502)
    return value


class NextcloudSetupStore(IntegrationStore):
    def release_claim(self, claim: str, pending: dict) -> None:
        """CAS release: an older HTTP request cannot resurrect a replaced flow."""
        pending = {k: v for k, v in pending.items() if k != "claim"}
        payload = {"status": "pending", "secret_enc": crypto.encrypt(json.dumps(pending))}
        with Session(self._store.engine) as session:
            result = session.execute(update(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(f"pending:{INTEGRATION_ID}"),
                RuntimeSetting.value_json == claim,
            ).values(value_json=json.dumps(payload)))
            if result.rowcount != 1:
                raise SetupError("login_superseded", "This login was replaced or disconnected. Start again.", 409)
            session.commit()


class NextcloudSetup:
    def __init__(self, config, *, store=None, client_factory=httpx.Client, bindings=None):
        self.config = config
        self.store = store if store is not None else NextcloudSetupStore(config)
        self.client_factory = client_factory
        self.bindings = bindings

    def provisioning(self) -> dict:
        return {"available": False, "reason": PROVISIONING_REASON}

    def status(self) -> dict:
        rec = self.store.load(INTEGRATION_ID)
        connected = bool(rec and rec.status == "connected" and rec.secret.get("app_password"))
        return {
            "instance_url": rec.meta.get("instance_url") if rec else getattr(self.config, "NEXTCLOUD_URL", None),
            "account": {"id": rec.account, "display_name": rec.meta.get("display_name", rec.account)} if connected else None,
            "account_connected": connected,
            "provisioning": self.provisioning(),
            "bindings": self.bindings.list() if self.bindings else [],
            "delivery_verified": False,
        }

    def descriptor(self) -> dict:
        status = self.status()
        return {**DEFINITION, "configured": status["account_connected"],
                "connection": {"provider": INTEGRATION_ID, "status": "connected" if status["account_connected"] else "disconnected",
                               "connected": status["account_connected"], "account": (status["account"] or {}).get("id"),
                               "auth_method": "nextcloud_login_v2", "meta": {"delivery_verified": False}},
                "provisioning": status["provisioning"]}

    def _client(self):
        return self.client_factory(timeout=20, follow_redirects=False, trust_env=False)

    def _instance(self, value: str) -> str:
        instance = canonical_instance(value)
        if instance != canonical_instance(self.config.NEXTCLOUD_URL):
            raise SetupError("instance_not_allowed", "This instance must match the server's NEXTCLOUD_URL. Ask the operator to configure it first.")
        return instance

    def start(self, instance_url: str) -> dict:
        instance = self._instance(instance_url)
        if not self.store.available:
            raise SetupError("storage_unavailable", "Secure integration storage is unavailable.", 503)
        browser = secrets.token_urlsafe(32)
        self.store.save(ConnectionRecord(provider=f"pending:{INTEGRATION_ID}", status="pending", secret={
            "state_hash": digest(browser), "browser_hash": digest(browser),
            "expires_at": time.time() + LOGIN_TTL, "instance_url": instance,
        }))
        pending = self._claim(browser)
        try:
            with self._client() as http:
                response = http.post(instance + "/index.php/login/v2")
            if response.status_code != 200:
                raise SetupError("login_start_failed", "Nextcloud Login Flow v2 is unavailable. Check the instance URL and server connectivity.", 502)
            body = response.json()
            login = pinned_url(body["login"], instance)
            endpoint = pinned_url(body["poll"]["endpoint"], instance, poll=True)
            token = body["poll"]["token"]
            if not isinstance(token, str) or not token or len(token) > 4096:
                raise ValueError
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise SetupError("login_start_failed", "Could not initiate Nextcloud login. Check connectivity and Login Flow v2 support.", 502) from None
        pending.update({"poll_url": endpoint, "poll_token": token})
        self.store.release_claim(pending["claim"], pending)
        return {"login_url": login, "expires_in": LOGIN_TTL, "browser": browser}

    def _claim(self, browser: str) -> dict:
        try:
            return self.store.consume_state(INTEGRATION_ID, browser, browser)
        except IntegrationError:
            raise SetupError("login_expired", "Login expired, is already being polled, or belongs to another browser. Start again if it persists.", 409) from None

    def poll(self, browser: str) -> dict:
        # A successful response may be lost. Allow an idempotent, browser-bound retry.
        rec = self.store.load(INTEGRATION_ID)
        if (browser and rec and rec.secret.get("login_expires_at", 0) > time.time()
                and secrets.compare_digest(rec.secret.get("login_browser_hash", ""), digest(browser))):
            return {"status": "connected"}
        pending = self._claim(browser)
        try:
            instance = self._instance(pending["instance_url"])
            endpoint = pinned_url(pending["poll_url"], instance, poll=True)
            with self._client() as http:
                response = http.post(endpoint, data={"token": pending["poll_token"]})
                if response.status_code == 404:
                    self.store.release_claim(pending["claim"], pending)
                    return {"status": "pending"}
                if response.status_code != 200:
                    raise SetupError("login_poll_failed", "Nextcloud could not complete login. Authorize again.", 502)
                result = response.json()
                if canonical_instance(result["server"]) != instance:
                    raise SetupError("untrusted_login_url", "The authorized account belongs to a different Nextcloud instance.", 502)
                user, password = result["loginName"], result["appPassword"]
                if not isinstance(user, str) or not user or not isinstance(password, str) or not password:
                    raise ValueError
                identity = self._ocs(http, instance, user, password, "/ocs/v2.php/cloud/user")
                if not isinstance(identity, dict) or not isinstance(identity.get("id"), str) or not identity["id"]:
                    raise ValueError
            if pending["expires_at"] <= time.time():
                raise SetupError("login_expired", "Login expired during authorization. Start again.", 409)
            self.store.commit_connection(INTEGRATION_ID, pending["claim"], ConnectionRecord(
                provider=INTEGRATION_ID, status="connected", auth_method="nextcloud_login_v2",
                account=identity["id"], meta={"instance_url": instance, "display_name": identity.get("display-name", identity.get("displayname", identity["id"])),
                                              "verified_at": time.time()},
                secret={"login_name": user, "app_password": password, "login_browser_hash": digest(browser),
                        "login_expires_at": pending["expires_at"]},
            ))
        except SetupError:
            raise
        except IntegrationError:
            raise SetupError("login_superseded", "This login was replaced or disconnected. Start again.", 409) from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise SetupError("login_poll_failed", "Could not verify Nextcloud authorization. Start a new login.", 502) from None
        return {"status": "connected"}

    def _ocs(self, http, instance, user, password, path, *, admin=False):
        response = http.get(instance + path, auth=(user, password),
                            headers={"OCS-APIRequest": "true", "Accept": "application/json"}, params={"format": "json"})
        if response.status_code == 401:
            raise SetupError("account_rejected", "Nextcloud rejected the app password. Reconnect the account.", 401)
        if response.status_code == 403:
            raise SetupError("admin_required" if admin else "access_denied",
                             "Installed bot discovery requires a Nextcloud administrator account." if admin else "This account cannot access Talk rooms.", 403)
        if response.status_code != 200:
            raise SetupError("nextcloud_unavailable", "Nextcloud could not complete discovery. Check server connectivity and Talk installation.", 502)
        body = response.json()
        ocs = body.get("ocs", {}) if isinstance(body, dict) else {}
        meta = ocs.get("meta") if isinstance(ocs, dict) else None
        code = meta.get("statuscode") if isinstance(meta, dict) else None
        if code in (401, 997):
            raise SetupError("account_rejected", "Nextcloud rejected the app password. Reconnect the account.", 401)
        if code == 403:
            raise SetupError("admin_required" if admin else "access_denied",
                             "Installed bot discovery requires a Nextcloud administrator account." if admin else "This account cannot access Talk rooms.", 403)
        if code not in (100, 200):
            raise SetupError("invalid_nextcloud_response", "Nextcloud returned an unsuccessful discovery response.", 502)
        return ocs.get("data")

    def _discover(self, path: str, *, admin=False):
        rec = self.store.load(INTEGRATION_ID)
        if not rec or rec.status != "connected" or not rec.secret.get("app_password"):
            raise SetupError("account_not_connected", "Authorize a Nextcloud account first.", 409)
        try:
            instance = self._instance(rec.meta["instance_url"])
            with self._client() as http:
                return self._ocs(http, instance, rec.secret["login_name"], rec.secret["app_password"], path, admin=admin)
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            raise SetupError("nextcloud_unavailable", "Unable to read Nextcloud. Check connectivity, then reconnect if needed.", 502) from None

    def rooms(self) -> dict:
        data = self._discover("/ocs/v2.php/apps/spreed/api/v4/room")
        if not isinstance(data, list):
            raise SetupError("invalid_nextcloud_response", "Nextcloud returned an invalid room list.", 502)
        return {"rooms": [{"token": room["token"], "name": room.get("displayName") or room.get("name") or room["token"],
                           "can_moderate": room.get("participantType") in (1, 2)}
                          for room in data if isinstance(room, dict) and isinstance(room.get("token"), str)]}

    def bots(self) -> dict:
        data = self._discover("/ocs/v2.php/apps/spreed/api/v1/bot/admin", admin=True)
        if not isinstance(data, list):
            raise SetupError("invalid_nextcloud_response", "Nextcloud returned an invalid bot list.", 502)
        # Deliberate allowlist: never relay secret/signature fields even if upstream changes.
        return {"bots": [{"id": bot["id"], "name": bot.get("name", ""),
                          "url": self._public_bot_url(bot.get("url", "")), "state": bot.get("state")}
                         for bot in data if isinstance(bot, dict) and isinstance(bot.get("id"), int)],
                "provisioning": self.provisioning()}

    @staticmethod
    def _public_bot_url(value) -> str:
        try:
            p = urlsplit(value)
            if not p.hostname or p.scheme not in ("http", "https"):
                return ""
            return urlunsplit((p.scheme, p.hostname + (f":{p.port}" if p.port else ""), p.path, "", ""))
        except (ValueError, TypeError):
            return ""

    def disconnect(self) -> dict:
        self.store.delete(f"pending:{INTEGRATION_ID}")
        self.store.delete(INTEGRATION_ID)
        return {"disconnected": True, "remote_rooms_preserved": True}
