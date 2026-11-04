"""GitHub provider adapter — App/device OAuth, or an encrypted tenant PAT.

App deployments use a GitHub App, authorized by **device flow** (no callback
URL needed). Tenants without App configuration connect a personal access token
through the existing API-key form. Both use encrypted CredentialStore records;
only the app holds the credentials. App configuration always wins over a PAT.
The App user approves
once at ``github.com/login/device``; we then read their App installations, let
them pick repos, and mint **short-lived installation tokens** on demand for git
over HTTPS. App configuration is app-side only, encrypted in the existing
credential store under ``github-app`` (separate from the user's connection).
Vault remains the operator's canonical source; this store is the runtime copy.

Legacy env fallback, only when no stored App configuration exists:
  * ``GITHUB_APP_CLIENT_ID``   — OAuth client id (device flow)
  * ``GITHUB_APP_ID``          — numeric App id (JWT ``iss``)
  * ``GITHUB_APP_PRIVATE_KEY`` — PEM contents  (or ``GITHUB_APP_PRIVATE_KEY_PATH``)

Network calls are sync (httpx) so the adapter stays testable without an event
loop; the async router invokes them via ``run_in_threadpool``.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import httpx

from .base import (
    AUTH_API_KEY,
    AUTH_DEVICE_OAUTH,
    HEALTH_BROKEN,
    STATUS_CONNECTED,
    ConnectionRecord,
    DeviceFlowStart,
    DevicePollResult,
    ValidateResult,
    health_block,
)
from .api_key import ApiKeyAdapter
from .github_deploy import _GitHubTokenAdapter

logger = logging.getLogger(__name__)

_DEVICE_CODE_URL = "https://github.com/login/device/code"
_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_API = "https://api.github.com"
_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
# Scope the user token needs to enumerate installations / repos.
_SCOPE = "read:user"
_UA = "llm-bawt-provider-connect"
_TIMEOUT = 15.0
_APP_CONFIG_PROVIDER = "github-app"


class GitHubConfigError(RuntimeError):
    """Raised when GitHub App configuration is missing or unusable."""


def parse_repository(repository: str) -> tuple[str, str]:
    """Accept only an OWNER/REPO slug, never a URL or path/query fragment."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}", repository):
        raise ValueError("repository must be OWNER/REPO")
    owner, repo = repository.split("/")
    if repo in {".", ".."}:
        raise ValueError("repository must be OWNER/REPO")
    return owner, repo


def _client_id() -> str:
    cid = os.getenv("GITHUB_APP_CLIENT_ID")
    if not cid:
        raise GitHubConfigError(
            "GITHUB_APP_CLIENT_ID not set — register the bawthub GitHub App and "
            "provide its client id."
        )
    return cid


def _app_id() -> str:
    aid = os.getenv("GITHUB_APP_ID")
    if not aid:
        raise GitHubConfigError("GITHUB_APP_ID not set (numeric App id for JWT).")
    return aid


def _private_key() -> str:
    pem = os.getenv("GITHUB_APP_PRIVATE_KEY")
    if pem:
        return pem.replace("\\n", "\n")
    path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read()
    raise GitHubConfigError(
        "GITHUB_APP_PRIVATE_KEY (or GITHUB_APP_PRIVATE_KEY_PATH) not set — needed "
        "to mint installation tokens."
    )


def _headers(token: str | None = None, *, bearer: bool = False) -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": _UA}
    if token:
        h["Authorization"] = f"{'Bearer' if bearer else 'token'} {token}"
    return h


def _app_jwt(app_id: str | None = None, private_key: str | None = None) -> str:
    """Signed RS256 JWT for App-level API calls (short-lived, <=10min)."""
    import jwt  # lazy — pyjwt[crypto]

    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 9 * 60, "iss": app_id or _app_id()}
    return jwt.encode(payload, private_key or _private_key(), algorithm="RS256")


class GitHubNotConnectedError(RuntimeError):
    """No usable tenant GitHub credential is connected."""


class GitHubAdapter(ApiKeyAdapter):
    id = "github"
    label = "GitHub"

    def uses_app(self) -> bool:
        """Configuration presence selects App mode, even when it is broken.

        Never bypass a configured App's repository restrictions with a PAT.
        """
        return self.store.load(_APP_CONFIG_PROVIDER) is not None or any(
            os.getenv(name) for name in (
                "GITHUB_APP_CLIENT_ID", "GITHUB_APP_ID",
                "GITHUB_APP_PRIVATE_KEY", "GITHUB_APP_PRIVATE_KEY_PATH",
            )
        )

    @property
    def auth_methods(self) -> tuple[str, ...]:
        return (AUTH_DEVICE_OAUTH,) if self.uses_app() else (AUTH_API_KEY,)

    def descriptor(self) -> dict[str, Any]:
        return {
            **super().descriptor(),
            "category": "git",
            "credential_label": "GitHub personal access token",
            "credential_help": (
                "Use a fine-grained token restricted to your repositories with "
                "Contents: Read for cloning, or Read and write for pushing. "
                "The token keeps its selected permissions; it is not narrowed per request. "
                "Stored encrypted; never copied into the bridge environment or files."
            ),
            "setup_url": "https://github.com/settings/personal-access-tokens/new",
        }

    def _probe(self, key: str) -> tuple[str | None, str | None]:
        return _GitHubTokenAdapter._probe(key)

    def set_api_key(self, api_key: str) -> ValidateResult:
        if self.uses_app():
            return ValidateResult(ok=False, detail="This deployment uses a GitHub App, not a personal access token")
        key = (api_key or "").strip()
        if not key or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in key):
            return ValidateResult(ok=False, detail="Enter a GitHub personal access token without whitespace")
        if self.store.durable_key_problem() is not None:
            return ValidateResult(ok=False, detail="Persistent provider encryption unavailable")
        return super().set_api_key(key)

    def git_credential_token(self, *, repository: str | None = None) -> str:
        if repository is not None:
            parse_repository(repository)
        if self.uses_app():
            if repository is None:
                return self.mint_installation_token()
            return self.mint_installation_token(repository=repository)
        record = self.store.load(self.id)
        token = record.secret.get("api_key") if record else None
        if (
            not record or record.status != STATUS_CONNECTED or record.auth_method != AUTH_API_KEY
            or not isinstance(token, str) or not token
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in token)
        ):
            raise GitHubNotConnectedError("GitHub not connected — connect GitHub in setup or provider settings")
        # GitHub enforces the PAT's repository/Contents permissions. Unlike an
        # App installation token it cannot be narrowed per request.
        return token

    def is_configured(self) -> bool:
        if not self.uses_app():
            return True  # PAT entry needs no operator App configuration.
        try:
            self._device_client_id()
            return True
        except GitHubConfigError:
            return False

    def health(self) -> dict[str, Any]:
        record = self.store.load(self.id)
        if record and record.auth_method == AUTH_API_KEY and self.uses_app():
            return health_block(
                HEALTH_BROKEN, detail="PAT inactive while a GitHub App is configured; connect the App",
                fix="reconnect",
            )
        if record and record.status == STATUS_CONNECTED and record.auth_method == AUTH_API_KEY:
            token = record.secret.get("api_key")
            if not isinstance(token, str) or not token:
                return health_block(HEALTH_BROKEN, detail="GitHub token unreadable; reconnect", fix="reconnect")
        return super().health()

    def configure_app(self, app_id: str, client_id: str, private_key: str) -> None:
        """Install a complete runtime copy, without touching OAuth/selection state.

        Local structural validation only: no upstream call and no key readback.
        All fields live in ``secret``, never public metadata.
        """
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

        app_id, client_id = app_id.strip(), client_id.strip()
        private_key = private_key.replace("\\n", "\n").strip()
        if not re.fullmatch(r"[1-9][0-9]*", app_id) or not re.fullmatch(r"[A-Za-z0-9_.-]+", client_id):
            raise ValueError("Invalid GitHub App IDs")
        try:
            key = serialization.load_pem_private_key(private_key.encode(), password=None)
        except (ValueError, TypeError):
            raise ValueError("Invalid GitHub App RSA private key") from None
        if not isinstance(key, RSAPrivateKey) or key.key_size < 2048:
            raise ValueError("GitHub App key must be RSA, at least 2048 bits")
        if self.store.durable_key_problem() is not None:
            raise GitHubConfigError("Persistent provider encryption unavailable")
        self.store.save(ConnectionRecord(
            provider=_APP_CONFIG_PROVIDER,
            status=STATUS_CONNECTED,
            auth_method="app",
            secret={"app_id": app_id, "client_id": client_id, "private_key": private_key},
        ))

    def _stored_app(self) -> dict[str, str] | None:
        record = self.store.load(_APP_CONFIG_PROVIDER)
        if record is None:
            return None
        fields = ("app_id", "client_id", "private_key")
        if record.status != STATUS_CONNECTED or any(
            not isinstance(record.secret.get(field), str) or not record.secret[field].strip()
            for field in fields
        ):
            # Never silently mix a broken stored configuration with env credentials.
            raise GitHubConfigError("Stored GitHub App configuration is unavailable")
        return {field: record.secret[field] for field in fields}

    def _device_client_id(self) -> str:
        stored = self._stored_app()
        return stored["client_id"] if stored is not None else _client_id()

    def _installation_jwt(self) -> str:
        stored = self._stored_app()
        if stored is None:
            return _app_jwt()
        return _app_jwt(stored["app_id"], stored["private_key"])

    # --- device flow --------------------------------------------------------
    def start_device_flow(self) -> DeviceFlowStart:
        resp = httpx.post(
            _DEVICE_CODE_URL,
            data={"client_id": self._device_client_id(), "scope": _SCOPE},
            headers={"Accept": "application/json", "User-Agent": _UA},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        d = resp.json()
        return DeviceFlowStart(
            user_code=d["user_code"],
            verification_uri=d.get("verification_uri", "https://github.com/login/device"),
            device_code=d["device_code"],
            interval=int(d.get("interval", 5)),
            expires_in=int(d.get("expires_in", 900)),
        )

    def poll_device_flow(self, device_code: str, user_code: str = "") -> DevicePollResult:
        resp = httpx.post(
            _ACCESS_TOKEN_URL,
            data={
                "client_id": self._device_client_id(),
                "device_code": device_code,
                "grant_type": _DEVICE_GRANT,
            },
            headers={"Accept": "application/json", "User-Agent": _UA},
            timeout=_TIMEOUT,
        )
        d = resp.json() if resp.content else {}
        err = d.get("error")
        if err == "authorization_pending":
            return DevicePollResult(status="pending")
        if err == "slow_down":
            return DevicePollResult(status="slow_down")
        if err in ("expired_token", "incorrect_device_code"):
            return DevicePollResult(status="expired", detail=err)
        if err == "access_denied":
            return DevicePollResult(status="denied", detail=err)
        if err:
            return DevicePollResult(status="error", detail=str(d.get("error_description") or err))

        token = d.get("access_token")
        if not token:
            return DevicePollResult(status="error", detail="no access_token in response")

        record = self._build_record(token)
        self.store.save(record)
        return DevicePollResult(status="authorized", record=record)

    # --- record assembly ----------------------------------------------------
    def _build_record(self, user_token: str) -> ConnectionRecord:
        account = None
        installations: list[dict[str, Any]] = []
        try:
            u = httpx.get(f"{_API}/user", headers=_headers(user_token), timeout=_TIMEOUT)
            if u.is_success:
                account = u.json().get("login")
            inst = httpx.get(
                f"{_API}/user/installations",
                headers=_headers(user_token),
                timeout=_TIMEOUT,
            )
            if inst.is_success:
                installations = inst.json().get("installations", [])
        except Exception as e:  # noqa: BLE001 — connection still recorded
            logger.warning("GitHub post-auth enrichment failed: %s", e)

        meta: dict[str, Any] = {
            "installations": [
                {
                    "id": i.get("id"),
                    "account": (i.get("account") or {}).get("login"),
                }
                for i in installations
            ],
        }
        # Auto-select when exactly one installation exists (common case).
        if len(installations) == 1:
            meta["installation_id"] = installations[0].get("id")

        return ConnectionRecord(
            provider=self.id,
            status=STATUS_CONNECTED,
            auth_method=AUTH_DEVICE_OAUTH,
            account=account,
            meta=meta,
            secret={"user_token": user_token},
        )

    # --- installation token (for git credential helper) ---------------------
    def mint_installation_token(
        self, record: ConnectionRecord | None = None, *, repository: str | None = None,
    ) -> str:
        """Mint a fresh token; optional repository selection is request-local only.

        Repository requests use App authority directly, independent of the device
        OAuth connection/global selection. GitHub checks the App installation's
        actual repository access; the minted token is narrowed to that repo.
        """
        token_options: dict[str, Any] = {}
        if repository is not None:
            owner, repo = parse_repository(repository)
            headers = _headers(self._installation_jwt(), bearer=True)
            installation = httpx.get(
                f"{_API}/repos/{owner}/{repo}/installation",
                headers=headers,
                timeout=_TIMEOUT,
            )
            installation.raise_for_status()
            installation_id = installation.json()["id"]
            if type(installation_id) is not int or installation_id <= 0:
                raise RuntimeError("Invalid GitHub installation response")
            token_options["json"] = {"repositories": [repo]}
        else:
            # Omitted query preserves the original connection/selection contract.
            rec = record or self.store.load(self.id)
            if rec is None or rec.status != STATUS_CONNECTED:
                raise RuntimeError("GitHub is not connected")
            installation_id = rec.meta.get("installation_id")
            if not installation_id:
                raise RuntimeError(
                    "No GitHub App installation selected — pick repos/installation first"
                )
            headers = _headers(self._installation_jwt(), bearer=True)
        resp = httpx.post(
            f"{_API}/app/installations/{installation_id}/access_tokens",
            headers=headers,
            timeout=_TIMEOUT,
            **token_options,
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        if not isinstance(token, str) or not token or any(char.isspace() for char in token):
            raise RuntimeError("Invalid GitHub token response")
        return token

    def list_installations(self) -> list[dict[str, Any]]:
        rec = self.store.load(self.id)
        if rec is None:
            return []
        return rec.meta.get("installations", [])

    def select_installation(self, installation_id: int, repos: list[str] | None = None) -> ConnectionRecord:
        rec = self.store.load(self.id)
        if rec is None:
            raise RuntimeError("GitHub is not connected")
        rec.meta["installation_id"] = installation_id
        if repos is not None:
            rec.meta["repos"] = repos
        self.store.save(rec)
        return rec

    def validate(self, record: ConnectionRecord) -> ValidateResult:
        if record.auth_method == AUTH_API_KEY:
            return super().validate(record)
        token = record.secret.get("user_token")
        if not token:
            return ValidateResult(ok=False, detail="no stored token")
        try:
            u = httpx.get(f"{_API}/user", headers=_headers(token), timeout=_TIMEOUT)
            if u.is_success:
                return ValidateResult(ok=True, account=u.json().get("login"))
            return ValidateResult(ok=False, detail=f"github /user -> {u.status_code}")
        except Exception as e:  # noqa: BLE001
            return ValidateResult(ok=False, detail=str(e))
