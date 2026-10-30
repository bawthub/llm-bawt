"""Fixed-route, workspace-free credential gateway for an isolated Preview bridge.

Deploy only in a pinned image with a read-only capability volume and a network
rule allowing this process (not Preview code) to reach the production broker.
The untrusted bridge sees this gateway, never the bearer file or broker origin.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import httpx

CAPABILITY_FILE_ENV = "PREVIEW_CREDENTIAL_CAPABILITY_FILE"
BROKER_ORIGIN_ENV = "PREVIEW_CREDENTIAL_BROKER_ORIGIN"
PROVIDER_ENV = "PREVIEW_CREDENTIAL_PROVIDER"
RUN_ID_ENV = "PREVIEW_CREDENTIAL_RUN_ID"
LISTEN_HOST_ENV = "PREVIEW_CREDENTIAL_LISTEN_HOST"
LISTEN_PORT_ENV = "PREVIEW_CREDENTIAL_LISTEN_PORT"
BROKER_PATH = "/v1/preview/credential"
PROVIDERS = frozenset({"claude", "openai_chatgpt"})
MIN_REMAINING_SECONDS = 60
MAX_CAPABILITY_SECONDS = 600


class GatewayUnavailable(ValueError):
    """Missing, malformed or expired grant, broker response, or configuration."""


@dataclass(frozen=True)
class GatewayConfig:
    capability_file: Path
    broker_origin: str
    provider: str
    run_id: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> GatewayConfig:
        from uuid import UUID

        env = os.environ if env is None else env
        path = env.get(CAPABILITY_FILE_ENV, "")
        origin = env.get(BROKER_ORIGIN_ENV, "")
        provider = env.get(PROVIDER_ENV, "")
        run_id = env.get(RUN_ID_ENV, "")
        parsed = urlsplit(origin)
        if (not path or not Path(path).is_absolute() or provider not in PROVIDERS
                or not origin or parsed.scheme not in {"http", "https"}
                or not parsed.hostname or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment
                or origin != f"{parsed.scheme}://{parsed.netloc}"):
            raise GatewayUnavailable("Preview credential gateway configuration unavailable")
        try:
            if str(UUID(run_id)) != run_id:
                raise ValueError("invalid run")
        except (ValueError, AttributeError) as exc:
            raise GatewayUnavailable("Preview credential gateway run unavailable") from exc
        return cls(Path(path), origin, provider, run_id)


def _capability(config: GatewayConfig, now: int) -> str:
    """Read on every request, so removal/rotation refuses immediately."""
    try:
        data = json.loads(config.capability_file.read_text(encoding="utf-8"))
        if (not isinstance(data, dict)
                or set(data) != {"run_id", "provider", "endpoint_id", "expires_at", "capability"}
                or data["run_id"] != config.run_id or data["provider"] != config.provider
                or type(data["endpoint_id"]) is not int or data["endpoint_id"] <= 0
                or type(data["expires_at"]) is not int or data["expires_at"] <= now
                or not isinstance(data["capability"], str)
                or len(data["capability"]) > 1024 or data["capability"].count(".") != 1):
            raise GatewayUnavailable("Preview credential capability unavailable")
        payload_part, _ = data["capability"].split(".", 1)
        raw = base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4))
        claims = json.loads(raw)
        if (not isinstance(claims, dict)
                or set(claims) != {"version", "run_id", "provider", "endpoint_id", "expires_at"}
                or type(claims["version"]) is not int or claims["version"] != 1
                or claims["run_id"] != data["run_id"] or claims["provider"] != data["provider"]
                or claims["endpoint_id"] != data["endpoint_id"]
                or type(claims["expires_at"]) is not int
                or not now < claims["expires_at"] <= min(data["expires_at"], now + MAX_CAPABILITY_SECONDS)):
            raise GatewayUnavailable("Preview credential capability unavailable")
        return data["capability"]
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, binascii.Error) as exc:
        raise GatewayUnavailable("Preview credential capability unavailable") from exc


def _validated_response(response: httpx.Response, provider: str, now: int) -> bytes:
    if response.status_code != 200:
        raise GatewayUnavailable("Preview credential broker refused")
    try:
        data = response.json()
        required = ({"access_token", "expires_at", "state"} if provider == "claude"
                    else {"access_token", "expires_at", "state", "account_id"})
        if (not isinstance(data, dict) or set(data) != required or data["state"] != "ok"
                or not isinstance(data["access_token"], str) or not data["access_token"]
                or type(data["expires_at"]) not in (int, float)
                or not math.isfinite(data["expires_at"])):
            raise GatewayUnavailable("Preview credential broker response unavailable")
        expires_at = data["expires_at"] / 1000 if provider == "claude" else data["expires_at"]
        if (expires_at < now + MIN_REMAINING_SECONDS
                or (provider == "claude" and data["expires_at"] < 100_000_000_000)
                or (provider == "openai_chatgpt" and data["expires_at"] >= 100_000_000_000)
                or (provider == "openai_chatgpt" and (
                    not isinstance(data["account_id"], str) or not data["account_id"]))):
            raise GatewayUnavailable("Preview credential broker token unavailable")
        return json.dumps(data, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, KeyError) as exc:
        raise GatewayUnavailable("Preview credential broker response unavailable") from exc


def fetch_credential(config: GatewayConfig, *, now: int | None = None,
                     transport: httpx.BaseTransport | None = None) -> bytes:
    """One fixed broker request; never forward caller headers, path or query."""
    now = int(time.time()) if now is None else now
    capability = _capability(config, now)
    try:
        with httpx.Client(transport=transport, timeout=10, follow_redirects=False,
                          trust_env=False) as client:
            result = client.get(config.broker_origin + BROKER_PATH,
                                headers={"X-Preview-Broker-Token": capability})
    except httpx.HTTPError as exc:
        raise GatewayUnavailable("Preview credential broker unreachable") from exc
    return _validated_response(result, config.provider, now)


class CredentialHandler(BaseHTTPRequestHandler):
    config: GatewayConfig
    _rate_lock = Lock()
    _rate_bucket_start = 0.0
    _rate_count = 0
    _rate_per_minute = 120

    def _allow(self, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        cls = type(self)
        with cls._rate_lock:
            if now - cls._rate_bucket_start >= 60:
                cls._rate_bucket_start = now
                cls._rate_count = 0
            if cls._rate_count >= cls._rate_per_minute:
                return False
            cls._rate_count += 1
            return True

    def do_GET(self) -> None:
        if self.path == "/health":
            try:
                _capability(self.config, int(time.time()))
            except GatewayUnavailable:
                self._reply(503)
                return
            self._reply(200, b'{"state":"ok"}')
            return
        path = f"/v1/providers/{self.config.provider}/token"
        if self.path != path:
            self._reply(404)
            return
        if not self._allow():
            self._reply(429)
            return
        try:
            body = fetch_credential(self.config)
        except GatewayUnavailable:
            self._reply(503)
            return
        self._reply(200, body)

    def do_POST(self) -> None:
        self._reply(405)

    def _reply(self, status: int, body: bytes = b"") -> None:
        if status != 200:
            body = json.dumps({"detail": "Preview credential unavailable"}).encode("utf-8")
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        # Never log a credential path, caller header or upstream token.
        pass


def main() -> None:
    config = GatewayConfig.from_env()
    _capability(config, int(time.time()))  # refuse startup without a live grant
    host = os.environ.get(LISTEN_HOST_ENV, "127.0.0.1")
    port = int(os.environ.get(LISTEN_PORT_ENV, "8642"))
    handler = type("ConfiguredCredentialHandler", (CredentialHandler,), {"config": config})
    ThreadingHTTPServer((host, port), handler).serve_forever()


if __name__ == "__main__":
    main()
